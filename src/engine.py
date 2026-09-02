"""
Event-driven forex backtester.

Design notes
------------
* Signals are computed on a resampled "signal timeframe" (M5/M15/...), but every
  trade is walked forward bar-by-bar on the underlying M1 series. That means when
  price touches both the stop and the target inside one signal bar we know which
  came first instead of guessing. Guessing is where most optimistic backtests die.
* Costs are charged explicitly: half-spread on entry and exit, slippage, and
  commission per lot. Dukascopy raw spreads are tighter than a retail account gets,
  so the spread used is max(feed spread, floor).
* Signals fire on bar close and fill at the next bar's open -> no look-ahead.
"""
import numpy as np, pandas as pd
from dataclasses import dataclass, field

def pip_size(sym):
    return 0.01 if "JPY" in sym else 0.0001

@dataclass
class Costs:
    spread_floor_pips: float = 1.0   # realistic retail spread floor
    slippage_pips: float = 0.2       # per side
    commission_per_lot: float = 7.0  # round-turn USD per 1.0 lot

@dataclass
class Trade:
    dir: int; entry_time: pd.Timestamp; entry: float
    sl: float; tp: float
    exit_time: pd.Timestamp = None; exit: float = None
    reason: str = ""; pips: float = 0.0; usd: float = 0.0
    bars_held: int = 0; mae_pips: float = 0.0; mfe_pips: float = 0.0

def resample(m1, tf):
    o = m1.resample(tf, label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last",
         "spread": "median", "volume": "sum"})
    return o.dropna(subset=["open"])

class Backtest:
    """
    signal_fn(sig_df) -> DataFrame with columns:
        dir  : +1 long, -1 short, 0 flat (evaluated on that bar's close)
        sl   : stop-loss price for the trade
        tp   : take-profit price
    Entry happens at the open of the following signal bar.
    """
    def __init__(self, m1, symbol, tf="5min", costs=Costs(), balance=200.0,
                 lot=0.01, risk_pct=3.0, max_bars=None, one_at_a_time=True):
        self.m1, self.sym, self.tf = m1, symbol, tf
        self.costs, self.balance0, self.lot = costs, balance, lot
        self.risk_pct, self.max_bars = risk_pct, max_bars
        self.pip = pip_size(symbol)
        self.sig = resample(m1, tf)

    def _pip_value(self, price):
        """USD per pip for `self.lot` lots. USD-quoted pairs are constant;
        XXXJPY needs the current rate to convert back to USD."""
        contract = 100_000 * self.lot
        if self.sym.endswith("USD"):
            return contract * self.pip
        if self.sym.startswith("USD"):
            return contract * self.pip / price
        return contract * self.pip / price  # XXXJPY quoted in JPY

    def run(self, signal_fn):
        sig = signal_fn(self.sig.copy())
        m1 = self.m1
        m1_idx = m1.index.values
        hi, lo = m1["high"].values, m1["low"].values
        op = m1["open"].values
        sp_m1 = np.maximum(m1["spread"].values, self.costs.spread_floor_pips * self.pip)
        slip = self.costs.slippage_pips * self.pip

        sig_times = sig.index.values
        dirs = sig["dir"].fillna(0).values.astype(int)
        sls, tps = sig["sl"].values, sig["tp"].values

        trades, i_m1 = [], 0
        n_m1 = len(m1_idx)
        balance = self.balance0
        equity_curve = []
        k = 0
        while k < len(sig_times) - 1:
            d = dirs[k]
            if d == 0 or not np.isfinite(sls[k]) or not np.isfinite(tps[k]):
                k += 1; continue
            # fill at first M1 bar strictly after this signal bar closes
            j = np.searchsorted(m1_idx, sig_times[k], side="right")
            if j >= n_m1:
                break
            half = sp_m1[j] / 2
            entry = op[j] + d * (half + slip)
            sl, tp = sls[k], tps[k]
            if (d == 1 and not (sl < entry < tp)) or (d == -1 and not (tp < entry < sl)):
                k += 1; continue

            tr = Trade(dir=d, entry_time=pd.Timestamp(m1_idx[j], tz="UTC"), entry=entry,
                       sl=sl, tp=tp)
            max_m1 = n_m1 - 1 if self.max_bars is None else min(
                n_m1 - 1, j + int(self.max_bars * pd.Timedelta(self.tf) / pd.Timedelta("1min")))
            exit_px = exit_i = None
            mae = mfe = 0.0
            for i in range(j, max_m1 + 1):
                h, l = hi[i], lo[i]
                if d == 1:
                    mfe = max(mfe, h - entry); mae = min(mae, l - entry)
                    hit_sl, hit_tp = l <= sl, h >= tp
                else:
                    mfe = max(mfe, entry - l); mae = min(mae, entry - h)
                    hit_sl, hit_tp = h >= sl, l <= tp
                if hit_sl and hit_tp:
                    # both inside one M1 bar: assume the adverse level first (conservative)
                    exit_px, tr.reason, exit_i = sl, "SL", i; break
                if hit_sl:
                    exit_px, tr.reason, exit_i = sl, "SL", i; break
                if hit_tp:
                    exit_px, tr.reason, exit_i = tp, "TP", i; break
            if exit_px is None:
                exit_i = max_m1
                exit_px, tr.reason = m1["close"].values[exit_i], "TIME"

            exit_fill = exit_px - d * (sp_m1[exit_i] / 2 + slip)
            tr.exit = exit_fill
            tr.exit_time = pd.Timestamp(m1_idx[exit_i], tz="UTC")
            tr.pips = d * (exit_fill - entry) / self.pip
            tr.mae_pips = mae / self.pip
            tr.mfe_pips = mfe / self.pip
            pv = self._pip_value(entry)
            tr.usd = tr.pips * pv - self.costs.commission_per_lot * self.lot
            balance += tr.usd
            equity_curve.append((tr.exit_time, balance))
            trades.append(tr)
            # no overlapping positions: resume signals after the exit
            k = int(np.searchsorted(sig_times, np.datetime64(tr.exit_time.tz_localize(None)), side="right"))
        return Result(trades, self.balance0, equity_curve, self.sym)

class Result:
    def __init__(self, trades, bal0, curve, sym):
        self.trades, self.bal0, self.sym = trades, bal0, sym
        self.df = pd.DataFrame([t.__dict__ for t in trades])
        self.curve = pd.DataFrame(curve, columns=["time", "balance"]).set_index("time") \
            if curve else pd.DataFrame(columns=["balance"])

    @property
    def stats(self):
        d = self.df
        if len(d) == 0:
            return {"trades": 0, "win_rate": 0, "pf": 0, "net_usd": 0, "expectancy_pips": 0,
                    "max_dd_pct": 0, "final_balance": self.bal0}
        wins, losses = d[d.usd > 0], d[d.usd <= 0]
        gp, gl = wins.usd.sum(), -losses.usd.sum()
        eq = self.bal0 + d.usd.cumsum()
        dd = (eq.cummax() - eq) / eq.cummax() * 100
        return {
            "trades": len(d),
            "win_rate": round(len(wins) / len(d) * 100, 2),
            "pf": round(gp / gl, 3) if gl > 0 else float("inf"),
            "net_usd": round(d.usd.sum(), 2),
            "return_pct": round(d.usd.sum() / self.bal0 * 100, 1),
            "expectancy_pips": round(d.pips.mean(), 2),
            "expectancy_usd": round(d.usd.mean(), 3),
            "avg_win_pips": round(wins.pips.mean(), 1) if len(wins) else 0,
            "avg_loss_pips": round(losses.pips.mean(), 1) if len(losses) else 0,
            "max_dd_pct": round(dd.max(), 2),
            "final_balance": round(self.bal0 + d.usd.sum(), 2),
        }

    def summary(self):
        s = self.stats
        return " | ".join(f"{k}={v}" for k, v in s.items())
