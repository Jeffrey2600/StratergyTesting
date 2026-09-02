"""
Backtester with scaled exits and breakeven stops.

Same M1-resolved execution as engine.py, but a trade can be closed in pieces at
several targets, and the stop can be pulled to breakeven once the first target
pays. That is what the original HRK KN idea was reaching for with TP1/TP2/TP3,
and it is also the honest way to lift a win rate: banking part of a move is a
real gain, not a redefinition of one.
"""
import numpy as np, pandas as pd
from dataclasses import dataclass
from engine import pip_size, Costs, resample

@dataclass
class ExitPlan:
    rr: tuple = (1.0, 2.0, 3.0)      # targets in R multiples
    frac: tuple = (0.5, 0.3, 0.2)    # fraction of the position closed at each
    breakeven_after: int = 1         # move stop to entry after target N (0 = off)
    trail_atr: float = 0.0           # if > 0, trail the stop by N*ATR
    max_bars: int = 200              # give up after this many signal bars

class Backtest2:
    def __init__(self, m1, symbol, tf="5min", costs=Costs(), balance=200.0,
                 lot=0.01, plan=ExitPlan()):
        self.m1, self.sym, self.tf = m1, symbol, tf
        self.costs, self.balance0, self.lot, self.plan = costs, balance, lot, plan
        self.pip = pip_size(symbol)
        self.sig = resample(m1, tf)

    def _pip_value(self, price):
        contract = 100_000 * self.lot
        if self.sym.endswith("USD"):
            return contract * self.pip
        return contract * self.pip / price

    def run(self, signal_fn):
        sig = signal_fn(self.sig.copy())
        m1 = self.m1
        idx = m1.index.values
        hi, lo, cl = m1["high"].values, m1["low"].values, m1["close"].values
        op = m1["open"].values
        sp = np.maximum(m1["spread"].values, self.costs.spread_floor_pips * self.pip)
        slip = self.costs.slippage_pips * self.pip
        P = self.plan
        bars_per_sig = int(pd.Timedelta(self.tf) / pd.Timedelta("1min"))

        st = sig.index.values
        dirs = sig["dir"].fillna(0).values.astype(int)
        sls, tps = sig["sl"].values, sig["tp"].values
        trades, k, n = [], 0, len(idx)

        while k < len(st) - 1:
            d = dirs[k]
            if d == 0 or not np.isfinite(sls[k]):
                k += 1; continue
            j = int(np.searchsorted(idx, st[k], side="right"))
            if j >= n: break
            entry = op[j] + d * (sp[j] / 2 + slip)
            sl0 = sls[k]
            risk = abs(entry - sl0)
            if risk <= 0 or not np.isfinite(risk):
                k += 1; continue
            targets = [entry + d * risk * r for r in P.rr]
            fracs = list(P.frac)
            stop = sl0
            remaining = 1.0
            realised_pips = 0.0
            hits = 0
            last = min(n - 1, j + P.max_bars * bars_per_sig)
            exit_i, reason = None, "TIME"

            for i in range(j, last + 1):
                h, l = hi[i], lo[i]
                stop_hit = (l <= stop) if d == 1 else (h >= stop)
                if stop_hit:
                    px = stop - d * (sp[i] / 2 + slip)
                    realised_pips += remaining * d * (px - entry) / self.pip
                    remaining = 0.0
                    exit_i, reason = i, ("BE" if hits and abs(stop - entry) < 1e-12
                                         else ("TRAIL" if hits else "SL"))
                    break
                # targets are checked after the stop: if both sit inside one M1
                # bar we take the adverse one, which is the conservative read.
                while hits < len(targets):
                    t = targets[hits]
                    if (h >= t) if d == 1 else (l <= t):
                        px = t - d * (sp[i] / 2 + slip)
                        realised_pips += fracs[hits] * d * (px - entry) / self.pip
                        remaining -= fracs[hits]
                        hits += 1
                        if P.breakeven_after and hits >= P.breakeven_after:
                            stop = entry if d == 1 else entry
                    else:
                        break
                if remaining <= 1e-9:
                    exit_i, reason = i, "TP%d" % hits
                    break
            if exit_i is None:
                exit_i = last
                px = cl[last] - d * (sp[last] / 2 + slip)
                realised_pips += remaining * d * (px - entry) / self.pip
                remaining = 0.0

            pv = self._pip_value(entry)
            usd = realised_pips * pv - self.costs.commission_per_lot * self.lot
            trades.append(dict(dir=d, entry_time=pd.Timestamp(idx[j], tz="UTC"),
                               exit_time=pd.Timestamp(idx[exit_i], tz="UTC"),
                               entry=entry, pips=realised_pips, usd=usd,
                               reason=reason, targets_hit=hits))
            k = int(np.searchsorted(st, idx[exit_i], side="right"))

        from engine import Result
        r = Result.__new__(Result)
        r.trades, r.bal0, r.sym = trades, self.balance0, self.sym
        r.df = pd.DataFrame(trades)
        r.curve = pd.DataFrame()
        return r
