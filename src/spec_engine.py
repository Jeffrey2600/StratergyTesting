"""
Multi-timeframe trend-pullback backtester, built to the supplied specification.

Structure: 4H trend gate (EMA50/200), 1H pullback to EMA20, 1H confirmation
candle, entry on a stop order at the confirmation candle's extreme, stop at
1.5*ATR(14) and target at 2R. Portfolio-level: equity-based sizing at 2.5% risk,
broker lot rounding with a skip rule, one position per pair, capped concurrent
positions and capped per-currency exposure.

EXECUTION ASSUMPTIONS (documented as the spec requires)
------------------------------------------------------
* Bars are built from 1-minute data, labelled at their CLOSE time, so a bar
  labelled 05:00 covers (04:00, 05:00]. A 1H decision at time T reads the most
  recent 4H bar whose label is <= T, which is by construction a completed
  candle. No partially formed higher-timeframe candle is ever visible.
* Indicators are read on the confirmation candle's close; the resulting order
  goes live only afterwards.
* The stop order triggers when the mid price trades through the level, and
  fills at the worse of the level and the triggering minute's open, plus half
  the prevailing spread plus slippage. Longs enter on ask and exit on bid;
  shorts do the reverse.
* Stop-loss exits pay half spread plus slippage; take-profit exits are limit
  fills at the level on the correct side of the book.
* When a minute's range covers both the stop and the target, the stop is taken.
  That is the conservative reading and it is applied consistently.
* Sizing uses equity at the moment of the fill, never final or peak equity.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from dataclasses import dataclass, field
import numpy as np, pandas as pd
from engine import resample, pip_size

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"]


@dataclass
class Params:
    tf_high: str = "240min"
    tf_exec: str = "60min"
    ema_fast: int = 50            # 4H trend
    ema_slow: int = 200           # 4H trend
    ema_pull: int = 20            # 1H pullback reference
    atr_len: int = 14
    sl_atr: float = 1.5
    tp_r: float = 2.0
    pullback_window: int = 5      # bars the pullback stays "armed"
    order_ttl: int = 1            # 1H bars the stop order rests before expiring
    risk_pct: float = 2.5
    max_positions: int = 3
    max_per_currency: int = 2
    min_lot: float = 0.01
    lot_step: float = 0.01
    max_lot: float = 100.0
    spread_floor_pips: float = 1.0
    max_spread_pips: float = 3.0
    slippage_pips: float = 0.2
    commission_per_lot: float = 7.0   # round turn, per 1.0 lot
    balance: float = 200.0
    fixed_lot: float = 0.0        # >0 pins lot size, for isolating signal quality


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def atr(df, n):
    pc = df.close.shift()
    tr = pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()],
                   axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def signals(sym, m1, p: Params):
    """Confirmation-candle signals for one pair, with no look-ahead."""
    h4 = resample(m1, p.tf_high)
    h1 = resample(m1, p.tf_exec)
    if len(h4) < p.ema_slow + 5 or len(h1) < 60:
        return pd.DataFrame()

    ef, es = ema(h4.close, p.ema_fast), ema(h4.close, p.ema_slow)
    bull4 = (ef > es) & (h4.close > es)
    bear4 = (ef < es) & (h4.close < es)
    trend = pd.Series(np.where(bull4, 1, np.where(bear4, -1, 0)), index=h4.index)
    # asof-join the last COMPLETED 4H candle onto each 1H bar
    tr1 = trend.reindex(trend.index.union(h1.index)).ffill().reindex(h1.index)

    e20 = ema(h1.close, p.ema_pull)
    a14 = atr(h1, p.atr_len)
    hi, lo = h1.high.values, h1.low.values
    op, cl = h1.open.values, h1.close.values
    e20v, av, tv = e20.values, a14.values, tr1.fillna(0).values
    idx = h1.index

    out, armed_long, armed_short = [], -99, -99
    for i in range(1, len(h1)):
        t = tv[i]
        if not np.isfinite(av[i]) or av[i] <= 0:
            continue
        if t == 1:
            armed_short = -99
            if lo[i] <= e20v[i]:
                armed_long = i
            if (i - armed_long) < p.pullback_window and cl[i] > op[i] and cl[i] > hi[i - 1]:
                out.append((idx[i], 1, hi[i], av[i]))
                armed_long = -99           # one entry per pullback
        elif t == -1:
            armed_long = -99
            if hi[i] >= e20v[i]:
                armed_short = i
            if (i - armed_short) < p.pullback_window and cl[i] < op[i] and cl[i] < lo[i - 1]:
                out.append((idx[i], -1, lo[i], av[i]))
                armed_short = -99
        else:
            armed_long = armed_short = -99
    return pd.DataFrame(out, columns=["t_signal", "dir", "trigger", "atr"])


def simulate_price(sym, m1, sig, p: Params):
    """Resolve each signal on 1-minute data: does it fill, and where does it exit?

    Purely price-driven, so it is independent of position size - which lets the
    portfolio pass size trades with the equity known at each fill time.
    """
    pip = pip_size(sym)
    idx = m1.index.values
    hi, lo, op, cl = (m1.high.values, m1.low.values, m1.open.values, m1.close.values)
    sp = np.maximum(m1.spread.values, p.spread_floor_pips * pip)
    slip = p.slippage_pips * pip
    ttl = pd.Timedelta(p.tf_exec) * p.order_ttl
    rows = []
    for t_sig, d, trigger, a in sig.itertuples(index=False):
        j0 = int(np.searchsorted(idx, np.datetime64(t_sig.tz_localize(None)), side="right"))
        j1 = int(np.searchsorted(idx, np.datetime64((t_sig + ttl).tz_localize(None)), side="right"))
        fill = None
        for j in range(j0, min(j1, len(idx))):
            if (d == 1 and hi[j] >= trigger) or (d == -1 and lo[j] <= trigger):
                if sp[j] > p.max_spread_pips * pip:
                    break                      # spread filter, per spec
                mid = max(trigger, op[j]) if d == 1 else min(trigger, op[j])
                fill = (j, mid + d * (sp[j] / 2 + slip), sp[j])
                break
        if fill is None:
            continue
        j, entry, sp_in = fill
        risk = p.sl_atr * a
        sl = entry - d * risk
        tp = entry + d * risk * p.tp_r

        ex_i, ex_px, reason, sp_out = None, None, "END", sp[-1]
        for k in range(j, len(idx)):
            half = sp[k] / 2
            bid_h, bid_l = hi[k] - half, lo[k] - half
            ask_h, ask_l = hi[k] + half, lo[k] + half
            if d == 1:
                hit_sl, hit_tp = bid_l <= sl, bid_h >= tp
            else:
                hit_sl, hit_tp = ask_h >= sl, ask_l <= tp
            if hit_sl:
                ex_i, ex_px, reason, sp_out = k, sl - d * slip, "SL", sp[k]
                break
            if hit_tp:
                ex_i, ex_px, reason, sp_out = k, tp, "TP", sp[k]
                break
        if ex_i is None:
            ex_i = len(idx) - 1
            ex_px = cl[ex_i] - d * (sp[ex_i] / 2 + slip)
            reason, sp_out = "END", sp[ex_i]

        rows.append(dict(
            pair=sym, dir=d, t_signal=t_sig,
            t_in=pd.Timestamp(idx[j], tz="UTC"), t_out=pd.Timestamp(idx[ex_i], tz="UTC"),
            entry=entry, sl=sl, tp=tp, exit=ex_px, reason=reason,
            stop_pips=risk / pip, pips=d * (ex_px - entry) / pip,
            spread_px=(sp_in / 2 + sp_out / 2), slip_px=2 * slip,
            entry_ref=entry))
    return pd.DataFrame(rows)


def pip_value(sym, price, lots, usdjpy=None):
    """USD value of one pip for `lots` lots."""
    c = 100_000 * lots
    if sym.endswith("USD"):
        return c * pip_size(sym)
    if sym.startswith("USD"):
        return c * pip_size(sym) / price
    return c * pip_size(sym) / (usdjpy or 150.0)


def size_position(sym, entry, stop_pips, equity, p: Params):
    """Lots for a 2.5%-of-equity risk, after broker rounding.

    Returns (lots, actual_risk_pct, skip_reason). Rounding is always DOWN, so
    only the minimum-lot floor can push risk above the cap - and that is exactly
    the case the spec says to skip rather than quietly accept.
    """
    per_lot = stop_pips * pip_value(sym, entry, 1.0)
    if per_lot <= 0:
        return 0.0, 0.0, "bad-stop"
    if p.fixed_lot > 0:
        return p.fixed_lot, p.fixed_lot * per_lot / equity * 100.0, None
    risk_usd = equity * p.risk_pct / 100.0
    raw = risk_usd / per_lot
    lots = np.floor(raw / p.lot_step) * p.lot_step
    if lots < p.min_lot:
        lots = p.min_lot
    lots = min(lots, p.max_lot)
    actual = lots * per_lot / equity * 100.0
    if actual > p.risk_pct + 1e-9:
        return lots, actual, "risk-too-big"
    return round(lots, 2), actual, None


def run_portfolio(data, p: Params, pairs=PAIRS):
    """Chronological portfolio pass: size and admit trades under the live limits."""
    cand = []
    for s in pairs:
        sig = signals(s, data[s], p)
        if len(sig):
            cand.append(simulate_price(s, data[s], sig, p))
    if not cand:
        return pd.DataFrame(), pd.DataFrame()
    c = pd.concat(cand, ignore_index=True).sort_values("t_in").reset_index(drop=True)

    usdjpy = resample(data["USDJPY"], "60min").close if "USDJPY" in data else None
    equity = p.balance
    open_pos, taken, skipped = [], [], {"limit": 0, "pair": 0, "ccy": 0, "risk": 0}

    for r in c.itertuples(index=False):
        # settle anything that closed before this fill
        still = []
        for q in open_pos:
            if q["t_out"] <= r.t_in:
                equity += q["pnl"]
                taken.append(q)
            else:
                still.append(q)
        open_pos = still

        if len(open_pos) >= p.max_positions:
            skipped["limit"] += 1; continue
        if any(q["pair"] == r.pair for q in open_pos):
            skipped["pair"] += 1; continue
        base, quote = r.pair[:3], r.pair[3:]
        exp = {}
        for q in open_pos:
            exp[q["pair"][:3]] = exp.get(q["pair"][:3], 0) + q["dir"]
            exp[q["pair"][3:]] = exp.get(q["pair"][3:], 0) - q["dir"]
        if (abs(exp.get(base, 0) + r.dir) > p.max_per_currency or
                abs(exp.get(quote, 0) - r.dir) > p.max_per_currency):
            skipped["ccy"] += 1; continue

        uj = float(usdjpy.asof(r.t_in)) if usdjpy is not None else 150.0
        lots, actual_risk, why = size_position(r.pair, r.entry, r.stop_pips, equity, p)
        if why:
            skipped["risk"] += 1; continue

        pv = pip_value(r.pair, r.entry, lots, uj)
        gross_pips = r.pips
        spread_cost = r.spread_px / pip_size(r.pair) * pv
        slip_cost = r.slip_px / pip_size(r.pair) * pv
        comm = p.commission_per_lot * lots
        pnl = gross_pips * pv - comm
        open_pos.append(dict(
            pair=r.pair, dir=r.dir, t_in=r.t_in, t_out=r.t_out, entry=r.entry,
            sl=r.sl, tp=r.tp, exit=r.exit, reason=r.reason, lots=lots,
            risk_pct=actual_risk, risk_usd=equity * p.risk_pct / 100,
            pips=gross_pips, pnl=pnl, spread_cost=spread_cost, stop_pips=r.stop_pips,
            slip_cost=slip_cost, commission=comm,
            equity_at_entry=equity,
            hold_h=(r.t_out - r.t_in).total_seconds() / 3600))

    for q in sorted(open_pos, key=lambda x: x["t_out"]):
        equity += q["pnl"]; taken.append(q)
    tr = pd.DataFrame(taken).sort_values("t_out").reset_index(drop=True)
    return tr, pd.Series(skipped)


def metrics(tr, p: Params):
    if len(tr) == 0:
        return {}
    eq = p.balance + tr.pnl.cumsum()
    peak = eq.cummax()
    dd = peak - eq
    wins, losses = tr[tr.pnl > 0], tr[tr.pnl <= 0]
    days = max((tr.t_out.iloc[-1] - tr.t_out.iloc[0]).days, 1)
    end = p.balance + tr.pnl.sum()
    daily = tr.set_index("t_out").pnl.resample("1D").sum()
    de = p.balance + daily.cumsum()
    dr = de.pct_change().dropna()
    sharpe = dr.mean() / dr.std() * np.sqrt(252) if len(dr) > 2 and dr.std() > 0 else 0
    downside = dr[dr < 0]
    sortino = (dr.mean() / downside.std() * np.sqrt(252)
               if len(downside) > 2 and downside.std() > 0 else 0)

    def streak(mask):
        best = cur = 0
        for v in mask:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        return best

    gp, gl = wins.pnl.sum(), -losses.pnl.sum()
    costs = tr.spread_cost.sum() + tr.slip_cost.sum() + tr.commission.sum()
    return dict(
        start=p.balance, end=round(end, 2), net=round(tr.pnl.sum(), 2),
        ret_pct=round(tr.pnl.sum() / p.balance * 100, 2),
        cagr=round(((max(end, 0.01) / p.balance) ** (365 / days) - 1) * 100, 2),
        trades=len(tr), win_rate=round(len(wins) / len(tr) * 100, 2),
        pf=round(gp / gl, 3) if gl > 0 else np.inf,
        expectancy=round(tr.pnl.mean(), 3),
        avg_win=round(wins.pnl.mean(), 2) if len(wins) else 0,
        avg_loss=round(losses.pnl.mean(), 2) if len(losses) else 0,
        largest_win=round(tr.pnl.max(), 2), largest_loss=round(tr.pnl.min(), 2),
        max_dd=round(dd.max(), 2), max_dd_pct=round((dd / peak.clip(lower=1e-9)).max() * 100, 2),
        sharpe=round(sharpe, 2), sortino=round(sortino, 2),
        recovery=round(tr.pnl.sum() / dd.max(), 2) if dd.max() > 0 else np.inf,
        max_cons_win=streak(tr.pnl > 0), max_cons_loss=streak(tr.pnl <= 0),
        avg_hold_h=round(tr.hold_h.mean(), 1),
        spread_cost=round(tr.spread_cost.sum(), 2),
        commission=round(tr.commission.sum(), 2),
        slippage_cost=round(tr.slip_cost.sum(), 2),
        total_costs=round(costs, 2),
        gross_before_costs=round(tr.pnl.sum() + costs, 2),
    )


def load(pairs=PAIRS, start=None, end=None):
    out = {}
    for s in pairs:
        d = pd.read_parquet(f"data/{s}_M1.parquet")
        if start:
            d = d.loc[start:end]
        out[s] = d
    return out
