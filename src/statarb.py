"""
Pairs mean-reversion backtest, two legs, full costs.

The premise checked out: spreads have variance ratios of 0.70-0.88 against
0.93-1.08 for the individual pairs, so they do revert more than the pairs
themselves. That is necessary, not sufficient - the question here is whether
what is left after paying spread, slippage and commission on TWO legs is
positive.

No look-ahead anywhere:
  * beta and the z-score mean/sd come from a trailing window only, never the
    full sample;
  * signals are read on a bar's close and filled at the next bar's open.

Sizing is a fixed 0.01 lot per leg, which is what a $200 account can actually
do - beta is used for the signal, not to size the hedge, since lots below 0.01
do not exist. That imperfection is real and is left in rather than assumed away.
"""
import sys, itertools
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from engine import resample, pip_size, Costs
from config import IS, OOS

TF = "60min"
LOT = 0.01

_UJ = None
def usdjpy_rate():
    """USDJPY close on the common timeframe, for converting JPY-cross P&L."""
    global _UJ
    if _UJ is None:
        _UJ = resample(pd.read_parquet("data/USDJPY_M1.parquet"), TF).close
    return _UJ

def series(sym):
    d = resample(pd.read_parquet(f"data/{sym}_M1.parquet"), TF)
    return d

def usd_per_pip(sym, price, usdjpy=None):
    """USD value of one pip on LOT lots. A JPY cross pays out in JPY, so it is
    converted at USDJPY - not at its own rate, which was quietly wrong by ~7%."""
    c = 100_000 * LOT
    if sym.endswith("USD"):
        return c * pip_size(sym)
    if sym.startswith("USD"):
        return c * pip_size(sym) / price
    if sym.endswith("JPY"):
        return c * pip_size(sym) / (usdjpy if usdjpy else 150.0)
    return c * pip_size(sym) / price

def backtest(a, b, lookback=500, entry=2.0, exit_z=0.5, stop_z=4.0,
             max_hold=480, costs=Costs()):
    da, db = series(a), series(b)
    idx = da.index.intersection(db.index)
    da, db = da.loc[idx], db.loc[idx]
    la, lb = np.log(da.close), np.log(db.close)

    # trailing beta and z-score - nothing here sees the future
    cov = la.rolling(lookback).cov(lb)
    var = lb.rolling(lookback).var()
    beta = (cov / var).shift(1)
    s = la - beta * lb
    z = ((s - s.rolling(lookback).mean()) / s.rolling(lookback).std()).shift(1)

    uj = pd.Series(usdjpy_rate(), name="uj").reindex(idx).ffill().bfill().values
    oa, ob = da.open.values, db.open.values
    pa = np.maximum(da.spread.values, costs.spread_floor_pips * pip_size(a))
    pb = np.maximum(db.spread.values, costs.spread_floor_pips * pip_size(b))
    slip_a = costs.slippage_pips * pip_size(a)
    slip_b = costs.slippage_pips * pip_size(b)
    zv = z.values
    n = len(idx)
    trades = []
    i = lookback + 1
    while i < n - 1:
        if not np.isfinite(zv[i]) or abs(zv[i]) < entry:
            i += 1; continue
        d = -1 if zv[i] > 0 else 1        # fade the stretch: short A / long B when z high
        j = i + 1                          # fill next bar open
        ea = oa[j] + d * (pa[j] / 2 + slip_a)
        eb = ob[j] - d * (pb[j] / 2 + slip_b)
        out = None
        for k in range(j, min(n - 1, j + max_hold)):
            if abs(zv[k]) < exit_z or abs(zv[k]) > stop_z:
                out = k; break
        if out is None:
            out = min(n - 1, j + max_hold)
        xa = oa[out] - d * (pa[out] / 2 + slip_a)
        xb = ob[out] + d * (pb[out] / 2 + slip_b)
        pips_a = d * (xa - ea) / pip_size(a)
        pips_b = -d * (xb - eb) / pip_size(b)
        usd = (pips_a * usd_per_pip(a, ea, uj[j]) + pips_b * usd_per_pip(b, eb, uj[j])
               - 2 * costs.commission_per_lot * LOT)
        trades.append(dict(t_in=idx[j], t_out=idx[out], usd=usd,
                           bars=out - j, z_in=zv[i]))
        i = out + 1
    return pd.DataFrame(trades)

def stats(tr, bal=200.0):
    if len(tr) == 0:
        return dict(n=0, win=0, pf=0, net=0, avg=0, dd=0)
    w = tr[tr.usd > 0].usd.sum()
    l = -tr[tr.usd <= 0].usd.sum()
    eq = bal + tr.usd.cumsum()
    return dict(n=len(tr), win=round((tr.usd > 0).mean() * 100, 1),
                pf=round(w / l, 3) if l > 0 else np.inf,
                net=round(tr.usd.sum(), 2), avg=round(tr.usd.mean(), 3),
                dd=round((eq.cummax() - eq).max(), 2),
                bars=int(tr.bars.mean()))

if __name__ == "__main__":
    combos = [("EURUSD", "USDCHF"), ("AUDUSD", "NZDUSD"), ("EURJPY", "GBPJPY"),
              ("EURUSD", "GBPUSD"), ("GBPUSD", "USDCHF"), ("GBPUSD", "USDJPY")]
    print(f"{'pair':>16}{'lb':>5}{'ent':>5} | {'IS n':>5}{'win':>6}{'pf':>7}{'net$':>8} | "
          f"{'OOS n':>6}{'win':>6}{'pf':>7}{'net$':>8}{'hold_h':>7}")
    for a, b in combos:
        for lookback in (300, 500):
            for entry in (2.0, 2.5):
                tr = backtest(a, b, lookback=lookback, entry=entry)
                if len(tr) == 0:
                    continue
                i = tr[(tr.t_in >= IS[0]) & (tr.t_in <= IS[1])]
                o = tr[(tr.t_in >= OOS[0]) & (tr.t_in <= OOS[1])]
                si, so = stats(i), stats(o)
                print(f"{a+'-'+b:>16}{lookback:5d}{entry:5.1f} | "
                      f"{si['n']:5d}{si['win']:6.1f}{si['pf']:7.2f}{si['net']:8.2f} | "
                      f"{so['n']:6d}{so['win']:6.1f}{so['pf']:7.2f}{so['net']:8.2f}"
                      f"{so.get('bars',0):7d}")
