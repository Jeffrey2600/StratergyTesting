"""
The controlled parameter experiments the specification asks for (section 30).

Deliberately small: trend filter, pullback EMA, stop multiple and target, which
is 72 combinations - not the unrestricted brute-force search the spec warns
against. Results are reported as expectancy in R, which is independent of
position size, so a config cannot look good merely by having traded when the
account happened to be large.

Sizing is pinned to a fixed lot here on purpose. With equity-based sizing the
baseline account was ruined inside the first year, after which the minimum-lot
rule blocked almost every later signal - which measures survival, not signal
quality. Fixing the lot lets every year contribute.
"""
import sys, os, itertools, json
sys.path.insert(0, os.path.dirname(__file__))
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from spec_engine import Params, PAIRS, load, run_portfolio
from config import DATA_START, DATA_END, IS, OOS

_D = None
def data():
    global _D
    if _D is None:
        _D = load(PAIRS, DATA_START, DATA_END)
    return _D

def evaluate(cfg):
    fast, slow, pull, sl, tp = cfg
    p = Params(ema_fast=fast, ema_slow=slow, ema_pull=pull, sl_atr=sl, tp_r=tp,
               fixed_lot=0.01)
    tr, _ = run_portfolio(data(), p)
    if len(tr) < 100:
        return None
    tr["R"] = tr.pips / tr.stop_pips
    i = tr[(tr.t_in >= IS[0]) & (tr.t_in <= IS[1])]
    o = tr[(tr.t_in >= OOS[0]) & (tr.t_in <= OOS[1])]
    if len(i) < 60 or len(o) < 40:
        return None
    def pf(d):
        g = d[d.pnl > 0].pnl.sum(); l = -d[d.pnl <= 0].pnl.sum()
        return round(g / l, 3) if l > 0 else np.inf
    return dict(fast=fast, slow=slow, pull=pull, sl_atr=sl, tp_r=tp,
                n=len(tr), R=round(tr.R.mean(), 4),
                win=round((tr.pnl > 0).mean() * 100, 2), pf=pf(tr),
                is_n=len(i), is_R=round(i.R.mean(), 4), is_pf=pf(i),
                oos_n=len(o), oos_R=round(o.R.mean(), 4), oos_pf=pf(o))

if __name__ == "__main__":
    grid = [(f, s, pu, sl, tp)
            for (f, s) in ((20, 100), (50, 200), (50, 100))
            for pu in (20, 50)
            for sl in (1.0, 1.5, 2.0)
            for tp in (1.5, 2.0, 2.5, 3.0)]
    print(f"{len(grid)} configurations", flush=True)
    rows = []
    with ProcessPoolExecutor(4) as ex:
        for k, r in enumerate(ex.map(evaluate, grid, chunksize=1)):
            if r: rows.append(r)
            if (k + 1) % 6 == 0: print(f"  {k+1}/{len(grid)}", flush=True)
    d = pd.DataFrame(rows).sort_values("R", ascending=False)
    d.to_csv("results/spec_experiments.csv", index=False)
    print(f"\n{len(d)} configs scored. Expectancy in R per trade "
          f"(0 = breakeven before commission):\n")
    print(d.to_string(index=False))
    print(f"\nconfigs with positive full-period R: {(d.R > 0).sum()} / {len(d)}")
    print(f"configs positive in BOTH windows:    "
          f"{((d.is_R > 0) & (d.oos_R > 0)).sum()} / {len(d)}")
