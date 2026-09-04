"""Two diagnostics on the specified strategy.

1. Fixed lot size, so a blown account cannot stop the test early and every
   year gets sampled. This measures signal quality, not survival.
2. The same run with spread, slippage and commission switched off. If the
   signal is still negative for free, no cost or broker change rescues it.
"""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from spec_engine import Params, PAIRS, load, run_portfolio, metrics
from config import DATA_START, DATA_END

data = load(PAIRS, DATA_START, DATA_END)

def show(tag, p):
    tr, sk = run_portfolio(data, p)
    if len(tr) == 0:
        print(f"{tag}: no trades"); return
    tr["R"] = tr.pips / tr.stop_pips
    m = metrics(tr, p)
    print(f"\n=== {tag}")
    print(f"  trades={m['trades']}  win={m['win_rate']}%  pf={m['pf']}  "
          f"net=${m['net']}  expectancy={tr.R.mean():+.3f}R")
    print(f"  by year: ", end="")
    for y, d in tr.groupby(tr.t_out.dt.year):
        print(f"{y}: n={len(d)} {d.R.mean():+.3f}R  ", end="")
    print()
    return tr

base = Params(fixed_lot=0.01)
show("fixed 0.01 lot, full costs", base)
show("fixed 0.01 lot, ZERO costs",
     Params(fixed_lot=0.01, spread_floor_pips=0.0, slippage_pips=0.0,
            commission_per_lot=0.0, max_spread_pips=99.0))
