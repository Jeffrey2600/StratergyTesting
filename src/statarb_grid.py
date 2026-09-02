"""Grid search over the pairs strategy, fitted in-sample and reported out."""
import sys, itertools, json
sys.path.insert(0, "src")
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from statarb import backtest, stats
from config import IS, OOS

COMBOS = [("EURUSD","USDCHF"),("AUDUSD","NZDUSD"),("EURJPY","GBPJPY"),
          ("EURUSD","GBPUSD"),("GBPUSD","USDCHF"),("GBPUSD","USDJPY"),
          ("EURUSD","USDJPY"),("AUDUSD","USDCAD"),("USDCAD","USDCHF"),
          ("AUDUSD","GBPUSD"),("NZDUSD","GBPUSD"),("EURGBP","EURUSD")]

def job(x):
    (a,b), lb, en, ex, st, mh = x
    try:
        tr = backtest(a, b, lookback=lb, entry=en, exit_z=ex, stop_z=st, max_hold=mh)
    except Exception:
        return None
    if len(tr) == 0:
        return None
    i = tr[(tr.t_in >= IS[0]) & (tr.t_in <= IS[1])]
    o = tr[(tr.t_in >= OOS[0]) & (tr.t_in <= OOS[1])]
    if len(i) < 25 or len(o) < 12:
        return None
    si, so = stats(i), stats(o)
    return dict(a=a, b=b, lb=lb, entry=en, exit=ex, stop=st, hold=mh,
                is_n=si["n"], is_pf=si["pf"], is_net=si["net"], is_win=si["win"],
                oos_n=so["n"], oos_pf=so["pf"], oos_net=so["net"], oos_win=so["win"])

if __name__ == "__main__":
    jobs = [(c, lb, en, ex, st, mh)
            for c in COMBOS
            for lb in (200, 300, 500)
            for en in (1.5, 2.0, 2.5, 3.0)
            for ex in (0.0, 0.5, 1.0)
            for st in (3.5, 5.0)
            for mh in (240, 480)]
    print(f"{len(jobs)} configs", flush=True)
    out = []
    with ProcessPoolExecutor(4) as ex_:
        for n, r in enumerate(ex_.map(job, jobs, chunksize=8)):
            if r: out.append(r)
            if (n+1) % 200 == 0: print(f"  {n+1}/{len(jobs)}", flush=True)
    d = pd.DataFrame(out)
    d.to_csv("results/statarb_grid.csv", index=False)
    print(f"\n{len(d)} configs produced enough trades")
    prof_is = d[d.is_pf > 1]
    print(f"profitable in-sample: {len(prof_is)}  ({len(prof_is)/max(len(d),1)*100:.1f}%)")
    both = d[(d.is_pf > 1) & (d.oos_pf > 1)]
    print(f"profitable in BOTH windows: {len(both)}  ({len(both)/max(len(d),1)*100:.1f}%)")
    print("\nIf the strategy had no edge, the both-windows share would sit near")
    print("the in-sample share times the ~50% chance of a positive holdout.\n")
    if len(both):
        print(both.sort_values("oos_pf", ascending=False).head(15).to_string(index=False))
