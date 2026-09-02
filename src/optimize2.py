"""
Round-2 selection: a config must work on several pairs, not just the one it was
tuned on.

Round 1 showed the failure mode clearly - heavy filtering cut samples to ~100
trades, the best in-sample params were fitting noise, and out-of-sample win
rates fell ~13 points. Two changes here:
  * a minimum trade count large enough that the numbers mean something;
  * scoring on the WORST pair rather than the average, so a config that only
    works on one symbol cannot win. Robustness across instruments is much
    harder to fake than a good single-market fit.
"""
import sys, os, itertools, json
sys.path.insert(0, os.path.dirname(__file__))
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from engine import Backtest, Costs
from engine2 import Backtest2, ExitPlan
import strategies as st

from config import IS, OOS
MIN_TRADES = 200

_D = {}
def m1(sym):
    if sym not in _D:
        _D[sym] = pd.read_parquet(f"data/{sym}_M1.parquet")
    return _D[sym]

def one(sym, fam, params, tf, plan, window):
    d = m1(sym).loc[window[0]:window[1]]
    fn = getattr(st, fam)(**params)
    if plan is None:
        return Backtest(d, sym, tf=tf, max_bars=200).run(fn).stats
    return Backtest2(d, sym, tf=tf, plan=plan).run(fn).stats

def evaluate(job):
    fam, params, tf, plan, syms = job
    per = {}
    for s in syms:
        try:
            per[s] = {"is": one(s, fam, params, tf, plan, IS),
                      "oos": one(s, fam, params, tf, plan, OOS)}
        except Exception:
            return None
    ist = [per[s]["is"] for s in syms]
    if min(x["trades"] for x in ist) < MIN_TRADES:
        return None
    worst_pf = min(x["pf"] for x in ist)
    mean_wr = float(np.mean([x["win_rate"] for x in ist]))
    # worst-pair profit factor is the score; the win-rate goal only scales it
    score = worst_pf * min(mean_wr / 65.0, 1.0) if worst_pf > 1.0 else -1.0
    return {"family": fam, "tf": tf, "params": params,
            "plan": None if plan is None else plan.__dict__,
            "per": per, "worst_is_pf": worst_pf, "mean_is_wr": round(mean_wr, 2),
            "worst_oos_pf": min(per[s]["oos"]["pf"] for s in syms),
            "mean_oos_wr": round(float(np.mean([per[s]["oos"]["win_rate"] for s in syms])), 2),
            "score": score}

def grid(d):
    ks = list(d)
    for v in itertools.product(*(d[k] for k in ks)):
        yield dict(zip(ks, v))

def run(jobs, workers=4, tag="round2"):
    print(f"{len(jobs)} configs x {len(jobs[0][4])} pairs", flush=True)
    res = []
    with ProcessPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(evaluate, jobs, chunksize=2)):
            if r: res.append(r)
            if (i + 1) % 50 == 0: print(f"  {i+1}/{len(jobs)}", flush=True)
    res.sort(key=lambda r: -r["score"])
    json.dump(res[:200], open(f"results/{tag}.json", "w"), indent=1, default=str)
    return res

def show(rows, syms, n=12):
    for r in rows[:n]:
        print(f"\n{r['family']} {r['tf']} score={r['score']:.3f} "
              f"IS: worstPF={r['worst_is_pf']:.2f} meanWR={r['mean_is_wr']}% || "
              f"OOS: worstPF={r['worst_oos_pf']:.2f} meanWR={r['mean_oos_wr']}%")
        print(f"   {r['params']}  plan={r['plan']}")
        for s in syms:
            i, o = r["per"][s]["is"], r["per"][s]["oos"]
            print(f"     {s}: IS wr={i['win_rate']:5.1f} pf={i['pf']:5.2f} n={i['trades']:4d} "
                  f"| OOS wr={o['win_rate']:5.1f} pf={o['pf']:5.2f} n={o['trades']:4d}")
