"""Grid search with a hard in-sample / out-of-sample split.

Params are chosen on the in-sample window only. The out-of-sample numbers are
reported but never used for selection - that is the whole point of the split.
"""
import sys, json, itertools, os
sys.path.insert(0, os.path.dirname(__file__))
from concurrent.futures import ProcessPoolExecutor
import pandas as pd, numpy as np
from engine import Backtest, Costs
import strategies as st

from config import IS, OOS
MIN_TRADES = 60          # per-window, so a result is not noise
TARGET_WR = 65.0

_DATA = {}
def m1(sym):
    if sym not in _DATA:
        _DATA[sym] = pd.read_parquet(f"data/{sym}_M1.parquet")
    return _DATA[sym]

def slice_(df, w):
    return df.loc[w[0]:w[1]]

def score(s):
    """Rank candidates: profit factor, but only credit ones that clear the
    win-rate goal and trade often enough to be believable."""
    if s["trades"] < MIN_TRADES or s["pf"] <= 1.0:
        return -1.0
    wr_bonus = 1.0 if s["win_rate"] >= TARGET_WR else s["win_rate"] / TARGET_WR
    return s["pf"] * wr_bonus * (1 - min(s["max_dd_pct"], 50) / 100)

def evaluate(job):
    fam, params, sym, tf = job
    fn = getattr(st, fam)(**params)
    out = {}
    for tag, w in (("is", IS), ("oos", OOS)):
        d = slice_(m1(sym), w)
        if len(d) < 5000:
            return None
        out[tag] = Backtest(d, sym, tf=tf, max_bars=200).run(fn).stats
    return {"family": fam, "symbol": sym, "tf": tf, "params": params,
            "is": out["is"], "oos": out["oos"], "score": score(out["is"])}

def grid(d):
    keys = list(d)
    for vals in itertools.product(*(d[k] for k in keys)):
        yield dict(zip(keys, vals))

def search(fam, space, syms, tfs, workers=4, top=15):
    jobs = [(fam, p, s, tf) for p in grid(space) for s in syms for tf in tfs]
    print(f"[{fam}] {len(jobs)} combinations", flush=True)
    res = []
    with ProcessPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(evaluate, jobs, chunksize=4)):
            if r: res.append(r)
            if (i + 1) % 100 == 0:
                print(f"  ...{i+1}/{len(jobs)}", flush=True)
    res.sort(key=lambda r: -r["score"])
    return res[:top], res

def show(rows, n=10):
    for r in rows[:n]:
        i, o = r["is"], r["oos"]
        print(f"{r['family']:14s} {r['symbol']} {r['tf']:>6s} score={r['score']:.3f} | "
              f"IS wr={i['win_rate']:5.1f} pf={i['pf']:5.2f} n={i['trades']:4d} ret={i['return_pct']:7.1f}% "
              f"|| OOS wr={o['win_rate']:5.1f} pf={o['pf']:5.2f} n={o['trades']:4d} ret={o['return_pct']:7.1f}%")
        print(f"    {r['params']}")
