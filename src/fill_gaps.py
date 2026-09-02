"""Re-fetch missing days until coverage stops improving.

The feed returns 503s under load and those days were being dropped, leaving
12-30% of weekdays missing - concentrated, in EURUSD's case, inside the holdout
window. Backtesting on that would quietly skip whole weeks of price action.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
from fetch_data import load, pip_size
from config import DATA_START, DATA_END

def missing(sym):
    try:
        d = pd.read_parquet(f"data/{sym}_M1.parquet")
    except FileNotFoundError:
        return None
    exp = {x.date() for x in pd.date_range(DATA_START, DATA_END, freq="D") if x.weekday() < 5}
    got = {x.date() for x in d.index.normalize().unique()}
    return sorted(exp - got)

if __name__ == "__main__":
    syms = sys.argv[1].split(",")
    for sym in syms:
        for attempt in range(6):
            miss = missing(sym)
            if miss is not None and len(miss) < 25:
                break
            df = load(sym, DATA_START, DATA_END)
            df.to_parquet(f"data/{sym}_M1.parquet")
            print(f"{sym} pass{attempt}: {len(missing(sym))} weekdays still missing", flush=True)
        print(f"{sym} FINAL missing={len(missing(sym))}", flush=True)
