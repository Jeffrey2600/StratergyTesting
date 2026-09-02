import sys, pandas as pd
sys.path.insert(0, "src")
from engine import Backtest, Costs
import strategies as st

def load(sym):
    return pd.read_parquet(f"data/{sym}_M1.parquet")

if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "EURUSD"
    m1 = load(sym)
    for tf in ["5min", "15min"]:
        bt = Backtest(m1, sym, tf=tf, max_bars=200)
        r = bt.run(st.hrk_kn())
        print(f"{sym} {tf} baseline :: {r.summary()}")
