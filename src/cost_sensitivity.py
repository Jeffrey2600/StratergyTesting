"""
Is there an edge that costs are eating, or is there no edge?

Re-runs the best round-2 configs at several cost levels. If profit factor at
zero cost is comfortably above 1, the signal has predictive value and the job
is to trade it less often or for bigger targets. If it sits near 1.0 even for
free, there is nothing there and no amount of tuning will help.
"""
import sys, json
sys.path.insert(0, "src")
import pandas as pd
from engine import Backtest, Costs
import strategies as st

SYMS = ["EURUSD", "GBPUSD", "AUDUSD", "USDJPY"]
LEVELS = {
    "zero cost":      Costs(spread_floor_pips=0.0, slippage_pips=0.0, commission_per_lot=0.0),
    "half retail":    Costs(spread_floor_pips=0.5, slippage_pips=0.1, commission_per_lot=3.5),
    "retail (used)":  Costs(spread_floor_pips=1.0, slippage_pips=0.2, commission_per_lot=7.0),
}
rows = json.load(open("results/round2.json"))
best = rows[0]
print("config:", best["params"], best["tf"], "\n")
for name, c in LEVELS.items():
    out = []
    for s in SYMS:
        d = pd.read_parquet(f"data/{s}_M1.parquet").loc["2022-01-01":"2023-12-31"]
        r = Backtest(d, s, tf=best["tf"], costs=c).run(st.bb_reversion(**best["params"]))
        out.append((s, r.stats))
    print(f"--- {name}")
    for s, x in out:
        print(f"    {s}: pf={x['pf']:5.2f} wr={x['win_rate']:5.1f}% n={x['trades']:5d} "
              f"exp={x['expectancy_pips']:+6.2f} pips")
