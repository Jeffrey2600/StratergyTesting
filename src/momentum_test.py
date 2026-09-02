"""
Time-series momentum on daily bars.

This is the one FX anomaly with serious academic support (Moskowitz/Ooi/Pedersen
and the managed-futures literature): instruments that rose over the past N months
tend to keep rising over the next month. It is slow and it is not a scalping
edge, but it is the most defensible thing left to test, so it gets tested rather
than assumed.

Caveat stated up front: 3.7 years on 4 pairs is far too little to confirm or
refute an effect this slow. Whatever comes out is indicative, not proof.
"""
import sys
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from engine import resample, pip_size

SYMS = ["EURUSD", "GBPUSD", "AUDUSD", "USDJPY"]

rows = []
for sym in SYMS:
    d = resample(pd.read_parquet(f"data/{sym}_M1.parquet"), "1440min")
    pip = pip_size(sym)
    for look in (20, 60, 120):
        for hold in (5, 10, 20):
            sig = np.sign(d.close.pct_change(look))
            fwd = (d.close.shift(-hold) - d.close) / pip
            # excess over the window drift, so a one-way market cannot masquerade
            pnl = (sig * (fwd - fwd.mean())).dropna()
            if len(pnl) < 100:
                continue
            t = pnl.mean() / (pnl.std() / np.sqrt(len(pnl)))
            rows.append(dict(sym=sym, look=look, hold=hold, n=len(pnl),
                             pips=round(pnl.mean(), 2), t=round(t, 2)))
r = pd.DataFrame(rows)
piv = r.pivot_table(index=["look", "hold"], columns="sym", values=["pips", "t"])
print(piv.round(2).to_string())
print("\nby lookback/hold, averaged over pairs (excess pips per trade, and how many")
print("of the 4 pairs are positive):")
g = r.groupby(["look", "hold"]).agg(mean_pips=("pips", "mean"),
                                    pairs_positive=("pips", lambda x: (x > 0).sum()),
                                    min_t=("t", "min"), max_t=("t", "max"))
print(g.round(2).to_string())
