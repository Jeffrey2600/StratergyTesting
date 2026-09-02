"""
Do any pair-spreads actually mean-revert?

Same discipline as the edge scans: test the premise before building a strategy
on it. For every combination of the 10 instruments we form the spread
    s = log(A) - beta * log(B)
with beta from an in-sample regression, then measure:

  * half-life - from the Ornstein-Uhlenbeck fit ds = -lambda*s*dt. A short
    half-life means the spread pulls back quickly; a very long one means it
    wanders.
  * variance ratio - Var(s over k bars) / (k * Var(s over 1 bar)). A random walk
    gives 1.0, mean reversion gives < 1, trending gives > 1. This is the honest
    test: a spread can look tight and still be a random walk.

Beta is fitted in-sample only and the statistics are reported for both windows,
so a relationship that holds in the fit period and breaks afterwards is visible
rather than hidden.
"""
import sys, itertools
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from engine import resample
from config import IS, OOS

PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "USDJPY", "USDCHF", "USDCAD",
         "NZDUSD", "EURGBP", "EURJPY", "GBPJPY"]
TF = "60min"

def closes():
    out = {}
    for p in PAIRS:
        d = resample(pd.read_parquet(f"data/{p}_M1.parquet"), TF)
        out[p] = np.log(d.close)
    return pd.DataFrame(out).dropna()

def half_life(s):
    s = s.dropna()
    ds = s.diff().dropna()
    x = s.shift().dropna().loc[ds.index]
    b = np.polyfit(x, ds, 1)[0]
    return -np.log(2) / b if b < 0 else np.inf

def var_ratio(s, k):
    r = s.diff().dropna()
    if r.std() == 0 or len(r) < k * 10:
        return np.nan
    return (r.rolling(k).sum().dropna().var() / k) / r.var()

if __name__ == "__main__":
    px = closes()
    rows = []
    for a, b in itertools.combinations(PAIRS, 2):
        i = px.loc[IS[0]:IS[1]]
        if len(i) < 500:
            continue
        beta = np.polyfit(i[b], i[a], 1)[0]          # fitted in-sample only
        s_all = px[a] - beta * px[b]
        si, so = s_all.loc[IS[0]:IS[1]], s_all.loc[OOS[0]:OOS[1]]
        corr = px[a].diff().corr(px[b].diff())
        rows.append(dict(A=a, B=b, corr=round(corr, 2), beta=round(beta, 2),
                         hl_is=round(half_life(si), 1), hl_oos=round(half_life(so), 1),
                         vr24_is=round(var_ratio(si, 24), 3),
                         vr24_oos=round(var_ratio(so, 24), 3)))
    r = pd.DataFrame(rows)
    r["mr_score"] = (1 - r.vr24_is) + (1 - r.vr24_oos)   # bigger = more reverting
    r = r.sort_values("mr_score", ascending=False)
    print(f"{TF} bars. Variance ratio at 24 bars: 1.0 = random walk, <1 = mean reverting\n")
    print(r.head(15).to_string(index=False))
    print("\nBaseline - the individual pairs themselves:")
    for p in PAIRS[:5]:
        print(f"  {p}: VR24 IS={var_ratio(px[p].loc[IS[0]:IS[1]],24):.3f} "
              f"OOS={var_ratio(px[p].loc[OOS[0]:OOS[1]],24):.3f}")
