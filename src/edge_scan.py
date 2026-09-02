"""
Edge discovery, before strategy building.

Eight thousand indicator configs failed because they were guesses. This asks a
narrower question instead: conditioned on some observable state, is the forward
return of the pair reliably different from zero?

Method
------
For each pair/timeframe we build simple state features (hour, volatility regime,
distance from a moving average, recent return sign, position in the Asian range)
and measure forward returns over several horizons. A candidate edge must:
  * have a t-statistic worth looking at IN-SAMPLE,
  * keep the SAME SIGN out-of-sample,
  * hold on more than one pair,
  * and beat the ~1.5 pip round-trip cost, not merely beat zero.
Everything reported in pips so cost comparison is direct.
"""
import sys, itertools, json
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from engine import resample, pip_size

SYMS = ["EURUSD", "GBPUSD", "AUDUSD", "USDJPY"]
from config import IS, OOS
COST_PIPS = 1.5

def features(df, sym):
    pip = pip_size(sym)
    f = pd.DataFrame(index=df.index)
    r = df.close.pct_change()
    f["hour"] = df.index.hour
    f["dow"] = df.index.dayofweek
    # volatility regime: where the recent range sits in its own history
    rng = (df.high - df.low) / pip
    f["vol_pct"] = rng.rolling(500).rank(pct=True)
    # stretch from a moving average, in units of its own deviation
    ma = df.close.rolling(50).mean()
    sd = df.close.rolling(50).std()
    f["z50"] = (df.close - ma) / sd
    # momentum over the last few bars
    f["ret1"] = r
    f["ret4"] = df.close.pct_change(4)
    f["ret24"] = df.close.pct_change(24)
    return f

def fwd_pips(df, sym, h):
    return (df.close.shift(-h) - df.close) / pip_size(sym)

def bucket(s, edges):
    return pd.cut(s, edges, labels=False)

def scan_one(sym, tf, horizons=(1, 2, 4, 8, 24)):
    m1 = pd.read_parquet(f"data/{sym}_M1.parquet")
    d = resample(m1, tf)
    f = features(d, sym)
    rows = []
    conds = {
        "hour":    [(f"h{h:02d}", f.hour == h) for h in range(24)],
        "dow":     [(f"dow{i}", f.dow == i) for i in range(5)],
        "vol":     [("vol_low", f.vol_pct < .33), ("vol_mid", f.vol_pct.between(.33, .66)),
                    ("vol_high", f.vol_pct > .66)],
        "z50":     [("z<-2", f.z50 < -2), ("z-2..-1", f.z50.between(-2, -1)),
                    ("z-1..1", f.z50.between(-1, 1)), ("z1..2", f.z50.between(1, 2)),
                    ("z>2", f.z50 > 2)],
        "mom4":    [("mom4-", f.ret4 < 0), ("mom4+", f.ret4 > 0)],
        "mom24":   [("mom24-", f.ret24 < 0), ("mom24+", f.ret24 > 0)],
    }
    for h in horizons:
        y = fwd_pips(d, sym, h)
        for group, items in conds.items():
            for name, mask in items:
                for tag, win in (("is", IS), ("oos", OOS)):
                    w = (d.index >= win[0]) & (d.index <= win[1])
                    v = y[mask & w].dropna()
                    if len(v) < 200:
                        continue
                    mu, se = v.mean(), v.std() / np.sqrt(len(v))
                    rows.append(dict(sym=sym, tf=tf, group=group, cond=name, h=h,
                                     win=tag, n=len(v), mean_pips=mu,
                                     t=mu / se if se > 0 else 0))
    return pd.DataFrame(rows)

if __name__ == "__main__":
    tfs = sys.argv[1].split(",") if len(sys.argv) > 1 else ["60min"]
    out = pd.concat([scan_one(s, tf) for s in SYMS for tf in tfs], ignore_index=True)
    out.to_csv("results/edge_scan.csv", index=False)
    p = out.pivot_table(index=["tf", "group", "cond", "h"], columns=["win", "sym"],
                        values=["mean_pips", "t"])
    # a candidate must be significant in-sample, same sign out-of-sample, on >=3 pairs
    isd = out[out.win == "is"].set_index(["tf", "group", "cond", "h", "sym"])
    oos = out[out.win == "oos"].set_index(["tf", "group", "cond", "h", "sym"])
    j = isd.join(oos, rsuffix="_o", how="inner").reset_index()
    j["same_sign"] = np.sign(j.mean_pips) == np.sign(j.mean_pips_o)
    j["beats_cost"] = j.mean_pips.abs() > COST_PIPS
    g = j.groupby(["tf", "group", "cond", "h"]).agg(
        pairs=("sym", "count"), agree=("same_sign", "sum"),
        min_absT=("t", lambda x: np.abs(x).min()),
        mean_is=("mean_pips", "mean"), mean_oos=("mean_pips_o", "mean"),
        cost_ok=("beats_cost", "sum")).reset_index()
    good = g[(g.agree >= 3) & (g.min_absT > 1.5) & (g.cost_ok >= 3)]
    good = good.reindex(good.mean_is.abs().sort_values(ascending=False).index)
    print(f"scanned {len(out)} cells; {len(good)} candidate edges survive\n")
    print(good.head(40).to_string(index=False))
