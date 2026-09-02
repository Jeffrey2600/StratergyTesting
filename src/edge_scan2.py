"""
Edge scan v2: event-conditioned states across timeframes.

v1 tested standing states (hour, volatility band, stretch from a mean) on H1 and
found essentially nothing. That matters more than it looks: with no conditional
drift, no arrangement of stops and targets can be profitable either - under zero
drift every stop/target combination has zero expectancy before costs, so a
directional edge is a necessary condition, not an optional extra.

So v2 looks where an edge is more plausible: at events rather than states.
Session-range breaks, volatility expansion after compression, sharp moves that
may over- or under-shoot, and the London/NY open. Same survival test as v1:
in-sample significance, matching out-of-sample sign, on at least three pairs,
and large enough to clear costs.

All returns are measured as EXCESS over the same window's unconditional drift.
A first pass without this showed 5-8 pip "edges" at the 4-hour/4-day scale that
turned out to be nothing but the trend of the period - GBPUSD alone drifted -8
pips per 4 days in-sample and +8.3 out-of-sample.
"""
import sys
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from engine import resample, pip_size

SYMS = ["EURUSD", "GBPUSD", "AUDUSD", "USDJPY"]
IS = ("2022-01-01", "2023-12-31")
OOS = ("2024-01-01", "2025-12-31")
COST = 1.5

def build(sym, tf):
    m1 = pd.read_parquet(f"data/{sym}_M1.parquet")
    d = resample(m1, tf)
    pip = pip_size(sym)
    x = pd.DataFrame(index=d.index)
    h = d.index.hour
    rng = (d.high - d.low) / pip
    atr = rng.rolling(20).mean()
    ret = d.close.diff() / pip

    # Asian session range (00:00-06:00 UTC), carried into the London session
    day = d.index.normalize()
    asia = d[(h >= 0) & (h < 6)]
    ah = asia.groupby(asia.index.normalize()).high.max().reindex(day).values
    al = asia.groupby(asia.index.normalize()).low.min().reindex(day).values
    x["asia_hi"], x["asia_lo"] = ah, al
    brk_up = (d.close.values > ah) & (d.close.shift().values <= ah) & (h >= 7) & (h < 12)
    brk_dn = (d.close.values < al) & (d.close.shift().values >= al) & (h >= 7) & (h < 12)

    # volatility compression then expansion
    comp = (rng.rolling(6).mean() / atr) < 0.7
    expand = rng > 1.5 * atr

    # sharp single-bar moves
    big_up = ret > 1.5 * atr
    big_dn = ret < -1.5 * atr

    conds = {
        "asia_break_up":   pd.Series(brk_up, index=d.index),
        "asia_break_dn":   pd.Series(brk_dn, index=d.index),
        "compress":        comp,
        "expand_up":       expand & (ret > 0),
        "expand_dn":       expand & (ret < 0),
        "spike_up":        big_up,
        "spike_dn":        big_dn,
        "london_open":     pd.Series((h == 7), index=d.index),
        "ny_open":         pd.Series((h == 13), index=d.index),
        "friday_pm":       pd.Series((d.index.dayofweek == 4) & (h >= 15), index=d.index),
        "sunday_open":     pd.Series((d.index.dayofweek == 6), index=d.index),
    }
    return d, pip, conds

def scan(tf, horizons=(1, 2, 4, 8, 12, 24)):
    rows = []
    for sym in SYMS:
        d, pip, conds = build(sym, tf)
        for name, mask in conds.items():
            mask = mask.fillna(False).astype(bool)
            for hz in horizons:
                y = (d.close.shift(-hz) - d.close) / pip
                for tag, w in (("is", IS), ("oos", OOS)):
                    sel = mask & (d.index >= w[0]) & (d.index <= w[1])
                    base = y[(d.index >= w[0]) & (d.index <= w[1])].dropna()
                    v = y[sel].dropna()
                    if len(v) < 100:
                        continue
                    # excess over the window's own drift. Without this, a period
                    # in which the pair simply fell shows up as an "edge" in every
                    # condition that happens to be short.
                    v = v - base.mean()
                    mu, se = v.mean(), v.std() / np.sqrt(len(v))
                    rows.append(dict(sym=sym, cond=name, h=hz, win=tag, n=len(v),
                                     mean=mu, t=mu / se if se else 0))
    return pd.DataFrame(rows)

if __name__ == "__main__":
    allr = []
    for tf in ["15min", "60min", "240min"]:
        r = scan(tf); r["tf"] = tf; allr.append(r)
    out = pd.concat(allr, ignore_index=True)
    out.to_csv("results/edge_scan2.csv", index=False)
    i = out[out.win == "is"].set_index(["tf", "cond", "h", "sym"])
    o = out[out.win == "oos"].set_index(["tf", "cond", "h", "sym"])
    j = i.join(o, rsuffix="_o", how="inner").reset_index()
    j["agree"] = np.sign(j["mean"]) == np.sign(j["mean_o"])
    g = j.groupby(["tf", "cond", "h"]).agg(
        pairs=("sym", "count"), agree=("agree", "sum"),
        minT=("t", lambda x: np.abs(x).min()),
        is_pips=("mean", "mean"), oos_pips=("mean_o", "mean")).reset_index()
    good = g[(g.pairs >= 3) & (g.agree >= 3) & (g.minT > 1.5) &
             (g.is_pips.abs() > COST) & (g.oos_pips.abs() > COST)]
    good = good.reindex(good.oos_pips.abs().sort_values(ascending=False).index)
    print(f"{len(g)} condition/horizon cells tested, {len(good)} survive\n")
    print(good.to_string(index=False) if len(good) else "(none)")
