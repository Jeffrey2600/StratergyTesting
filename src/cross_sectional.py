"""
Cross-sectional currency momentum.

Single-pair timing failed every test so far. This asks a different question: not
"will EURUSD go up" but "is EUR stronger than JPY" - a relative-value question,
which is where the documented FX anomalies (cross-sectional momentum and carry)
actually live.

Individual currency returns are recovered from the pair matrix: each currency's
daily return is the average of its returns against every pair it appears in,
sign-adjusted for whether it is the base or the quote. Currencies are then ranked
on trailing return, going long the strongest and short the weakest.

Cross-sectional returns need no drift correction - a long/short portfolio with
equal weights is already market-neutral, so a period where everything fell
against the dollar nets out.
"""
import sys, itertools
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from engine import resample

PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "USDJPY", "USDCHF", "USDCAD",
         "NZDUSD", "EURGBP", "EURJPY", "GBPJPY"]

def daily_returns():
    out = {}
    for p in PAIRS:
        try:
            d = resample(pd.read_parquet(f"data/{p}_M1.parquet"), "1440min")
        except FileNotFoundError:
            continue
        out[p] = np.log(d.close).diff()
    return pd.DataFrame(out).dropna(how="all")

def currency_returns(px):
    ccys = sorted({c for p in px.columns for c in (p[:3], p[3:])})
    r = pd.DataFrame(index=px.index, columns=ccys, dtype=float)
    for c in ccys:
        parts = []
        for p in px.columns:
            if p[:3] == c:
                parts.append(px[p])
            elif p[3:] == c:
                parts.append(-px[p])
        r[c] = pd.concat(parts, axis=1).mean(axis=1)
    return r

def backtest_xs(cr, look, hold, n_side=2, cost_bp=2.0):
    """Long the n_side strongest currencies, short the n_side weakest.
    cost_bp is charged on every rebalance, per unit of turnover."""
    sig = cr.rolling(look).sum()
    rets, dates = [], []
    for i in range(look, len(cr) - hold, hold):
        s = sig.iloc[i].dropna()
        if len(s) < 4:
            continue
        rank = s.rank()
        longs = rank.nlargest(n_side).index
        shorts = rank.nsmallest(n_side).index
        fwd = cr.iloc[i + 1:i + 1 + hold].sum()
        pnl = fwd[longs].mean() - fwd[shorts].mean()
        rets.append(pnl - cost_bp / 10000 * 2)
        dates.append(cr.index[i])
    return pd.Series(rets, index=dates)

if __name__ == "__main__":
    px = daily_returns()
    print("pairs loaded:", list(px.columns), "\n")
    cr = currency_returns(px)
    print("currencies:", list(cr.columns), "\n")
    rows = []
    for look in (20, 60, 120):
        for hold in (5, 10, 20):
            s = backtest_xs(cr, look, hold)
            if len(s) < 20:
                continue
            t = s.mean() / (s.std() / np.sqrt(len(s)))
            ann = s.mean() * (252 / hold) * 100
            rows.append(dict(look=look, hold=hold, n=len(s),
                             mean_bp=round(s.mean() * 10000, 1),
                             ann_pct=round(ann, 1), t=round(t, 2),
                             hit=round((s > 0).mean() * 100, 1)))
    r = pd.DataFrame(rows)
    print(r.to_string(index=False))
    print("\nn is the number of NON-OVERLAPPING rebalances - t-stats are honest.")
