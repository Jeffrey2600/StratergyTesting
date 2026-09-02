"""
Cross-sectional currency momentum, direction chosen in-sample only.

The full-sample run came out negative at every lookback, which hints at
reversal rather than momentum. Flipping the sign after seeing that would be
fitting to the answer, so the sign is decided on the fit window alone and then
applied unchanged to the holdout.
"""
import sys
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from cross_sectional import daily_returns, currency_returns, backtest_xs
from config import IS, OOS

cr = currency_returns(daily_returns())
print(f"fit {IS[0]}..{IS[1]}   holdout {OOS[0]}..{OOS[1]}\n")
print(f"{'look':>5}{'hold':>5} | {'IS bp':>8}{'IS t':>7}{'n':>5} | sign | "
      f"{'OOS bp':>8}{'OOS t':>7}{'n':>5} {'ann%':>7}")
for look in (20, 60, 120):
    for hold in (5, 10, 20):
        s = backtest_xs(cr, look, hold)
        i = s[(s.index >= IS[0]) & (s.index <= IS[1])]
        o = s[(s.index >= OOS[0]) & (s.index <= OOS[1])]
        if len(i) < 15 or len(o) < 8:
            continue
        ti = i.mean() / (i.std() / np.sqrt(len(i)))
        sign = 1 if i.mean() > 0 else -1        # decided on IS only
        oo = o * sign
        to = oo.mean() / (oo.std() / np.sqrt(len(oo)))
        ann = oo.mean() * (252 / hold) * 100
        print(f"{look:5d}{hold:5d} | {i.mean()*1e4:8.1f}{ti:7.2f}{len(i):5d} | "
              f"{'mom' if sign>0 else 'rev':>4} | {oo.mean()*1e4:8.1f}{to:7.2f}{len(o):5d}{ann:7.1f}")
