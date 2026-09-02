"""
Round 2: refine the only family that survived out-of-sample.

Round 1 verdict:
  * hrk_kn (the original)  - every one of 60 configs lost. Dropped.
  * ema_pullback           - best in-sample scores, but win rates fell ~13 points
                             out-of-sample on ~100-trade samples. Overfit. Dropped.
  * bb_reversion           - the only family with configs that held or improved
                             out-of-sample, and it converged independently on wide
                             stops (3.0 ATR) with a 1.5R target on M15.
  * donchian_break         - nothing cleared the bar.

So this round searches around that wide-stop / larger-target region, and every
config must clear 200 trades on ALL FOUR pairs to be scored at all.
"""
import sys
sys.path.insert(0, "src")
from optimize2 import grid, run, show

SYMS = ["EURUSD", "GBPUSD", "AUDUSD", "USDJPY"]

space = dict(
    bb_len=[20, 30], bb_std=[2.0, 2.5, 3.0], atr_len=[14],
    sl_mult=[2.0, 2.5, 3.0, 4.0], rr=[1.0, 1.2, 1.5, 2.0],
    rsi_ob=[65, 70, 75], rsi_os=[35, 30, 25], sess=[(7, 17), (0, 24)],
)
# rsi thresholds stay symmetric - independent long/short levels invite fitting
jobs = [("bb_reversion", p, tf, None, SYMS)
        for p in grid(space) if p["rsi_ob"] + p["rsi_os"] == 100
        for tf in ["15min", "30min"]]

res = run(jobs, tag="round2")
print(f"\n{len(res)} configs scored (of {len(jobs)})")
print("\n########## ROUND 2 TOP ##########")
show(res, SYMS, 12)
