"""Round 1: broad sweep over all three families."""
import sys, os, json
sys.path.insert(0, "src")
from optimize import search, show

SYMS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["EURUSD"]
TFS = ["5min", "15min"]

SPACES = {
    "hrk_kn": dict(fast=[5, 8], slow=[13, 21], atr_len=[14],
                   sl_mult=[1.0, 1.5, 2.0], rr=[0.6, 1.0, 1.5, 2.0, 3.0]),
    "ema_pullback": dict(fast=[5, 8, 13], slow=[21, 34], trend=[100, 200],
                         atr_len=[14], sl_mult=[1.0, 1.5, 2.0],
                         rr=[0.6, 0.8, 1.0, 1.5], adx_min=[0, 20],
                         sess=[(7, 17), (0, 24), (12, 20)]),
    "bb_reversion": dict(bb_len=[20, 30], bb_std=[2.0, 2.5], atr_len=[14],
                         sl_mult=[1.5, 2.0, 3.0], rr=[0.5, 0.8, 1.0, 1.5],
                         rsi_ob=[65, 70, 75], rsi_os=[35, 30, 25],
                         sess=[(7, 17), (0, 24)]),
    "donchian_break": dict(ch=[20, 40, 60], atr_len=[14], sl_mult=[1.0, 1.5],
                           rr=[0.8, 1.0, 1.5, 2.0], trend=[100, 200],
                           sess=[(7, 17), (0, 24)]),
}

all_rows = []
for fam, space in SPACES.items():
    top, rows = search(fam, space, SYMS, TFS)
    print(f"\n===== TOP {fam} =====")
    show(top)
    all_rows += rows
all_rows.sort(key=lambda r: -r["score"])
json.dump(all_rows[:400], open("results/round1.json", "w"), indent=1, default=str)
print("\n\n########## OVERALL TOP 20 ##########")
show(all_rows, 20)
