"""Baseline run of the specified trend-pullback strategy, in the spec's format."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from spec_engine import Params, PAIRS, load, run_portfolio, metrics, signals, simulate_price
from config import DATA_START, DATA_END

def report(tr, sk, p, title="TREND PULLBACK STRATEGY", per_pair=True):
    m = metrics(tr, p)
    L = "=" * 54
    print(L); print(title); print(L)
    if not m:
        print("no trades"); return m
    print(f"Account:\n  Starting Balance: ${p.balance:.2f}")
    print(f"  Risk Per Trade:   {p.risk_pct}%   (initial ${p.balance*p.risk_pct/100:.2f})")
    print(f"  Pairs: {len(PAIRS)}   Timeframes: {p.tf_high} / {p.tf_exec}")
    print(f"\nPerformance:")
    print(f"  Ending Balance:   ${m['end']:.2f}")
    print(f"  Net Profit:       ${m['net']:.2f}")
    print(f"  Return:           {m['ret_pct']:.2f}%")
    print(f"  CAGR:             {m['cagr']:.2f}%")
    print(f"  Profit Factor:    {m['pf']}")
    print(f"  Expectancy:       ${m['expectancy']:.3f} / trade")
    print(f"  Win Rate:         {m['win_rate']:.2f}%")
    print(f"  Max Drawdown:     ${m['max_dd']:.2f}  ({m['max_dd_pct']:.1f}%)")
    print(f"  Sharpe / Sortino: {m['sharpe']} / {m['sortino']}")
    print(f"  Recovery Factor:  {m['recovery']}")
    print(f"\nTrades:")
    print(f"  Total:            {m['trades']}")
    print(f"  Average Trade:    ${m['expectancy']:.3f}")
    print(f"  Average Win:      ${m['avg_win']:.2f}")
    print(f"  Average Loss:     ${m['avg_loss']:.2f}")
    print(f"  Largest Win:      ${m['largest_win']:.2f}")
    print(f"  Largest Loss:     ${m['largest_loss']:.2f}")
    print(f"  Max Cons. Losses: {m['max_cons_loss']}   (max cons. wins {m['max_cons_win']})")
    print(f"  Avg Duration:     {m['avg_hold_h']:.1f} h")
    print(f"\nCosts:")
    print(f"  Spread:           ${m['spread_cost']:.2f}")
    print(f"  Commission:       ${m['commission']:.2f}")
    print(f"  Slippage:         ${m['slippage_cost']:.2f}")
    print(f"  Total:            ${m['total_costs']:.2f}")
    print(f"  Net BEFORE costs: ${m['gross_before_costs']:.2f}")
    if len(sk):
        print(f"\nSignals skipped: {dict(sk)}")
    if per_pair:
        print(f"\nPer-Pair Results:")
        for s in PAIRS:
            d = tr[tr.pair == s]
            if len(d) == 0:
                print(f"  {s}:  no trades"); continue
            w = d[d.pnl > 0].pnl.sum(); l = -d[d.pnl <= 0].pnl.sum()
            print(f"  {s}:  n={len(d):4d}  win={(d.pnl>0).mean()*100:5.1f}%  "
                  f"pf={w/l if l>0 else np.inf:5.2f}  net=${d.pnl.sum():8.2f}")
        print(f"\nProfit by year:")
        for y, d in tr.groupby(tr.t_out.dt.year):
            print(f"  {y}:  n={len(d):4d}  net=${d.pnl.sum():8.2f}  "
                  f"win={(d.pnl>0).mean()*100:5.1f}%")
    return m


if __name__ == "__main__":
    p = Params()
    print(f"Loading {len(PAIRS)} pairs, {DATA_START} .. {DATA_END}", flush=True)
    data = load(PAIRS, DATA_START, DATA_END)
    tr, sk = run_portfolio(data, p)
    m = report(tr, sk, p)
    if len(tr):
        tr.to_csv("results/spec_baseline_trades.csv", index=False)
        print(f"\ntrade log -> results/spec_baseline_trades.csv ({len(tr)} rows)")
