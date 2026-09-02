"""Shared backtest windows.

Restricted to recent history at the user's direction: nothing before 2024. The
tradeoff is real and worth stating - 2.7 years is a thin sample, so a config can
look good on it by luck more easily than on a longer history. The out-of-sample
window is therefore kept long (12 months) rather than a token holdout.
"""
DATA_START = "2024-01-01"
DATA_END   = "2026-08-31"

IS  = ("2024-01-01", "2025-08-31")   # 20 months - parameters are fitted here
OOS = ("2025-09-01", "2026-08-31")   # 12 months - never used for selection
