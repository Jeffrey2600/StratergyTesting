"""Strategy signal generators. Each returns a df with dir/sl/tp columns."""
import numpy as np, pandas as pd

def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def atr(df, n):
    pc = df.close.shift()
    tr = pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))

def adx(df, n=14):
    up = df.high.diff(); dn = -df.low.diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    a = atr(df, n)
    pdi = 100 * pd.Series(plus, index=df.index).ewm(alpha=1/n, adjust=False).mean() / a
    mdi = 100 * pd.Series(minus, index=df.index).ewm(alpha=1/n, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean()


def hrk_kn(fast=5, slow=13, atr_len=14, sl_mult=1.5, rr=3.0, longs=True, shorts=True):
    """Baseline: the original 'HRK KN Smart TP SL' EMA-cross strategy."""
    def f(df):
        ef, es = ema(df.close, fast), ema(df.close, slow)
        a = atr(df, atr_len)
        buy = (ef > es) & (ef.shift() <= es.shift())
        sell = (ef < es) & (ef.shift() >= es.shift())
        d = np.where(buy & longs, 1, np.where(sell & shorts, -1, 0))
        risk = a * sl_mult
        df["dir"] = d
        df["sl"] = np.where(d == 1, df.close - risk, df.close + risk)
        df["tp"] = np.where(d == 1, df.close + risk * rr, df.close - risk * rr)
        return df
    return f
