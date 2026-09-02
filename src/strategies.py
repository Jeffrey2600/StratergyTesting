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


def session_mask(idx, start_h, end_h):
    """UTC hour window. Handles wrap-around windows (e.g. 22->6)."""
    h = idx.hour
    if start_h <= end_h:
        return (h >= start_h) & (h < end_h)
    return (h >= start_h) | (h < end_h)


def trend_ema(df, n):
    return ema(df.close, n)


def ema_pullback(fast=8, slow=21, trend=200, atr_len=14, sl_mult=1.2, rr=1.0,
                 adx_min=0.0, sess=(7, 17), longs=True, shorts=True,
                 rsi_len=14, rsi_lo=0, rsi_hi=100):
    """Trend-following pullback: trade EMA crosses only in the direction of a
    long-term EMA, optionally gated by ADX strength and a session window."""
    def f(df):
        ef, es = ema(df.close, fast), ema(df.close, slow)
        et = ema(df.close, trend)
        a = atr(df, atr_len)
        ax = adx(df, atr_len) if adx_min > 0 else pd.Series(100.0, index=df.index)
        rs = rsi(df.close, rsi_len)
        ok = session_mask(df.index, *sess) & (ax >= adx_min) & rs.between(rsi_lo, rsi_hi)
        buy = (ef > es) & (ef.shift() <= es.shift()) & (df.close > et) & ok & longs
        sell = (ef < es) & (ef.shift() >= es.shift()) & (df.close < et) & ok & shorts
        d = np.where(buy, 1, np.where(sell, -1, 0))
        risk = a * sl_mult
        df["dir"] = d
        df["sl"] = np.where(d == 1, df.close - risk, df.close + risk)
        df["tp"] = np.where(d == 1, df.close + risk * rr, df.close - risk * rr)
        return df
    return f


def bb_reversion(bb_len=20, bb_std=2.0, atr_len=14, sl_mult=1.5, rr=0.6,
                 rsi_len=14, rsi_ob=70, rsi_os=30, trend=0, sess=(7, 17),
                 longs=True, shorts=True):
    """Mean reversion: fade a close outside the Bollinger band when RSI is
    stretched. Small TP, wider SL -> aims for a high hit rate."""
    def f(df):
        ma = df.close.rolling(bb_len).mean()
        sd = df.close.rolling(bb_len).std()
        up, lo = ma + bb_std * sd, ma - bb_std * sd
        a = atr(df, atr_len)
        rs = rsi(df.close, rsi_len)
        ok = session_mask(df.index, *sess)
        if trend:
            et = ema(df.close, trend)
            up_ok, dn_ok = df.close > et, df.close < et
        else:
            up_ok = dn_ok = pd.Series(True, index=df.index)
        buy = (df.close < lo) & (rs < rsi_os) & ok & up_ok & longs
        sell = (df.close > up) & (rs > rsi_ob) & ok & dn_ok & shorts
        d = np.where(buy, 1, np.where(sell, -1, 0))
        risk = a * sl_mult
        df["dir"] = d
        df["sl"] = np.where(d == 1, df.close - risk, df.close + risk)
        df["tp"] = np.where(d == 1, df.close + risk * rr, df.close - risk * rr)
        return df
    return f


def donchian_break(ch=20, atr_len=14, sl_mult=1.0, rr=1.5, trend=200,
                   sess=(7, 17), longs=True, shorts=True):
    """Breakout of an N-bar channel, filtered by a long EMA."""
    def f(df):
        hh = df.high.rolling(ch).max().shift()
        ll = df.low.rolling(ch).min().shift()
        et = ema(df.close, trend)
        a = atr(df, atr_len)
        ok = session_mask(df.index, *sess)
        buy = (df.close > hh) & (df.close > et) & ok & longs
        sell = (df.close < ll) & (df.close < et) & ok & shorts
        d = np.where(buy, 1, np.where(sell, -1, 0))
        risk = a * sl_mult
        df["dir"] = d
        df["sl"] = np.where(d == 1, df.close - risk, df.close + risk)
        df["tp"] = np.where(d == 1, df.close + risk * rr, df.close - risk * rr)
        return df
    return f
