"""Download Dukascopy daily M1 candle files and build clean OHLC parquet/CSV."""
import lzma, struct, os, sys, time, io
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import numpy as np, pandas as pd, requests

BASE = "https://datafeed.dukascopy.com/datafeed"
CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(CACHE, exist_ok=True)
S = requests.Session()

def point(sym):
    """Price of one point (1/10 pip) for the instrument."""
    return 1e-3 if "JPY" in sym else 1e-5

def pip_size(sym):
    return 0.01 if "JPY" in sym else 0.0001

def _url(sym, d, side):
    return f"{BASE}/{sym}/{d.year:04d}/{d.month-1:02d}/{d.day:02d}/{side}_candles_min_1.bi5"

def _fetch(sym, d, side, retries=4):
    fn = os.path.join(CACHE, f"{sym}_{side}_{d:%Y%m%d}.bi5")
    if os.path.exists(fn):
        return open(fn, "rb").read()
    url = _url(sym, d, side)
    for i in range(retries):
        try:
            r = S.get(url, timeout=30)
            if r.status_code == 200:
                open(fn, "wb").write(r.content)
                return r.content
            if r.status_code == 404:
                open(fn, "wb").write(b"")
                return b""
        except Exception:
            pass
        time.sleep(2 ** i)
    return None

def _decode(raw, sym, d):
    if not raw:
        return None
    try:
        data = lzma.LZMADecompressor().decompress(raw)
    except Exception:
        return None
    n = len(data) // 24
    if n == 0:
        return None
    arr = np.frombuffer(data[:n*24], dtype=np.dtype([
        ("t", ">i4"), ("o", ">i4"), ("c", ">i4"), ("l", ">i4"), ("h", ">i4"), ("v", ">f4")]))
    pt = point(sym)
    ts = pd.Timestamp(d, tz="UTC") + pd.to_timedelta(arr["t"].astype("int64"), unit="s")
    return pd.DataFrame({"open": arr["o"] * pt, "high": arr["h"] * pt, "low": arr["l"] * pt,
                         "close": arr["c"] * pt, "volume": arr["v"].astype("float64")}, index=ts)

def load(sym, start, end, sides=("BID", "ASK")):
    days = pd.date_range(start, end, freq="D")
    days = [d for d in days if d.weekday() != 5]  # skip Saturday
    out = {}
    for side in sides:
        with ThreadPoolExecutor(16) as ex:
            raws = list(ex.map(lambda d: _fetch(sym, d, side), days))
        frames = [f for f in (_decode(r, sym, d) for r, d in zip(raws, days)) if f is not None]
        if not frames:
            raise RuntimeError(f"no data {sym} {side}")
        df = pd.concat(frames).sort_index()
        df = df[~df.index.duplicated()]
        out[side] = df
    if len(sides) == 2:
        b, a = out["BID"], out["ASK"]
        idx = b.index.intersection(a.index)
        b, a = b.loc[idx], a.loc[idx]
        mid = (b[["open","high","low","close"]].values + a[["open","high","low","close"]].values) / 2
        df = pd.DataFrame(mid, index=idx, columns=["open","high","low","close"])
        df["spread"] = (a["close"] - b["close"]).values
        df["volume"] = b["volume"].values + a["volume"].values
        return df
    return out[sides[0]]

if __name__ == "__main__":
    sym, start, end = sys.argv[1], sys.argv[2], sys.argv[3]
    df = load(sym, start, end)
    p = os.path.join(os.path.dirname(__file__), "..", "data", f"{sym}_M1.parquet")
    df.to_parquet(p)
    print(sym, len(df), df.index[0], df.index[-1],
          "median spread (pips): %.2f" % (df.spread.median() / pip_size(sym)))
