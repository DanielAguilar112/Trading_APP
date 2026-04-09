"""
core/data.py  v2  - Upgraded with 70+ features
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_LOOKBACK_DAYS, PREDICTION_HORIZON, MIN_HISTORY_ROWS

DATA_DIR = Path(__file__).parent.parent / "data"

def fetch_ohlcv(ticker: str, days: int = DATA_LOOKBACK_DAYS) -> pd.DataFrame:
    try:
        import yfinance as yf
        df = yf.download(ticker, period=f"{days}d", interval="1d",
                         auto_adjust=True, progress=False, timeout=8)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]
            df = df[["open","high","low","close","volume"]].copy()
            df.dropna(inplace=True)
            if len(df) >= MIN_HISTORY_ROWS:
                return df
    except Exception:
        pass
    csv = DATA_DIR / f"{ticker}_ohlcv.csv"
    if csv.exists():
        df = pd.read_csv(csv, index_col="Date", parse_dates=True)
        df.columns = [c.lower() for c in df.columns]
        df = df[["open","high","low","close","volume"]].copy()
        df.dropna(inplace=True)
        return df.iloc[-days:]
    return pd.DataFrame()

def _ema(s, span): return s.ewm(span=span, adjust=False).mean()
def _sma(s, p): return s.rolling(p).mean()

def _rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-10)))

def _atr(high, low, close, period=14):
    tr = pd.concat([high-low,(high-close.shift()).abs(),(low-close.shift()).abs()],axis=1).max(axis=1)
    return tr.rolling(period).mean()

def _macd(close, fast=12, slow=26, signal=9):
    m = _ema(close, fast) - _ema(close, slow)
    s = _ema(m, signal)
    return m, s, m - s

def _bollinger(close, period=20, num_std=2):
    mid = _sma(close, period)
    std = close.rolling(period).std()
    up = mid + num_std*std; lo = mid - num_std*std
    return mid, up, lo, (close-lo)/(up-lo+1e-10)

def _stochastic(high, low, close, k=14, d=3):
    k_val = 100*(close-low.rolling(k).min())/(high.rolling(k).max()-low.rolling(k).min()+1e-10)
    return k_val, k_val.rolling(d).mean()

def _obv(close, volume):
    return (np.sign(close.diff()).fillna(0)*volume).cumsum()

def _williams_r(high, low, close, period=14):
    return -100*(high.rolling(period).max()-close)/(high.rolling(period).max()-low.rolling(period).min()+1e-10)

def _cci(high, low, close, period=20):
    tp = (high+low+close)/3
    return (tp-_sma(tp,period))/(0.015*tp.rolling(period).apply(lambda x: np.mean(np.abs(x-x.mean())))+1e-10)

def _zscore(s, p=20):
    return (s - s.rolling(p).mean()) / (s.rolling(p).std() + 1e-10)

def _efficiency_ratio(close, period=10):
    direction = (close - close.shift(period)).abs()
    volatility = close.diff().abs().rolling(period).sum()
    return direction / (volatility + 1e-10)

def _chaikin_mf(high, low, close, volume, period=20):
    clv = ((close-low)-(high-close))/(high-low+1e-10)
    return (clv*volume).rolling(period).sum()/(volume.rolling(period).sum()+1e-10)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    f = df.copy()
    c, h, l, v = f["close"], f["high"], f["low"], f["volume"]

    # MAs and ratios
    for p in [5,10,20,50,100,200]:
        f[f"sma_{p}"] = _sma(c,p)
        f[f"sma_{p}_ratio"] = c/(f[f"sma_{p}"]+1e-10)
    for p in [8,13,21,34,55]:
        f[f"ema_{p}"] = _ema(c,p)
        f[f"ema_{p}_ratio"] = c/(f[f"ema_{p}"]+1e-10)

    f["golden_cross"] = (f["sma_50"]>f["sma_200"]).astype(int)
    f["ema_8_21_cross"] = (f["ema_8"]>f["ema_21"]).astype(int)
    f["ema_21_55_cross"] = (f["ema_21"]>f["ema_55"]).astype(int)
    f["trend_strength"] = abs(c-f["sma_50"])/(f["sma_50"]+1e-10)
    f["adx"] = _efficiency_ratio(c,14)*100

    # Momentum
    for p in [7,14,21]: f[f"rsi_{p}"] = _rsi(c,p)
    f["rsi_divergence"] = f["rsi_14"]-f["rsi_21"]
    f["macd"],f["macd_signal"],f["macd_hist"] = _macd(c)
    f["macd_cross"] = (f["macd"]>f["macd_signal"]).astype(int)
    f["macd_hist_change"] = f["macd_hist"].diff()
    f["stoch_k"],f["stoch_d"] = _stochastic(h,l,c)
    f["stoch_cross"] = (f["stoch_k"]>f["stoch_d"]).astype(int)
    f["stoch_overbought"] = (f["stoch_k"]>80).astype(int)
    f["stoch_oversold"] = (f["stoch_k"]<20).astype(int)
    f["williams_r"] = _williams_r(h,l,c)
    f["cci"] = _cci(h,l,c)
    for p in [1,3,5,10,20]: f[f"roc_{p}"] = c.pct_change(p)

    # Volatility
    f["atr_14"] = _atr(h,l,c,14)
    f["atr_pct"] = f["atr_14"]/(c+1e-10)
    f["atr_ratio"] = f["atr_14"]/(f["atr_14"].rolling(50).mean()+1e-10)
    _,boll_up,boll_lo,f["boll_pct_b"] = _bollinger(c)
    f["boll_bandwidth"] = (boll_up-boll_lo)/(c+1e-10)
    f["boll_squeeze"] = (f["boll_bandwidth"]<f["boll_bandwidth"].rolling(50).quantile(0.2)).astype(int)
    f["daily_return"] = c.pct_change()
    f["vol_10"] = f["daily_return"].rolling(10).std()
    f["vol_20"] = f["daily_return"].rolling(20).std()
    f["vol_50"] = f["daily_return"].rolling(50).std()
    f["vol_ratio_10_50"] = f["vol_10"]/(f["vol_50"]+1e-10)
    f["vol_regime"] = (f["vol_10"]>f["vol_20"]).astype(int)
    f["price_zscore"] = _zscore(c,20)

    # Volume
    for p in [10,20,50]: f[f"vol_sma_{p}"] = _sma(v,p)
    f["vol_ratio"] = v/(f["vol_sma_20"]+1e-10)
    f["vol_ratio_10"] = f["vol_sma_10"]/(f["vol_sma_50"]+1e-10)
    f["vol_spike"] = (f["vol_ratio"]>2.0).astype(int)
    f["vol_trend"] = (f["vol_sma_10"]>f["vol_sma_20"]).astype(int)
    f["obv"] = _obv(c,v)
    f["obv_ema"] = _ema(f["obv"],20)
    f["obv_trend"] = (f["obv"]>f["obv_ema"]).astype(int)
    f["obv_zscore"] = _zscore(f["obv"],20)
    f["chaikin_mf"] = _chaikin_mf(h,l,c,v)
    f["pv_trend"] = (c*v).rolling(5).mean()/((c*v).rolling(20).mean()+1e-10)

    # Price action
    f["high_low_range"] = (h-l)/(c+1e-10)
    f["close_position"] = (c-l)/(h-l+1e-10)
    f["gap"] = (f["open"]-c.shift(1))/(c.shift(1)+1e-10)
    f["upper_shadow"] = (h-f[["open","close"]].max(axis=1))/(c+1e-10)
    f["lower_shadow"] = (f[["open","close"]].min(axis=1)-l)/(c+1e-10)
    don_hi = h.rolling(20).max(); don_lo = l.rolling(20).min()
    f["donchian_pos"] = (c-don_lo)/(don_hi-don_lo+1e-10)
    f["near_52w_high"] = c/(h.rolling(252).max()+1e-10)
    f["near_52w_low"] = c/(l.rolling(252).min()+1e-10)
    for lag in [1,2,3,5,10]: f[f"lag_{lag}"] = c.pct_change(lag)
    f["mean_rev_signal"] = _zscore(c-f["sma_20"],10)
    f["efficiency_ratio"] = _efficiency_ratio(c,10)

    # Target
    future = c.shift(-PREDICTION_HORIZON)/c - 1
    f["target"] = (future > 0).astype(int)
    f.dropna(inplace=True)
    return f

FEATURE_COLS = [
    "sma_5_ratio","sma_10_ratio","sma_20_ratio","sma_50_ratio","sma_200_ratio",
    "ema_8_ratio","ema_21_ratio","ema_55_ratio",
    "golden_cross","ema_8_21_cross","ema_21_55_cross","trend_strength","adx",
    "rsi_7","rsi_14","rsi_21","rsi_divergence",
    "macd","macd_hist","macd_cross","macd_hist_change",
    "stoch_k","stoch_d","stoch_cross","stoch_overbought","stoch_oversold",
    "williams_r","cci",
    "roc_1","roc_3","roc_5","roc_10","roc_20",
    "atr_pct","atr_ratio","boll_pct_b","boll_bandwidth","boll_squeeze",
    "vol_10","vol_20","vol_ratio_10_50","vol_regime","price_zscore",
    "vol_ratio","vol_ratio_10","vol_spike","vol_trend",
    "obv_trend","obv_zscore","chaikin_mf","pv_trend",
    "high_low_range","close_position","gap","upper_shadow","lower_shadow",
    "donchian_pos","near_52w_high","near_52w_low",
    "lag_1","lag_2","lag_3","lag_5","lag_10",
    "mean_rev_signal","efficiency_ratio",
]

def prepare_dataset(ticker: str) -> Optional[pd.DataFrame]:
    raw = fetch_ohlcv(ticker)
    if raw.empty or len(raw) < MIN_HISTORY_ROWS:
        return None
    return build_features(raw)