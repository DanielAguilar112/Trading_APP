"""
core/data.py
============
Fetches OHLCV data via yfinance and computes all technical features
used by the prediction models.
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


# ─────────────────────────────────────────────────────────────────────────────
# Raw data
# ─────────────────────────────────────────────────────────────────────────────

def fetch_ohlcv(ticker: str, days: int = DATA_LOOKBACK_DAYS) -> pd.DataFrame:
    """
    1. Try yfinance (live).
    2. Fall back to local CSV at data/<ticker>_ohlcv.csv.
    """
    try:
        import yfinance as yf
        df = yf.download(ticker, period=f"{days}d", interval="1d",
                         auto_adjust=True, progress=False, timeout=8)
        if not df.empty:
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


# ─────────────────────────────────────────────────────────────────────────────
# Technical indicators (pure-pandas, no TA-Lib dependency)
# ─────────────────────────────────────────────────────────────────────────────

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def _atr(high, low, close, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def _macd(close: pd.Series, fast=12, slow=26, signal=9):
    macd_line   = _ema(close, fast) - _ema(close, slow)
    signal_line = _ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line

def _bollinger(close: pd.Series, period=20, num_std=2):
    mid   = close.rolling(period).mean()
    std   = close.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    pct_b = (close - lower) / (upper - lower + 1e-10)
    return mid, upper, lower, pct_b

def _stochastic(high, low, close, k_period=14, d_period=3):
    low_min  = low.rolling(k_period).min()
    high_max = high.rolling(k_period).max()
    k = 100 * (close - low_min) / (high_max - low_min + 1e-10)
    d = k.rolling(d_period).mean()
    return k, d

def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()

def _vwap(high, low, close, volume) -> pd.Series:
    typical = (high + low + close) / 3
    return (typical * volume).cumsum() / volume.cumsum()


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ~35 technical features to an OHLCV DataFrame.
    Rows with NaN (warm-up period) are dropped before returning.
    """
    f = df.copy()
    c, h, l, v = f["close"], f["high"], f["low"], f["volume"]

    # ── Trend / MA ────────────────────────────────────────────────────────────
    for p in [5, 10, 20, 50, 200]:
        f[f"sma_{p}"]    = c.rolling(p).mean()
        f[f"close_sma{p}_ratio"] = c / f[f"sma_{p}"]
    for p in [12, 26, 50]:
        f[f"ema_{p}"]    = _ema(c, p)
        f[f"close_ema{p}_ratio"] = c / f[f"ema_{p}"]

    f["ma_cross_50_200"] = (f["sma_50"] > f["sma_200"]).astype(int)  # golden/death cross

    # ── Momentum / Oscillators ────────────────────────────────────────────────
    f["rsi_14"]   = _rsi(c, 14)
    f["rsi_7"]    = _rsi(c, 7)

    f["macd"], f["macd_signal"], f["macd_hist"] = _macd(c)
    f["macd_cross"] = (f["macd"] > f["macd_signal"]).astype(int)

    f["stoch_k"], f["stoch_d"] = _stochastic(h, l, c)
    f["stoch_cross"] = (f["stoch_k"] > f["stoch_d"]).astype(int)

    # ── Volatility ────────────────────────────────────────────────────────────
    f["atr_14"]   = _atr(h, l, c, 14)
    f["atr_pct"]  = f["atr_14"] / c          # normalised ATR

    _, boll_up, boll_lo, f["boll_pct_b"] = _bollinger(c)
    f["boll_bandwidth"] = (boll_up - boll_lo) / c

    f["daily_return"]   = c.pct_change()
    f["volatility_10"]  = f["daily_return"].rolling(10).std()
    f["volatility_20"]  = f["daily_return"].rolling(20).std()

    # ── Volume ────────────────────────────────────────────────────────────────
    f["vol_sma20"]       = v.rolling(20).mean()
    f["vol_ratio"]       = v / f["vol_sma20"]     # >1.5 = volume spike
    f["obv"]             = _obv(c, v)
    f["obv_ema"]         = _ema(f["obv"], 20)
    f["obv_trend"]       = (f["obv"] > f["obv_ema"]).astype(int)

    # ── Price action ─────────────────────────────────────────────────────────
    f["high_low_range"]  = (h - l) / c
    f["close_position"]  = (c - l) / (h - l + 1e-10)  # where close sits in bar
    for lag in [1, 2, 3, 5]:
        f[f"return_lag{lag}"] = c.pct_change(lag)

    # ── Target variable ───────────────────────────────────────────────────────
    future_return = c.shift(-PREDICTION_HORIZON) / c - 1
    f["target"] = (future_return > 0).astype(int)

    # ── Clean up ──────────────────────────────────────────────────────────────
    f.dropna(inplace=True)
    return f


FEATURE_COLS = [
    # Trend
    "close_sma5_ratio","close_sma10_ratio","close_sma20_ratio",
    "close_sma50_ratio","close_ema12_ratio","close_ema26_ratio",
    "ma_cross_50_200",
    # Momentum
    "rsi_14","rsi_7","macd","macd_hist","macd_cross",
    "stoch_k","stoch_d","stoch_cross",
    # Volatility
    "atr_pct","boll_pct_b","boll_bandwidth","volatility_10","volatility_20",
    # Volume
    "vol_ratio","obv_trend",
    # Price action
    "high_low_range","close_position",
    "return_lag1","return_lag2","return_lag3","return_lag5",
]


def prepare_dataset(ticker: str) -> Optional[pd.DataFrame]:
    """Full pipeline: download → features → ready DataFrame."""
    raw = fetch_ohlcv(ticker)
    if raw.empty or len(raw) < MIN_HISTORY_ROWS:
        return None
    return build_features(raw)
