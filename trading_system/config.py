"""
AI Trading System — Central Configuration
==========================================
Edit these values to tune the system behaviour.
"""
from dataclasses import dataclass, field
from typing import List

# ── Assets to watch ──────────────────────────────────────────────────────────
WATCHLIST = [
    "AAPL", "NVDA", "MSFT", "GOOGL", "META", "AMZN", "AMD", "TSLA",
    "CRM", "ORCL", "ADBE", "INTC", "QCOM", "JPM", "BAC", "GS", "V",
    "MA", "JNJ", "UNH", "PFE", "ABBV", "QQQ", "IWM", "DIA", "XLF",
    "XLK", "COIN", "PLTR", "RBLX", "UBER", "LYFT", "NFLX", "SNOW",
    "MSTR", "SPY"
]

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_LOOKBACK_DAYS   = 365 * 2   # training window
PREDICTION_HORIZON   = 3         # trading days ahead (24-72 h)
MIN_HISTORY_ROWS     = 60       # minimum bars required to generate a signal

# ── Model ────────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.65      # only trade when P(up) > this
ENSEMBLE_WEIGHTS     = {         # must sum to 1.0
    "xgboost":  0.45,
    "lstm":     0.35,
    "logreg":   0.20,
}
RETRAIN_EVERY_N_DAYS = 7         # auto-retrain cadence
DECAY_ALERT_THRESHOLD = 0.05     # flag if accuracy drops > 5 pp vs baseline

# ── Risk / Position Sizing ────────────────────────────────────────────────────
CAPITAL              = 100_000   # starting / current portfolio value ($)
MAX_RISK_PER_TRADE   = 0.02      # 2 % of capital max per trade
MAX_PORTFOLIO_HEAT   = 0.08      # total open risk across all trades
KELLY_FRACTION       = 0.50      # half-Kelly to reduce variance
ATR_STOP_MULT        = 2.0       # stop-loss = entry ± ATR_STOP_MULT × ATR
REWARD_RISK_RATIO    = 2.0       # take-profit = entry + ratio × risk

# ── Strategy Filters ─────────────────────────────────────────────────────────
RSI_OVERSOLD         = 35
RSI_OVERBOUGHT       = 70
VOLUME_SPIKE_RATIO   = 1.5       # volume must be > 1.5× 20-day avg to qualify

# ── Backtesting ───────────────────────────────────────────────────────────────
BACKTEST_COMMISSION  = 0.001     # 0.10 % per side
BACKTEST_SLIPPAGE    = 0.0005    # 0.05 % slippage

# ── Paths ─────────────────────────────────────────────────────────────────────
import pathlib
BASE_DIR    = pathlib.Path(__file__).parent
MODEL_DIR   = BASE_DIR / "models"
OUTPUT_DIR  = BASE_DIR / "outputs"
LOG_DIR     = BASE_DIR / "logs"
for _d in [MODEL_DIR, OUTPUT_DIR, LOG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)
