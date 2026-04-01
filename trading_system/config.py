"""
AI Trading System — Central Configuration
==========================================
Edit these values to tune the system behaviour.
"""
from dataclasses import dataclass, field
from typing import List

# ── Assets to watch ──────────────────────────────────────────────────────────
WATCHLIST = ['AAPL', 'ABBV', 'ABT', 'ACN', 'ADBE', 'ADI', 'AEP', 'AFRM', 'ALNY', 'AMAT', 'AMD', 'AMGN', 'AMZN', 'AON', 'APD', 'APP', 'ARKK', 'ARM', 'AVGO', 'AXP', 'BA', 'BABA', 'BAC', 'BIIB', 'BILL', 'BKNG', 'BLK', 'BMRN', 'BMY', 'BNTX', 'BRK-B', 'BURL', 'C', 'CAG', 'CARR', 'CAT', 'CB', 'CDNS', 'CFLT', 'CHD', 'CI', 'CL', 'CME', 'COF', 'COIN', 'COP', 'COST', 'CPB', 'CRM', 'CRWD', 'CSCO', 'CTAS', 'CVX', 'D', 'DDOG', 'DE', 'DG', 'DHR', 'DIA', 'DLTR', 'DOCU', 'DUK', 'DVN', 'DXCM', 'ED', 'EEM', 'EFA', 'ELV', 'EMR', 'EOG', 'ES', 'ESTC', 'ETN', 'EXC', 'F', 'FCX', 'FDX', 'FTNT', 'FXI', 'GE', 'GEV', 'GILD', 'GIS', 'GLD', 'GM', 'GOOG', 'GOOGL', 'GS', 'HD', 'HON', 'HOOD', 'HRL', 'HUBS', 'HUM', 'HYG', 'IBM', 'ICE', 'INTC', 'INTU', 'ISRG', 'ITW', 'IWM', 'JNJ', 'JPM', 'KHC', 'KLAC', 'KO', 'KR', 'KWEB', 'LCID', 'LIN', 'LLY', 'LOW', 'LRCX', 'LULU', 'LYFT', 'MA', 'MAR', 'MCD', 'MCO', 'MDB', 'MDT', 'META', 'MKC', 'MNST', 'MPC', 'MRK', 'MRNA', 'MRVL', 'MS', 'MSFT', 'MSI', 'MSTR', 'MU', 'NEE', 'NET', 'NFLX', 'NIO', 'NKE', 'NOW', 'NSC', 'NVAX', 'NVDA', 'OKTA', 'OPEN', 'ORCL', 'ORLY', 'OXY', 'PANW', 'PAYC', 'PCAR', 'PCG', 'PEP', 'PFE', 'PG', 'PH', 'PLD', 'PLTR', 'PM', 'PSX', 'QCOM', 'QQQ', 'RBLX', 'REGN', 'RIVN', 'ROKU', 'ROP', 'ROST', 'RTX', 'SBUX', 'SCHW', 'SHOP', 'SJM', 'SLB', 'SLV', 'SMCI', 'SNOW', 'SNPS', 'SO', 'SOFI', 'SPGI', 'SPY', 'SRE', 'SYK', 'SYY', 'T', 'TDG', 'TEAM', 'TGT', 'TJX', 'TLT', 'TMO', 'TSLA', 'TSM', 'TSN', 'TT', 'TTD', 'TXN', 'U', 'UBER', 'UNH', 'UPST', 'USO', 'V', 'VEEV', 'VLO', 'VRTX', 'VXX', 'WDAY', 'WELL', 'WFC', 'WM', 'WMT', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY', 'XOM', 'ZM', 'ZS', 'ZTS']
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
