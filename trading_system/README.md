# AI Trading System

Probabilistic trading system with full broker execution layer.

## Architecture

```
trading_system/
├── config.py               ← ALL tunable parameters in one place
├── main.py                 ← Master CLI entry point
├── requirements.txt
│
├── core/
│   ├── data.py             ← yfinance fetch + 35 technical features
│   ├── models.py           ← XGBoost + LSTM + LogReg + Ensemble
│   ├── strategy.py         ← Signal gen, Kelly sizing, ATR stops
│   └── trainer.py          ← Retrain scheduler + decay detection
│
├── backtest/
│   └── engine.py           ← Walk-forward backtester + analytics
│
└── broker/
    ├── client.py           ← BrokerClient factory (DRY_RUN/PAPER/LIVE)
    ├── dry_run.py          ← Full simulation, no credentials needed
    ├── alpaca_broker.py    ← Alpaca Markets (paper + live)
    ├── executor.py         ← Signal → Order with risk management
    └── live_loop.py        ← Continuous trading loop + scheduler
```

## Quick Start

```bash
pip install -r requirements.txt

# 1. Train models
python main.py --mode train --tickers AAPL NVDA MSFT TSLA BTC-USD

# 2. Generate signals
python main.py --mode signals

# 3. Signals + execute (dry run — safest, no credentials needed)
python main.py --mode execute --broker dry_run

# 4. Backtest
python main.py --mode backtest --ticker AAPL --days 365

# 5. Check model health
python main.py --mode status
```

## Alpaca Paper Trading

1. Sign up at https://alpaca.markets (free)
2. Go to Paper Trading → API Keys → Generate New Key
3. Run:

```bash
export ALPACA_API_KEY="PKxxxxxxxxxxxxxxxxxx"
export ALPACA_SECRET_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Single cycle (test it once)
python main.py --mode loop --broker paper --once

# Continuous loop (runs every 30 min during market hours)
python main.py --mode loop --broker paper
```

## Live Trading

Live trading requires the `--live` flag as an explicit safety confirmation.

```bash
export ALPACA_API_KEY="AKxxxxxxxxxxxxxxxxxx"   # Note: AK prefix for live
export ALPACA_SECRET_KEY="xxxxxxxxxxxxxxxxxxxx"

python main.py --mode loop --broker live --live
```

**IMPORTANT**: Only trade with capital you can afford to lose.
This system is for educational purposes.

## Key Parameters (config.py)

| Parameter              | Default  | Description                        |
|------------------------|----------|------------------------------------|
| `CONFIDENCE_THRESHOLD` | 0.60     | Min P(up/down) to take a trade     |
| `MAX_RISK_PER_TRADE`   | 0.02     | Max 2% of capital per trade        |
| `MAX_PORTFOLIO_HEAT`   | 0.08     | Max 8% total open risk             |
| `KELLY_FRACTION`       | 0.50     | Half-Kelly (reduces variance)      |
| `ATR_STOP_MULT`        | 2.0      | Stop = entry ± 2 × ATR             |
| `REWARD_RISK_RATIO`    | 2.0      | TP = entry + 2 × risk              |
| `RETRAIN_EVERY_N_DAYS` | 7        | Weekly model retraining            |

## Broker Modes

| Mode       | Real money | Credentials | Use case                   |
|------------|-----------|-------------|----------------------------|
| `dry_run`  | No        | None        | Testing, development       |
| `paper`    | No        | Alpaca keys | Pre-live validation        |
| `live`     | YES       | Alpaca keys | Production (use carefully) |

## Output Files

| File                         | Contents                              |
|------------------------------|---------------------------------------|
| `outputs/signals_latest.json`| Latest signal recommendations         |
| `outputs/trade_journal.json` | Full execution log                    |
| `outputs/dry_run_state.json` | Dry-run portfolio snapshot            |
| `outputs/backtest_TICKER.csv`| Equity curve vs buy & hold            |
| `logs/trainer.log`           | Model training history                |
| `logs/live_loop.log`         | Live loop execution log               |
| `models/xgboost_TICKER.pkl`  | Saved XGBoost model                   |
| `models/lstm_TICKER.pkl`     | Saved LSTM model                      |
| `models/logreg_TICKER.pkl`   | Saved logistic regression model       |
| `models/registry.json`       | Model training registry + metrics     |
