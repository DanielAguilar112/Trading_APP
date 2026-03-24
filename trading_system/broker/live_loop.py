"""
broker/live_loop.py
===================
Production trading loop.

Runs continuously and:
  1. Waits for market open (NYSE hours)
  2. Retrains models if scheduled
  3. Generates signals at configurable intervals
  4. Executes via the Executor
  5. Monitors open positions and updates trailing stops
  6. Saves state on market close

Usage
-----
  # Dry-run (safest — no real money, no credentials needed)
  python broker/live_loop.py --mode dry_run

  # Paper trading (Alpaca paper account)
  python broker/live_loop.py --mode paper \
      --api-key PKXXX --secret-key XXXXX

  # Live trading (requires explicit --live flag)
  python broker/live_loop.py --mode live \
      --api-key AKXXX --secret-key XXXXX --live

  # Single-shot (train + signal + execute once, then exit)
  python broker/live_loop.py --mode paper --once \
      --api-key PKXXX --secret-key XXXXX
"""
import argparse
import logging
import os
import time
from datetime import datetime, time as dtime
from pathlib  import Path
from typing   import Dict, List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config          import WATCHLIST, CAPITAL, RETRAIN_EVERY_N_DAYS, OUTPUT_DIR
from core.data       import prepare_dataset
from core.models     import EnsembleModel
from core.strategy   import SignalGenerator, TradeSignal
from core.trainer    import ModelTrainer
from broker.client   import BrokerClient, BrokerMode
from broker.executor import Executor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_DIR / "live_loop.log"),
    ],
)
log = logging.getLogger(__name__)

# Market hours (Eastern) — adjust for crypto (24/7)
MARKET_OPEN  = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)
SIGNAL_INTERVAL_MINUTES = 30   # re-check signals every 30 min during session


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def is_market_hours(broker=None) -> bool:
    """Check market hours — prefers broker clock, falls back to local time."""
    if broker and hasattr(broker, "is_market_open"):
        try:
            return broker.is_market_open()
        except Exception:
            pass
    now = datetime.now().time()
    weekday = datetime.now().weekday()
    return weekday < 5 and MARKET_OPEN <= now <= MARKET_CLOSE


def seconds_until_open() -> int:
    now = datetime.now()
    if now.weekday() >= 5:
        days_ahead = 7 - now.weekday()
        target = now.replace(hour=9, minute=30, second=0, microsecond=0)
        return int((target - now).total_seconds()) + days_ahead * 86400
    target = now.replace(hour=9, minute=30, second=0, microsecond=0)
    if now.time() > MARKET_OPEN:
        return 0   # already open
    return max(0, int((target - now).total_seconds()))


# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline (one full signal → execution cycle)
# ─────────────────────────────────────────────────────────────────────────────

def run_cycle(tickers: List[str], trainer: ModelTrainer,
              generator: SignalGenerator, executor: Executor,
              broker) -> Dict:
    """One complete: fetch → predict → signal → execute cycle."""
    log.info("--- Starting cycle at %s ---", datetime.now().strftime("%H:%M:%S"))
    signals: List[TradeSignal] = []

    # ── 1. Refresh prices ─────────────────────────────────────────────────
    price_map: Dict[str, float] = {}

    # ── 2. For each ticker: load/retrain model → predict → signal ─────────
    open_heat = 0.0
    for ticker in tickers:
        try:
            # Retrain if needed
            model = EnsembleModel()
            if not model.load(ticker) or trainer.needs_retrain(ticker):
                log.info("Retraining %s …", ticker)
                m = trainer.train(ticker, force=True)
                if m:
                    model = m
                else:
                    continue

            df = prepare_dataset(ticker)
            if df is None or len(df) < 30:
                log.warning("Skipping %s — insufficient data", ticker)
                continue

            # Track last known price
            last_price = float(df.iloc[-1]["close"])
            price_map[ticker] = last_price

            prediction = model.predict(df)
            signal     = generator.generate(ticker, df, prediction, open_heat)
            signals.append(signal)

            if signal.action != "HOLD":
                open_heat += signal.position_size_pct / 100

            log.info("  %s -> %s  conf=%d%%  p_up=%.2f",
                     ticker, signal.action, signal.confidence_pct, signal.p_up)

        except Exception as e:
            log.error("Error processing %s: %s", ticker, e)

    # ── 3. Tick broker (bracket exit checks for dry-run) ──────────────────
    executor.tick(price_map)

    # ── 4. Execute signals ────────────────────────────────────────────────
    if signals:
        # Pass price map into executor for accurate order sizing
        executor._price_map.update(price_map)
        results = executor.execute_signals(signals)
    else:
        results = {"executed": [], "skipped": [], "errors": []}
        log.info("No actionable signals this cycle.")

    # ── 5. Print account state ────────────────────────────────────────────
    broker.print_summary()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def run_loop(mode: BrokerMode, tickers: List[str],
             api_key: str = "", secret_key: str = "",
             live: bool = False, once: bool = False,
             capital: float = CAPITAL):

    print(f"""
╔═══════════════════════════════════════════════════════╗
║  AI TRADING SYSTEM — LIVE LOOP                        ║
║  Mode    : {mode.upper():12}                          ║
║  Tickers : {(str(len(tickers))+' tickers'):30}        ║
║  Capital : ${capital:>10,.0f}                         ║
╚═══════════════════════════════════════════════════════╝
""")

    # ── Broker ────────────────────────────────────────────────────────────
    broker = BrokerClient(
        mode=mode, api_key=api_key, secret_key=secret_key, live=live
    )
    if not broker.connect():
        log.error("Could not connect to broker — aborting")
        return

    # ── Components ────────────────────────────────────────────────────────
    trainer   = ModelTrainer()
    generator = SignalGenerator(capital=capital)
    executor  = Executor(broker)

    # ── Single-shot mode ──────────────────────────────────────────────────
    if once:
        log.info("Running single-shot cycle …")
        run_cycle(tickers, trainer, generator, executor, broker)
        broker.print_summary()
        if hasattr(broker, "save_state"):
            broker.save_state()
        return

    # ── Continuous loop ───────────────────────────────────────────────────
    log.info("Entering continuous loop. Press Ctrl+C to stop.")
    last_cycle = None

    try:
        while True:
            now = datetime.now()

            if not is_market_hours(broker):
                wait = seconds_until_open()
                log.info("Market closed. Next open in ~%dh %dm.",
                         wait // 3600, (wait % 3600) // 60)
                # Save state at end of day
                if hasattr(broker, "save_state"):
                    broker.save_state()
                # Sleep until 5 min before open (or 1h if far away)
                sleep_sec = min(wait - 300, 3600) if wait > 300 else 60
                time.sleep(max(sleep_sec, 60))
                continue

            # Run cycle at most every SIGNAL_INTERVAL_MINUTES
            if (last_cycle is None or
                    (now - last_cycle).total_seconds() >= SIGNAL_INTERVAL_MINUTES * 60):
                run_cycle(tickers, trainer, generator, executor, broker)
                last_cycle = now

            # Poll every 60 seconds
            time.sleep(60)

    except KeyboardInterrupt:
        log.info("Shutting down …")
        broker.print_summary()
        if hasattr(broker, "save_state"):
            broker.save_state()
        log.info("State saved. Goodbye.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="AI Trading Live Loop")
    p.add_argument("--mode",       choices=["dry_run","paper","live"], default="dry_run")
    p.add_argument("--tickers",    nargs="+", default=WATCHLIST)
    p.add_argument("--api-key",    default=os.getenv("ALPACA_API_KEY",""))
    p.add_argument("--secret-key", default=os.getenv("ALPACA_SECRET_KEY",""))
    p.add_argument("--live",       action="store_true")
    p.add_argument("--once",       action="store_true", help="Single cycle then exit")
    p.add_argument("--capital",    type=float, default=CAPITAL)
    args = p.parse_args()

    run_loop(
        mode       = BrokerMode(args.mode),
        tickers    = args.tickers,
        api_key    = args.api_key,
        secret_key = args.secret_key,
        live       = args.live,
        once       = args.once,
        capital    = args.capital,
    )
