"""
broker/executor.py
==================
The Executor bridges AI signals → broker orders.

Responsibilities
----------------
  1. Deduplication    — don't double-enter an existing position
  2. Heat check       — total open risk stays under MAX_PORTFOLIO_HEAT
  3. Order sizing     — shares = dollar_risk / risk_per_share
  4. Bracket entry    — market in + OCO stop-loss / take-profit
  5. Exit management  — update stops, close on signal reversal
  6. Journalling      — every action written to trade_journal.json

Usage
-----
  from broker.executor import Executor
  from broker.client   import BrokerClient, BrokerMode

  client = BrokerClient(mode=BrokerMode.DRY_RUN)
  client.connect()

  executor = Executor(client)
  executor.execute_signals(signals)   # list of TradeSignal objects
"""
import json
import logging
from datetime import datetime
from pathlib  import Path
from typing   import Dict, List, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config          import (CAPITAL, MAX_PORTFOLIO_HEAT, MAX_RISK_PER_TRADE,
                              OUTPUT_DIR)
from core.strategy   import TradeSignal
from broker.client   import Order, OrderType, OrderStatus, Position

log = logging.getLogger(__name__)
JOURNAL_PATH = OUTPUT_DIR / "trade_journal.json"


# ─────────────────────────────────────────────────────────────────────────────
# Executor
# ─────────────────────────────────────────────────────────────────────────────

class Executor:

    def __init__(self, broker, dry_run_price_map: Dict[str, float] = None):
        """
        broker            : DryRunBroker or AlpacaBroker instance
        dry_run_price_map : {ticker: price} — used in dry-run mode when no
                            live quote feed is available
        """
        self.broker       = broker
        self._price_map   = dry_run_price_map or {}
        self._journal: List[Dict] = self._load_journal()

    # ── Main entry point ─────────────────────────────────────────────────────

    def execute_signals(self, signals: List[TradeSignal]) -> Dict:
        """
        Process a list of TradeSignal objects and execute appropriate orders.
        Returns a summary dict.
        """
        results = {
            "executed": [], "skipped": [], "closed": [],
            "errors":   [], "timestamp": str(datetime.now()),
        }

        if not self.broker.is_connected():
            log.error("[Executor] Broker not connected")
            results["errors"].append("Broker not connected")
            return results

        # ── Refresh price map ─────────────────────────────────────────────
        self._refresh_prices([s.ticker for s in signals])

        # ── Portfolio heat before we start ────────────────────────────────
        open_positions = self.broker.get_positions()
        portfolio_value = self._get_portfolio_value()
        open_heat = self._calc_heat(open_positions, portfolio_value)
        log.info("[Executor] Portfolio heat = %.1f%% open positions = %d",
                 open_heat * 100, len(open_positions))

        # ── Process each signal ───────────────────────────────────────────
        for signal in signals:
            try:
                outcome = self._process_signal(
                    signal, open_positions, open_heat, portfolio_value
                )
                if outcome["action"] == "ENTERED":
                    results["executed"].append(outcome)
                    # Update heat estimate
                    open_heat += signal.position_size_pct / 100
                elif outcome["action"] == "CLOSED":
                    results["closed"].append(outcome)
                else:
                    results["skipped"].append(outcome)
            except Exception as e:
                log.error("[Executor] Error processing %s: %s", signal.ticker, e)
                results["errors"].append({"ticker": signal.ticker, "error": str(e)})

        self._save_journal()
        self._print_execution_summary(results)
        return results

    # ── Signal processor ─────────────────────────────────────────────────────

    def _process_signal(self, signal: TradeSignal,
                         open_positions: Dict[str, Position],
                         open_heat: float,
                         portfolio_value: float) -> Dict:
        ticker  = signal.ticker
        action  = signal.action
        price   = self._price_map.get(ticker, signal.entry_price or 0)

        # ── HOLD: do nothing (or close if we have a conflicting position) ──
        if action == "HOLD":
            # If we're long and signal flipped to HOLD, keep position
            return self._skip(signal, "Below confidence threshold")

        # ── Check if we already have a position ───────────────────────────
        existing = open_positions.get(ticker)

        if existing:
            same_direction = (action == "BUY"  and existing.side == "long") or \
                             (action == "SELL" and existing.side == "short")
            if same_direction:
                return self._skip(signal, "Already in position (same direction)")

            # Opposite direction → close existing first
            log.info("[Executor] %s signal reversal — closing existing %s",
                     ticker, existing.side)
            close_result = self._close_position(ticker, price)
            if close_result:
                return {"action": "CLOSED", "ticker": ticker,
                        "reason": "Signal reversal",
                        "close_price": price}

        # ── Heat gate ─────────────────────────────────────────────────────
        if open_heat + signal.position_size_pct / 100 > MAX_PORTFOLIO_HEAT:
            return self._skip(signal,
                f"Portfolio heat {open_heat:.1%} would exceed "
                f"limit {MAX_PORTFOLIO_HEAT:.1%} — skipping")

        # ── Size in shares ────────────────────────────────────────────────
        if price <= 0:
            return self._skip(signal, "Could not determine entry price")

        qty = self._calc_qty(signal, price, portfolio_value)
        if qty <= 0:
            return self._skip(signal, "Calculated qty = 0 — position too small")

        # ── Submit bracket order ──────────────────────────────────────────
        sl = round(price * 0.98 if action == "BUY" else price * 1.02, 2)
        tp = round(price * 1.04 if action == "BUY" else price * 0.96, 2)

        log.info("[Executor] Entering %s %s qty=%.4f  entry=%.4f  SL=%.4f  TP=%.4f",
                 action, ticker, qty, price, sl, tp)

        orders = self.broker.submit_bracket(
            ticker=ticker, side=action.lower(), qty=qty,
            entry_price=price, stop_loss=sl, take_profit=tp,
            confidence=signal.confidence,
        )

        entry_order = orders["entry"]
        success = entry_order.status not in (OrderStatus.REJECTED,)

        outcome = {
            "action":     "ENTERED" if success else "FAILED",
            "ticker":     ticker,
            "signal":     action,
            "qty":        qty,
            "entry":      price,
            "stop_loss":  sl,
            "take_profit": tp,
            "confidence": signal.confidence_pct,
            "position_pct": signal.position_size_pct,
            "order_id":   entry_order.id,
            "timestamp":  str(datetime.now()),
        }
        self._journal_entry(outcome)
        return outcome

    # ── Position management ───────────────────────────────────────────────────

    def update_stops(self, ticker: str, new_stop: float) -> bool:
        """Trail the stop-loss up (for long) — cancel old, submit new stop."""
        log.info("[Executor] Updating stop for %s → %.4f", ticker, new_stop)
        # Cancel existing stop orders for this ticker
        for order in self.broker.get_open_orders() if hasattr(self.broker, "get_open_orders") else []:
            if order.ticker == ticker and order.tag == "stop_loss":
                self.broker.cancel_order(order.id)

        # Submit new stop
        pos = self.broker.get_position(ticker)
        if pos is None:
            return False
        sl_side = "sell" if pos.side == "long" else "buy"
        stop_order = Order(ticker=ticker, side=sl_side, qty=pos.qty,
                           order_type=OrderType.STOP,
                           stop_price=new_stop, tag="stop_loss")
        self.broker.submit_order(stop_order, new_stop)
        return True

    def _close_position(self, ticker: str, price: float) -> bool:
        order = self.broker.close_position(ticker, price)
        return order is not None

    # ── Tick: called on each price update (for dry-run bracket checks) ────────

    def tick(self, price_map: Dict[str, float]):
        """
        Feed new prices to the broker (triggers bracket exits in dry-run mode).
        Call this once per bar in live operation.
        """
        self._price_map.update(price_map)
        if hasattr(self.broker, "tick"):
            self.broker.tick(price_map)

    # ── Risk helpers ─────────────────────────────────────────────────────────

    def _calc_qty(self, signal: TradeSignal,
                  price: float, portfolio_value: float) -> float:
        """Shares = max_dollar_risk / risk_per_share, rounded down."""
        dollar_alloc  = portfolio_value * (signal.position_size_pct / 100)
        sl_price      = signal.stop_loss or (price * 0.98)
        risk_per_share = abs(price - sl_price)
        if risk_per_share < 1e-6:
            risk_per_share = price * 0.02
        max_dollar_risk = portfolio_value * MAX_RISK_PER_TRADE
        qty_by_risk     = max_dollar_risk / risk_per_share
        qty_by_alloc    = dollar_alloc / price
        qty = min(qty_by_risk, qty_by_alloc)
        # Allow fractional shares (crypto); for equities round down to 2dp
        return max(int(qty), 1)

    def _calc_heat(self, positions: Dict[str, Position],
                   portfolio_value: float) -> float:
        """Total open risk as fraction of portfolio."""
        if not positions or portfolio_value <= 0:
            return 0.0
        total_risk = sum(
            abs(p.avg_entry - (p.stop_loss or p.avg_entry * 0.98)) * p.qty
            for p in positions.values()
        )
        return total_risk / portfolio_value

    def _get_portfolio_value(self) -> float:
        try:
            acct = self.broker.get_account()
            return float(acct.get("portfolio_value", CAPITAL))
        except Exception:
            return CAPITAL

    def _refresh_prices(self, tickers: List[str]):
        """Try to get live quotes; fall back to stored prices."""
        if hasattr(self.broker, "get_latest_prices"):
            try:
                prices = self.broker.get_latest_prices(tickers)
                self._price_map.update(prices)
            except Exception:
                pass

    # ── Journal ───────────────────────────────────────────────────────────────

    def _load_journal(self) -> List[Dict]:
        if JOURNAL_PATH.exists():
            try:
                return json.loads(JOURNAL_PATH.read_text())
            except Exception:
                pass
        return []

    def _journal_entry(self, entry: Dict):
        self._journal.append(entry)

    def _save_journal(self):
        try:
            JOURNAL_PATH.write_text(
                json.dumps(self._journal, indent=2, default=str)
            )
        except Exception as e:
            log.warning("[Executor] Could not save journal: %s", e)

    @staticmethod
    def _skip(signal: TradeSignal, reason: str) -> Dict:
        log.info("[Executor] SKIP %s — %s", signal.ticker, reason)
        return {"action": "SKIPPED", "ticker": signal.ticker, "reason": reason}

    # ── Summary printer ───────────────────────────────────────────────────────

    @staticmethod
    def _print_execution_summary(results: Dict):
        print(f"\n{'─'*58}")
        print("  EXECUTION SUMMARY")
        print(f"{'─'*58}")
        if results["executed"]:
            print(f"  ✓ Entered ({len(results['executed'])}):")
            for r in results["executed"]:
                print(f"    {r['ticker']:8} {r['signal']:4}  qty={r['qty']:.4f}"
                      f"  @{r['entry']:.4f}  SL={r['stop_loss']:.4f}"
                      f"  TP={r['take_profit']:.4f}  conf={r['confidence']}%")
        if results["closed"]:
            print(f"  ↩ Closed ({len(results['closed'])}):")
            for r in results["closed"]:
                print(f"    {r['ticker']:8} — {r['reason']}")
        if results["skipped"]:
            print(f"  ○ Skipped ({len(results['skipped'])}):")
            for r in results["skipped"]:
                print(f"    {r['ticker']:8} — {r.get('reason', r.get('error', 'unknown'))}")
        if results["errors"]:
            print(f"  ✗ Errors ({len(results['errors'])}):")
            for r in results["errors"]:
                print(f"    {r}")
        print()
