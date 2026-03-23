"""
broker/dry_run.py
=================
Full paper-trading simulation that requires zero credentials.
Simulates:
  - Market / limit / stop / stop-limit orders
  - Realistic slippage and commission
  - OCO bracket order logic (stop-loss + take-profit)
  - Portfolio equity tracking
  - Position mark-to-market
"""
import logging
import uuid
from datetime import datetime
from pathlib  import Path
from typing   import Dict, List, Optional
import json, sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config   import CAPITAL, BACKTEST_COMMISSION, BACKTEST_SLIPPAGE, OUTPUT_DIR
from broker.client import (
    BrokerMode, Order, OrderStatus, OrderSide, OrderType, Position
)

log = logging.getLogger(__name__)


class DryRunBroker:
    """Simulated broker — no network, perfect for CI and offline testing."""

    def __init__(self, mode=BrokerMode.DRY_RUN, **kwargs):
        self.mode        = BrokerMode.DRY_RUN
        self.cash        = CAPITAL
        self.commission  = BACKTEST_COMMISSION
        self.slippage    = BACKTEST_SLIPPAGE
        self.positions:  Dict[str, Position] = {}
        self.orders:     List[Order]         = []
        self.order_map:  Dict[str, Order]    = {}
        self.trade_log:  List[Dict]          = []
        self._connected  = False
        log.info("[DryRun] Broker initialised - capital $%.0f", self.cash)

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        self._connected = True
        log.info("[DryRun] Connected (simulation mode)")
        return True

    def disconnect(self):
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    # ── Account ──────────────────────────────────────────────────────────────

    def get_account(self) -> Dict:
        equity = self.cash + sum(
            p.unrealized_pnl for p in self.positions.values()
        )
        return {
            "mode":              "dry_run",
            "cash":              round(self.cash, 2),
            "portfolio_value":   round(equity, 2),
            "buying_power":      round(self.cash, 2),
            "num_positions":     len(self.positions),
            "unrealized_pnl":    round(equity - CAPITAL, 2),
            "unrealized_pnl_pct": round((equity - CAPITAL) / CAPITAL * 100, 3),
        }

    # ── Quotes (simulated) ───────────────────────────────────────────────────

    def get_latest_price(self, ticker: str) -> Optional[float]:
        """Returns last known price from open position or None."""
        if ticker in self.positions:
            return self.positions[ticker].current_price
        return None

    def update_prices(self, price_map: Dict[str, float]):
        """Feed current prices into open positions (called by the executor)."""
        for ticker, price in price_map.items():
            if ticker in self.positions:
                self.positions[ticker].current_price = price

    # ── Orders ───────────────────────────────────────────────────────────────

    def submit_order(self, order: Order,
                     current_price: float = None) -> Order:
        """
        Submit an order.  Market orders fill immediately at current_price
        (with slippage). Limit/stop orders are stored pending.
        """
        if not self._connected:
            order.status = OrderStatus.REJECTED
            return order

        order.broker_order_id = str(uuid.uuid4())
        self.orders.append(order)
        self.order_map[order.id] = order

        if order.order_type == OrderType.MARKET and current_price:
            self._fill_market(order, current_price)
        else:
            log.info("[DryRun] Order queued: %s", order)

        return order

    def cancel_order(self, order_id: str) -> bool:
        order = self.order_map.get(order_id)
        if order and order.status == OrderStatus.PENDING:
            order.status = OrderStatus.CANCELLED
            log.info("[DryRun] Order cancelled: %s", order_id)
            return True
        return False

    def get_order(self, order_id: str) -> Optional[Order]:
        return self.order_map.get(order_id)

    def get_open_orders(self) -> List[Order]:
        return [o for o in self.orders if o.status == OrderStatus.PENDING]

    # ── Positions ─────────────────────────────────────────────────────────────

    def get_positions(self) -> Dict[str, Position]:
        return dict(self.positions)

    def get_position(self, ticker: str) -> Optional[Position]:
        return self.positions.get(ticker)

    def close_position(self, ticker: str,
                        current_price: float = None) -> Optional[Order]:
        pos = self.positions.get(ticker)
        if not pos:
            log.warning("[DryRun] No position in %s", ticker)
            return None
        side   = "sell" if pos.side == "long" else "buy"
        qty    = pos.qty
        order  = Order(ticker=ticker, side=side, qty=qty,
                       order_type=OrderType.MARKET, tag="close")
        self.submit_order(order, current_price or pos.current_price)
        return order

    def close_all_positions(self, price_map: Dict[str, float] = None):
        for ticker in list(self.positions.keys()):
            price = (price_map or {}).get(ticker,
                     self.positions[ticker].current_price)
            self.close_position(ticker, price)

    # ── Bracket helper (entry + OCO stop/TP) ─────────────────────────────────

    def submit_bracket(self, ticker: str, side: str, qty: float,
                        entry_price: float,
                        stop_loss: float, take_profit: float,
                        confidence: float = None) -> Dict[str, Order]:
        """
        Submit a 3-legged bracket:
          1. Market entry
          2. Stop-loss (stored pending)
          3. Take-profit limit (stored pending)
        Returns dict of {entry, stop_loss, take_profit} orders.
        """
        entry_order = Order(ticker=ticker, side=side, qty=qty,
                            order_type=OrderType.MARKET, tag="entry")
        sl_side = "sell" if side.lower() == "buy" else "buy"
        sl_order = Order(ticker=ticker, side=sl_side, qty=qty,
                         order_type=OrderType.STOP,
                         stop_price=stop_loss, tag="stop_loss")
        tp_order = Order(ticker=ticker, side=sl_side, qty=qty,
                         order_type=OrderType.LIMIT,
                         limit_price=take_profit, tag="take_profit")

        self.submit_order(entry_order, entry_price)

        if entry_order.status == OrderStatus.FILLED:
            pos = self.positions.get(ticker)
            if pos:
                pos.stop_loss     = stop_loss
                pos.take_profit   = take_profit
                pos.signal_confidence = confidence
                pos.sl_order_id   = sl_order.id
                pos.tp_order_id   = tp_order.id

            self.orders.append(sl_order);  self.order_map[sl_order.id] = sl_order
            self.orders.append(tp_order); self.order_map[tp_order.id] = tp_order
            log.info("[DryRun] Bracket placed — SL=%.4f  TP=%.4f",
                     stop_loss, take_profit)

        return {"entry": entry_order, "stop_loss": sl_order, "take_profit": tp_order}

    # ── Tick: process pending orders against new price ────────────────────────

    def tick(self, price_map: Dict[str, float]):
        """
        Call once per bar / price update.
        Checks all pending limit/stop orders and triggers bracket exits.
        """
        self.update_prices(price_map)

        for order in list(self.get_open_orders()):
            price = price_map.get(order.ticker)
            if price is None:
                continue

            if order.order_type == OrderType.STOP:
                pos = self.positions.get(order.ticker)
                if pos and pos.side == "long" and price <= order.stop_price:
                    self._fill_market(order, price)
                    self._cancel_sibling(order)
                elif pos and pos.side == "short" and price >= order.stop_price:
                    self._fill_market(order, price)
                    self._cancel_sibling(order)

            elif order.order_type == OrderType.LIMIT:
                pos = self.positions.get(order.ticker)
                if pos and pos.side == "long" and price >= order.limit_price:
                    self._fill_market(order, order.limit_price)
                    self._cancel_sibling(order)
                elif pos and pos.side == "short" and price <= order.limit_price:
                    self._fill_market(order, order.limit_price)
                    self._cancel_sibling(order)

    # ── Internal fill logic ───────────────────────────────────────────────────

    def _fill_market(self, order: Order, price: float):
        slip  = price * self.slippage
        if order.side == OrderSide.BUY:
            fill_price = price + slip
        else:
            fill_price = price - slip

        commission = fill_price * order.qty * self.commission

        existing = self.positions.get(order.ticker)
        if order.side == OrderSide.BUY:
            if existing and existing.side == "short":
                # Covering a short
                self.cash += fill_price * order.qty - commission
                self._reduce_position(order.ticker, order.qty, fill_price)
            else:
                # Opening / adding to a long
                cost = fill_price * order.qty + commission
                if cost > self.cash:
                    order.status = OrderStatus.REJECTED
                    log.warning("[DryRun] Rejected %s - insufficient cash", order.ticker)
                    return
                self.cash -= cost
                self._open_or_add_position(order.ticker, order.qty, fill_price, "long")
        else:  # SELL
            if existing and existing.side == "long":
                # Closing / reducing a long
                self.cash += fill_price * order.qty - commission
                self._reduce_position(order.ticker, order.qty, fill_price)
            else:
                # Opening a short (simplified margin - receive proceeds)
                self.cash += fill_price * order.qty - commission
                self._open_or_add_position(order.ticker, order.qty, fill_price, "short")

        order.filled_price = round(fill_price, 4)
        order.filled_qty   = order.qty
        order.status       = OrderStatus.FILLED
        order.filled_at    = datetime.now()

        self.trade_log.append({
            "ts":         str(datetime.now()),
            "ticker":     order.ticker,
            "side":       order.side,
            "qty":        order.qty,
            "fill_price": fill_price,
            "commission": round(commission, 4),
            "tag":        order.tag,
            "cash_after": round(self.cash, 2),
        })
        log.info("[DryRun] Filled %s %s qty=%.4f @ %.4f cash=%.2f",
                 order.side.upper(), order.ticker, order.qty,
                 fill_price, self.cash)

    def _open_or_add_position(self, ticker, qty, price, side):
        if ticker in self.positions:
            pos = self.positions[ticker]
            total_qty = pos.qty + qty
            pos.avg_entry = (pos.avg_entry * pos.qty + price * qty) / total_qty
            pos.qty = total_qty
        else:
            self.positions[ticker] = Position(ticker, qty, price, side)

    def _reduce_position(self, ticker, qty, price):
        pos = self.positions.get(ticker)
        if pos is None:
            return
        pos.current_price = price
        if qty >= pos.qty:
            del self.positions[ticker]
        else:
            pos.qty -= qty

    def _cancel_sibling(self, filled_order: Order):
        """Cancel the other leg of a bracket after one fills."""
        pos = self.positions.get(filled_order.ticker)
        if pos is None:
            return
        sibling_ids = [pos.sl_order_id, pos.tp_order_id]
        for oid in sibling_ids:
            if oid and oid != filled_order.id:
                self.cancel_order(oid)

    # ── Reporting ─────────────────────────────────────────────────────────────

    def get_trade_log(self) -> List[Dict]:
        return list(self.trade_log)

    def save_state(self, path: Path = None):
        path = path or OUTPUT_DIR / "dry_run_state.json"
        state = {
            "account":    self.get_account(),
            "positions":  {t: p.to_dict() for t, p in self.positions.items()},
            "trade_log":  self.trade_log,
            "saved_at":   str(datetime.now()),
        }
        path.write_text(json.dumps(state, indent=2, default=str))
        log.info("[DryRun] State saved → %s", path)

    def print_summary(self):
        acct = self.get_account()
        pnl_color = "+" if acct["unrealized_pnl"] >= 0 else ""
        print(f"""
┌─────────────────────────────────────────────────┐
│  DRY-RUN ACCOUNT SUMMARY                        │
├─────────────────────────────────────────────────┤
│  Cash              ${acct['cash']:>12,.2f}             │
│  Portfolio value   ${acct['portfolio_value']:>12,.2f}             │
│  Unrealized P&L    {pnl_color}${acct['unrealized_pnl']:>11,.2f}             │
│  Return            {pnl_color}{acct['unrealized_pnl_pct']:>10.2f}%             │
│  Open positions    {acct['num_positions']:>12}             │
└─────────────────────────────────────────────────┘""")
        if self.positions:
            print("  OPEN POSITIONS")
            print(f"  {'Ticker':8} {'Side':6} {'Qty':>10} {'Entry':>10} "
                  f"{'Current':>10} {'P&L':>10} {'SL':>10} {'TP':>10}")
            print("  " + "─" * 76)
            for t, p in self.positions.items():
                pnl_str = f"{p.unrealized_pnl_pct*100:+.2f}%"
                print(f"  {t:8} {p.side:6} {p.qty:>10.4f} "
                      f"${p.avg_entry:>9.2f} ${p.current_price:>9.2f} "
                      f"{pnl_str:>10} "
                      f"${p.stop_loss or 0:>9.2f} ${p.take_profit or 0:>9.2f}")
        print()
