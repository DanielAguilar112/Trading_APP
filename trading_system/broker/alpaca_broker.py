"""
broker/alpaca_broker.py
=======================
Alpaca Markets broker implementation.
Supports both paper trading and live trading.

Setup
-----
1. Sign up at https://alpaca.markets (free paper account)
2. Get API keys from the dashboard
3. Set environment variables (recommended):
     export ALPACA_API_KEY="PKXXXXXXXXXX"
     export ALPACA_SECRET_KEY="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
   OR pass them directly to BrokerClient().

Paper vs Live
-------------
  Paper : base_url = https://paper-api.alpaca.markets
  Live  : base_url = https://api.alpaca.markets
          Must pass live=True to BrokerClient — safety guard.

Order types supported
---------------------
  Market, Limit, Stop, Stop-limit, Bracket (OCO)
  Fractional shares supported for most symbols.
"""

import os
import logging
from datetime import datetime
from pathlib  import Path
from typing   import Dict, List, Optional
import json, sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OUTPUT_DIR
from broker.client import (
    BrokerMode, Order, OrderStatus, OrderSide, OrderType, Position,
    ALPACA_PAPER_URL, ALPACA_LIVE_URL, ALPACA_DATA_URL,
)

log = logging.getLogger(__name__)


class AlpacaBroker:
    """
    Wraps the alpaca-py SDK.
    Provides the same interface as DryRunBroker so the executor
    can swap brokers without any code changes.
    """

    def __init__(self, mode: BrokerMode = BrokerMode.PAPER,
                 api_key: str = "", secret_key: str = "",
                 live: bool = False, **kwargs):
        self.mode       = mode
        self.api_key    = api_key or os.getenv("ALPACA_API_KEY", "")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY", "")
        self.live       = live
        self.base_url   = ALPACA_LIVE_URL if live else ALPACA_PAPER_URL
        self._trading   = None   # alpaca.trading.TradingClient
        self._data      = None   # alpaca.data.StockHistoricalDataClient
        self._connected = False
        self.order_map: Dict[str, Order] = {}   # local id → Order

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        if not self.api_key or not self.secret_key:
            log.error("[Alpaca] Missing API credentials. Set ALPACA_API_KEY / ALPACA_SECRET_KEY.")
            return False
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient
            paper = not self.live
            self._trading = TradingClient(
                api_key=self.api_key, secret_key=self.secret_key, paper=paper
            )
            self._data = StockHistoricalDataClient(
                api_key=self.api_key, secret_key=self.secret_key
            )
            # Verify credentials with a test call
            acct = self._trading.get_account()
            self._connected = True
            log.info("[Alpaca] Connected (%s) — equity=$%s",
                     "PAPER" if paper else "LIVE", acct.equity)
            return True
        except Exception as e:
            log.error("[Alpaca] Connection failed: %s", e)
            return False

    def disconnect(self):
        self._connected = False
        log.info("[Alpaca] Disconnected")

    def is_connected(self) -> bool:
        return self._connected

    # ── Account ──────────────────────────────────────────────────────────────

    def get_account(self) -> Dict:
        self._require_connection()
        acct = self._trading.get_account()
        return {
            "mode":            self.mode,
            "account_id":      str(acct.id),
            "status":          str(acct.status),
            "cash":            float(acct.cash),
            "portfolio_value": float(acct.equity),
            "buying_power":    float(acct.buying_power),
            "day_trade_count": int(acct.daytrade_count),
            "pdt_flag":        acct.pattern_day_trader,
            "currency":        str(acct.currency),
        }

    def is_market_open(self) -> bool:
        self._require_connection()
        clock = self._trading.get_clock()
        return clock.is_open

    def get_next_open(self) -> str:
        self._require_connection()
        clock = self._trading.get_clock()
        return str(clock.next_open)

    # ── Quotes ────────────────────────────────────────────────────────────────

    def get_latest_price(self, ticker: str) -> Optional[float]:
        self._require_connection()
        try:
            from alpaca.data.requests import StockLatestQuoteRequest
            request = StockLatestQuoteRequest(symbol_or_symbols=ticker)
            quote = self._data.get_stock_latest_quote(request)
            if ticker in quote:
                q = quote[ticker]
                return float((q.ask_price + q.bid_price) / 2)
        except Exception as e:
            log.warning("[Alpaca] Could not fetch quote for %s: %s", ticker, e)
        return None

    def get_latest_prices(self, tickers: List[str]) -> Dict[str, float]:
        self._require_connection()
        prices = {}
        try:
            from alpaca.data.requests import StockLatestQuoteRequest
            request = StockLatestQuoteRequest(symbol_or_symbols=tickers)
            quotes = self._data.get_stock_latest_quote(request)
            for ticker, q in quotes.items():
                prices[ticker] = float((q.ask_price + q.bid_price) / 2)
        except Exception as e:
            log.warning("[Alpaca] Bulk quote error: %s", e)
        return prices

    # ── Orders ────────────────────────────────────────────────────────────────

    def submit_order(self, order: Order,
                     current_price: float = None) -> Order:
        self._require_connection()
        try:
            from alpaca.trading.requests import (
                MarketOrderRequest, LimitOrderRequest,
                StopOrderRequest, StopLimitOrderRequest,
            )
            from alpaca.trading.enums import (
                OrderSide as AlpacaSide, TimeInForce,
            )

            side = AlpacaSide.BUY if order.side == OrderSide.BUY else AlpacaSide.SELL
            tif  = TimeInForce.DAY

            if order.order_type == OrderType.MARKET:
                req = MarketOrderRequest(
                    symbol=order.ticker, qty=order.qty,
                    side=side, time_in_force=tif,
                    client_order_id=order.client_order_id,
                )
            elif order.order_type == OrderType.LIMIT:
                req = LimitOrderRequest(
                    symbol=order.ticker, qty=order.qty, side=side,
                    limit_price=order.limit_price, time_in_force=tif,
                    client_order_id=order.client_order_id,
                )
            elif order.order_type == OrderType.STOP:
                req = StopOrderRequest(
                    symbol=order.ticker, qty=order.qty, side=side,
                    stop_price=order.stop_price, time_in_force=tif,
                    client_order_id=order.client_order_id,
                )
            elif order.order_type == OrderType.STOP_LIMIT:
                req = StopLimitOrderRequest(
                    symbol=order.ticker, qty=order.qty, side=side,
                    stop_price=order.stop_price,
                    limit_price=order.limit_price,
                    time_in_force=tif,
                    client_order_id=order.client_order_id,
                )
            else:
                raise ValueError(f"Unsupported order type: {order.order_type}")

            result = self._trading.submit_order(order_data=req)
            order.broker_order_id = str(result.id)
            order.status          = OrderStatus.PENDING
            self.order_map[order.id] = order
            log.info("[Alpaca] Order submitted: %s → broker_id=%s",
                     order, order.broker_order_id)

        except Exception as e:
            order.status = OrderStatus.REJECTED
            log.error("[Alpaca] Order rejected %s: %s", order.ticker, e)

        return order

    def submit_bracket(self, ticker: str, side: str, qty: float,
                        entry_price: float,
                        stop_loss: float, take_profit: float,
                        confidence: float = None) -> Dict[str, Order]:
        """
        Alpaca native bracket order (OCO legs submitted atomically).
        """
        self._require_connection()
        entry_order = Order(ticker=ticker, side=side, qty=qty,
                            order_type=OrderType.MARKET, tag="entry")
        sl_side = "sell" if side.lower() == "buy" else "buy"
        sl_order = Order(ticker=ticker, side=sl_side, qty=qty,
                         order_type=OrderType.STOP,
                         stop_price=stop_loss, tag="stop_loss")
        tp_order = Order(ticker=ticker, side=sl_side, qty=qty,
                         order_type=OrderType.LIMIT,
                         limit_price=take_profit, tag="take_profit")
        try:
            from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
            from alpaca.trading.enums    import OrderSide as AlpacaSide, TimeInForce, OrderClass

            a_side = AlpacaSide.BUY if side.lower() == "buy" else AlpacaSide.SELL
            req = MarketOrderRequest(
                symbol=ticker, qty=qty, side=a_side,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=take_profit),
                stop_loss=StopLossRequest(stop_price=stop_loss),
                client_order_id=entry_order.client_order_id,
            )
            result = self._trading.submit_order(order_data=req)
            entry_order.broker_order_id = str(result.id)
            entry_order.status = OrderStatus.PENDING

            # Extract child order IDs if available
            for leg in (result.legs or []):
                if leg.order_type.lower() == "stop":
                    sl_order.broker_order_id = str(leg.id)
                elif leg.order_type.lower() == "limit":
                    tp_order.broker_order_id = str(leg.id)

            log.info("[Alpaca] Bracket submitted for %s — SL=%.4f  TP=%.4f",
                     ticker, stop_loss, take_profit)
        except Exception as e:
            entry_order.status = OrderStatus.REJECTED
            log.error("[Alpaca] Bracket order failed for %s: %s", ticker, e)

        for o in [entry_order, sl_order, tp_order]:
            self.order_map[o.id] = o
        return {"entry": entry_order, "stop_loss": sl_order, "take_profit": tp_order}

    def cancel_order(self, order_id: str) -> bool:
        local = self.order_map.get(order_id)
        if local and local.broker_order_id:
            try:
                self._trading.cancel_order_by_id(local.broker_order_id)
                local.status = OrderStatus.CANCELLED
                return True
            except Exception as e:
                log.warning("[Alpaca] Cancel failed for %s: %s", order_id, e)
        return False

    def cancel_all_orders(self):
        self._require_connection()
        self._trading.cancel_orders()
        log.info("[Alpaca] All open orders cancelled")

    def get_order(self, order_id: str) -> Optional[Order]:
        return self.order_map.get(order_id)

    def sync_order_status(self, order_id: str) -> Optional[Order]:
        """Pull latest status from Alpaca and update local record."""
        local = self.order_map.get(order_id)
        if not local or not local.broker_order_id:
            return None
        try:
            result = self._trading.get_order_by_id(local.broker_order_id)
            local.status       = self._map_status(str(result.status))
            local.filled_qty   = float(result.filled_qty or 0)
            local.filled_price = float(result.filled_avg_price or 0) or None
            if result.filled_at:
                local.filled_at = result.filled_at
        except Exception as e:
            log.warning("[Alpaca] sync_order_status error: %s", e)
        return local

    # ── Positions ─────────────────────────────────────────────────────────────

    def get_positions(self) -> Dict[str, Position]:
        self._require_connection()
        positions = {}
        try:
            for ap in self._trading.get_all_positions():
                side  = "long" if float(ap.qty) > 0 else "short"
                qty   = abs(float(ap.qty))
                pos   = Position(
                    ticker    = ap.symbol,
                    qty       = qty,
                    avg_entry = float(ap.avg_entry_price),
                    side      = side,
                )
                pos.current_price = float(ap.current_price)
                positions[ap.symbol] = pos
        except Exception as e:
            log.error("[Alpaca] get_positions error: %s", e)
        return positions

    def get_position(self, ticker: str) -> Optional[Position]:
        self._require_connection()
        try:
            ap = self._trading.get_open_position(ticker)
            pos = Position(
                ticker    = ap.symbol,
                qty       = abs(float(ap.qty)),
                avg_entry = float(ap.avg_entry_price),
                side      = "long" if float(ap.qty) > 0 else "short",
            )
            pos.current_price = float(ap.current_price)
            return pos
        except Exception:
            return None

    def close_position(self, ticker: str,
                        current_price: float = None) -> Optional[Order]:
        self._require_connection()
        try:
            result = self._trading.close_position(ticker)
            order  = Order(ticker=ticker, side="sell", qty=0,
                           order_type=OrderType.MARKET, tag="close")
            order.broker_order_id = str(result.id)
            order.status          = OrderStatus.PENDING
            self.order_map[order.id] = order
            log.info("[Alpaca] Close position submitted for %s", ticker)
            return order
        except Exception as e:
            log.error("[Alpaca] close_position error %s: %s", ticker, e)
            return None

    def close_all_positions(self, price_map=None):
        self._require_connection()
        self._trading.close_all_positions(cancel_orders=True)
        log.info("[Alpaca] All positions closed")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _require_connection(self):
        if not self._connected:
            raise RuntimeError("Broker not connected. Call client.connect() first.")

    @staticmethod
    def _map_status(alpaca_status: str) -> OrderStatus:
        mapping = {
            "new":           OrderStatus.PENDING,
            "accepted":      OrderStatus.PENDING,
            "pending_new":   OrderStatus.PENDING,
            "partially_filled": OrderStatus.PARTIAL,
            "filled":        OrderStatus.FILLED,
            "done_for_day":  OrderStatus.CANCELLED,
            "canceled":      OrderStatus.CANCELLED,
            "expired":       OrderStatus.CANCELLED,
            "replaced":      OrderStatus.CANCELLED,
            "rejected":      OrderStatus.REJECTED,
        }
        return mapping.get(alpaca_status.lower(), OrderStatus.PENDING)

    def print_summary(self):
        try:
            acct = self.get_account()
            mode_label = "PAPER" if self.mode == BrokerMode.PAPER else "LIVE"
            print(f"""
┌─────────────────────────────────────────────────┐
│  ALPACA {mode_label} ACCOUNT                         │
├─────────────────────────────────────────────────┤
│  Cash            ${acct['cash']:>12,.2f}             │
│  Portfolio value ${acct['portfolio_value']:>12,.2f}             │
│  Buying power    ${acct['buying_power']:>12,.2f}             │
└─────────────────────────────────────────────────┘""")
        except Exception as e:
            print(f"[Alpaca] Could not fetch account: {e}")
