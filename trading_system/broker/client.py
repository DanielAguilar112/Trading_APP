"""
broker/client.py
================
Unified broker interface.

Three operating modes
---------------------
  DRY_RUN   — fully simulated, no network calls, perfect for testing
  PAPER     — Alpaca paper trading (real API, fake money)
  LIVE      — Alpaca live trading (real API, real money) ← requires explicit flag

Usage
-----
  from broker.client import BrokerClient, BrokerMode
  client = BrokerClient(mode=BrokerMode.PAPER,
                        api_key="YOUR_KEY",
                        secret_key="YOUR_SECRET")
  client.connect()
  order = client.submit_order(signal)
"""

import uuid
import logging
from datetime import datetime, timedelta
from enum    import Enum
from typing  import Dict, List, Optional, Tuple
from pathlib import Path
import json, sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CAPITAL, OUTPUT_DIR

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

ALPACA_PAPER_URL = "https://paper-api.alpaca.markets"
ALPACA_LIVE_URL  = "https://api.alpaca.markets"
ALPACA_DATA_URL  = "https://data.alpaca.markets"


class BrokerMode(str, Enum):
    DRY_RUN = "dry_run"
    PAPER   = "paper"
    LIVE    = "live"


# ─────────────────────────────────────────────────────────────────────────────
# Order / Position data structures
# ─────────────────────────────────────────────────────────────────────────────

class OrderStatus(str, Enum):
    PENDING   = "pending"
    FILLED    = "filled"
    PARTIAL   = "partial"
    CANCELLED = "cancelled"
    REJECTED  = "rejected"

class OrderSide(str, Enum):
    BUY  = "buy"
    SELL = "sell"

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT  = "limit"
    STOP   = "stop"
    STOP_LIMIT = "stop_limit"


class Order:
    def __init__(self, *, ticker, side, qty, order_type=OrderType.MARKET,
                 limit_price=None, stop_price=None,
                 client_order_id=None, tag=""):
        self.id               = str(uuid.uuid4())
        self.client_order_id  = client_order_id or f"ats_{ticker}_{int(datetime.now().timestamp())}"
        self.ticker           = ticker
        self.side             = OrderSide(side.lower())
        self.qty              = qty
        self.order_type       = order_type
        self.limit_price      = limit_price
        self.stop_price       = stop_price
        self.filled_price     = None
        self.filled_qty       = 0
        self.status           = OrderStatus.PENDING
        self.created_at       = datetime.now()
        self.filled_at        = None
        self.tag              = tag          # "entry" | "stop_loss" | "take_profit"
        self.broker_order_id  = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id, "ticker": self.ticker,
            "side": self.side, "qty": self.qty,
            "type": self.order_type,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "filled_price": self.filled_price,
            "filled_qty": self.filled_qty,
            "status": self.status,
            "created_at": str(self.created_at),
            "filled_at": str(self.filled_at),
            "tag": self.tag,
        }

    def __repr__(self):
        return (f"Order({self.ticker} {self.side.upper()} {self.qty:.4f} "
                f"@ {self.order_type} | {self.status})")


class Position:
    def __init__(self, ticker, qty, avg_entry, side="long",
                 stop_loss=None, take_profit=None,
                 signal_confidence=None, tag=""):
        self.ticker            = ticker
        self.qty               = qty
        self.avg_entry         = avg_entry
        self.side              = side          # "long" | "short"
        self.stop_loss         = stop_loss
        self.take_profit       = take_profit
        self.signal_confidence = signal_confidence
        self.tag               = tag
        self.opened_at         = datetime.now()
        self.current_price     = avg_entry
        self.sl_order_id       = None
        self.tp_order_id       = None

    @property
    def unrealized_pnl(self):
        if self.side == "long":
            return (self.current_price - self.avg_entry) * self.qty
        return (self.avg_entry - self.current_price) * self.qty

    @property
    def unrealized_pnl_pct(self):
        return self.unrealized_pnl / (self.avg_entry * self.qty + 1e-10)

    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker, "qty": self.qty,
            "avg_entry": self.avg_entry, "side": self.side,
            "current_price": self.current_price,
            "stop_loss": self.stop_loss, "take_profit": self.take_profit,
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "unrealized_pnl_pct": round(self.unrealized_pnl_pct * 100, 3),
            "opened_at": str(self.opened_at),
            "confidence": self.signal_confidence,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Base interface
# ─────────────────────────────────────────────────────────────────────────────

class BrokerClient:
    """
    Factory: returns the appropriate backend based on `mode`.

    Examples
    --------
    # Simulation (no credentials needed)
    client = BrokerClient(mode=BrokerMode.DRY_RUN)

    # Alpaca paper trading
    client = BrokerClient(mode=BrokerMode.PAPER,
                          api_key="PKXXX", secret_key="XXXXX")

    # Alpaca live  ← must pass live=True explicitly as a safety guard
    client = BrokerClient(mode=BrokerMode.LIVE,
                          api_key="AKXXX", secret_key="XXXXX",
                          live=True)
    """

    def __new__(cls, mode: BrokerMode = BrokerMode.DRY_RUN,
                api_key: str = "", secret_key: str = "",
                live: bool = False, **kwargs):
        from broker.dry_run      import DryRunBroker
        from broker.alpaca_broker import AlpacaBroker
        if mode == BrokerMode.DRY_RUN:
            instance = object.__new__(DryRunBroker)
        elif mode == BrokerMode.PAPER:
            instance = object.__new__(AlpacaBroker)
        elif mode == BrokerMode.LIVE:
            if not live:
                raise RuntimeError(
                    "LIVE mode requires live=True as an explicit safety confirmation.")
            instance = object.__new__(AlpacaBroker)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        instance.__init__(mode=mode, api_key=api_key,
                          secret_key=secret_key, live=live, **kwargs)
        return instance
