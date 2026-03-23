"""
backtest/engine.py
==================
Event-driven backtesting engine.

Usage
-----
    from backtest.engine import Backtester
    bt = Backtester(capital=100_000)
    results = bt.run("AAPL", df_with_features, signals_series)
    bt.print_report(results)
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (BACKTEST_COMMISSION, BACKTEST_SLIPPAGE,
                    CONFIDENCE_THRESHOLD, ATR_STOP_MULT, REWARD_RISK_RATIO,
                    CAPITAL, MAX_RISK_PER_TRADE, KELLY_FRACTION)
from core.strategy import kelly_size, atr_stops


# ─────────────────────────────────────────────────────────────────────────────
# Trade record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    entry_date:  pd.Timestamp
    exit_date:   Optional[pd.Timestamp] = None
    direction:   str = "BUY"          # BUY or SELL
    entry_price: float = 0.0
    exit_price:  float = 0.0
    stop_loss:   float = 0.0
    take_profit: float = 0.0
    shares:      float = 0.0
    pnl:         float = 0.0
    pnl_pct:     float = 0.0
    exit_reason: str = ""             # "stop_loss" | "take_profit" | "signal" | "eod"


# ─────────────────────────────────────────────────────────────────────────────
# Backtesting engine
# ─────────────────────────────────────────────────────────────────────────────

class Backtester:

    def __init__(self, capital: float = CAPITAL,
                 commission: float = BACKTEST_COMMISSION,
                 slippage: float = BACKTEST_SLIPPAGE):
        self.initial_capital = capital
        self.commission = commission
        self.slippage   = slippage

    # ── Public entry point ───────────────────────────────────────────────────

    def run(self, ticker: str, df: pd.DataFrame,
            model_probs: pd.Series) -> Dict:
        """
        Parameters
        ----------
        df          : feature DataFrame with OHLCV columns
        model_probs : Series (same index as df) of P(up) from the ensemble.
                      NaN rows are treated as "no signal".

        Returns
        -------
        dict with equity_curve, trades, and performance metrics
        """
        capital    = self.initial_capital
        equity     = [capital]
        dates      = [df.index[0]]
        trades: List[Trade] = []
        open_trade: Optional[Trade] = None

        for i in range(1, len(df)):
            row  = df.iloc[i]
            prev = df.iloc[i - 1]
            date = df.index[i]
            price = float(row["close"])
            high  = float(row["high"])
            low   = float(row["low"])

            # ── Check open trade for stop/TP ─────────────────────────────
            if open_trade is not None:
                sl, tp = open_trade.stop_loss, open_trade.take_profit

                if open_trade.direction == "BUY":
                    if low <= sl:
                        open_trade = self._close(open_trade, sl, date, "stop_loss", capital)
                        capital   += open_trade.pnl
                        trades.append(open_trade); open_trade = None
                    elif high >= tp:
                        open_trade = self._close(open_trade, tp, date, "take_profit", capital)
                        capital   += open_trade.pnl
                        trades.append(open_trade); open_trade = None
                else:  # SELL / short
                    if high >= sl:
                        open_trade = self._close(open_trade, sl, date, "stop_loss", capital)
                        capital   += open_trade.pnl
                        trades.append(open_trade); open_trade = None
                    elif low <= tp:
                        open_trade = self._close(open_trade, tp, date, "take_profit", capital)
                        capital   += open_trade.pnl
                        trades.append(open_trade); open_trade = None

            # ── New signal? ───────────────────────────────────────────────
            p_up = model_probs.iloc[i] if i < len(model_probs) else np.nan
            if open_trade is None and not np.isnan(p_up):
                p_down = 1 - p_up
                direction = None
                if p_up >= CONFIDENCE_THRESHOLD:
                    direction = "BUY"
                elif p_down >= CONFIDENCE_THRESHOLD:
                    direction = "SELL"

                if direction:
                    atr = float(prev.get("atr_14", price * 0.02))
                    sl, tp = atr_stops(price, atr, direction)
                    size_frac = kelly_size(
                        p_win=(p_up if direction == "BUY" else p_down),
                        win_loss_ratio=REWARD_RISK_RATIO,
                        capital=capital,
                        max_risk=MAX_RISK_PER_TRADE,
                        fraction=KELLY_FRACTION,
                    )
                    entry = price * (1 + self.slippage if direction == "BUY"
                                     else 1 - self.slippage)
                    shares = (capital * size_frac) / entry
                    cost   = shares * entry * self.commission

                    open_trade = Trade(
                        entry_date=date, direction=direction,
                        entry_price=round(entry, 4),
                        stop_loss=round(sl, 4),
                        take_profit=round(tp, 4),
                        shares=round(shares, 4),
                    )
                    capital -= cost   # deduct entry commission

            equity.append(round(capital + self._mark_to_market(open_trade, price), 2))
            dates.append(date)

        # ── Close any remaining trade at end ──────────────────────────────
        if open_trade is not None:
            last_price = float(df.iloc[-1]["close"])
            open_trade = self._close(open_trade, last_price, df.index[-1], "eod", capital)
            capital   += open_trade.pnl
            trades.append(open_trade)

        equity_series = pd.Series(equity, index=dates, name="equity")
        return {
            "ticker":        ticker,
            "equity":        equity_series,
            "trades":        trades,
            "final_capital": round(capital, 2),
            **self._performance(equity_series, trades),
        }

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _close(self, t: Trade, exit_price: float,
               date: pd.Timestamp, reason: str,
               capital: float) -> Trade:
        exit_with_slip = exit_price * (1 - self.slippage if t.direction == "BUY"
                                       else 1 + self.slippage)
        if t.direction == "BUY":
            gross = (exit_with_slip - t.entry_price) * t.shares
        else:
            gross = (t.entry_price - exit_with_slip) * t.shares
        commission = exit_with_slip * t.shares * self.commission
        pnl = gross - commission
        t.exit_date   = date
        t.exit_price  = round(exit_with_slip, 4)
        t.exit_reason = reason
        t.pnl         = round(pnl, 2)
        t.pnl_pct     = round(pnl / (t.entry_price * t.shares) * 100, 3)
        return t

    @staticmethod
    def _mark_to_market(trade: Optional[Trade], price: float) -> float:
        if trade is None:
            return 0.0
        if trade.direction == "BUY":
            return (price - trade.entry_price) * trade.shares
        return (trade.entry_price - price) * trade.shares

    def _performance(self, equity: pd.Series, trades: List[Trade]) -> Dict:
        if not trades:
            return {k: 0 for k in ["win_rate","profit_factor","max_drawdown",
                                    "sharpe","total_return","num_trades",
                                    "avg_win","avg_loss","expectancy"]}

        pnls       = [t.pnl for t in trades]
        wins       = [p for p in pnls if p > 0]
        losses     = [p for p in pnls if p <= 0]
        win_rate   = len(wins) / len(pnls)
        gross_w    = sum(wins)
        gross_l    = abs(sum(losses)) + 1e-10
        pf         = gross_w / gross_l

        # Max drawdown
        eq_arr  = equity.values
        peak    = np.maximum.accumulate(eq_arr)
        dd      = (eq_arr - peak) / peak
        max_dd  = float(dd.min())

        # Sharpe (annualised, risk-free ≈ 0)
        daily_ret = equity.pct_change().dropna()
        sharpe    = (daily_ret.mean() / (daily_ret.std() + 1e-10)) * np.sqrt(252)

        total_ret = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]

        avg_win  = np.mean(wins)  if wins   else 0
        avg_loss = np.mean(losses) if losses else 0
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

        return {
            "win_rate":      round(win_rate, 4),
            "profit_factor": round(pf, 3),
            "max_drawdown":  round(max_dd, 4),
            "sharpe":        round(float(sharpe), 3),
            "total_return":  round(float(total_ret), 4),
            "num_trades":    len(trades),
            "avg_win":       round(avg_win, 2),
            "avg_loss":      round(avg_loss, 2),
            "expectancy":    round(expectancy, 2),
        }

    # ── Report ───────────────────────────────────────────────────────────────

    @staticmethod
    def print_report(results: Dict):
        t = results["trades"]
        wins   = [x for x in t if x.pnl > 0]
        losses = [x for x in t if x.pnl <= 0]
        sl_ex  = sum(1 for x in t if x.exit_reason == "stop_loss")
        tp_ex  = sum(1 for x in t if x.exit_reason == "take_profit")

        print(f"""
╔══════════════════════════════════════════════════════╗
║  BACKTEST REPORT — {results['ticker']:8s}                       ║
╠══════════════════════════════════════════════════════╣
  Total trades     : {results['num_trades']}
  Win rate         : {results['win_rate']:.1%}
  Profit factor    : {results['profit_factor']:.2f}
  Max drawdown     : {results['max_drawdown']:.2%}
  Sharpe ratio     : {results['sharpe']:.2f}
  Total return     : {results['total_return']:.2%}
  Final capital    : ${results['final_capital']:,.2f}

  Avg win          : ${results['avg_win']:,.2f}
  Avg loss         : ${results['avg_loss']:,.2f}
  Expectancy/trade : ${results['expectancy']:,.2f}

  Stop-loss exits  : {sl_ex}  ({sl_ex/max(len(t),1):.0%})
  Take-profit exits: {tp_ex}  ({tp_ex/max(len(t),1):.0%})
╚══════════════════════════════════════════════════════╝""")
