"""
core/strategy.py
================
Converts raw model probabilities + technical indicators into
actionable trade signals with full risk parameters.

Flow
----
  model prediction → indicator filters → confidence gate →
  Kelly sizing → ATR-based stops → TradeSignal
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    CONFIDENCE_THRESHOLD, CAPITAL, MAX_RISK_PER_TRADE,
    MAX_PORTFOLIO_HEAT, KELLY_FRACTION, ATR_STOP_MULT,
    REWARD_RISK_RATIO, RSI_OVERSOLD, RSI_OVERBOUGHT,
    VOLUME_SPIKE_RATIO,
)


# ─────────────────────────────────────────────────────────────────────────────
# Signal dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TradeSignal:
    ticker:         str
    action:         str          # "BUY" | "SELL" | "HOLD"
    confidence:     float        # 0–1
    confidence_pct: int          # 0–100 (display)
    p_up:           float
    p_down:         float

    # Prices
    entry_price:    Optional[float] = None
    stop_loss:      Optional[float] = None
    take_profit:    Optional[float] = None
    atr:            Optional[float] = None

    # Risk
    position_size_pct:    float = 0.0   # % of portfolio
    position_size_dollar: float = 0.0
    risk_dollar:          float = 0.0
    reward_dollar:        float = 0.0
    rr_ratio:             float = 0.0

    # Indicators snapshot
    rsi:            Optional[float] = None
    ma_trend:       str = ""
    volume_signal:  str = ""
    macd_signal:    str = ""
    boll_position:  str = ""

    # Model breakdown
    xgboost_prob:   float = 0.0
    lstm_prob:      float = 0.0
    logreg_prob:    float = 0.0

    # Explanation
    reasoning:      str = ""
    warnings:       list = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}

    def __str__(self):
        lines = [
            f"{'='*60}",
            f"  {self.ticker:8s}  │  {self.action:4s}  │  Confidence: {self.confidence_pct}%",
            f"{'='*60}",
            f"  Entry:       ${self.entry_price:>10.4f}" if self.entry_price else "",
            f"  Stop-loss:   ${self.stop_loss:>10.4f}  ({self._sl_pct():+.2f}%)" if self.stop_loss else "",
            f"  Take-profit: ${self.take_profit:>10.4f}  ({self._tp_pct():+.2f}%)" if self.take_profit else "",
            f"  Position:    {self.position_size_pct:.2f}%  (${self.position_size_dollar:,.0f})",
            f"  R/R ratio:   {self.rr_ratio:.2f}:1",
            f"  RSI:         {self.rsi:.1f}" if self.rsi else "",
            f"  MA Trend:    {self.ma_trend}",
            f"  Volume:      {self.volume_signal}",
            f"",
            f"  Model probs  XGB={self.xgboost_prob:.0%}  LSTM={self.lstm_prob:.0%}  LR={self.logreg_prob:.0%}",
            f"",
            f"  Reasoning:   {self.reasoning}",
        ]
        if self.warnings:
            lines.append(f"  ⚠ Warnings:  {', '.join(self.warnings)}")
        lines.append(f"{'='*60}")
        return "\n".join(l for l in lines if l)

    def _sl_pct(self):
        if self.stop_loss and self.entry_price:
            return (self.stop_loss - self.entry_price) / self.entry_price * 100
        return 0.0

    def _tp_pct(self):
        if self.take_profit and self.entry_price:
            return (self.take_profit - self.entry_price) / self.entry_price * 100
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Position sizing
# ─────────────────────────────────────────────────────────────────────────────

def kelly_size(p_win: float, win_loss_ratio: float,
               capital: float = CAPITAL,
               max_risk: float = MAX_RISK_PER_TRADE,
               fraction: float = KELLY_FRACTION) -> float:
    """
    Half-Kelly position size, capped at max_risk.
    Returns fraction of capital to allocate (0-1).
    """
    if p_win <= 0 or win_loss_ratio <= 0:
        return 0.0
    kelly_full = (p_win - (1 - p_win) / win_loss_ratio)
    kelly_frac = kelly_full * fraction
    return min(max(kelly_frac, 0.0), max_risk)


def atr_stops(entry: float, atr: float, direction: str,
              mult: float = ATR_STOP_MULT,
              rr: float = REWARD_RISK_RATIO):
    """Return (stop_loss, take_profit) based on ATR."""
    risk = mult * atr
    if direction == "BUY":
        sl = entry - risk
        tp = entry + rr * risk
    else:  # SELL / short
        sl = entry + risk
        tp = entry - rr * risk
    return round(sl, 4), round(tp, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Indicator helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ma_trend(row: pd.Series) -> str:
    above = []
    for p in [20, 50, 200]:
        col = f"sma_{p}"
        if col in row and not pd.isna(row[col]):
            above.append(row["close"] > row[col])
    if all(above):
        return "Bullish — above all MAs"
    if not any(above):
        return "Bearish — below all MAs"
    return "Mixed MA signal"

def _volume_signal(row: pd.Series) -> str:
    if "vol_ratio" not in row or pd.isna(row["vol_ratio"]):
        return "Unknown"
    r = row["vol_ratio"]
    if r >= VOLUME_SPIKE_RATIO * 1.5:
        return f"Strong spike ({r:.1f}×)"
    if r >= VOLUME_SPIKE_RATIO:
        return f"Elevated ({r:.1f}×)"
    if r < 0.7:
        return f"Low ({r:.1f}×)"
    return f"Normal ({r:.1f}×)"

def _macd_sig(row: pd.Series) -> str:
    if "macd" not in row or "macd_signal" not in row:
        return "Unknown"
    if row["macd"] > row["macd_signal"]:
        return "Bullish crossover"
    return "Bearish crossover"

def _boll_pos(row: pd.Series) -> str:
    b = row.get("boll_pct_b", 0.5)
    if b > 0.9:  return "Near upper band (overbought zone)"
    if b < 0.1:  return "Near lower band (oversold zone)"
    return "Mid-band"


# ─────────────────────────────────────────────────────────────────────────────
# Main signal generator
# ─────────────────────────────────────────────────────────────────────────────

class SignalGenerator:

    def __init__(self, capital: float = CAPITAL):
        self.capital = capital

    def generate(self, ticker: str, df: pd.DataFrame,
                 prediction: Dict,
                 open_heat: float = 0.0) -> TradeSignal:
        """
        Parameters
        ----------
        ticker     : asset symbol
        df         : feature DataFrame (must include OHLCV + features)
        prediction : output of EnsembleModel.predict()
        open_heat  : existing portfolio risk (0–1 fraction)

        Returns
        -------
        TradeSignal
        """
        p_up   = prediction["p_up"]
        p_down = prediction["p_down"]
        row    = df.iloc[-1]
        price  = float(row["close"])
        atr    = float(row.get("atr_14", price * 0.02))
        rsi    = float(row.get("rsi_14", 50))

        warnings = []

        # ── Step 1: direction & base confidence ──────────────────────────────
        if p_up >= CONFIDENCE_THRESHOLD:
            direction = "BUY"
            conf = p_up
        elif p_down >= CONFIDENCE_THRESHOLD:
            direction = "SELL"
            conf = p_down
        else:
            # Below threshold → HOLD, no position
            return self._hold(ticker, price, prediction, row, rsi,
                              "Ensemble probability below confidence threshold "
                              f"({max(p_up, p_down):.0%} < {CONFIDENCE_THRESHOLD:.0%}).")

        # ── Step 2: indicator confirmation ───────────────────────────────────
        indicator_score = 0
        reasons = []

        # RSI filter
        if direction == "BUY":
            if rsi > RSI_OVERBOUGHT:
                warnings.append(f"RSI {rsi:.0f} is overbought — reduced size")
                indicator_score -= 1
            elif rsi < RSI_OVERSOLD:
                reasons.append(f"RSI {rsi:.0f} oversold bounce setup")
                indicator_score += 2
            else:
                reasons.append(f"RSI {rsi:.0f} neutral/healthy")
                indicator_score += 1
        else:  # SELL
            if rsi < RSI_OVERSOLD:
                warnings.append(f"RSI {rsi:.0f} already oversold — risky short")
                indicator_score -= 1
            elif rsi > RSI_OVERBOUGHT:
                reasons.append(f"RSI {rsi:.0f} overbought — short confirmation")
                indicator_score += 2
            else:
                reasons.append(f"RSI {rsi:.0f} neutral")
                indicator_score += 1

        # MA trend
        ma_trend = _ma_trend(row)
        if direction == "BUY" and "Bullish" in ma_trend:
            reasons.append("price above all MAs"); indicator_score += 1
        elif direction == "SELL" and "Bearish" in ma_trend:
            reasons.append("price below key MAs"); indicator_score += 1
        else:
            warnings.append("MA trend conflicts with direction")

        # Volume
        vol_ratio = float(row.get("vol_ratio", 1.0))
        vol_sig   = _volume_signal(row)
        if vol_ratio >= VOLUME_SPIKE_RATIO:
            reasons.append(f"volume spike {vol_ratio:.1f}×")
            indicator_score += 1
        elif vol_ratio < 0.7:
            warnings.append("low volume — lower conviction")

        # MACD
        if "macd_hist" in row:
            hist = float(row["macd_hist"])
            if (direction == "BUY" and hist > 0) or (direction == "SELL" and hist < 0):
                reasons.append("MACD confirms direction"); indicator_score += 1

        # ── Step 3: confidence adjustment ────────────────────────────────────
        adjusted_conf = min(conf + indicator_score * 0.01, 0.95)
        if indicator_score < 0:
            adjusted_conf = max(adjusted_conf - 0.03, CONFIDENCE_THRESHOLD)

        # ── Step 4: portfolio heat check ─────────────────────────────────────
        if open_heat >= MAX_PORTFOLIO_HEAT:
            return self._hold(ticker, price, prediction, row, rsi,
                              f"Portfolio heat {open_heat:.1%} at maximum — skipping new trade.")

        # ── Step 5: Kelly position sizing ────────────────────────────────────
        tp_price_est = price * (1 + REWARD_RISK_RATIO * ATR_STOP_MULT * atr / price)
        sl_price_est = price * (1 - ATR_STOP_MULT * atr / price)
        win_loss_r   = REWARD_RISK_RATIO  # simplified

        size_frac = kelly_size(p_win=adjusted_conf,
                               win_loss_ratio=win_loss_r,
                               capital=self.capital,
                               max_risk=MAX_RISK_PER_TRADE,
                               fraction=KELLY_FRACTION)
        # Reduce size if indicator confirmation is weak
        if indicator_score <= 0:
            size_frac *= 0.5

        sl, tp = atr_stops(price, atr, direction)
        risk_per_share  = abs(price - sl)
        shares          = (self.capital * size_frac) / price
        risk_dollar     = shares * risk_per_share
        reward_dollar   = shares * abs(tp - price)
        rr_ratio        = reward_dollar / (risk_dollar + 1e-10)

        # ── Step 6: build reasoning string ───────────────────────────────────
        model_summary = (
            f"XGBoost {prediction['xgboost_prob']:.0%} · "
            f"LSTM {prediction['lstm_prob']:.0%} · "
            f"LogReg {prediction['logreg_prob']:.0%}. "
        )
        reasoning = model_summary + " | ".join(reasons)

        return TradeSignal(
            ticker=ticker,
            action=direction,
            confidence=adjusted_conf,
            confidence_pct=round(adjusted_conf * 100),
            p_up=p_up,
            p_down=p_down,
            entry_price=round(price, 4),
            stop_loss=round(sl, 4),
            take_profit=round(tp, 4),
            atr=round(atr, 4),
            position_size_pct=round(size_frac * 100, 2),
            position_size_dollar=round(self.capital * size_frac, 2),
            risk_dollar=round(risk_dollar, 2),
            reward_dollar=round(reward_dollar, 2),
            rr_ratio=round(rr_ratio, 2),
            rsi=round(rsi, 1),
            ma_trend=ma_trend,
            volume_signal=vol_sig,
            macd_signal=_macd_sig(row),
            boll_position=_boll_pos(row),
            xgboost_prob=prediction["xgboost_prob"],
            lstm_prob=prediction["lstm_prob"],
            logreg_prob=prediction["logreg_prob"],
            reasoning=reasoning,
            warnings=warnings,
        )

    def _hold(self, ticker, price, prediction, row, rsi, reason) -> TradeSignal:
        return TradeSignal(
            ticker=ticker, action="HOLD",
            confidence=max(prediction["p_up"], prediction["p_down"]),
            confidence_pct=round(max(prediction["p_up"], prediction["p_down"]) * 100),
            p_up=prediction["p_up"], p_down=prediction["p_down"],
            entry_price=round(price, 4),
            rsi=round(rsi, 1),
            ma_trend=_ma_trend(row),
            volume_signal=_volume_signal(row),
            xgboost_prob=prediction.get("xgboost_prob", 0),
            lstm_prob=prediction.get("lstm_prob", 0),
            logreg_prob=prediction.get("logreg_prob", 0),
            reasoning=reason,
        )
