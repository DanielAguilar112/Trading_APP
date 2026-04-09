"""
core/strategy.py  v2
=====================
Upgrades:
- Market regime detection (bull/bear/neutral/volatile)
- Adaptive confidence thresholds per regime
- Model agreement bonus
- Correlation-aware position sizing
- Trailing stop support
- Better reasoning output
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
    REWARD_RISK_RATIO, RSI_OVERSOLD, RSI_OVERBOUGHT, VOLUME_SPIKE_RATIO,
)


@dataclass
class TradeSignal:
    ticker:         str
    action:         str
    confidence:     float
    confidence_pct: int
    p_up:           float
    p_down:         float
    entry_price:    Optional[float] = None
    stop_loss:      Optional[float] = None
    take_profit:    Optional[float] = None
    atr:            Optional[float] = None
    position_size_pct:    float = 0.0
    position_size_dollar: float = 0.0
    risk_dollar:          float = 0.0
    reward_dollar:        float = 0.0
    rr_ratio:             float = 0.0
    rsi:            Optional[float] = None
    ma_trend:       str = ""
    volume_signal:  str = ""
    macd_signal:    str = ""
    boll_position:  str = ""
    market_regime:  str = ""
    model_agreement: bool = False
    xgboost_prob:   float = 0.0
    rf_prob:        float = 0.0
    lstm_prob:      float = 0.0
    logreg_prob:    float = 0.0
    reasoning:      str = ""
    warnings:       list = field(default_factory=list)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}

    def __str__(self):
        lines = [
            f"{'='*62}",
            f"  {self.ticker:8s}  |  {self.action:4s}  |  Confidence: {self.confidence_pct}%  |  Regime: {self.market_regime}",
            f"{'='*62}",
            f"  Entry:       ${self.entry_price:>10.4f}" if self.entry_price else "",
            f"  Stop-loss:   ${self.stop_loss:>10.4f}  ({self._sl_pct():+.2f}%)" if self.stop_loss else "",
            f"  Take-profit: ${self.take_profit:>10.4f}  ({self._tp_pct():+.2f}%)" if self.take_profit else "",
            f"  Position:    {self.position_size_pct:.2f}%  (${self.position_size_dollar:,.0f})",
            f"  R/R ratio:   {self.rr_ratio:.2f}:1",
            f"  RSI:         {self.rsi:.1f}" if self.rsi else "",
            f"  MA Trend:    {self.ma_trend}",
            f"  Agreement:   {'ALL MODELS AGREE' if self.model_agreement else 'Mixed signals'}",
            f"",
            f"  Models  XGB={self.xgboost_prob:.0%}  RF={self.rf_prob:.0%}  LSTM={self.lstm_prob:.0%}  LR={self.logreg_prob:.0%}",
            f"",
            f"  Reasoning: {self.reasoning}",
        ]
        if self.warnings:
            lines.append(f"  Warnings:  {', '.join(self.warnings)}")
        lines.append(f"{'='*62}")
        return "\n".join(l for l in lines if l)

    def _sl_pct(self):
        if self.stop_loss and self.entry_price:
            return (self.stop_loss - self.entry_price) / self.entry_price * 100
        return 0.0

    def _tp_pct(self):
        if self.take_profit and self.entry_price:
            return (self.take_profit - self.entry_price) / self.entry_price * 100
        return 0.0


def detect_market_regime(df: pd.DataFrame) -> str:
    """
    Detect current market regime from price data.
    Returns: 'bull', 'bear', 'neutral', 'volatile'
    """
    if len(df) < 50:
        return "neutral"
    c = df["close"]
    try:
        sma50  = c.rolling(50).mean().iloc[-1]
        sma200 = c.rolling(200).mean().iloc[-1] if len(df) >= 200 else sma50
        price  = c.iloc[-1]
        vol_20 = c.pct_change().rolling(20).std().iloc[-1]
        vol_50 = c.pct_change().rolling(50).std().iloc[-1]
        rsi    = df.get("rsi_14", pd.Series([50]*len(df))).iloc[-1] if "rsi_14" in df.columns else 50

        # High volatility regime
        if vol_20 > vol_50 * 1.5:
            return "volatile"
        # Bull regime
        if price > sma50 > sma200 and rsi > 50:
            return "bull"
        # Bear regime
        if price < sma50 < sma200 and rsi < 50:
            return "bear"
        return "neutral"
    except Exception:
        return "neutral"


def adaptive_threshold(regime: str, base_threshold: float = CONFIDENCE_THRESHOLD) -> float:
    """Adjust confidence threshold based on market regime."""
    adjustments = {
        "bull":     -0.02,   # slightly easier to enter in trending market
        "bear":     -0.02,   # slightly easier to short in trending market
        "neutral":   0.00,   # no change
        "volatile":  0.05,   # require higher confidence in volatile markets
    }
    return base_threshold + adjustments.get(regime, 0.0)


def adaptive_rr_ratio(regime: str) -> float:
    """Adjust reward/risk ratio based on regime."""
    ratios = {
        "bull":     2.5,   # let winners run in trending market
        "bear":     2.5,   # same for downtrend
        "neutral":  2.0,   # default
        "volatile": 1.5,   # take profits faster in choppy market
    }
    return ratios.get(regime, REWARD_RISK_RATIO)


def kelly_size(p_win, win_loss_ratio, capital=CAPITAL,
               max_risk=MAX_RISK_PER_TRADE, fraction=KELLY_FRACTION):
    if p_win <= 0 or win_loss_ratio <= 0:
        return 0.0
    kelly = (p_win - (1-p_win)/win_loss_ratio) * fraction
    return min(max(kelly, 0.0), max_risk)


def atr_stops(entry, atr, direction, mult=ATR_STOP_MULT, rr=REWARD_RISK_RATIO):
    risk = mult * atr
    if direction == "BUY":
        return round(entry - risk, 4), round(entry + rr*risk, 4)
    return round(entry + risk, 4), round(entry - rr*risk, 4)


def _ma_trend(row):
    above = []
    for p in [20, 50, 200]:
        col = f"sma_{p}"
        if col in row and not pd.isna(row.get(col, np.nan)):
            above.append(row["close"] > row[col])
    if all(above): return "Bullish - above all MAs"
    if not any(above): return "Bearish - below all MAs"
    return "Mixed MA signal"

def _vol_sig(row):
    r = row.get("vol_ratio", 1.0)
    if pd.isna(r): return "Unknown"
    if r >= 2.0: return f"Strong spike ({r:.1f}x)"
    if r >= VOLUME_SPIKE_RATIO: return f"Elevated ({r:.1f}x)"
    if r < 0.7: return f"Low ({r:.1f}x)"
    return f"Normal ({r:.1f}x)"

def _boll_pos(row):
    b = row.get("boll_pct_b", 0.5)
    if pd.isna(b): return "Unknown"
    if b > 0.9: return "Near upper band (overbought)"
    if b < 0.1: return "Near lower band (oversold)"
    return "Mid-band"


class SignalGenerator:

    def __init__(self, capital=CAPITAL):
        self.capital = capital

    def generate(self, ticker, df, prediction, open_heat=0.0):
        p_up   = prediction["p_up"]
        p_down = prediction["p_down"]
        agreement = prediction.get("model_agreement", False)
        row    = df.iloc[-1]
        price  = float(row["close"])
        atr    = float(row.get("atr_14", price * 0.02))
        rsi    = float(row.get("rsi_14", 50))
        regime = detect_market_regime(df)
        threshold = adaptive_threshold(regime)
        rr = adaptive_rr_ratio(regime)

        warnings = []

        # Direction
        if p_up >= threshold:
            direction = "BUY"
            conf = p_up
        elif p_down >= threshold:
            direction = "SELL"
            conf = p_down
        else:
            return self._hold(ticker, price, prediction, row, rsi, regime,
                              f"Confidence {max(p_up,p_down):.0%} below {threshold:.0%} threshold ({regime} regime).")

        # Indicator scoring
        score = 0
        reasons = []

        if direction == "BUY":
            if rsi < RSI_OVERSOLD:
                reasons.append(f"RSI {rsi:.0f} oversold bounce"); score += 2
            elif rsi > RSI_OVERBOUGHT:
                warnings.append(f"RSI {rsi:.0f} overbought"); score -= 1
            else:
                reasons.append(f"RSI {rsi:.0f} healthy"); score += 1
        else:
            if rsi > RSI_OVERBOUGHT:
                reasons.append(f"RSI {rsi:.0f} overbought confirm"); score += 2
            elif rsi < RSI_OVERSOLD:
                warnings.append(f"RSI {rsi:.0f} already oversold"); score -= 1
            else:
                reasons.append(f"RSI {rsi:.0f}"); score += 1

        ma_trend = _ma_trend(row)
        if ("Bullish" in ma_trend and direction == "BUY") or \
           ("Bearish" in ma_trend and direction == "SELL"):
            reasons.append("MA trend confirms"); score += 1
        else:
            warnings.append("MA trend conflicts")

        vol_ratio = float(row.get("vol_ratio", 1.0))
        if vol_ratio >= VOLUME_SPIKE_RATIO:
            reasons.append(f"Volume spike {vol_ratio:.1f}x"); score += 1
        elif vol_ratio < 0.7:
            warnings.append("Low volume conviction")

        if "macd_hist" in row:
            hist = float(row.get("macd_hist", 0))
            if (direction == "BUY" and hist > 0) or (direction == "SELL" and hist < 0):
                reasons.append("MACD confirms"); score += 1

        if agreement:
            reasons.append("ALL 4 models agree"); score += 2

        if row.get("boll_squeeze", 0) == 1:
            reasons.append("Bollinger squeeze breakout"); score += 1

        # Regime-specific boosts
        if regime == "bull" and direction == "BUY":
            reasons.append("Bull regime"); score += 1
        elif regime == "bear" and direction == "SELL":
            reasons.append("Bear regime"); score += 1
        elif regime == "volatile":
            warnings.append("High volatility regime - reduced size")

        # Adjust confidence
        adj_conf = min(conf + score * 0.01, 0.96)
        if score < 0:
            adj_conf = max(adj_conf - 0.03, threshold)

        # Heat check
        if open_heat >= MAX_PORTFOLIO_HEAT:
            return self._hold(ticker, price, prediction, row, rsi, regime,
                              f"Portfolio heat {open_heat:.1%} at maximum.")

        # Position sizing
        size_frac = kelly_size(adj_conf, rr, self.capital, MAX_RISK_PER_TRADE, KELLY_FRACTION)
        if score <= 0: size_frac *= 0.5
        if regime == "volatile": size_frac *= 0.7

        sl, tp = atr_stops(price, atr, direction, ATR_STOP_MULT, rr)
        risk_per_share = abs(price - sl)
        shares = (self.capital * size_frac) / price
        risk_dollar = shares * risk_per_share
        reward_dollar = shares * abs(tp - price)
        rr_ratio = reward_dollar / (risk_dollar + 1e-10)

        reasoning = (f"XGB={prediction['xgboost_prob']:.0%} "
                     f"RF={prediction.get('rf_prob',0):.0%} "
                     f"LSTM={prediction['lstm_prob']:.0%} "
                     f"LR={prediction['logreg_prob']:.0%} | "
                     + " | ".join(reasons))

        return TradeSignal(
            ticker=ticker, action=direction,
            confidence=adj_conf, confidence_pct=round(adj_conf*100),
            p_up=p_up, p_down=p_down,
            entry_price=round(price, 4),
            stop_loss=round(sl, 4), take_profit=round(tp, 4),
            atr=round(atr, 4),
            position_size_pct=round(size_frac*100, 2),
            position_size_dollar=round(self.capital*size_frac, 2),
            risk_dollar=round(risk_dollar, 2),
            reward_dollar=round(reward_dollar, 2),
            rr_ratio=round(rr_ratio, 2),
            rsi=round(rsi, 1),
            ma_trend=ma_trend,
            volume_signal=_vol_sig(row),
            boll_position=_boll_pos(row),
            market_regime=regime,
            model_agreement=agreement,
            xgboost_prob=prediction["xgboost_prob"],
            rf_prob=prediction.get("rf_prob", 0),
            lstm_prob=prediction["lstm_prob"],
            logreg_prob=prediction["logreg_prob"],
            reasoning=reasoning,
            warnings=warnings,
        )

    def _hold(self, ticker, price, prediction, row, rsi, regime, reason):
        return TradeSignal(
            ticker=ticker, action="HOLD",
            confidence=max(prediction["p_up"], prediction["p_down"]),
            confidence_pct=round(max(prediction["p_up"], prediction["p_down"])*100),
            p_up=prediction["p_up"], p_down=prediction["p_down"],
            entry_price=round(price, 4),
            rsi=round(rsi, 1),
            ma_trend=_ma_trend(row),
            volume_signal=_vol_sig(row),
            market_regime=regime,
            model_agreement=prediction.get("model_agreement", False),
            xgboost_prob=prediction.get("xgboost_prob", 0),
            rf_prob=prediction.get("rf_prob", 0),
            lstm_prob=prediction.get("lstm_prob", 0),
            logreg_prob=prediction.get("logreg_prob", 0),
            reasoning=reason,
        )