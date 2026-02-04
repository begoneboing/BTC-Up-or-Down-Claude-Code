"""
Prediction Engine for Polymarket
Combines technical indicators to predict price direction

This module provides robust prediction capabilities with comprehensive error handling
to ensure trading bot stability during edge cases and unexpected data conditions.
"""

from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Import with error handling
try:
    from indicators import TechnicalIndicators, calculate_all_indicators_optimized
except ImportError as e:
    logger.error(f"Failed to import indicators module: {e}")
    raise


class PredictionError(Exception):
    """Base exception for prediction engine errors."""
    pass


class InsufficientDataError(PredictionError):
    """Raised when there is not enough price data for analysis."""
    pass


class InvalidPriceDataError(PredictionError):
    """Raised when price data contains invalid values."""
    pass


class IndicatorCalculationError(PredictionError):
    """Raised when indicator calculation fails."""
    pass


class Direction(Enum):
    STRONG_UP = "STRONG_UP"
    UP = "UP"
    NEUTRAL = "NEUTRAL"
    DOWN = "DOWN"
    STRONG_DOWN = "STRONG_DOWN"


class MarketRegime(Enum):
    """
    Market regime classification to filter out unreliable signals.
    Trading in CHOPPY regimes leads to false positives.
    """
    TRENDING_UP = "TRENDING_UP"      # Clear uptrend - trade with trend
    TRENDING_DOWN = "TRENDING_DOWN"  # Clear downtrend - trade with trend
    CHOPPY = "CHOPPY"                # Sideways/noisy - avoid trading
    LOW_VOLATILITY = "LOW_VOLATILITY"  # Too quiet - wait for breakout


@dataclass
class Prediction:
    direction: Direction
    confidence: float  # 0-100
    signals: Dict[str, str]
    reasoning: List[str]
    price_target: Optional[float] = None
    market_regime: Optional["MarketRegime"] = None  # Market regime for signal filtering


class PredictionEngine:
    """
    Combines multiple indicators to predict price direction
    Uses signal confirmation to reduce false positives
    Uses adaptive weighting to give stronger signals more influence

    Error Handling:
        - Validates input price data for NaN, Inf, and invalid values
        - Handles indicator calculation failures gracefully
        - Returns NEUTRAL predictions when data is insufficient or invalid
        - Logs all errors for debugging and monitoring
    """

    # Minimum number of indicators that must agree for a signal
    MIN_INDICATOR_AGREEMENT = 3

    # Base weights for each indicator category
    BASE_WEIGHTS = {
        "rsi": 18,
        "ma": 22,
        "momentum": 18,
        "trend": 18,
        "bollinger": 12,
        "macd": 12  # New: MACD crossover and divergence signals
    }

    # Valid price range for Polymarket (0-1 probabilities)
    MIN_VALID_PRICE = 0.0
    MAX_VALID_PRICE = 1000000.0  # Support BTC and other asset prices
    MIN_DATA_POINTS = 20  # Minimum data points required for analysis

    def __init__(self):
        self.ti = TechnicalIndicators()
        self._last_error: Optional[str] = None

    def validate_prices(self, prices: List[float]) -> tuple[bool, str]:
        """
        Validate price data before analysis.

        Args:
            prices: List of price values to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if prices is None:
            return False, "Price data is None"

        if not isinstance(prices, (list, tuple)):
            return False, f"Price data must be a list, got {type(prices).__name__}"

        if len(prices) < self.MIN_DATA_POINTS:
            return False, f"Insufficient data: {len(prices)} points, need at least {self.MIN_DATA_POINTS}"

        # Check for invalid values
        for i, price in enumerate(prices):
            if price is None:
                return False, f"None value at index {i}"

            try:
                price_float = float(price)
            except (TypeError, ValueError):
                return False, f"Non-numeric value at index {i}: {price}"

            # Check for NaN and Inf
            if price_float != price_float:  # NaN check
                return False, f"NaN value at index {i}"

            if abs(price_float) == float('inf'):
                return False, f"Infinite value at index {i}"

            # Check valid price range (with small tolerance for float precision)
            if price_float < self.MIN_VALID_PRICE - 0.001 or price_float > self.MAX_VALID_PRICE + 0.001:
                return False, f"Price {price_float} at index {i} outside valid range [{self.MIN_VALID_PRICE}, {self.MAX_VALID_PRICE}]"

        return True, ""

    def _safe_get_indicator(self, indicators: Dict, key: str, default=None):
        """
        Safely retrieve an indicator value with error handling.

        Args:
            indicators: Dictionary of calculated indicators
            key: The indicator key to retrieve
            default: Default value if key not found or value is invalid

        Returns:
            The indicator value or default
        """
        try:
            value = indicators.get(key, default)
            if value is None:
                return default
            # Check for NaN/Inf
            if isinstance(value, (int, float)):
                if value != value or abs(value) == float('inf'):
                    logger.warning(f"Invalid indicator value for {key}: {value}")
                    return default
            return value
        except Exception as e:
            logger.warning(f"Error retrieving indicator {key}: {e}")
            return default

    def _calculate_signal_multiplier(self, indicator: str, signal_strength: str, value: float = None) -> float:
        """
        Calculate weight multiplier based on signal strength.
        Stronger/clearer signals get higher multipliers (up to 1.5x).
        Weak signals get reduced multipliers (down to 0.5x).
        """
        multipliers = {
            # RSI - extreme readings are more reliable
            "rsi": {
                "extreme_oversold": 1.5,   # RSI < 20
                "oversold": 1.2,           # RSI < 30
                "extreme_overbought": 1.5, # RSI > 80
                "overbought": 1.2,         # RSI > 70
                "weak_bearish": 0.7,       # 30 <= RSI < 45
                "weak_bullish": 0.7,       # 55 < RSI <= 70
                "neutral": 0.5             # 45 <= RSI <= 55
            },
            # Moving Averages - alignment strength matters
            "ma": {
                "perfect_alignment": 1.5,  # Price > SMA5 > SMA10 > SMA20
                "strong_alignment": 1.2,   # Strong bullish/bearish
                "moderate": 0.8,           # Price above/below SMA10
                "neutral": 0.5             # No clear alignment
            },
            # Momentum - magnitude matters
            "momentum": {
                "very_strong": 1.5,        # |momentum| > 0.05
                "strong": 1.2,             # |momentum| > 0.02
                "moderate": 0.8,           # |momentum| > 0
                "flat": 0.5                # momentum near 0
            },
            # Trend - strength score matters
            "trend": {
                "very_strong": 1.5,        # strength > 70
                "strong": 1.2,             # strength > 50
                "moderate": 0.8,           # strength > 25
                "weak": 0.5                # strength <= 25
            },
            # Bollinger - position relative to bands
            "bollinger": {
                "extreme": 1.5,            # At or beyond bands
                "near_band": 1.2,          # Close to bands
                "mid_zone": 0.7,           # Near middle band
                "neutral": 0.5             # No clear signal
            },
            # MACD - crossovers and divergences
            "macd": {
                "crossover_with_divergence": 1.5,  # Both crossover and divergence
                "crossover": 1.3,          # MACD line crosses signal line
                "divergence": 1.4,         # Price/MACD divergence (reversal warning)
                "histogram_strong": 1.2,   # Strong histogram
                "histogram_weak": 0.7,     # Weak histogram
                "neutral": 0.5             # No clear signal
            }
        }

        return multipliers.get(indicator, {}).get(signal_strength, 1.0)

    def detect_market_regime(self, prices: List[float], indicators: Dict) -> MarketRegime:
        """
        Detect the current market regime to filter out unreliable signals.

        Choppy markets have:
        - Low trend strength (sideways movement)
        - Price oscillating around the middle Bollinger band
        - Conflicting indicator signals
        - High rate of direction changes

        Returns:
            MarketRegime indicating whether the market is suitable for trading

        Error Handling:
            - Returns CHOPPY for any errors (safest default - avoids trading)
            - Handles missing or invalid indicator values
        """
        try:
            if not prices or len(prices) < 20:
                return MarketRegime.CHOPPY  # Insufficient data

            if not indicators or not isinstance(indicators, dict):
                logger.warning("Invalid indicators dict in detect_market_regime")
                return MarketRegime.CHOPPY

            # Get trend information with safe access
            trend = self._safe_get_indicator(indicators, "trend", {})
            if isinstance(trend, dict):
                trend_strength = trend.get("strength", 0) or 0
                trend_direction = trend.get("direction", "neutral") or "neutral"
            else:
                trend_strength = 0
                trend_direction = "neutral"

            # Get volatility info with safe access
            volatility = self._safe_get_indicator(indicators, "volatility_20", 0) or 0
            bb = self._safe_get_indicator(indicators, "bollinger", None)

            # Get momentum indicators with safe access
            momentum = self._safe_get_indicator(indicators, "momentum_10", 0) or 0
            roc = self._safe_get_indicator(indicators, "roc_10", 0) or 0

            # Count direction changes in recent prices (choppiness indicator)
            recent_prices = prices[-14:]  # Last 14 periods
            direction_changes = 0
            for i in range(2, len(recent_prices)):
                prev_dir = recent_prices[i-1] - recent_prices[i-2]
                curr_dir = recent_prices[i] - recent_prices[i-1]
                if (prev_dir > 0 and curr_dir < 0) or (prev_dir < 0 and curr_dir > 0):
                    direction_changes += 1

            # Choppiness ratio: how often direction changes
            choppiness_ratio = direction_changes / (len(recent_prices) - 2) if len(recent_prices) > 2 else 0

            # Calculate price range relative to volatility
            price_range = max(recent_prices) - min(recent_prices)
            avg_price = sum(recent_prices) / len(recent_prices)
            range_ratio = price_range / avg_price if avg_price > 0 else 0

            # LOW VOLATILITY: Very tight range, no clear movement
            if volatility < 0.005 and range_ratio < 0.02 and abs(roc) < 2:
                return MarketRegime.LOW_VOLATILITY

            # TRENDING: Strong rate of change indicates clear directional movement
            # ROC > 10% or < -10% with low choppiness = strong trend
            if abs(roc) > 10 and choppiness_ratio < 0.3:
                if roc > 0 and momentum > 0:
                    return MarketRegime.TRENDING_UP
                elif roc < 0 and momentum < 0:
                    return MarketRegime.TRENDING_DOWN

            # TRENDING: Moderate ROC with consistent direction
            if abs(roc) > 5 and choppiness_ratio < 0.4:
                if roc > 0 and momentum > 0 and trend_direction == "up":
                    return MarketRegime.TRENDING_UP
                elif roc < 0 and momentum < 0 and trend_direction == "down":
                    return MarketRegime.TRENDING_DOWN

            # TRENDING: Strong trend strength indicator
            if trend_strength > 40:
                if trend_direction == "up" and momentum > 0:
                    return MarketRegime.TRENDING_UP
                elif trend_direction == "down" and momentum < 0:
                    return MarketRegime.TRENDING_DOWN

            # CHOPPY: High direction changes indicates uncertain market
            if choppiness_ratio > 0.5:
                return MarketRegime.CHOPPY

            # CHOPPY: Weak trend + price in middle of Bollinger bands
            if bb and trend_strength < 15 and abs(roc) < 5:
                band_width = bb["upper"] - bb["lower"]
                if band_width > 0:
                    position_in_band = (prices[-1] - bb["lower"]) / band_width
                    if 0.3 < position_in_band < 0.7:  # In the middle zone
                        return MarketRegime.CHOPPY

            # TRENDING: Weak trend but consistent momentum direction
            if trend_direction != "neutral" and abs(roc) > 3 and choppiness_ratio < 0.35:
                if momentum > 0 and roc > 0:
                    return MarketRegime.TRENDING_UP
                elif momentum < 0 and roc < 0:
                    return MarketRegime.TRENDING_DOWN

            # Default to CHOPPY if no clear regime
            return MarketRegime.CHOPPY

        except Exception as e:
            logger.error(f"Error in detect_market_regime: {e}", exc_info=True)
            return MarketRegime.CHOPPY  # Safe default: avoid trading on errors

    def get_last_error(self) -> Optional[str]:
        """
        Get the last error message from analysis.

        Returns:
            Error message string or None if no error occurred
        """
        return self._last_error

    def analyze(self, prices: List[float]) -> Optional[Prediction]:
        """
        Analyze price data and generate prediction with comprehensive error handling.

        Args:
            prices: List of historical prices (0-1 range for Polymarket)

        Returns:
            Prediction object with direction, confidence, and signals, or None if analysis fails

        Error Handling:
            - Returns None for invalid/insufficient data
            - Catches indicator calculation errors
            - Logs all errors for debugging
            - Never raises exceptions to caller (fail-safe for trading)
        """
        self._last_error = None

        # Validate input data
        is_valid, error_msg = self.validate_prices(prices)
        if not is_valid:
            self._last_error = error_msg
            logger.warning(f"Price validation failed: {error_msg}")
            return None

        # Calculate indicators with error handling
        try:
            indicators = calculate_all_indicators_optimized(prices)
            if indicators is None or not isinstance(indicators, dict):
                self._last_error = "Indicator calculation returned invalid result"
                logger.error(self._last_error)
                return None
        except Exception as e:
            self._last_error = f"Indicator calculation error: {e}"
            logger.error(self._last_error, exc_info=True)
            return None

        # Safely get current price
        try:
            current_price = float(prices[-1])
        except (IndexError, TypeError, ValueError) as e:
            self._last_error = f"Failed to get current price: {e}"
            logger.error(self._last_error)
            return None

        # Collect signals from each indicator
        signals = {}
        bullish_score = 0
        bearish_score = 0
        reasoning = []

        # Track indicator agreement for signal confirmation
        bullish_indicators = 0  # Count of indicators signaling bullish
        bearish_indicators = 0  # Count of indicators signaling bearish
        total_indicators = 0    # Total indicators with valid signals

        # 1. RSI Analysis (base weight: 20%, adaptive)
        rsi = indicators.get("rsi_14")
        base_weight = self.BASE_WEIGHTS["rsi"]
        if rsi is not None:
            total_indicators += 1
            if rsi < 20:
                # Extreme oversold - very strong signal
                multiplier = self._calculate_signal_multiplier("rsi", "extreme_oversold")
                signals["rsi"] = "extreme_oversold"
                bullish_score += int(base_weight * multiplier)
                bullish_indicators += 1
                reasoning.append(f"RSI extreme oversold at {rsi:.0f} - strong reversal signal (weight: {multiplier:.1f}x)")
            elif rsi < 30:
                multiplier = self._calculate_signal_multiplier("rsi", "oversold")
                signals["rsi"] = "oversold"
                bullish_score += int(base_weight * multiplier)
                bullish_indicators += 1
                reasoning.append(f"RSI oversold at {rsi:.0f} - potential reversal up (weight: {multiplier:.1f}x)")
            elif rsi > 80:
                # Extreme overbought - very strong signal
                multiplier = self._calculate_signal_multiplier("rsi", "extreme_overbought")
                signals["rsi"] = "extreme_overbought"
                bearish_score += int(base_weight * multiplier)
                bearish_indicators += 1
                reasoning.append(f"RSI extreme overbought at {rsi:.0f} - strong reversal signal (weight: {multiplier:.1f}x)")
            elif rsi > 70:
                multiplier = self._calculate_signal_multiplier("rsi", "overbought")
                signals["rsi"] = "overbought"
                bearish_score += int(base_weight * multiplier)
                bearish_indicators += 1
                reasoning.append(f"RSI overbought at {rsi:.0f} - potential reversal down (weight: {multiplier:.1f}x)")
            elif rsi < 45:
                multiplier = self._calculate_signal_multiplier("rsi", "weak_bearish")
                signals["rsi"] = "weak"
                bearish_score += int((base_weight / 2) * multiplier)
                bearish_indicators += 1
            elif rsi > 55:
                multiplier = self._calculate_signal_multiplier("rsi", "weak_bullish")
                signals["rsi"] = "strong"
                bullish_score += int((base_weight / 2) * multiplier)
                bullish_indicators += 1
            else:
                signals["rsi"] = "neutral"

        # 2. Moving Average Analysis (base weight: 25%, adaptive)
        sma_5 = indicators.get("sma_5")
        sma_10 = indicators.get("sma_10")
        sma_20 = indicators.get("sma_20")
        base_weight = self.BASE_WEIGHTS["ma"]

        if sma_5 and sma_10 and sma_20:
            total_indicators += 1
            if current_price > sma_5 > sma_10 > sma_20:
                # Perfect bullish alignment
                multiplier = self._calculate_signal_multiplier("ma", "perfect_alignment")
                signals["ma"] = "strong_bullish"
                bullish_score += int(base_weight * multiplier)
                bullish_indicators += 1
                reasoning.append(f"Price above all MAs in perfect bullish alignment (weight: {multiplier:.1f}x)")
            elif current_price < sma_5 < sma_10 < sma_20:
                # Perfect bearish alignment
                multiplier = self._calculate_signal_multiplier("ma", "perfect_alignment")
                signals["ma"] = "strong_bearish"
                bearish_score += int(base_weight * multiplier)
                bearish_indicators += 1
                reasoning.append(f"Price below all MAs in perfect bearish alignment (weight: {multiplier:.1f}x)")
            elif current_price > sma_10:
                multiplier = self._calculate_signal_multiplier("ma", "moderate")
                signals["ma"] = "bullish"
                bullish_score += int((base_weight * 0.6) * multiplier)
                bullish_indicators += 1
                reasoning.append("Price above SMA(10)")
            elif current_price < sma_10:
                multiplier = self._calculate_signal_multiplier("ma", "moderate")
                signals["ma"] = "bearish"
                bearish_score += int((base_weight * 0.6) * multiplier)
                bearish_indicators += 1
                reasoning.append("Price below SMA(10)")
            else:
                signals["ma"] = "neutral"

        # 3. Momentum Analysis (base weight: 20%, adaptive)
        momentum = indicators.get("momentum_10")
        roc = indicators.get("roc_10")
        base_weight = self.BASE_WEIGHTS["momentum"]

        if momentum is not None:
            total_indicators += 1
            abs_momentum = abs(momentum)
            if abs_momentum > 0.05:
                # Very strong momentum
                multiplier = self._calculate_signal_multiplier("momentum", "very_strong")
                if momentum > 0:
                    signals["momentum"] = "very_strong_positive"
                    bullish_score += int(base_weight * multiplier)
                    bullish_indicators += 1
                    reasoning.append(f"Very strong positive momentum: +{momentum:.2%} (weight: {multiplier:.1f}x)")
                else:
                    signals["momentum"] = "very_strong_negative"
                    bearish_score += int(base_weight * multiplier)
                    bearish_indicators += 1
                    reasoning.append(f"Very strong negative momentum: {momentum:.2%} (weight: {multiplier:.1f}x)")
            elif abs_momentum > 0.02:
                # Strong momentum
                multiplier = self._calculate_signal_multiplier("momentum", "strong")
                if momentum > 0:
                    signals["momentum"] = "strong_positive"
                    bullish_score += int(base_weight * multiplier)
                    bullish_indicators += 1
                    reasoning.append(f"Strong positive momentum: +{momentum:.2%} (weight: {multiplier:.1f}x)")
                else:
                    signals["momentum"] = "strong_negative"
                    bearish_score += int(base_weight * multiplier)
                    bearish_indicators += 1
                    reasoning.append(f"Strong negative momentum: {momentum:.2%} (weight: {multiplier:.1f}x)")
            elif abs_momentum > 0:
                # Moderate momentum
                multiplier = self._calculate_signal_multiplier("momentum", "moderate")
                if momentum > 0:
                    signals["momentum"] = "positive"
                    bullish_score += int((base_weight / 2) * multiplier)
                    bullish_indicators += 1
                else:
                    signals["momentum"] = "negative"
                    bearish_score += int((base_weight / 2) * multiplier)
                    bearish_indicators += 1
            else:
                signals["momentum"] = "flat"

        # 4. Trend Analysis (base weight: 20%, adaptive)
        trend = indicators.get("trend")
        base_weight = self.BASE_WEIGHTS["trend"]
        if trend:
            total_indicators += 1
            trend_strength = trend.get("strength", 0)

            if trend_strength > 70:
                # Very strong trend
                multiplier = self._calculate_signal_multiplier("trend", "very_strong")
                if trend["direction"] == "up":
                    signals["trend"] = "very_strong_uptrend"
                    bullish_score += int(base_weight * multiplier)
                    bullish_indicators += 1
                    reasoning.append(f"Very strong uptrend (strength: {trend_strength:.0f}, weight: {multiplier:.1f}x)")
                else:
                    signals["trend"] = "very_strong_downtrend"
                    bearish_score += int(base_weight * multiplier)
                    bearish_indicators += 1
                    reasoning.append(f"Very strong downtrend (strength: {trend_strength:.0f}, weight: {multiplier:.1f}x)")
            elif trend_strength > 50:
                # Strong trend
                multiplier = self._calculate_signal_multiplier("trend", "strong")
                if trend["direction"] == "up":
                    signals["trend"] = "strong_uptrend"
                    bullish_score += int(base_weight * multiplier)
                    bullish_indicators += 1
                    reasoning.append(f"Strong uptrend (strength: {trend_strength:.0f}, weight: {multiplier:.1f}x)")
                else:
                    signals["trend"] = "strong_downtrend"
                    bearish_score += int(base_weight * multiplier)
                    bearish_indicators += 1
                    reasoning.append(f"Strong downtrend (strength: {trend_strength:.0f}, weight: {multiplier:.1f}x)")
            elif trend_strength > 25 and trend["direction"] != "neutral":
                # Moderate trend
                multiplier = self._calculate_signal_multiplier("trend", "moderate")
                if trend["direction"] == "up":
                    signals["trend"] = "uptrend"
                    bullish_score += int((base_weight / 2) * multiplier)
                    bullish_indicators += 1
                else:
                    signals["trend"] = "downtrend"
                    bearish_score += int((base_weight / 2) * multiplier)
                    bearish_indicators += 1
            else:
                signals["trend"] = "sideways"

        # 5. Bollinger Bands Analysis (base weight: 15%, adaptive)
        bb = indicators.get("bollinger")
        base_weight = self.BASE_WEIGHTS["bollinger"]
        if bb:
            total_indicators += 1
            # Calculate position within bands (0 = lower, 1 = upper)
            band_width = bb["upper"] - bb["lower"]
            if band_width > 0:
                position_in_band = (current_price - bb["lower"]) / band_width
            else:
                position_in_band = 0.5

            if current_price <= bb["lower"]:
                # At or below lower band - extreme oversold
                multiplier = self._calculate_signal_multiplier("bollinger", "extreme")
                signals["bollinger"] = "extreme_oversold"
                bullish_score += int(base_weight * multiplier)
                bullish_indicators += 1
                reasoning.append(f"Price at/below lower Bollinger Band - extreme oversold (weight: {multiplier:.1f}x)")
            elif current_price >= bb["upper"]:
                # At or above upper band - extreme overbought
                multiplier = self._calculate_signal_multiplier("bollinger", "extreme")
                signals["bollinger"] = "extreme_overbought"
                bearish_score += int(base_weight * multiplier)
                bearish_indicators += 1
                reasoning.append(f"Price at/above upper Bollinger Band - extreme overbought (weight: {multiplier:.1f}x)")
            elif position_in_band < 0.2:
                # Near lower band
                multiplier = self._calculate_signal_multiplier("bollinger", "near_band")
                signals["bollinger"] = "near_lower"
                bullish_score += int(base_weight * multiplier)
                bullish_indicators += 1
                reasoning.append(f"Price near lower Bollinger Band (weight: {multiplier:.1f}x)")
            elif position_in_band > 0.8:
                # Near upper band
                multiplier = self._calculate_signal_multiplier("bollinger", "near_band")
                signals["bollinger"] = "near_upper"
                bearish_score += int(base_weight * multiplier)
                bearish_indicators += 1
                reasoning.append(f"Price near upper Bollinger Band (weight: {multiplier:.1f}x)")
            elif current_price > bb["middle"]:
                multiplier = self._calculate_signal_multiplier("bollinger", "mid_zone")
                signals["bollinger"] = "above_mid"
                bullish_score += int((base_weight / 3) * multiplier)
                bullish_indicators += 1
            else:
                multiplier = self._calculate_signal_multiplier("bollinger", "mid_zone")
                signals["bollinger"] = "below_mid"
                bearish_score += int((base_weight / 3) * multiplier)
                bearish_indicators += 1

        # 6. MACD Analysis (base weight: 12%, adaptive)
        # Detects crossovers and divergences for entry/exit timing
        macd = indicators.get("macd")
        base_weight = self.BASE_WEIGHTS["macd"]
        if macd:
            total_indicators += 1
            histogram = macd.get("histogram", 0)
            crossover = macd.get("crossover")
            divergence = macd.get("divergence")

            # Track signal strength for weighting
            has_crossover = crossover is not None
            has_divergence = divergence is not None

            if has_crossover and has_divergence and crossover == divergence:
                # Both crossover and divergence agree - very strong signal
                multiplier = self._calculate_signal_multiplier("macd", "crossover_with_divergence")
                if crossover == "bullish":
                    signals["macd"] = "bullish_crossover_divergence"
                    bullish_score += int(base_weight * multiplier)
                    bullish_indicators += 1
                    reasoning.append(f"MACD bullish crossover confirmed by divergence (weight: {multiplier:.1f}x)")
                else:
                    signals["macd"] = "bearish_crossover_divergence"
                    bearish_score += int(base_weight * multiplier)
                    bearish_indicators += 1
                    reasoning.append(f"MACD bearish crossover confirmed by divergence (weight: {multiplier:.1f}x)")
            elif has_crossover:
                # Crossover alone - good signal
                multiplier = self._calculate_signal_multiplier("macd", "crossover")
                if crossover == "bullish":
                    signals["macd"] = "bullish_crossover"
                    bullish_score += int(base_weight * multiplier)
                    bullish_indicators += 1
                    reasoning.append(f"MACD bullish crossover - entry signal (weight: {multiplier:.1f}x)")
                else:
                    signals["macd"] = "bearish_crossover"
                    bearish_score += int(base_weight * multiplier)
                    bearish_indicators += 1
                    reasoning.append(f"MACD bearish crossover - exit signal (weight: {multiplier:.1f}x)")
            elif has_divergence:
                # Divergence alone - potential reversal warning
                multiplier = self._calculate_signal_multiplier("macd", "divergence")
                if divergence == "bullish":
                    signals["macd"] = "bullish_divergence"
                    bullish_score += int(base_weight * multiplier)
                    bullish_indicators += 1
                    reasoning.append(f"MACD bullish divergence - trend weakening (weight: {multiplier:.1f}x)")
                else:
                    signals["macd"] = "bearish_divergence"
                    bearish_score += int(base_weight * multiplier)
                    bearish_indicators += 1
                    reasoning.append(f"MACD bearish divergence - trend may reverse (weight: {multiplier:.1f}x)")
            elif histogram != 0:
                # Use histogram for direction
                abs_histogram = abs(histogram)
                if abs_histogram > 0.01:
                    multiplier = self._calculate_signal_multiplier("macd", "histogram_strong")
                    if histogram > 0:
                        signals["macd"] = "histogram_positive"
                        bullish_score += int((base_weight / 2) * multiplier)
                        bullish_indicators += 1
                    else:
                        signals["macd"] = "histogram_negative"
                        bearish_score += int((base_weight / 2) * multiplier)
                        bearish_indicators += 1
                else:
                    multiplier = self._calculate_signal_multiplier("macd", "histogram_weak")
                    if histogram > 0:
                        signals["macd"] = "histogram_weak_positive"
                        bullish_score += int((base_weight / 4) * multiplier)
                        bullish_indicators += 1
                    else:
                        signals["macd"] = "histogram_weak_negative"
                        bearish_score += int((base_weight / 4) * multiplier)
                        bearish_indicators += 1
            else:
                signals["macd"] = "neutral"

        # Calculate net score and determine direction
        net_score = bullish_score - bearish_score
        total_score = bullish_score + bearish_score

        if total_score == 0:
            confidence = 0
        else:
            confidence = abs(net_score) / total_score * 100

        # Store indicator agreement info in signals for transparency
        signals["_agreement"] = {
            "bullish_count": bullish_indicators,
            "bearish_count": bearish_indicators,
            "total_indicators": total_indicators
        }

        # Signal Confirmation: Require minimum indicator agreement
        # This reduces false positives by ensuring multiple indicators agree
        has_bullish_confirmation = bullish_indicators >= self.MIN_INDICATOR_AGREEMENT
        has_bearish_confirmation = bearish_indicators >= self.MIN_INDICATOR_AGREEMENT

        # Determine direction with confirmation requirement
        if net_score >= 40 and has_bullish_confirmation:
            direction = Direction.STRONG_UP
            reasoning.append(f"Signal confirmed: {bullish_indicators}/{total_indicators} indicators bullish")
        elif net_score >= 15 and has_bullish_confirmation:
            direction = Direction.UP
            reasoning.append(f"Signal confirmed: {bullish_indicators}/{total_indicators} indicators bullish")
        elif net_score <= -40 and has_bearish_confirmation:
            direction = Direction.STRONG_DOWN
            reasoning.append(f"Signal confirmed: {bearish_indicators}/{total_indicators} indicators bearish")
        elif net_score <= -15 and has_bearish_confirmation:
            direction = Direction.DOWN
            reasoning.append(f"Signal confirmed: {bearish_indicators}/{total_indicators} indicators bearish")
        else:
            direction = Direction.NEUTRAL
            # Add reasoning for why signal was not confirmed
            if net_score >= 15 and not has_bullish_confirmation:
                reasoning.append(f"Bullish not confirmed: only {bullish_indicators}/{self.MIN_INDICATOR_AGREEMENT} required indicators agree")
            elif net_score <= -15 and not has_bearish_confirmation:
                reasoning.append(f"Bearish not confirmed: only {bearish_indicators}/{self.MIN_INDICATOR_AGREEMENT} required indicators agree")

        # Detect market regime for signal filtering
        market_regime = self.detect_market_regime(prices, indicators)

        # Filter signals in unfavorable market regimes
        # This reduces false positives by avoiding trades in choppy/unclear markets
        if market_regime == MarketRegime.CHOPPY:
            # In choppy markets, only allow STRONG signals with high confidence
            if direction not in [Direction.STRONG_UP, Direction.STRONG_DOWN] or confidence < 70:
                direction = Direction.NEUTRAL
                reasoning.append(f"Signal filtered: CHOPPY market regime - avoiding unreliable signals")
            else:
                reasoning.append(f"Market regime: CHOPPY but strong signal ({confidence:.0f}%) accepted")
        elif market_regime == MarketRegime.LOW_VOLATILITY:
            # In low volatility markets, signals are weak - wait for breakout
            if confidence < 80:
                direction = Direction.NEUTRAL
                reasoning.append(f"Signal filtered: LOW_VOLATILITY regime - waiting for breakout")
            else:
                reasoning.append(f"Market regime: LOW_VOLATILITY but very strong signal accepted")
        else:
            # Trending market - good for trading
            reasoning.append(f"Market regime: {market_regime.value} - favorable for trading")

        # Check for regime/direction mismatch (counter-trend signals are riskier)
        if market_regime == MarketRegime.TRENDING_UP and direction in [Direction.DOWN, Direction.STRONG_DOWN]:
            if confidence < 75:
                direction = Direction.NEUTRAL
                reasoning.append(f"Counter-trend signal filtered: market trending UP but bearish signal")
        elif market_regime == MarketRegime.TRENDING_DOWN and direction in [Direction.UP, Direction.STRONG_UP]:
            if confidence < 75:
                direction = Direction.NEUTRAL
                reasoning.append(f"Counter-trend signal filtered: market trending DOWN but bullish signal")

        # Calculate price target
        levels = indicators.get("levels")
        if levels:
            if direction in [Direction.STRONG_UP, Direction.UP]:
                price_target = levels["resistance"]
            elif direction in [Direction.STRONG_DOWN, Direction.DOWN]:
                price_target = levels["support"]
            else:
                price_target = current_price
        else:
            price_target = None

        return Prediction(
            direction=direction,
            confidence=confidence,
            signals=signals,
            reasoning=reasoning,
            price_target=price_target,
            market_regime=market_regime
        )

    def get_summary(self, prediction: Prediction, current_price: float) -> str:
        """Generate a human-readable summary"""
        direction_emoji = {
            Direction.STRONG_UP: "🚀",
            Direction.UP: "📈",
            Direction.NEUTRAL: "➡️",
            Direction.DOWN: "📉",
            Direction.STRONG_DOWN: "💥"
        }

        lines = [
            f"Direction: {direction_emoji.get(prediction.direction, '')} {prediction.direction.value}",
            f"Confidence: {prediction.confidence:.0f}%",
            f"Current Price: {current_price:.2%}",
        ]

        if prediction.price_target:
            change = (prediction.price_target - current_price) / current_price * 100
            lines.append(f"Target: {prediction.price_target:.2%} ({change:+.1f}%)")

        lines.append("\nSignals:")
        for indicator, signal in prediction.signals.items():
            lines.append(f"  {indicator}: {signal}")

        if prediction.reasoning:
            lines.append("\nReasoning:")
            for reason in prediction.reasoning:
                lines.append(f"  - {reason}")

        return "\n".join(lines)


def predict_direction(prices: List[float]) -> Optional[Prediction]:
    """Quick helper to get prediction"""
    engine = PredictionEngine()
    return engine.analyze(prices)


if __name__ == "__main__":
    # Test with uptrending data
    uptrend_prices = [
        0.40, 0.41, 0.42, 0.41, 0.43, 0.44, 0.45, 0.44, 0.46, 0.47,
        0.46, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56,
        0.57, 0.58, 0.59, 0.60, 0.61
    ]

    # Test with downtrending data
    downtrend_prices = [
        0.70, 0.69, 0.68, 0.67, 0.68, 0.66, 0.65, 0.64, 0.65, 0.63,
        0.62, 0.61, 0.60, 0.59, 0.58, 0.57, 0.56, 0.55, 0.54, 0.53,
        0.52, 0.51, 0.50, 0.49, 0.48
    ]

    engine = PredictionEngine()

    print("=" * 60)
    print("UPTREND TEST")
    print("=" * 60)
    prediction = engine.analyze(uptrend_prices)
    if prediction:
        print(engine.get_summary(prediction, uptrend_prices[-1]))

    print("\n" + "=" * 60)
    print("DOWNTREND TEST")
    print("=" * 60)
    prediction = engine.analyze(downtrend_prices)
    if prediction:
        print(engine.get_summary(prediction, downtrend_prices[-1]))
