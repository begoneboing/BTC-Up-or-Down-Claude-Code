"""
Signal Generator for Polymarket
Generates buy/sell signals based on technical analysis

This module provides robust signal generation with comprehensive error handling
to ensure trading bot stability during edge cases and unexpected data conditions.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Import with error handling
try:
    from indicators import TechnicalIndicators, calculate_all_indicators_optimized
except ImportError as e:
    logger.error(f"Failed to import indicators module: {e}")
    raise

try:
    from prediction_engine import PredictionEngine, Direction
except ImportError as e:
    logger.error(f"Failed to import prediction_engine module: {e}")
    raise


class SignalError(Exception):
    """Base exception for signal generator errors."""
    pass


class InvalidSignalDataError(SignalError):
    """Raised when input data for signal generation is invalid."""
    pass


class SignalType(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class Signal:
    signal_type: SignalType
    strength: float  # 0-100
    entry_price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    reasoning: List[str]
    timestamp: str


class SignalGenerator:
    """
    Generates actionable trading signals from market data

    Error Handling:
        - Validates input price data before processing
        - Returns HOLD signal for invalid/insufficient data (safe default)
        - Catches prediction and indicator calculation errors
        - Logs all errors for debugging and monitoring
        - Never raises exceptions to caller (fail-safe for trading)
    """

    # Constants
    MIN_DATA_POINTS = 20
    MIN_VALID_PRICE = 0.0
    MAX_VALID_PRICE = 1.0

    def __init__(self, risk_percent: float = 5.0):
        self.engine = PredictionEngine()
        self.ti = TechnicalIndicators()
        self.risk_percent = risk_percent  # Stop loss distance
        self._last_error: Optional[str] = None

    def validate_inputs(
        self,
        prices: List[float],
        volume: float = 0,
        liquidity: float = 0
    ) -> tuple[bool, str]:
        """
        Validate all inputs before signal generation.

        Args:
            prices: List of price values
            volume: Market volume
            liquidity: Market liquidity

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Validate prices
        if prices is None:
            return False, "Price data is None"

        if not isinstance(prices, (list, tuple)):
            return False, f"Price data must be a list, got {type(prices).__name__}"

        if len(prices) < self.MIN_DATA_POINTS:
            return False, f"Insufficient data: {len(prices)} points, need at least {self.MIN_DATA_POINTS}"

        # Check for invalid price values
        for i, price in enumerate(prices):
            if price is None:
                return False, f"None value at index {i}"

            try:
                price_float = float(price)
            except (TypeError, ValueError):
                return False, f"Non-numeric price at index {i}: {price}"

            # NaN check
            if price_float != price_float:
                return False, f"NaN value at index {i}"

            # Inf check
            if abs(price_float) == float('inf'):
                return False, f"Infinite value at index {i}"

            # Range check (with tolerance)
            if price_float < self.MIN_VALID_PRICE - 0.001 or price_float > self.MAX_VALID_PRICE + 0.001:
                return False, f"Price {price_float} at index {i} outside valid range [0, 1]"

        # Validate volume and liquidity (non-negative)
        if volume is not None:
            try:
                vol = float(volume)
                if vol < 0:
                    return False, f"Volume cannot be negative: {volume}"
            except (TypeError, ValueError):
                return False, f"Invalid volume value: {volume}"

        if liquidity is not None:
            try:
                liq = float(liquidity)
                if liq < 0:
                    return False, f"Liquidity cannot be negative: {liquidity}"
            except (TypeError, ValueError):
                return False, f"Invalid liquidity value: {liquidity}"

        return True, ""

    def get_last_error(self) -> Optional[str]:
        """
        Get the last error message from signal generation.

        Returns:
            Error message string or None if no error occurred
        """
        return self._last_error

    def generate_signal(self, prices: List[float], volume: float = 0, liquidity: float = 0) -> Optional[Signal]:
        """
        Generate a trading signal from price data with comprehensive error handling.

        Args:
            prices: List of historical prices (0-1 range for Polymarket)
            volume: Market trading volume
            liquidity: Market liquidity

        Returns:
            Signal object with signal type, entry/exit levels, or None if generation fails

        Error Handling:
            - Returns None for invalid/insufficient data
            - Catches prediction and indicator errors
            - Logs all errors for debugging
            - Never raises exceptions to caller (fail-safe for trading)
        """
        self._last_error = None

        # Validate inputs
        is_valid, error_msg = self.validate_inputs(prices, volume, liquidity)
        if not is_valid:
            self._last_error = error_msg
            logger.warning(f"Input validation failed: {error_msg}")
            return None

        # Safely get current price
        try:
            current_price = float(prices[-1])
        except (IndexError, TypeError, ValueError) as e:
            self._last_error = f"Failed to get current price: {e}"
            logger.error(self._last_error)
            return None

        # Get prediction with error handling
        try:
            prediction = self.engine.analyze(prices)

            if not prediction:
                # Check if engine has error info
                engine_error = self.engine.get_last_error() if hasattr(self.engine, 'get_last_error') else None
                if engine_error:
                    self._last_error = f"Prediction failed: {engine_error}"
                else:
                    self._last_error = "Prediction returned None (insufficient data or neutral market)"
                logger.debug(self._last_error)
                return None
        except Exception as e:
            self._last_error = f"Prediction error: {e}"
            logger.error(self._last_error, exc_info=True)
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

        reasoning = list(prediction.reasoning)

        # Determine signal type based on prediction
        if prediction.direction == Direction.STRONG_UP and prediction.confidence >= 60:
            signal_type = SignalType.STRONG_BUY
        elif prediction.direction in [Direction.STRONG_UP, Direction.UP] and prediction.confidence >= 40:
            signal_type = SignalType.BUY
        elif prediction.direction == Direction.STRONG_DOWN and prediction.confidence >= 60:
            signal_type = SignalType.STRONG_SELL
        elif prediction.direction in [Direction.STRONG_DOWN, Direction.DOWN] and prediction.confidence >= 40:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD

        # Calculate stop loss and take profit with error handling
        try:
            levels = indicators.get("levels")
            volatility = indicators.get("volatility_20")

            # Validate volatility value
            if volatility is None or not isinstance(volatility, (int, float)):
                volatility = 0.02
            elif volatility != volatility or abs(volatility) == float('inf'):  # NaN/Inf check
                logger.warning(f"Invalid volatility value: {volatility}, using default")
                volatility = 0.02
            else:
                volatility = max(0.001, min(0.5, float(volatility)))  # Clamp to reasonable range

            if signal_type in [SignalType.STRONG_BUY, SignalType.BUY]:
                # For buys: stop below support, target at resistance
                if levels and isinstance(levels, dict):
                    support = levels.get("support")
                    resistance = levels.get("resistance")

                    # Validate support/resistance values
                    if support is not None and resistance is not None:
                        try:
                            support = float(support)
                            resistance = float(resistance)
                            # Sanity check
                            if 0 <= support <= 1 and 0 <= resistance <= 1:
                                stop_loss = max(support - volatility, 0.01)
                                take_profit = min(resistance + volatility, 0.99)
                            else:
                                raise ValueError("Support/resistance outside valid range")
                        except (TypeError, ValueError) as e:
                            logger.warning(f"Invalid support/resistance levels: {e}, using defaults")
                            stop_loss = max(current_price * (1 - self.risk_percent / 100), 0.01)
                            take_profit = min(current_price * 1.15, 0.99)
                    else:
                        stop_loss = max(current_price * (1 - self.risk_percent / 100), 0.01)
                        take_profit = min(current_price * 1.15, 0.99)
                else:
                    stop_loss = max(current_price * (1 - self.risk_percent / 100), 0.01)
                    take_profit = min(current_price * 1.15, 0.99)

                reasoning.append(f"Entry: {current_price:.2%}, Stop: {stop_loss:.2%}, Target: {take_profit:.2%}")

            elif signal_type in [SignalType.STRONG_SELL, SignalType.SELL]:
                # For sells: stop above resistance, target at support
                if levels and isinstance(levels, dict):
                    support = levels.get("support")
                    resistance = levels.get("resistance")

                    if support is not None and resistance is not None:
                        try:
                            support = float(support)
                            resistance = float(resistance)
                            if 0 <= support <= 1 and 0 <= resistance <= 1:
                                stop_loss = min(resistance + volatility, 0.99)
                                take_profit = max(support - volatility, 0.01)
                            else:
                                raise ValueError("Support/resistance outside valid range")
                        except (TypeError, ValueError) as e:
                            logger.warning(f"Invalid support/resistance levels: {e}, using defaults")
                            stop_loss = min(current_price * (1 + self.risk_percent / 100), 0.99)
                            take_profit = max(current_price * 0.85, 0.01)
                    else:
                        stop_loss = min(current_price * (1 + self.risk_percent / 100), 0.99)
                        take_profit = max(current_price * 0.85, 0.01)
                else:
                    stop_loss = min(current_price * (1 + self.risk_percent / 100), 0.99)
                    take_profit = max(current_price * 0.85, 0.01)

                reasoning.append(f"Entry: {current_price:.2%}, Stop: {stop_loss:.2%}, Target: {take_profit:.2%}")

            else:
                stop_loss = None
                take_profit = None

        except Exception as e:
            logger.warning(f"Error calculating stop loss/take profit: {e}, using defaults")
            if signal_type in [SignalType.STRONG_BUY, SignalType.BUY]:
                stop_loss = max(current_price * 0.95, 0.01)
                take_profit = min(current_price * 1.15, 0.99)
            elif signal_type in [SignalType.STRONG_SELL, SignalType.SELL]:
                stop_loss = min(current_price * 1.05, 0.99)
                take_profit = max(current_price * 0.85, 0.01)
            else:
                stop_loss = None
                take_profit = None

        # Add volume/liquidity context
        if volume > 100000:
            reasoning.append(f"High volume: ${volume:,.0f}")
        if liquidity > 50000:
            reasoning.append(f"Good liquidity: ${liquidity:,.0f}")

        return Signal(
            signal_type=signal_type,
            strength=prediction.confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=reasoning,
            timestamp=datetime.now().isoformat()
        )

    def format_signal(self, signal: Signal, market_name: str = "") -> str:
        """Format signal for display"""
        emoji = {
            SignalType.STRONG_BUY: "🟢🟢",
            SignalType.BUY: "🟢",
            SignalType.HOLD: "⚪",
            SignalType.SELL: "🔴",
            SignalType.STRONG_SELL: "🔴🔴"
        }

        lines = []

        if market_name:
            lines.append(f"Market: {market_name}")

        lines.extend([
            f"Signal: {emoji.get(signal.signal_type, '')} {signal.signal_type.value}",
            f"Strength: {signal.strength:.0f}%",
            f"Entry Price: {signal.entry_price:.2%}",
        ])

        if signal.stop_loss:
            sl_pct = (signal.stop_loss - signal.entry_price) / signal.entry_price * 100
            lines.append(f"Stop Loss: {signal.stop_loss:.2%} ({sl_pct:+.1f}%)")

        if signal.take_profit:
            tp_pct = (signal.take_profit - signal.entry_price) / signal.entry_price * 100
            lines.append(f"Take Profit: {signal.take_profit:.2%} ({tp_pct:+.1f}%)")

        if signal.stop_loss and signal.take_profit:
            risk = abs(signal.entry_price - signal.stop_loss)
            reward = abs(signal.take_profit - signal.entry_price)
            if risk > 0:
                rr_ratio = reward / risk
                lines.append(f"Risk/Reward: 1:{rr_ratio:.1f}")

        lines.append(f"\nReasoning:")
        for reason in signal.reasoning:
            lines.append(f"  - {reason}")

        return "\n".join(lines)


class AlertManager:
    """
    Manages alerts and notifications for signals
    """

    def __init__(self):
        self.alerts: List[Dict] = []

    def add_alert(self, signal: Signal, market_name: str, market_id: str):
        """Add a signal alert"""
        if signal.signal_type == SignalType.HOLD:
            return  # Don't alert on holds

        alert = {
            "timestamp": signal.timestamp,
            "market_name": market_name,
            "market_id": market_id,
            "signal": signal.signal_type.value,
            "strength": signal.strength,
            "entry_price": signal.entry_price,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit
        }

        self.alerts.append(alert)

    def get_active_alerts(self, min_strength: float = 50) -> List[Dict]:
        """Get alerts above minimum strength"""
        return [a for a in self.alerts if a["strength"] >= min_strength]

    def get_buy_signals(self) -> List[Dict]:
        """Get all buy signals"""
        return [a for a in self.alerts if "BUY" in a["signal"]]

    def get_sell_signals(self) -> List[Dict]:
        """Get all sell signals"""
        return [a for a in self.alerts if "SELL" in a["signal"]]

    def print_summary(self):
        """Print alert summary"""
        buys = self.get_buy_signals()
        sells = self.get_sell_signals()

        print(f"\n{'='*60}")
        print("SIGNAL SUMMARY")
        print(f"{'='*60}")
        print(f"Total Alerts: {len(self.alerts)}")
        print(f"Buy Signals: {len(buys)}")
        print(f"Sell Signals: {len(sells)}")

        if buys:
            print(f"\n{'='*40}")
            print("TOP BUY SIGNALS")
            print(f"{'='*40}")
            for alert in sorted(buys, key=lambda x: -x["strength"])[:5]:
                print(f"\n{alert['market_name'][:50]}")
                print(f"  Signal: {alert['signal']} ({alert['strength']:.0f}%)")
                print(f"  Entry: {alert['entry_price']:.2%}")

        if sells:
            print(f"\n{'='*40}")
            print("TOP SELL SIGNALS")
            print(f"{'='*40}")
            for alert in sorted(sells, key=lambda x: -x["strength"])[:5]:
                print(f"\n{alert['market_name'][:50]}")
                print(f"  Signal: {alert['signal']} ({alert['strength']:.0f}%)")
                print(f"  Entry: {alert['entry_price']:.2%}")


if __name__ == "__main__":
    # Test signal generation
    generator = SignalGenerator()

    # Uptrend data
    uptrend = [
        0.40, 0.41, 0.42, 0.41, 0.43, 0.44, 0.45, 0.44, 0.46, 0.47,
        0.46, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56,
        0.57, 0.58, 0.59, 0.60, 0.61
    ]

    # Downtrend data
    downtrend = [
        0.70, 0.69, 0.68, 0.67, 0.68, 0.66, 0.65, 0.64, 0.65, 0.63,
        0.62, 0.61, 0.60, 0.59, 0.58, 0.57, 0.56, 0.55, 0.54, 0.53,
        0.52, 0.51, 0.50, 0.49, 0.48
    ]

    print("=" * 60)
    print("BUY SIGNAL TEST (Uptrend)")
    print("=" * 60)
    signal = generator.generate_signal(uptrend, volume=50000, liquidity=25000)
    if signal:
        print(generator.format_signal(signal, "Test Uptrend Market"))

    print("\n" + "=" * 60)
    print("SELL SIGNAL TEST (Downtrend)")
    print("=" * 60)
    signal = generator.generate_signal(downtrend, volume=50000, liquidity=25000)
    if signal:
        print(generator.format_signal(signal, "Test Downtrend Market"))
