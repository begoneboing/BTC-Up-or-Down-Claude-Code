"""
Technical Indicators for Price Analysis
Implements various technical indicators for up/down prediction

Includes both standard and optimized calculation methods:
- TechnicalIndicators: Original implementation
- OptimizedIndicators: Batch processing with caching for better performance
"""

from typing import List, Dict, Optional, Tuple
import statistics
from functools import lru_cache


class PriceCache:
    """
    Pre-computed price data for efficient indicator calculation.
    Computes common values once and reuses them across indicators.
    """

    def __init__(self, prices: List[float]):
        self.prices = prices
        self.n = len(prices)

        # Pre-compute price changes (used by RSI, momentum, etc.)
        self._changes: Optional[List[float]] = None
        self._gains: Optional[List[float]] = None
        self._losses: Optional[List[float]] = None

        # EMA cache to avoid recalculation
        self._ema_cache: Dict[int, List[float]] = {}

    @property
    def changes(self) -> List[float]:
        """Price changes from previous period"""
        if self._changes is None:
            self._changes = [
                self.prices[i] - self.prices[i-1]
                for i in range(1, self.n)
            ]
        return self._changes

    @property
    def gains(self) -> List[float]:
        """Positive price changes (gains)"""
        if self._gains is None:
            self._gains = [max(0, c) for c in self.changes]
        return self._gains

    @property
    def losses(self) -> List[float]:
        """Absolute negative price changes (losses)"""
        if self._losses is None:
            self._losses = [abs(min(0, c)) for c in self.changes]
        return self._losses

    def get_ema_series(self, period: int) -> List[float]:
        """
        Get full EMA series for a period (cached).
        Returns EMA values for each point starting from index period-1.
        """
        if period in self._ema_cache:
            return self._ema_cache[period]

        if self.n < period:
            self._ema_cache[period] = []
            return []

        multiplier = 2 / (period + 1)
        ema_values = []

        # First EMA value is SMA
        ema = sum(self.prices[:period]) / period
        ema_values.append(ema)

        # Calculate rest of EMA series
        for price in self.prices[period:]:
            ema = (price - ema) * multiplier + ema
            ema_values.append(ema)

        self._ema_cache[period] = ema_values
        return ema_values


class TechnicalIndicators:
    """Calculate technical indicators from price data"""

    @staticmethod
    def extract_prices(history: List[Dict]) -> List[float]:
        """Extract price values from history data"""
        prices = []
        for point in history:
            if isinstance(point, dict):
                price = point.get("p") or point.get("price") or point.get("c")
                if price is not None:
                    prices.append(float(price))
            elif isinstance(point, (int, float)):
                prices.append(float(point))
        return prices

    @staticmethod
    def sma(prices: List[float], period: int) -> Optional[float]:
        """Simple Moving Average"""
        if len(prices) < period:
            return None
        return sum(prices[-period:]) / period

    @staticmethod
    def ema(prices: List[float], period: int) -> Optional[float]:
        """Exponential Moving Average"""
        if len(prices) < period:
            return None

        multiplier = 2 / (period + 1)
        ema_value = sum(prices[:period]) / period  # Start with SMA

        for price in prices[period:]:
            ema_value = (price - ema_value) * multiplier + ema_value

        return ema_value

    @staticmethod
    def rsi(prices: List[float], period: int = 14) -> Optional[float]:
        """
        Relative Strength Index
        Returns value between 0-100
        >70 = overbought, <30 = oversold
        """
        if len(prices) < period + 1:
            return None

        gains = []
        losses = []

        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))

        if len(gains) < period:
            return None

        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        rsi_value = 100 - (100 / (1 + rs))

        return rsi_value

    @staticmethod
    def momentum(prices: List[float], period: int = 10) -> Optional[float]:
        """
        Price Momentum
        Positive = upward momentum, Negative = downward momentum
        """
        if len(prices) < period:
            return None
        return prices[-1] - prices[-period]

    @staticmethod
    def rate_of_change(prices: List[float], period: int = 10) -> Optional[float]:
        """
        Rate of Change (ROC)
        Returns percentage change over period
        """
        if len(prices) < period or prices[-period] == 0:
            return None
        return ((prices[-1] - prices[-period]) / prices[-period]) * 100

    @staticmethod
    def volatility(prices: List[float], period: int = 20) -> Optional[float]:
        """
        Price Volatility (Standard Deviation)
        Higher = more volatile
        """
        if len(prices) < period:
            return None
        return statistics.stdev(prices[-period:])

    @staticmethod
    def bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2.0) -> Optional[Dict]:
        """
        Bollinger Bands
        Returns upper, middle, lower bands
        """
        if len(prices) < period:
            return None

        middle = sum(prices[-period:]) / period
        std = statistics.stdev(prices[-period:])

        return {
            "upper": middle + (std * std_dev),
            "middle": middle,
            "lower": middle - (std * std_dev),
            "bandwidth": (std * std_dev * 2) / middle if middle > 0 else 0
        }

    @staticmethod
    def macd(prices: List[float], fast: int = 12, slow: int = 26, signal_period: int = 9) -> Optional[Dict]:
        """
        MACD (Moving Average Convergence Divergence)
        Positive histogram = bullish, Negative = bearish
        Returns MACD line, signal line, histogram, and divergence detection
        """
        if len(prices) < slow + signal_period:
            return None

        # Calculate MACD line for each point (we need history for signal line)
        macd_history = []
        for i in range(slow, len(prices) + 1):
            subset = prices[:i]
            ema_fast = TechnicalIndicators.ema(subset, fast)
            ema_slow = TechnicalIndicators.ema(subset, slow)
            if ema_fast is not None and ema_slow is not None:
                macd_history.append(ema_fast - ema_slow)

        if len(macd_history) < signal_period:
            return None

        # Calculate signal line as EMA of MACD
        signal_line = TechnicalIndicators.ema(macd_history, signal_period)

        if signal_line is None:
            return None

        macd_line = macd_history[-1]
        histogram = macd_line - signal_line

        # Previous values for divergence detection
        prev_histogram = macd_history[-2] - TechnicalIndicators.ema(macd_history[:-1], signal_period) if len(macd_history) >= signal_period + 1 else None

        # Detect MACD crossovers and divergences
        crossover = None
        if prev_histogram is not None:
            if prev_histogram < 0 and histogram > 0:
                crossover = "bullish"  # MACD crossed above signal
            elif prev_histogram > 0 and histogram < 0:
                crossover = "bearish"  # MACD crossed below signal

        # Detect price/MACD divergence (potential reversal signal)
        divergence = None
        if len(prices) >= 10 and len(macd_history) >= 10:
            # Compare price direction vs MACD direction over last 10 periods
            price_change = prices[-1] - prices[-10]
            macd_change = macd_history[-1] - macd_history[-10]

            if price_change > 0 and macd_change < 0:
                divergence = "bearish"  # Price up but MACD down = weakening trend
            elif price_change < 0 and macd_change > 0:
                divergence = "bullish"  # Price down but MACD up = potential reversal

        return {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram,
            "crossover": crossover,
            "divergence": divergence
        }

    @staticmethod
    def trend_strength(prices: List[float], period: int = 14) -> Optional[Dict]:
        """
        Analyze trend strength and direction
        Returns trend direction and strength score
        """
        if len(prices) < period:
            return None

        recent_prices = prices[-period:]

        # Calculate linear regression slope
        n = len(recent_prices)
        x_sum = sum(range(n))
        y_sum = sum(recent_prices)
        xy_sum = sum(i * p for i, p in enumerate(recent_prices))
        x2_sum = sum(i * i for i in range(n))

        denominator = n * x2_sum - x_sum * x_sum
        if denominator == 0:
            return {"direction": "neutral", "strength": 0, "slope": 0}

        slope = (n * xy_sum - x_sum * y_sum) / denominator

        # Normalize slope to price range
        price_range = max(recent_prices) - min(recent_prices)
        if price_range > 0:
            normalized_slope = slope / price_range
        else:
            normalized_slope = 0

        # Determine direction and strength
        if normalized_slope > 0.02:
            direction = "up"
            strength = min(abs(normalized_slope) * 50, 100)
        elif normalized_slope < -0.02:
            direction = "down"
            strength = min(abs(normalized_slope) * 50, 100)
        else:
            direction = "neutral"
            strength = 0

        return {
            "direction": direction,
            "strength": strength,
            "slope": slope
        }

    @staticmethod
    def support_resistance(prices: List[float], period: int = 20) -> Optional[Dict]:
        """
        Identify support and resistance levels
        """
        if len(prices) < period:
            return None

        recent = prices[-period:]
        current = prices[-1]

        high = max(recent)
        low = min(recent)

        # Distance to support/resistance
        resistance_distance = (high - current) / current if current > 0 else 0
        support_distance = (current - low) / current if current > 0 else 0

        return {
            "resistance": high,
            "support": low,
            "current": current,
            "resistance_distance": resistance_distance,
            "support_distance": support_distance,
            "position": (current - low) / (high - low) if high != low else 0.5
        }


class OptimizedIndicators:
    """
    Optimized technical indicator calculator using pre-computed caching.

    This class calculates all indicators in a single pass by:
    1. Pre-computing price changes, gains, and losses once
    2. Caching EMA series to avoid redundant calculations
    3. Computing dependent indicators (MACD) using cached EMAs

    Performance improvement: ~2-3x faster for full indicator calculation
    on typical price series (50+ data points).
    """

    def __init__(self, prices: List[float]):
        """
        Initialize with price data.

        Args:
            prices: List of price values
        """
        self.prices = prices
        self.n = len(prices)
        self.cache = PriceCache(prices)

    def sma(self, period: int) -> Optional[float]:
        """Simple Moving Average using last N prices"""
        if self.n < period:
            return None
        return sum(self.prices[-period:]) / period

    def ema(self, period: int) -> Optional[float]:
        """Exponential Moving Average (uses cached series)"""
        ema_series = self.cache.get_ema_series(period)
        return ema_series[-1] if ema_series else None

    def rsi(self, period: int = 14) -> Optional[float]:
        """
        RSI using pre-computed gains/losses.
        More efficient than recalculating changes each time.
        """
        if self.n < period + 1:
            return None

        gains = self.cache.gains
        losses = self.cache.losses

        if len(gains) < period:
            return None

        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def momentum(self, period: int = 10) -> Optional[float]:
        """Price momentum"""
        if self.n < period:
            return None
        return self.prices[-1] - self.prices[-period]

    def rate_of_change(self, period: int = 10) -> Optional[float]:
        """Rate of Change percentage"""
        if self.n < period or self.prices[-period] == 0:
            return None
        return ((self.prices[-1] - self.prices[-period]) / self.prices[-period]) * 100

    def volatility(self, period: int = 20) -> Optional[float]:
        """Standard deviation of prices"""
        if self.n < period:
            return None
        return statistics.stdev(self.prices[-period:])

    def bollinger_bands(self, period: int = 20, std_dev: float = 2.0) -> Optional[Dict]:
        """Bollinger Bands with single-pass std calculation"""
        if self.n < period:
            return None

        recent = self.prices[-period:]
        middle = sum(recent) / period
        std = statistics.stdev(recent)

        return {
            "upper": middle + (std * std_dev),
            "middle": middle,
            "lower": middle - (std * std_dev),
            "bandwidth": (std * std_dev * 2) / middle if middle > 0 else 0
        }

    def macd(self, fast: int = 12, slow: int = 26, signal_period: int = 9) -> Optional[Dict]:
        """
        Optimized MACD using cached EMA series.

        This is significantly faster than the original because:
        1. EMA series are computed once and cached
        2. MACD line history is computed directly from cached EMAs
        """
        if self.n < slow + signal_period:
            return None

        # Get cached EMA series
        ema_fast_series = self.cache.get_ema_series(fast)
        ema_slow_series = self.cache.get_ema_series(slow)

        if not ema_fast_series or not ema_slow_series:
            return None

        # Compute MACD line series (difference of EMAs where both exist)
        # Fast EMA starts at index fast-1, slow EMA starts at index slow-1
        # MACD starts when both exist: index slow-1 of prices
        macd_history = []

        # Align the series: slow EMA is shorter, use it as reference
        fast_offset = slow - fast  # How many extra fast EMA values we have
        for i in range(len(ema_slow_series)):
            macd_val = ema_fast_series[fast_offset + i] - ema_slow_series[i]
            macd_history.append(macd_val)

        if len(macd_history) < signal_period:
            return None

        # Calculate signal line as EMA of MACD history
        multiplier = 2 / (signal_period + 1)
        signal_value = sum(macd_history[:signal_period]) / signal_period

        for val in macd_history[signal_period:]:
            signal_value = (val - signal_value) * multiplier + signal_value

        macd_line = macd_history[-1]
        histogram = macd_line - signal_value

        # Previous histogram for crossover detection
        prev_histogram = None
        if len(macd_history) >= signal_period + 1:
            # Recalculate previous signal
            prev_signal = sum(macd_history[:signal_period]) / signal_period
            for val in macd_history[signal_period:-1]:
                prev_signal = (val - prev_signal) * multiplier + prev_signal
            prev_histogram = macd_history[-2] - prev_signal

        # Detect crossovers
        crossover = None
        if prev_histogram is not None:
            if prev_histogram < 0 and histogram > 0:
                crossover = "bullish"
            elif prev_histogram > 0 and histogram < 0:
                crossover = "bearish"

        # Detect divergence
        divergence = None
        if self.n >= 10 and len(macd_history) >= 10:
            price_change = self.prices[-1] - self.prices[-10]
            macd_change = macd_history[-1] - macd_history[-10]

            if price_change > 0 and macd_change < 0:
                divergence = "bearish"
            elif price_change < 0 and macd_change > 0:
                divergence = "bullish"

        return {
            "macd": macd_line,
            "signal": signal_value,
            "histogram": histogram,
            "crossover": crossover,
            "divergence": divergence
        }

    def trend_strength(self, period: int = 14) -> Optional[Dict]:
        """Analyze trend using linear regression"""
        if self.n < period:
            return None

        recent = self.prices[-period:]
        n = period

        # Pre-compute sums for linear regression
        x_sum = n * (n - 1) // 2  # sum(0..n-1) = n(n-1)/2
        x2_sum = (n - 1) * n * (2 * n - 1) // 6  # sum of squares formula
        y_sum = sum(recent)
        xy_sum = sum(i * p for i, p in enumerate(recent))

        denominator = n * x2_sum - x_sum * x_sum
        if denominator == 0:
            return {"direction": "neutral", "strength": 0, "slope": 0}

        slope = (n * xy_sum - x_sum * y_sum) / denominator

        price_range = max(recent) - min(recent)
        normalized_slope = slope / price_range if price_range > 0 else 0

        if normalized_slope > 0.02:
            direction = "up"
            strength = min(abs(normalized_slope) * 50, 100)
        elif normalized_slope < -0.02:
            direction = "down"
            strength = min(abs(normalized_slope) * 50, 100)
        else:
            direction = "neutral"
            strength = 0

        return {
            "direction": direction,
            "strength": strength,
            "slope": slope
        }

    def support_resistance(self, period: int = 20) -> Optional[Dict]:
        """Identify support and resistance levels"""
        if self.n < period:
            return None

        recent = self.prices[-period:]
        current = self.prices[-1]
        high = max(recent)
        low = min(recent)

        resistance_distance = (high - current) / current if current > 0 else 0
        support_distance = (current - low) / current if current > 0 else 0

        return {
            "resistance": high,
            "support": low,
            "current": current,
            "resistance_distance": resistance_distance,
            "support_distance": support_distance,
            "position": (current - low) / (high - low) if high != low else 0.5
        }

    def calculate_all(self) -> Dict:
        """
        Calculate all indicators in a single optimized pass.

        Returns the same structure as calculate_all_indicators() but faster.
        """
        return {
            "price_current": self.prices[-1] if self.prices else None,
            "price_count": self.n,

            # Moving Averages
            "sma_5": self.sma(5),
            "sma_10": self.sma(10),
            "sma_20": self.sma(20),
            "ema_5": self.ema(5),
            "ema_10": self.ema(10),

            # Momentum Indicators
            "rsi_14": self.rsi(14),
            "momentum_10": self.momentum(10),
            "roc_10": self.rate_of_change(10),

            # Volatility
            "volatility_20": self.volatility(20),
            "bollinger": self.bollinger_bands(20),

            # Trend
            "macd": self.macd(),
            "trend": self.trend_strength(14),

            # Support/Resistance
            "levels": self.support_resistance(20)
        }


def calculate_all_indicators_optimized(prices: List[float]) -> Dict:
    """
    Calculate all technical indicators using optimized batch processing.

    This is the recommended function for production use.
    Uses caching and single-pass calculations for better performance.

    Args:
        prices: List of price values

    Returns:
        Dictionary of all indicator values
    """
    if not prices:
        return {"price_current": None, "price_count": 0}

    optimizer = OptimizedIndicators(prices)
    return optimizer.calculate_all()


def calculate_all_indicators(prices: List[float]) -> Dict:
    """Calculate all technical indicators for a price series"""
    ti = TechnicalIndicators()

    indicators = {
        "price_current": prices[-1] if prices else None,
        "price_count": len(prices),

        # Moving Averages
        "sma_5": ti.sma(prices, 5),
        "sma_10": ti.sma(prices, 10),
        "sma_20": ti.sma(prices, 20),
        "ema_5": ti.ema(prices, 5),
        "ema_10": ti.ema(prices, 10),

        # Momentum Indicators
        "rsi_14": ti.rsi(prices, 14),
        "momentum_10": ti.momentum(prices, 10),
        "roc_10": ti.rate_of_change(prices, 10),

        # Volatility
        "volatility_20": ti.volatility(prices, 20),
        "bollinger": ti.bollinger_bands(prices, 20),

        # Trend
        "macd": ti.macd(prices),
        "trend": ti.trend_strength(prices, 14),

        # Support/Resistance
        "levels": ti.support_resistance(prices, 20)
    }

    return indicators


if __name__ == "__main__":
    import time

    # Test with sample data - small dataset
    sample_prices = [
        0.50, 0.51, 0.52, 0.51, 0.53, 0.54, 0.55, 0.54, 0.56, 0.57,
        0.56, 0.58, 0.59, 0.58, 0.60, 0.61, 0.60, 0.62, 0.63, 0.64,
        0.63, 0.65, 0.66, 0.65, 0.67
    ]

    # Larger dataset for benchmarking (100 data points)
    large_prices = sample_prices * 4

    print("Testing Technical Indicators")
    print("=" * 60)

    # Test original implementation
    indicators = calculate_all_indicators(sample_prices)
    print(f"\nCurrent Price: {indicators['price_current']:.2%}")
    print(f"RSI (14): {indicators['rsi_14']:.1f}" if indicators['rsi_14'] else "RSI: N/A")
    print(f"SMA (10): {indicators['sma_10']:.2%}" if indicators['sma_10'] else "SMA: N/A")
    print(f"Momentum: {indicators['momentum_10']:.4f}" if indicators['momentum_10'] else "Momentum: N/A")

    if indicators['trend']:
        print(f"Trend: {indicators['trend']['direction'].upper()} (strength: {indicators['trend']['strength']:.0f})")

    if indicators['bollinger']:
        bb = indicators['bollinger']
        print(f"Bollinger: {bb['lower']:.2%} - {bb['middle']:.2%} - {bb['upper']:.2%}")

    # Test optimized implementation
    print("\n" + "=" * 60)
    print("OPTIMIZED vs ORIGINAL Comparison")
    print("=" * 60)

    opt_indicators = calculate_all_indicators_optimized(sample_prices)

    # Verify results match
    print("\nResult verification (should match):")
    print(f"  SMA(10): Original={indicators['sma_10']:.4f}, Optimized={opt_indicators['sma_10']:.4f}")
    print(f"  EMA(10): Original={indicators['ema_10']:.4f}, Optimized={opt_indicators['ema_10']:.4f}")
    print(f"  RSI(14): Original={indicators['rsi_14']:.2f}, Optimized={opt_indicators['rsi_14']:.2f}")

    if indicators['macd'] and opt_indicators['macd']:
        print(f"  MACD: Original={indicators['macd']['macd']:.6f}, Optimized={opt_indicators['macd']['macd']:.6f}")
        print(f"  Signal: Original={indicators['macd']['signal']:.6f}, Optimized={opt_indicators['macd']['signal']:.6f}")

    # Benchmark performance
    print("\n" + "-" * 60)
    print("PERFORMANCE BENCHMARK (1000 iterations on 100 data points)")
    print("-" * 60)

    iterations = 1000

    # Benchmark original
    start = time.perf_counter()
    for _ in range(iterations):
        calculate_all_indicators(large_prices)
    original_time = time.perf_counter() - start

    # Benchmark optimized
    start = time.perf_counter()
    for _ in range(iterations):
        calculate_all_indicators_optimized(large_prices)
    optimized_time = time.perf_counter() - start

    print(f"Original:  {original_time:.3f}s ({original_time/iterations*1000:.3f}ms per call)")
    print(f"Optimized: {optimized_time:.3f}s ({optimized_time/iterations*1000:.3f}ms per call)")

    speedup = original_time / optimized_time if optimized_time > 0 else 0
    print(f"\nSpeedup: {speedup:.2f}x faster")

    # Verify MACD results match more closely
    print("\n" + "-" * 60)
    print("MACD DETAILED COMPARISON")
    print("-" * 60)

    orig_macd = TechnicalIndicators.macd(large_prices)
    opt = OptimizedIndicators(large_prices)
    opt_macd = opt.macd()

    if orig_macd and opt_macd:
        print(f"  MACD Line - Original: {orig_macd['macd']:.8f}")
        print(f"  MACD Line - Optimized: {opt_macd['macd']:.8f}")
        print(f"  Difference: {abs(orig_macd['macd'] - opt_macd['macd']):.10f}")
        print(f"  Signal - Original: {orig_macd['signal']:.8f}")
        print(f"  Signal - Optimized: {opt_macd['signal']:.8f}")
        print(f"  Histogram Match: {abs(orig_macd['histogram'] - opt_macd['histogram']) < 0.0001}")
        print(f"  Crossover Match: {orig_macd['crossover'] == opt_macd['crossover']}")
        print(f"  Divergence Match: {orig_macd['divergence'] == opt_macd['divergence']}")

    print("\n" + "=" * 60)
    print("Technical Indicators Test Complete")
    print("=" * 60)
