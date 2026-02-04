"""
Ensemble Predictor for Polymarket
Combines multiple prediction strategies to improve signal quality and reduce false positives.

This module implements an ensemble approach that:
1. Combines market metrics prediction (price momentum, volume) with technical indicators (RSI, MACD)
2. Requires agreement between predictors to reduce false signals
3. Weights predictions based on market conditions
4. Provides confidence boosting when multiple strategies align

Iteration 12 Enhancement - Phase 5: Advanced Features
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Import prediction engines
try:
    from predictor import PolymarketPredictor, Prediction as MarketPrediction, Signal as MarketSignal
except ImportError as e:
    logger.error(f"Failed to import predictor module: {e}")
    raise

try:
    from prediction_engine import PredictionEngine, Prediction as TechPrediction, Direction, MarketRegime
except ImportError as e:
    logger.error(f"Failed to import prediction_engine module: {e}")
    raise

try:
    from data_collector import DataCollector
except ImportError as e:
    logger.error(f"Failed to import data_collector module: {e}")
    raise


class EnsembleSignal(Enum):
    """Ensemble signal types with stricter criteria than individual predictors"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class EnsemblePrediction:
    """
    Combined prediction from multiple strategies.

    Attributes:
        market_question: Market being analyzed
        condition_id: Market condition ID
        token_id: Token ID for trading
        current_price: Current market price
        signal: Final ensemble signal
        confidence: Combined confidence score (0-100)
        agreement_level: How well predictors agree (0-100)
        market_prediction: Underlying market metrics prediction
        tech_prediction: Underlying technical analysis prediction
        reasoning: Explanation of ensemble decision
        metrics: Combined metrics from all sources
    """
    market_question: str
    condition_id: str
    token_id: str
    current_price: float
    signal: EnsembleSignal
    confidence: float
    agreement_level: float
    market_prediction: Optional[MarketPrediction] = None
    tech_prediction: Optional[TechPrediction] = None
    reasoning: List[str] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class EnsembleConfig:
    """Configuration for ensemble prediction behavior"""

    def __init__(
        self,
        require_agreement: bool = True,
        min_agreement_pct: float = 60.0,
        market_weight: float = 0.4,
        tech_weight: float = 0.6,
        confidence_boost_on_agreement: float = 15.0,
        confidence_penalty_on_conflict: float = 25.0,
        min_confidence_threshold: float = 50.0,
        regime_weight_adjustments: Dict[str, Tuple[float, float]] = None
    ):
        """
        Initialize ensemble configuration.

        Args:
            require_agreement: If True, both predictors must agree on direction
            min_agreement_pct: Minimum agreement level to generate signal
            market_weight: Weight for market metrics prediction (0-1)
            tech_weight: Weight for technical analysis prediction (0-1)
            confidence_boost_on_agreement: Bonus confidence when predictors agree
            confidence_penalty_on_conflict: Penalty when predictors conflict
            min_confidence_threshold: Minimum confidence to generate non-neutral signal
            regime_weight_adjustments: Dict mapping regime to (market_weight, tech_weight)
        """
        self.require_agreement = require_agreement
        self.min_agreement_pct = min_agreement_pct
        self.market_weight = market_weight
        self.tech_weight = tech_weight
        self.confidence_boost_on_agreement = confidence_boost_on_agreement
        self.confidence_penalty_on_conflict = confidence_penalty_on_conflict
        self.min_confidence_threshold = min_confidence_threshold

        # Default regime-based weight adjustments
        # In trending markets, technical analysis is more reliable
        # In choppy markets, neither is reliable (both get reduced)
        self.regime_weight_adjustments = regime_weight_adjustments or {
            "TRENDING_UP": (0.35, 0.65),      # More tech weight in trends
            "TRENDING_DOWN": (0.35, 0.65),
            "CHOPPY": (0.5, 0.5),              # Equal weight, but both less reliable
            "LOW_VOLATILITY": (0.45, 0.55),   # Slight tech preference
        }


class EnsemblePredictor:
    """
    Combines multiple prediction strategies to improve signal quality.

    The ensemble approach reduces false positives by:
    1. Requiring agreement between different prediction methods
    2. Boosting confidence when signals align
    3. Penalizing conflicting signals
    4. Adjusting weights based on market regime

    Example usage:
        ensemble = EnsemblePredictor()
        predictions = ensemble.scan(limit=30)
        for pred in predictions:
            if pred.signal != EnsembleSignal.NEUTRAL:
                print(f"{pred.signal.value}: {pred.market_question}")
    """

    def __init__(self, config: EnsembleConfig = None):
        """Initialize the ensemble predictor with optional configuration."""
        self.config = config or EnsembleConfig()
        self.market_predictor = PolymarketPredictor()
        self.tech_engine = PredictionEngine()
        self.data_collector = DataCollector()
        self._last_error: Optional[str] = None

    def get_last_error(self) -> Optional[str]:
        """Get the last error message from prediction."""
        return self._last_error

    def _map_market_signal_to_direction(self, signal: MarketSignal) -> Optional[str]:
        """Map market predictor signal to a direction string."""
        mapping = {
            MarketSignal.STRONG_BUY: "bullish",
            MarketSignal.BUY: "bullish",
            MarketSignal.NEUTRAL: "neutral",
            MarketSignal.SELL: "bearish",
            MarketSignal.STRONG_SELL: "bearish"
        }
        return mapping.get(signal, "neutral")

    def _map_direction_to_string(self, direction: Direction) -> str:
        """Map technical direction to a direction string."""
        mapping = {
            Direction.STRONG_UP: "bullish",
            Direction.UP: "bullish",
            Direction.NEUTRAL: "neutral",
            Direction.DOWN: "bearish",
            Direction.STRONG_DOWN: "bearish"
        }
        return mapping.get(direction, "neutral")

    def _calculate_agreement(
        self,
        market_pred: Optional[MarketPrediction],
        tech_pred: Optional[TechPrediction]
    ) -> Tuple[float, str]:
        """
        Calculate agreement level between predictors.

        Returns:
            Tuple of (agreement_percentage, agreement_description)
        """
        if market_pred is None and tech_pred is None:
            return 0.0, "No predictions available"

        if market_pred is None:
            return 50.0, "Only technical prediction available"

        if tech_pred is None:
            return 50.0, "Only market prediction available"

        market_dir = self._map_market_signal_to_direction(market_pred.signal)
        tech_dir = self._map_direction_to_string(tech_pred.direction)

        # Perfect agreement: same direction
        if market_dir == tech_dir and market_dir != "neutral":
            # Bonus for both being strong signals
            market_strong = market_pred.signal in [MarketSignal.STRONG_BUY, MarketSignal.STRONG_SELL]
            tech_strong = tech_pred.direction in [Direction.STRONG_UP, Direction.STRONG_DOWN]

            if market_strong and tech_strong:
                return 100.0, "Strong agreement: both predictors strongly agree"
            elif market_strong or tech_strong:
                return 85.0, "Good agreement: predictors agree, one is strong"
            else:
                return 75.0, "Agreement: predictors agree on direction"

        # Partial agreement: one neutral, one directional
        if market_dir == "neutral" or tech_dir == "neutral":
            return 50.0, "Partial: one predictor neutral"

        # Disagreement: opposite directions
        if market_dir != tech_dir:
            return 25.0, "Conflict: predictors disagree on direction"

        # Neutral agreement
        return 60.0, "Both neutral"

    def _get_regime_adjusted_weights(
        self,
        regime: Optional[MarketRegime]
    ) -> Tuple[float, float]:
        """Get weights adjusted for current market regime."""
        if regime is None:
            return self.config.market_weight, self.config.tech_weight

        regime_str = regime.value if hasattr(regime, 'value') else str(regime)
        weights = self.config.regime_weight_adjustments.get(
            regime_str,
            (self.config.market_weight, self.config.tech_weight)
        )
        return weights

    def _combine_predictions(
        self,
        market_pred: Optional[MarketPrediction],
        tech_pred: Optional[TechPrediction],
        price_data: List[float]
    ) -> EnsemblePrediction:
        """
        Combine predictions from both engines into a single ensemble prediction.

        Args:
            market_pred: Prediction from market metrics
            tech_pred: Prediction from technical analysis
            price_data: Historical price data for context

        Returns:
            Combined EnsemblePrediction
        """
        reasoning = []

        # Handle case where neither prediction is available
        if market_pred is None and tech_pred is None:
            self._last_error = "No predictions available from either engine"
            return EnsemblePrediction(
                market_question="Unknown",
                condition_id="",
                token_id="",
                current_price=0.5,
                signal=EnsembleSignal.NEUTRAL,
                confidence=0.0,
                agreement_level=0.0,
                reasoning=["No predictions available"]
            )

        # Extract common fields
        if market_pred:
            market_question = market_pred.market_question
            condition_id = market_pred.condition_id
            token_id = market_pred.token_id
            current_price = market_pred.current_price
        else:
            market_question = "Unknown"
            condition_id = ""
            token_id = ""
            current_price = price_data[-1] if price_data else 0.5

        # Calculate agreement
        agreement_pct, agreement_desc = self._calculate_agreement(market_pred, tech_pred)
        reasoning.append(f"Agreement: {agreement_desc} ({agreement_pct:.0f}%)")

        # Get market regime from tech prediction
        regime = tech_pred.market_regime if tech_pred else None
        if regime:
            reasoning.append(f"Market regime: {regime.value}")

        # Get regime-adjusted weights
        market_weight, tech_weight = self._get_regime_adjusted_weights(regime)

        # Calculate weighted scores
        market_score = 0.0
        tech_score = 0.0

        if market_pred:
            # Convert market signal to score (-100 to +100)
            signal_scores = {
                MarketSignal.STRONG_BUY: 100,
                MarketSignal.BUY: 50,
                MarketSignal.NEUTRAL: 0,
                MarketSignal.SELL: -50,
                MarketSignal.STRONG_SELL: -100
            }
            market_score = signal_scores.get(market_pred.signal, 0) * (market_pred.confidence / 100)
            reasoning.append(f"Market score: {market_score:+.1f} ({market_pred.signal.value}, {market_pred.confidence:.0f}%)")

        if tech_pred:
            # Convert tech direction to score (-100 to +100)
            direction_scores = {
                Direction.STRONG_UP: 100,
                Direction.UP: 50,
                Direction.NEUTRAL: 0,
                Direction.DOWN: -50,
                Direction.STRONG_DOWN: -100
            }
            tech_score = direction_scores.get(tech_pred.direction, 0) * (tech_pred.confidence / 100)
            reasoning.append(f"Tech score: {tech_score:+.1f} ({tech_pred.direction.value}, {tech_pred.confidence:.0f}%)")

        # Calculate combined score
        if market_pred and tech_pred:
            combined_score = (market_score * market_weight) + (tech_score * tech_weight)
        elif market_pred:
            combined_score = market_score
        else:
            combined_score = tech_score

        # Calculate base confidence from individual confidences
        if market_pred and tech_pred:
            base_confidence = (market_pred.confidence * market_weight + tech_pred.confidence * tech_weight)
        elif market_pred:
            base_confidence = market_pred.confidence
        else:
            base_confidence = tech_pred.confidence if tech_pred else 0

        # Apply agreement adjustments
        if agreement_pct >= 75:
            base_confidence = min(100, base_confidence + self.config.confidence_boost_on_agreement)
            reasoning.append(f"Confidence boosted +{self.config.confidence_boost_on_agreement:.0f}% for agreement")
        elif agreement_pct <= 30:
            base_confidence = max(0, base_confidence - self.config.confidence_penalty_on_conflict)
            reasoning.append(f"Confidence penalized -{self.config.confidence_penalty_on_conflict:.0f}% for conflict")

        # Check if we require agreement
        if self.config.require_agreement and agreement_pct < self.config.min_agreement_pct:
            combined_score = 0  # Force neutral if not enough agreement
            reasoning.append(f"Signal filtered: agreement {agreement_pct:.0f}% below threshold {self.config.min_agreement_pct:.0f}%")

        # Determine final signal
        if combined_score >= 60 and base_confidence >= self.config.min_confidence_threshold:
            signal = EnsembleSignal.STRONG_BUY
        elif combined_score >= 25 and base_confidence >= self.config.min_confidence_threshold:
            signal = EnsembleSignal.BUY
        elif combined_score <= -60 and base_confidence >= self.config.min_confidence_threshold:
            signal = EnsembleSignal.STRONG_SELL
        elif combined_score <= -25 and base_confidence >= self.config.min_confidence_threshold:
            signal = EnsembleSignal.SELL
        else:
            signal = EnsembleSignal.NEUTRAL
            if base_confidence < self.config.min_confidence_threshold:
                reasoning.append(f"Signal filtered: confidence {base_confidence:.0f}% below threshold {self.config.min_confidence_threshold:.0f}%")

        # Add regime-based filtering for choppy markets
        if regime == MarketRegime.CHOPPY and signal != EnsembleSignal.NEUTRAL:
            if base_confidence < 70:
                signal = EnsembleSignal.NEUTRAL
                reasoning.append("Signal filtered: choppy market requires higher confidence")

        # Combine metrics from both predictions
        combined_metrics = {}
        if market_pred:
            combined_metrics.update(market_pred.metrics)
        if tech_pred:
            combined_metrics["tech_signals"] = tech_pred.signals
            combined_metrics["market_regime"] = regime.value if regime else None
        combined_metrics["combined_score"] = combined_score
        combined_metrics["market_weight"] = market_weight
        combined_metrics["tech_weight"] = tech_weight

        # Get stop loss and take profit from tech prediction
        stop_loss = None
        take_profit = None
        if tech_pred and tech_pred.price_target:
            if signal in [EnsembleSignal.STRONG_BUY, EnsembleSignal.BUY]:
                take_profit = tech_pred.price_target
                # Calculate stop loss based on support level
                levels = combined_metrics.get("tech_signals", {}).get("levels") if tech_pred else None
                if levels:
                    stop_loss = levels.get("support")
            elif signal in [EnsembleSignal.STRONG_SELL, EnsembleSignal.SELL]:
                take_profit = tech_pred.price_target
                levels = combined_metrics.get("tech_signals", {}).get("levels") if tech_pred else None
                if levels:
                    stop_loss = levels.get("resistance")

        return EnsemblePrediction(
            market_question=market_question,
            condition_id=condition_id,
            token_id=token_id,
            current_price=current_price,
            signal=signal,
            confidence=base_confidence,
            agreement_level=agreement_pct,
            market_prediction=market_pred,
            tech_prediction=tech_pred,
            reasoning=reasoning,
            metrics=combined_metrics,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

    def predict_single(
        self,
        market: Dict,
        fetch_price_history: bool = True
    ) -> Optional[EnsemblePrediction]:
        """
        Generate ensemble prediction for a single market.

        Args:
            market: Market data dictionary
            fetch_price_history: Whether to fetch price history for technical analysis

        Returns:
            EnsemblePrediction or None if prediction fails
        """
        self._last_error = None

        try:
            # Get market prediction
            market_pred = self.market_predictor.analyze_market(market)

            # Get technical prediction (requires price history)
            tech_pred = None
            price_data = []

            if fetch_price_history:
                token_id = market_pred.token_id if market_pred else None
                if token_id:
                    try:
                        history = self.data_collector.get_price_history(
                            token_id, interval="1h", fidelity=48
                        )
                        if history and len(history) >= 20:
                            # Extract prices from history
                            price_data = []
                            for point in history:
                                if isinstance(point, dict):
                                    price = point.get("p") or point.get("price")
                                    if price is not None:
                                        price_data.append(float(price))
                                elif isinstance(point, (int, float)):
                                    price_data.append(float(point))

                            if len(price_data) >= 20:
                                tech_pred = self.tech_engine.analyze(price_data)
                    except Exception as e:
                        logger.warning(f"Failed to get price history for tech analysis: {e}")

            # Combine predictions
            return self._combine_predictions(market_pred, tech_pred, price_data)

        except Exception as e:
            self._last_error = f"Prediction error: {e}"
            logger.error(self._last_error, exc_info=True)
            return None

    def scan(
        self,
        limit: int = 30,
        min_volume: float = 10000,
        min_liquidity: float = 1000,
        fetch_price_history: bool = True
    ) -> List[EnsemblePrediction]:
        """
        Scan markets and generate ensemble predictions.

        Args:
            limit: Maximum number of markets to scan
            min_volume: Minimum 24h volume filter
            min_liquidity: Minimum liquidity filter
            fetch_price_history: Whether to fetch price history for tech analysis

        Returns:
            List of EnsemblePrediction objects
        """
        logger.info(f"Ensemble scan starting: limit={limit}, min_volume={min_volume}")

        # Fetch markets
        markets = self.market_predictor.fetch_markets(
            limit=limit,
            min_volume=min_volume,
            min_liquidity=min_liquidity
        )

        predictions = []
        for market in markets:
            pred = self.predict_single(market, fetch_price_history=fetch_price_history)
            if pred:
                predictions.append(pred)

        logger.info(f"Ensemble scan complete: {len(predictions)} predictions generated")
        return predictions

    def format_prediction(self, pred: EnsemblePrediction) -> str:
        """Format ensemble prediction for display."""
        emoji = {
            EnsembleSignal.STRONG_BUY: "[++]",
            EnsembleSignal.BUY: "[+]",
            EnsembleSignal.NEUTRAL: "[=]",
            EnsembleSignal.SELL: "[-]",
            EnsembleSignal.STRONG_SELL: "[--]"
        }

        lines = [
            f"Market: {pred.market_question[:60]}",
            f"Ensemble Signal: {emoji.get(pred.signal, '')} {pred.signal.value}",
            f"Confidence: {pred.confidence:.0f}% | Agreement: {pred.agreement_level:.0f}%",
            f"Price: {pred.current_price:.1%}",
        ]

        if pred.stop_loss:
            lines.append(f"Stop Loss: {pred.stop_loss:.1%}")
        if pred.take_profit:
            lines.append(f"Take Profit: {pred.take_profit:.1%}")

        if pred.reasoning:
            lines.append("Reasoning:")
            for r in pred.reasoning:
                lines.append(f"  - {r}")

        return "\n".join(lines)


def main():
    """Test ensemble predictor with a scan."""
    from datetime import datetime

    print("=" * 70)
    print("ENSEMBLE PREDICTOR - Combining Market Metrics + Technical Analysis")
    print(f"Scan Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

    # Create ensemble predictor with default config
    ensemble = EnsemblePredictor()

    print("Scanning markets with ensemble approach...")
    print("(Requires both market metrics AND technical indicators to agree)")
    print()

    # Scan markets
    predictions = ensemble.scan(
        limit=20,
        min_volume=5000,
        min_liquidity=500,
        fetch_price_history=True
    )

    # Separate by signal
    strong_buys = [p for p in predictions if p.signal == EnsembleSignal.STRONG_BUY]
    buys = [p for p in predictions if p.signal == EnsembleSignal.BUY]
    sells = [p for p in predictions if p.signal == EnsembleSignal.SELL]
    strong_sells = [p for p in predictions if p.signal == EnsembleSignal.STRONG_SELL]
    neutrals = [p for p in predictions if p.signal == EnsembleSignal.NEUTRAL]

    print(f"Found {len(predictions)} markets")
    print(f"  Strong Buys: {len(strong_buys)}")
    print(f"  Buys: {len(buys)}")
    print(f"  Neutral: {len(neutrals)}")
    print(f"  Sells: {len(sells)}")
    print(f"  Strong Sells: {len(strong_sells)}")

    # Calculate agreement stats
    high_agreement = [p for p in predictions if p.agreement_level >= 75]
    print(f"\nHigh Agreement Signals (>=75%): {len(high_agreement)}")

    # Print strong signals with high agreement
    actionable = [p for p in predictions if p.signal != EnsembleSignal.NEUTRAL and p.agreement_level >= 60]

    if actionable:
        print("\n" + "=" * 70)
        print("ACTIONABLE SIGNALS (Non-neutral with >=60% agreement)")
        print("=" * 70)
        for pred in sorted(actionable, key=lambda x: (-x.confidence, -x.agreement_level))[:10]:
            print()
            print(ensemble.format_prediction(pred))
            print("-" * 50)
    else:
        print("\nNo actionable signals found (this is normal - ensemble is selective)")

    # Show filtering stats
    filtered = len([p for p in predictions if p.signal == EnsembleSignal.NEUTRAL])
    print(f"\n{filtered}/{len(predictions)} signals filtered (neutral) - ensemble reduces false positives")

    print("\n" + "=" * 70)
    print("ENSEMBLE SCAN COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
