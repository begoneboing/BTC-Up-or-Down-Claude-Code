"""
Up or Down Predictor for Polymarket
Uses available market metrics to predict price direction
"""

import os
import requests
import json
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

GAMMA_API_URL = "https://gamma-api.polymarket.com"


class Signal(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class Prediction:
    market_question: str
    condition_id: str
    token_id: str
    current_price: float
    signal: Signal
    confidence: float
    reasons: List[str]
    metrics: Dict


class PolymarketPredictor:
    """
    Predicts price direction using available market metrics:
    - Price changes (1h, 1d, 1w, 1m)
    - Volume trends
    - Spread analysis
    - Momentum indicators
    """

    def __init__(self):
        pass

    def fetch_markets(self, limit=100, min_volume=10000, min_liquidity=1000):
        """Fetch active markets with sufficient volume"""
        url = f"{GAMMA_API_URL}/markets"
        params = {
            "limit": limit * 2,
            "active": "true",
            "closed": "false"
        }

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        markets = response.json()

        # Filter by volume and liquidity
        filtered = []
        for m in markets:
            vol = float(m.get("volume", 0) or 0)
            liq = float(m.get("liquidity", 0) or 0)
            if vol >= min_volume and liq >= min_liquidity:
                filtered.append(m)
                if len(filtered) >= limit:
                    break

        return filtered

    def analyze_market(self, market: Dict) -> Optional[Prediction]:
        """Analyze a single market and generate prediction"""
        question = market.get("question", "Unknown")
        condition_id = market.get("conditionId", "")

        # Get token ID from clobTokenIds
        clob_tokens = market.get("clobTokenIds", "[]")
        if isinstance(clob_tokens, str):
            try:
                clob_tokens = json.loads(clob_tokens)
            except:
                clob_tokens = []
        token_id = clob_tokens[0] if clob_tokens else ""

        # Get current prices
        outcome_prices = market.get("outcomePrices", "[]")
        if isinstance(outcome_prices, str):
            try:
                outcome_prices = json.loads(outcome_prices)
            except:
                outcome_prices = []

        if not outcome_prices:
            return None

        current_price = float(outcome_prices[0])  # Yes price

        # Get price changes
        hour_change = float(market.get("oneHourPriceChange", 0) or 0)
        day_change = float(market.get("oneDayPriceChange", 0) or 0)
        week_change = float(market.get("oneWeekPriceChange", 0) or 0)
        month_change = float(market.get("oneMonthPriceChange", 0) or 0)

        # Get volume metrics
        volume_24h = float(market.get("volume24hr", 0) or 0)
        volume_1w = float(market.get("volume1wk", 0) or 0)
        volume_1m = float(market.get("volume1mo", 0) or 0)
        total_volume = float(market.get("volume", 0) or 0)

        # Get spread
        best_bid = float(market.get("bestBid", 0) or 0)
        best_ask = float(market.get("bestAsk", 1) or 1)
        spread = best_ask - best_bid

        # Get liquidity
        liquidity = float(market.get("liquidity", 0) or 0)

        # Calculate signals
        bullish_score = 0
        bearish_score = 0
        reasons = []

        # 1. Short-term momentum (1h change) - Weight: 15%
        if hour_change > 0.02:
            bullish_score += 15
            reasons.append(f"Strong 1h momentum: +{hour_change:.1%}")
        elif hour_change > 0.005:
            bullish_score += 8
        elif hour_change < -0.02:
            bearish_score += 15
            reasons.append(f"Weak 1h momentum: {hour_change:.1%}")
        elif hour_change < -0.005:
            bearish_score += 8

        # 2. Daily trend (1d change) - Weight: 25%
        if day_change > 0.05:
            bullish_score += 25
            reasons.append(f"Strong daily uptrend: +{day_change:.1%}")
        elif day_change > 0.02:
            bullish_score += 15
            reasons.append(f"Daily uptrend: +{day_change:.1%}")
        elif day_change > 0:
            bullish_score += 5
        elif day_change < -0.05:
            bearish_score += 25
            reasons.append(f"Strong daily downtrend: {day_change:.1%}")
        elif day_change < -0.02:
            bearish_score += 15
            reasons.append(f"Daily downtrend: {day_change:.1%}")
        elif day_change < 0:
            bearish_score += 5

        # 3. Weekly trend - Weight: 20%
        if week_change > 0.10:
            bullish_score += 20
            reasons.append(f"Strong weekly uptrend: +{week_change:.1%}")
        elif week_change > 0.03:
            bullish_score += 12
        elif week_change < -0.10:
            bearish_score += 20
            reasons.append(f"Strong weekly downtrend: {week_change:.1%}")
        elif week_change < -0.03:
            bearish_score += 12

        # 4. Monthly trend - Weight: 15%
        if month_change > 0.15:
            bullish_score += 15
            reasons.append(f"Strong monthly trend: +{month_change:.1%}")
        elif month_change > 0.05:
            bullish_score += 8
        elif month_change < -0.15:
            bearish_score += 15
            reasons.append(f"Weak monthly trend: {month_change:.1%}")
        elif month_change < -0.05:
            bearish_score += 8

        # 5. Volume trend analysis - Weight: 15%
        if volume_1w > 0:
            daily_avg_1w = volume_1w / 7
            if volume_24h > daily_avg_1w * 1.5:
                # Volume spike - confirms trend
                if day_change > 0:
                    bullish_score += 15
                    reasons.append(f"Volume spike confirms uptrend: ${volume_24h:,.0f}")
                elif day_change < 0:
                    bearish_score += 15
                    reasons.append(f"Volume spike confirms downtrend: ${volume_24h:,.0f}")
            elif volume_24h > daily_avg_1w:
                if day_change > 0:
                    bullish_score += 8
                elif day_change < 0:
                    bearish_score += 8

        # 6. Price position analysis - Weight: 10%
        if current_price < 0.20:
            # Low price - potential upside
            if day_change > 0:
                bullish_score += 10
                reasons.append(f"Low price with upward momentum")
        elif current_price > 0.80:
            # High price - potential downside
            if day_change < 0:
                bearish_score += 10
                reasons.append(f"High price with downward momentum")

        # Calculate net score
        net_score = bullish_score - bearish_score
        total_weight = max(bullish_score + bearish_score, 1)
        confidence = min(abs(net_score) / total_weight * 100, 100)

        # Determine signal
        if net_score >= 35:
            signal = Signal.STRONG_BUY
        elif net_score >= 15:
            signal = Signal.BUY
        elif net_score <= -35:
            signal = Signal.STRONG_SELL
        elif net_score <= -15:
            signal = Signal.SELL
        else:
            signal = Signal.NEUTRAL

        # Calculate implied volatility from price changes
        # Uses variance of available price changes as a proxy for volatility
        price_changes = [
            abs(hour_change) * 24,  # Annualize hourly (rough estimate)
            abs(day_change) * 365 ** 0.5,  # Annualize daily
            abs(week_change) * (52 ** 0.5),  # Annualize weekly
        ]
        # Filter out zero changes and calculate mean volatility
        valid_changes = [c for c in price_changes if c > 0]
        if valid_changes:
            implied_volatility = sum(valid_changes) / len(valid_changes)
        else:
            implied_volatility = 0.5  # Default to 50% if no data

        # Normalize to a 0-1 scale where 0 = low vol, 1 = extreme vol
        # Most prediction market volatility ranges from 20% to 200% annualized
        volatility_normalized = min(max(implied_volatility / 2.0, 0.1), 1.0)

        metrics = {
            "hour_change": hour_change,
            "day_change": day_change,
            "week_change": week_change,
            "month_change": month_change,
            "volume_24h": volume_24h,
            "spread": spread,
            "liquidity": liquidity,
            "bullish_score": bullish_score,
            "bearish_score": bearish_score,
            "implied_volatility": implied_volatility,
            "volatility_normalized": volatility_normalized
        }

        return Prediction(
            market_question=question,
            condition_id=condition_id,
            token_id=token_id,
            current_price=current_price,
            signal=signal,
            confidence=confidence,
            reasons=reasons,
            metrics=metrics
        )

    def scan(self, limit=30, min_volume=10000, min_liquidity=1000):
        """Scan markets and return predictions"""
        markets = self.fetch_markets(limit, min_volume, min_liquidity)

        predictions = []
        for market in markets:
            pred = self.analyze_market(market)
            if pred:
                predictions.append(pred)

        return predictions

    def format_prediction(self, pred: Prediction) -> str:
        """Format prediction for display"""
        emoji = {
            Signal.STRONG_BUY: "[++]",
            Signal.BUY: "[+]",
            Signal.NEUTRAL: "[=]",
            Signal.SELL: "[-]",
            Signal.STRONG_SELL: "[--]"
        }

        lines = [
            f"Market: {pred.market_question[:60]}",
            f"Signal: {emoji.get(pred.signal, '')} {pred.signal.value} ({pred.confidence:.0f}%)",
            f"Price: {pred.current_price:.1%}",
            f"Changes: 1h={pred.metrics['hour_change']:+.1%} | 1d={pred.metrics['day_change']:+.1%} | 1w={pred.metrics['week_change']:+.1%}",
            f"Volume 24h: ${pred.metrics['volume_24h']:,.0f}",
        ]

        if pred.reasons:
            lines.append("Reasons:")
            for r in pred.reasons:
                lines.append(f"  - {r}")

        return "\n".join(lines)


def main():
    print("=" * 70)
    print("UP OR DOWN - Polymarket Price Predictor")
    print(f"Scan Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

    predictor = PolymarketPredictor()

    print("Scanning markets...")
    predictions = predictor.scan(limit=30, min_volume=5000, min_liquidity=500)

    # Separate by signal
    strong_buys = [p for p in predictions if p.signal == Signal.STRONG_BUY]
    buys = [p for p in predictions if p.signal == Signal.BUY]
    sells = [p for p in predictions if p.signal == Signal.SELL]
    strong_sells = [p for p in predictions if p.signal == Signal.STRONG_SELL]

    print(f"\nFound {len(predictions)} markets")
    print(f"  Strong Buys: {len(strong_buys)}")
    print(f"  Buys: {len(buys)}")
    print(f"  Sells: {len(sells)}")
    print(f"  Strong Sells: {len(strong_sells)}")

    # Print strong signals
    if strong_buys:
        print("\n" + "=" * 70)
        print("STRONG BUY SIGNALS")
        print("=" * 70)
        for pred in sorted(strong_buys, key=lambda x: -x.confidence)[:5]:
            print()
            print(predictor.format_prediction(pred))

    if buys:
        print("\n" + "=" * 70)
        print("BUY SIGNALS")
        print("=" * 70)
        for pred in sorted(buys, key=lambda x: -x.confidence)[:5]:
            print()
            print(predictor.format_prediction(pred))

    if strong_sells:
        print("\n" + "=" * 70)
        print("STRONG SELL SIGNALS")
        print("=" * 70)
        for pred in sorted(strong_sells, key=lambda x: -x.confidence)[:5]:
            print()
            print(predictor.format_prediction(pred))

    if sells:
        print("\n" + "=" * 70)
        print("SELL SIGNALS")
        print("=" * 70)
        for pred in sorted(sells, key=lambda x: -x.confidence)[:5]:
            print()
            print(predictor.format_prediction(pred))

    print("\n" + "=" * 70)
    print("SCAN COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
