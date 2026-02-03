"""
Up or Down - Polymarket Price Prediction System
Main entry point for scanning markets and generating predictions
"""

import os
import sys
import time
import argparse
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from data_collector import DataCollector
from indicators import TechnicalIndicators, calculate_all_indicators_optimized
from prediction_engine import PredictionEngine, Direction
from signal_generator import SignalGenerator, AlertManager, SignalType


def scan_markets(limit: int = 20, min_volume: float = 5000, min_liquidity: float = 1000):
    """
    Scan Polymarket for trading opportunities
    """
    print("=" * 70)
    print("UP OR DOWN - Polymarket Price Prediction")
    print("=" * 70)
    print(f"Scan Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Parameters: limit={limit}, min_volume=${min_volume:,.0f}, min_liquidity=${min_liquidity:,.0f}")
    print("=" * 70)
    print()

    collector = DataCollector()
    generator = SignalGenerator()
    alerts = AlertManager()

    # Fetch markets
    print("Fetching active markets...")
    markets = collector.get_active_markets(limit=limit * 3)

    # Filter by volume/liquidity
    filtered_markets = []
    for market in markets:
        vol = float(market.get("volume", 0) or 0)
        liq = float(market.get("liquidity", 0) or 0)
        tokens = market.get("tokens", [])

        if vol >= min_volume and liq >= min_liquidity and tokens:
            filtered_markets.append(market)
            if len(filtered_markets) >= limit:
                break

    print(f"Found {len(filtered_markets)} markets meeting criteria\n")

    results = []

    for i, market in enumerate(filtered_markets):
        question = market.get("question", "Unknown")[:60]
        condition_id = market.get("conditionId", "")
        volume = float(market.get("volume", 0) or 0)
        liquidity = float(market.get("liquidity", 0) or 0)

        print(f"[{i+1}/{len(filtered_markets)}] Analyzing: {question}...")

        tokens = market.get("tokens", [])
        if not tokens:
            continue

        # Analyze first outcome (usually "Yes")
        token = tokens[0]
        token_id = token.get("token_id", "")
        outcome = token.get("outcome", "Yes")

        if not token_id:
            continue

        # Get price history
        history = collector.get_price_history(token_id, interval="1h", fidelity=48)

        if not history or len(history) < 20:
            print(f"  -> Insufficient price history ({len(history) if history else 0} points)")
            continue

        # Extract prices
        prices = TechnicalIndicators.extract_prices(history)

        if len(prices) < 20:
            continue

        # Generate signal
        signal = generator.generate_signal(prices, volume=volume, liquidity=liquidity)

        if signal:
            result = {
                "market": question,
                "condition_id": condition_id,
                "outcome": outcome,
                "signal": signal,
                "volume": volume,
                "liquidity": liquidity,
                "current_price": prices[-1]
            }
            results.append(result)

            # Add to alert manager
            alerts.add_alert(signal, question, condition_id)

            # Print result
            signal_str = signal.signal_type.value
            strength = signal.strength

            if signal.signal_type in [SignalType.STRONG_BUY, SignalType.BUY]:
                print(f"  -> 🟢 {signal_str} ({strength:.0f}%) @ {prices[-1]:.2%}")
            elif signal.signal_type in [SignalType.STRONG_SELL, SignalType.SELL]:
                print(f"  -> 🔴 {signal_str} ({strength:.0f}%) @ {prices[-1]:.2%}")
            else:
                print(f"  -> ⚪ HOLD @ {prices[-1]:.2%}")

        time.sleep(0.15)  # Rate limiting

    # Print summary
    alerts.print_summary()

    # Print detailed top signals
    print_top_signals(results, generator)

    return results


def print_top_signals(results, generator):
    """Print detailed analysis of top signals"""
    # Filter for actionable signals
    actionable = [r for r in results if r["signal"].signal_type != SignalType.HOLD]

    if not actionable:
        print("\nNo strong signals found in this scan.")
        return

    # Sort by strength
    actionable.sort(key=lambda x: x["signal"].strength, reverse=True)

    # Top buys
    buys = [r for r in actionable if "BUY" in r["signal"].signal_type.value]
    if buys:
        print(f"\n{'='*70}")
        print("DETAILED BUY SIGNALS")
        print(f"{'='*70}")
        for r in buys[:3]:
            print(f"\n{'-'*70}")
            print(generator.format_signal(r["signal"], r["market"]))
            print(f"Volume: ${r['volume']:,.0f} | Liquidity: ${r['liquidity']:,.0f}")

    # Top sells
    sells = [r for r in actionable if "SELL" in r["signal"].signal_type.value]
    if sells:
        print(f"\n{'='*70}")
        print("DETAILED SELL SIGNALS")
        print(f"{'='*70}")
        for r in sells[:3]:
            print(f"\n{'-'*70}")
            print(generator.format_signal(r["signal"], r["market"]))
            print(f"Volume: ${r['volume']:,.0f} | Liquidity: ${r['liquidity']:,.0f}")


def analyze_single_market(market_id: str):
    """
    Analyze a specific market by ID
    """
    print(f"Analyzing market: {market_id}")

    collector = DataCollector()
    generator = SignalGenerator()
    engine = PredictionEngine()

    # Get market details
    try:
        market = collector.get_market_details(market_id)
    except Exception as e:
        print(f"Error fetching market: {e}")
        return None

    question = market.get("question", "Unknown")
    tokens = market.get("tokens", [])

    print(f"\nMarket: {question}")
    print("=" * 70)

    if not tokens:
        print("No tokens found for this market")
        return None

    for token in tokens:
        token_id = token.get("token_id", "")
        outcome = token.get("outcome", "")

        if not token_id:
            continue

        print(f"\nOutcome: {outcome}")
        print("-" * 50)

        # Get price history
        history = collector.get_price_history(token_id, interval="1h", fidelity=72)
        prices = TechnicalIndicators.extract_prices(history)

        if len(prices) < 20:
            print("Insufficient price history")
            continue

        # Calculate indicators (using optimized batch processing)
        indicators = calculate_all_indicators_optimized(prices)

        print(f"Current Price: {prices[-1]:.2%}")
        print(f"24h High: {max(prices[-24:]):.2%}" if len(prices) >= 24 else "")
        print(f"24h Low: {min(prices[-24:]):.2%}" if len(prices) >= 24 else "")

        if indicators["rsi_14"]:
            print(f"RSI (14): {indicators['rsi_14']:.1f}")

        if indicators["trend"]:
            t = indicators["trend"]
            print(f"Trend: {t['direction'].upper()} (strength: {t['strength']:.0f})")

        # Generate signal
        signal = generator.generate_signal(prices)
        if signal:
            print(f"\n{generator.format_signal(signal)}")

    return market


def watch_mode(interval_minutes: int = 5, limit: int = 15):
    """
    Continuous monitoring mode
    """
    print("Starting watch mode...")
    print(f"Scan interval: {interval_minutes} minutes")
    print("Press Ctrl+C to stop\n")

    while True:
        try:
            scan_markets(limit=limit)
            print(f"\nNext scan in {interval_minutes} minutes...")
            time.sleep(interval_minutes * 60)
        except KeyboardInterrupt:
            print("\nWatch mode stopped.")
            break
        except Exception as e:
            print(f"Error during scan: {e}")
            time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description="Polymarket Up/Down Prediction System")
    parser.add_argument("--scan", action="store_true", help="Scan markets for signals")
    parser.add_argument("--market", type=str, help="Analyze specific market by ID")
    parser.add_argument("--watch", action="store_true", help="Continuous monitoring mode")
    parser.add_argument("--limit", type=int, default=20, help="Number of markets to scan")
    parser.add_argument("--min-volume", type=float, default=5000, help="Minimum volume filter")
    parser.add_argument("--min-liquidity", type=float, default=1000, help="Minimum liquidity filter")
    parser.add_argument("--interval", type=int, default=5, help="Watch mode interval in minutes")

    args = parser.parse_args()

    if args.market:
        analyze_single_market(args.market)
    elif args.watch:
        watch_mode(interval_minutes=args.interval, limit=args.limit)
    else:
        # Default: scan markets
        scan_markets(
            limit=args.limit,
            min_volume=args.min_volume,
            min_liquidity=args.min_liquidity
        )


if __name__ == "__main__":
    main()
