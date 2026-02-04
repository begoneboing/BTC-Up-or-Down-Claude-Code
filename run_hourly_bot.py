#!/usr/bin/env python
"""
BTC Hourly Up/Down Trading Bot

Improvements over 15-minute bot:
1. Hourly timeframe - more time for trends to develop
2. Uses 15-minute candles for better prediction accuracy
3. Contrarian strategy - bet against heavily skewed markets
4. Kelly criterion position sizing based on edge
5. Stronger confidence requirements
"""
import os
import sys
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
os.environ['TRADING_MODE'] = 'live'

import requests
import json
import time
from datetime import datetime, timezone
from trade_executor import TradeExecutor, OrderSide
from prediction_engine import PredictionEngine, Direction as PredDirection

# Configuration
TARGET_TRADES = 3
BASE_POSITION_SIZE = 5.0  # Base $5 per trade
MAX_POSITION_SIZE = 15.0  # Max with Kelly sizing
MIN_CONFIDENCE = 60  # Higher threshold for hourly
CONTRARIAN_THRESHOLD = 0.70  # Bet against if one side > 70%
BINANCE_URL = "https://api.binance.com/api/v3"
GAMMA_URL = "https://gamma-api.polymarket.com"


def fetch_btc_prices(interval="15m", limit=48):
    """Fetch BTC price history - use 15m candles for hourly trading"""
    try:
        response = requests.get(
            f"{BINANCE_URL}/klines",
            params={"symbol": "BTCUSDT", "interval": interval, "limit": limit},
            timeout=30
        )
        if response.status_code == 200:
            klines = response.json()
            return [float(k[4]) for k in klines]
    except Exception as e:
        print(f"Error fetching BTC prices: {e}", flush=True)
    return []


def get_prediction():
    """Get BTC direction prediction using 15-minute candles"""
    # Use 15-minute candles for better hourly prediction
    prices = fetch_btc_prices(interval="15m", limit=48)  # 12 hours of data

    if len(prices) < 20:
        print("Insufficient price data", flush=True)
        return None, 0, []

    engine = PredictionEngine()
    prediction = engine.analyze(prices)

    if not prediction:
        return None, 0, ["No prediction"]

    if prediction.direction in [PredDirection.STRONG_UP, PredDirection.UP]:
        direction = "UP"
    elif prediction.direction in [PredDirection.STRONG_DOWN, PredDirection.DOWN]:
        direction = "DOWN"
    else:
        direction = None

    return direction, prediction.confidence, prediction.reasoning[:4]


def calculate_kelly_size(edge: float, odds: float, base_size: float) -> float:
    """
    Calculate position size using Kelly Criterion

    Kelly % = (bp - q) / b
    where b = odds, p = win probability, q = 1 - p

    We use fractional Kelly (25%) for safety
    """
    if edge <= 0 or odds <= 0:
        return base_size

    win_prob = 0.5 + edge  # Our estimated win probability
    lose_prob = 1 - win_prob

    # For binary markets, odds are based on price
    # If we buy at 0.40, we win 0.60 if correct (odds = 0.60/0.40 = 1.5)
    kelly_pct = (odds * win_prob - lose_prob) / odds

    # Use 25% Kelly for safety
    kelly_pct = max(0, kelly_pct * 0.25)

    # Scale position size
    size = base_size * (1 + kelly_pct * 2)  # 1x to 3x base size
    return min(size, MAX_POSITION_SIZE)


def find_hourly_markets():
    """Find open hourly BTC Up/Down markets"""
    try:
        response = requests.get(
            f"{GAMMA_URL}/markets",
            params={'active': 'true', 'limit': 500, 'order': 'volume24hr', 'ascending': 'false'},
            timeout=60
        )
        if response.status_code != 200:
            return []

        markets = response.json()
        hourly = []

        for m in markets:
            question = m.get('question', '')
            # Hourly format: "Bitcoin Up or Down - February 4, 2PM ET"
            if 'Bitcoin Up or Down' in question:
                if ('AM ET' in question or 'PM ET' in question) and 'AM-' not in question and 'PM-' not in question:
                    accepting = m.get('acceptingOrders', False)
                    closed = m.get('closed', False)

                    prices = json.loads(m.get('outcomePrices', '[]'))
                    resolved = set(str(p) for p in prices) == {'0', '1'}

                    if accepting and not closed and not resolved:
                        tokens = json.loads(m.get('clobTokenIds', '[]'))
                        outcomes = json.loads(m.get('outcomes', '[]'))

                        up_idx = 0 if outcomes[0].lower() == 'up' else 1
                        down_idx = 1 - up_idx

                        hourly.append({
                            'question': question,
                            'slug': m.get('slug'),
                            'up_token': tokens[up_idx],
                            'down_token': tokens[down_idx],
                            'up_price': float(prices[up_idx]),
                            'down_price': float(prices[down_idx]),
                            'volume': float(m.get('volume24hr', 0) or 0)
                        })

        # Sort by volume
        hourly.sort(key=lambda x: x['volume'], reverse=True)
        return hourly

    except Exception as e:
        print(f"Error finding markets: {e}", flush=True)
        return []


def determine_trade(market: dict, pred_direction: str, confidence: float) -> dict:
    """
    Determine trade direction using multiple signals:
    1. Technical prediction
    2. Contrarian strategy (bet against skewed markets)
    3. Value betting (buy underpriced side)
    """
    up_price = market['up_price']
    down_price = market['down_price']

    signals = []
    direction = None
    edge = 0

    # Signal 1: Contrarian - bet against heavily skewed markets
    if up_price >= CONTRARIAN_THRESHOLD:
        signals.append(f"CONTRARIAN: UP at {up_price:.0%} is overpriced, bet DOWN")
        direction = "DOWN"
        edge = (up_price - 0.5) * 0.5  # Edge based on mispricing
    elif down_price >= CONTRARIAN_THRESHOLD:
        signals.append(f"CONTRARIAN: DOWN at {down_price:.0%} is overpriced, bet UP")
        direction = "UP"
        edge = (down_price - 0.5) * 0.5

    # Signal 2: Technical prediction (if confident)
    if pred_direction and confidence >= MIN_CONFIDENCE:
        pred_edge = (confidence - 50) / 100

        if direction is None:
            direction = pred_direction
            edge = pred_edge
            signals.append(f"PREDICTION: {pred_direction} with {confidence:.0f}% confidence")
        elif direction == pred_direction:
            # Contrarian and prediction agree - stronger signal
            edge = edge + pred_edge
            signals.append(f"CONFIRMED: Prediction agrees ({confidence:.0f}% conf)")
        else:
            # Conflict - reduce edge but keep contrarian
            edge = edge * 0.5
            signals.append(f"CONFLICT: Prediction says {pred_direction}, keeping contrarian")

    # Signal 3: Value bet (buy cheaper side if no other signal)
    if direction is None:
        if up_price < 0.45:
            direction = "UP"
            edge = (0.5 - up_price) * 0.3
            signals.append(f"VALUE: UP underpriced at {up_price:.0%}")
        elif down_price < 0.45:
            direction = "DOWN"
            edge = (0.5 - down_price) * 0.3
            signals.append(f"VALUE: DOWN underpriced at {down_price:.0%}")
        else:
            # No clear edge - use prediction or skip
            if pred_direction:
                direction = pred_direction
                edge = 0.02
                signals.append(f"WEAK: Using prediction {pred_direction} (low edge)")
            else:
                direction = "DOWN"  # Default slight bearish
                edge = 0.01
                signals.append("DEFAULT: No clear signal, slight bearish bias")

    return {
        'direction': direction,
        'edge': min(edge, 0.30),  # Cap edge at 30%
        'signals': signals
    }


def main():
    print('='*60, flush=True)
    print('BTC HOURLY UP/DOWN BOT', flush=True)
    print('='*60, flush=True)
    print(f'Target trades: {TARGET_TRADES}', flush=True)
    print(f'Min confidence: {MIN_CONFIDENCE}%', flush=True)
    print(f'Contrarian threshold: {CONTRARIAN_THRESHOLD:.0%}', flush=True)
    print(f'Base size: ${BASE_POSITION_SIZE}, Max: ${MAX_POSITION_SIZE}', flush=True)
    print('='*60, flush=True)

    executor = TradeExecutor()
    trades_completed = 0
    results = []
    markets_traded = set()

    while trades_completed < TARGET_TRADES:
        print(f'\n{"="*50}', flush=True)
        print(f'Trade {trades_completed + 1} of {TARGET_TRADES}', flush=True)
        print(f'{"="*50}', flush=True)

        now = datetime.now(timezone.utc)
        print(f'UTC: {now.strftime("%H:%M:%S")}', flush=True)

        # Get prediction
        print('\nAnalyzing BTC (15m candles)...', flush=True)
        pred_direction, confidence, reasons = get_prediction()

        print(f'Prediction: {pred_direction or "NEUTRAL"}', flush=True)
        print(f'Confidence: {confidence:.0f}%', flush=True)
        for r in reasons[:3]:
            print(f'  - {r}', flush=True)

        # Find hourly markets
        markets = find_hourly_markets()
        print(f'\nFound {len(markets)} hourly markets', flush=True)

        if not markets:
            print('No markets available, waiting 5 min...', flush=True)
            time.sleep(300)
            continue

        # Find untrade market
        market = None
        for m in markets:
            if m['slug'] not in markets_traded:
                market = m
                break

        if not market:
            print('All available markets traded, waiting 30 min...', flush=True)
            time.sleep(1800)
            continue

        print(f'\nMarket: {market["question"]}', flush=True)
        print(f'UP: {market["up_price"]:.1%} | DOWN: {market["down_price"]:.1%}', flush=True)

        # Determine trade
        trade_decision = determine_trade(market, pred_direction, confidence)
        direction = trade_decision['direction']
        edge = trade_decision['edge']

        print(f'\nTrade Analysis:', flush=True)
        for sig in trade_decision['signals']:
            print(f'  {sig}', flush=True)
        print(f'Direction: {direction}, Edge: {edge:.1%}', flush=True)

        # Calculate position size with Kelly
        if direction == "UP":
            token = market["up_token"]
            price = market["up_price"]
            odds = (1 - price) / price if price > 0 else 1
        else:
            token = market["down_token"]
            price = market["down_price"]
            odds = (1 - price) / price if price > 0 else 1

        position_size = calculate_kelly_size(edge, odds, BASE_POSITION_SIZE)
        size = round(position_size / price, 2)

        print(f'\nPosition: ${position_size:.2f} ({size} shares @ {price:.1%})', flush=True)

        # Execute trade
        try:
            order = executor.place_order(
                token_id=token,
                side=OrderSide.BUY,
                price=min(price + 0.02, 0.99),
                size=size,
                market_question=f'BTC Hourly {direction}'
            )

            if order:
                print(f'\nORDER {order.status.value}: {order.order_id[:30]}...', flush=True)
                trades_completed += 1
                markets_traded.add(market['slug'])
                results.append({
                    'market': market['question'],
                    'direction': direction,
                    'price': price,
                    'size': size,
                    'position': position_size,
                    'edge': edge,
                    'signals': trade_decision['signals']
                })
            else:
                print('Order failed!', flush=True)
        except Exception as e:
            print(f'Trade error: {e}', flush=True)

        # Wait before next trade (hourly markets, so wait 15-30 min)
        if trades_completed < TARGET_TRADES:
            wait_time = 900  # 15 minutes between trades
            print(f'\nWaiting {wait_time//60} minutes...', flush=True)
            time.sleep(wait_time)

    # Summary
    print(f'\n{"="*60}', flush=True)
    print('HOURLY TRADING COMPLETE', flush=True)
    print(f'{"="*60}', flush=True)
    print(f'Trades: {trades_completed}', flush=True)
    print(f'Total invested: ${sum(r["position"] for r in results):.2f}', flush=True)

    print('\nPositions:', flush=True)
    for r in results:
        print(f'\n  {r["direction"]} @ {r["price"]:.1%} (${r["position"]:.2f})', flush=True)
        print(f'  Edge: {r["edge"]:.1%}', flush=True)
        print(f'  Market: {r["market"]}', flush=True)
        for sig in r['signals']:
            print(f'    - {sig}', flush=True)


if __name__ == '__main__':
    main()
