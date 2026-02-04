#!/usr/bin/env python
"""
BTC 15-minute bot with improved prediction logic
Uses technical analysis (RSI, MACD, momentum) to determine direction
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
TARGET_TRADES = 4
POSITION_SIZE = 5.0  # $5 per trade
MIN_CONFIDENCE = 55  # Minimum prediction confidence to trade
BINANCE_URL = "https://api.binance.com/api/v3"

def fetch_btc_prices(interval="1m", limit=60):
    """Fetch BTC price history from Binance"""
    try:
        response = requests.get(
            f"{BINANCE_URL}/klines",
            params={"symbol": "BTCUSDT", "interval": interval, "limit": limit},
            timeout=30
        )
        if response.status_code == 200:
            klines = response.json()
            return [float(k[4]) for k in klines]  # Closing prices
    except Exception as e:
        print(f"Error fetching BTC prices: {e}", flush=True)
    return []

def get_prediction():
    """Get BTC direction prediction using technical analysis"""
    prices = fetch_btc_prices(interval="1m", limit=60)

    if len(prices) < 20:
        return None, 0, []

    engine = PredictionEngine()
    prediction = engine.analyze(prices)

    if not prediction:
        return None, 0, ["No prediction"]

    # Map to UP/DOWN
    if prediction.direction in [PredDirection.STRONG_UP, PredDirection.UP]:
        direction = "UP"
    elif prediction.direction in [PredDirection.STRONG_DOWN, PredDirection.DOWN]:
        direction = "DOWN"
    else:
        direction = None

    return direction, prediction.confidence, prediction.reasoning[:3]

def find_open_market():
    """Find the current open BTC 15-minute market"""
    now = datetime.now(timezone.utc)
    minute = (now.minute // 15) * 15
    slot_time = now.replace(minute=minute, second=0, microsecond=0)
    current_ts = int(slot_time.timestamp())

    for offset in range(0, 6):
        ts = current_ts + (offset * 15 * 60)
        slug = f'btc-updown-15m-{ts}'

        try:
            response = requests.get(
                f'https://gamma-api.polymarket.com/events/slug/{slug}',
                timeout=10
            )
            if response.status_code != 200:
                continue

            event = response.json()
            if event.get('closed') or not event.get('markets'):
                continue

            market = event['markets'][0]
            if not market.get('acceptingOrders'):
                continue

            prices = json.loads(market.get('outcomePrices', '[]'))
            if set(str(p) for p in prices) == {'0', '1'}:
                continue

            tokens = json.loads(market.get('clobTokenIds', '[]'))
            outcomes = json.loads(market.get('outcomes', '[]'))

            up_idx = 0 if outcomes[0].lower() == 'up' else 1
            down_idx = 1 - up_idx

            return {
                'title': event.get('title', ''),
                'slug': slug,
                'up_token': tokens[up_idx],
                'down_token': tokens[down_idx],
                'up_price': float(prices[up_idx]),
                'down_price': float(prices[down_idx])
            }
        except:
            continue

    return None

def main():
    print('='*60, flush=True)
    print('BTC 15-MINUTE BOT - SMART TRADING', flush=True)
    print('Using Technical Analysis for Direction', flush=True)
    print(f'Target: {TARGET_TRADES} trades', flush=True)
    print(f'Min Confidence: {MIN_CONFIDENCE}%', flush=True)
    print('='*60, flush=True)

    executor = TradeExecutor()
    trades_completed = 0
    results = []

    while trades_completed < TARGET_TRADES:
        print(f'\n{"="*50}', flush=True)
        print(f'Trade {trades_completed + 1} of {TARGET_TRADES}', flush=True)
        print(f'{"="*50}', flush=True)

        now = datetime.now(timezone.utc)
        print(f'UTC: {now.strftime("%H:%M:%S")}', flush=True)

        # Get prediction
        print('\nAnalyzing BTC price action...', flush=True)
        pred_direction, confidence, reasons = get_prediction()

        print(f'Prediction: {pred_direction or "NEUTRAL"}', flush=True)
        print(f'Confidence: {confidence:.0f}%', flush=True)
        for r in reasons:
            print(f'  - {r}', flush=True)

        # Find market
        market = find_open_market()
        if not market:
            print('\nNo open market found, waiting 60s...', flush=True)
            time.sleep(60)
            continue

        print(f'\nMarket: {market["title"]}', flush=True)
        print(f'UP: {market["up_price"]:.1%} | DOWN: {market["down_price"]:.1%}', flush=True)

        # Decide trade direction
        if pred_direction and confidence >= MIN_CONFIDENCE:
            # Use prediction
            direction = pred_direction
            print(f'\n-> Using PREDICTION: {direction} ({confidence:.0f}% conf)', flush=True)
        else:
            # Fallback: buy cheaper side with slight DOWN bias (BTC tends to be volatile)
            if market["down_price"] < market["up_price"] - 0.05:
                direction = "DOWN"
            elif market["up_price"] < market["down_price"] - 0.05:
                direction = "UP"
            else:
                # Close to 50/50 - use momentum hint or skip
                if pred_direction:
                    direction = pred_direction
                else:
                    direction = "DOWN"  # Slight bearish bias in uncertain markets
            print(f'\n-> Using FALLBACK: {direction} (low confidence)', flush=True)

        # Execute trade
        if direction == "UP":
            token = market["up_token"]
            price = market["up_price"]
        else:
            token = market["down_token"]
            price = market["down_price"]

        size = round(POSITION_SIZE / price, 2)

        print(f'\nTrading: {direction} at {price:.1%}', flush=True)
        print(f'Size: {size} shares (${size * price:.2f})', flush=True)

        try:
            order = executor.place_order(
                token_id=token,
                side=OrderSide.BUY,
                price=min(price + 0.02, 0.99),
                size=size,
                market_question=f'BTC 15M {direction}'
            )

            if order:
                print(f'\nORDER {order.status.value}: {order.order_id[:30]}...', flush=True)
                trades_completed += 1
                results.append({
                    'market': market['title'],
                    'direction': direction,
                    'price': price,
                    'size': size,
                    'confidence': confidence,
                    'prediction_used': confidence >= MIN_CONFIDENCE
                })
            else:
                print('Order failed!', flush=True)
        except Exception as e:
            print(f'Trade error: {e}', flush=True)

        # Wait for next slot
        if trades_completed < TARGET_TRADES:
            now = datetime.now(timezone.utc)
            next_slot = ((now.minute // 15) + 1) * 15
            if next_slot >= 60:
                wait_seconds = (60 - now.minute) * 60 - now.second
            else:
                wait_seconds = (next_slot - now.minute) * 60 - now.second

            wait_seconds = max(wait_seconds, 30)
            wait_seconds = min(wait_seconds, 900)

            print(f'\nWaiting {wait_seconds//60}m {wait_seconds%60}s...', flush=True)
            time.sleep(wait_seconds)

    # Summary
    print(f'\n{"="*60}', flush=True)
    print('TRADING COMPLETE', flush=True)
    print(f'{"="*60}', flush=True)
    print(f'Trades: {trades_completed}', flush=True)
    print(f'Total invested: ${sum(r["price"] * r["size"] for r in results):.2f}', flush=True)
    print(f'\nTrades using prediction: {sum(1 for r in results if r["prediction_used"])}', flush=True)
    print(f'Trades using fallback: {sum(1 for r in results if not r["prediction_used"])}', flush=True)

    print('\nPositions:', flush=True)
    for r in results:
        pred_mark = '*' if r['prediction_used'] else ''
        print(f'  {r["direction"]}{pred_mark} @ {r["price"]:.1%} - {r["market"]}', flush=True)

if __name__ == '__main__':
    main()
