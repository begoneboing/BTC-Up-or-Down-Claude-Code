#!/usr/bin/env python
"""Run BTC 15-minute bot for 4 consecutive events"""
import os
import sys

# Unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

os.environ['TRADING_MODE'] = 'live'

import requests
import json
import time
from datetime import datetime, timezone
from trade_executor import TradeExecutor, OrderSide

print('='*60, flush=True)
print('BTC 15-MINUTE BOT - LIVE TRADING', flush=True)
print('Target: 4 events', flush=True)
print('='*60, flush=True)

executor = TradeExecutor()
trades_completed = 0

while trades_completed < 4:
    print(f'\n--- Trade {trades_completed + 1} of 4 ---', flush=True)

    now = datetime.now(timezone.utc)
    minute = (now.minute // 15) * 15
    slot_time = now.replace(minute=minute, second=0, microsecond=0)
    current_ts = int(slot_time.timestamp())

    print(f'Current UTC: {now.strftime("%H:%M:%S")}', flush=True)

    market_found = False
    for offset in range(0, 6):
        ts = current_ts + (offset * 15 * 60)
        slug = f'btc-updown-15m-{ts}'

        try:
            response = requests.get(f'https://gamma-api.polymarket.com/events/slug/{slug}', timeout=10)
        except Exception as e:
            print(f'API error: {e}', flush=True)
            continue

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

        title = event.get('title', '')
        print(f'\nMarket: {title}', flush=True)

        tokens = json.loads(market.get('clobTokenIds', '[]'))
        outcomes = json.loads(market.get('outcomes', '[]'))

        up_idx = 0 if outcomes[0].lower() == 'up' else 1
        down_idx = 1 - up_idx

        up_price = float(prices[up_idx])
        down_price = float(prices[down_idx])

        print(f'UP: {up_price:.1%} | DOWN: {down_price:.1%}', flush=True)

        # Trading logic
        if down_price < 0.45:
            direction = 'DOWN'
            token = tokens[down_idx]
            price = down_price
        elif up_price < 0.45:
            direction = 'UP'
            token = tokens[up_idx]
            price = up_price
        elif up_price < down_price:
            direction = 'UP'
            token = tokens[up_idx]
            price = up_price
        else:
            direction = 'DOWN'
            token = tokens[down_idx]
            price = down_price

        size = round(5.0 / price, 2)
        print(f'Trading: {direction} at {price:.1%} ({size} shares, ${size*price:.2f})', flush=True)

        try:
            order = executor.place_order(
                token_id=token,
                side=OrderSide.BUY,
                price=min(price + 0.02, 0.99),
                size=size,
                market_question=f'BTC 15M {direction}'
            )

            if order:
                print(f'ORDER {order.status.value}: {order.order_id[:30]}...', flush=True)
                trades_completed += 1
                market_found = True
            else:
                print('Order failed', flush=True)
        except Exception as e:
            print(f'Trade error: {e}', flush=True)

        break

    if not market_found:
        print('No open market found, waiting 60s...', flush=True)
        time.sleep(60)
        continue

    if trades_completed < 4:
        now = datetime.now(timezone.utc)
        next_slot = ((now.minute // 15) + 1) * 15
        if next_slot >= 60:
            wait_seconds = (60 - now.minute) * 60 - now.second
        else:
            wait_seconds = (next_slot - now.minute) * 60 - now.second

        wait_seconds = max(wait_seconds, 30)
        wait_seconds = min(wait_seconds, 900)

        print(f'\nWaiting {wait_seconds//60}m {wait_seconds%60}s for next market...', flush=True)
        time.sleep(wait_seconds)

print('\n' + '='*60, flush=True)
print(f'COMPLETED: {trades_completed} trades', flush=True)
print('='*60, flush=True)
