#!/usr/bin/env python
"""Monitor BTC 15-minute positions and exit when profitable"""
import os
import sys
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
os.environ['TRADING_MODE'] = 'live'

import requests
import json
import time
from datetime import datetime, timezone
from trade_executor import TradeExecutor, OrderSide

# Positions from recent trades
POSITIONS = [
    # Original trades
    {'direction': 'UP', 'entry_price': 0.365, 'size': 13.7, 'market': '11:00AM-11:15AM ET'},
    {'direction': 'DOWN', 'entry_price': 0.465, 'size': 10.75, 'market': '11:15AM-11:30AM ET'},
    {'direction': 'DOWN', 'entry_price': 0.455, 'size': 10.99, 'market': '11:30AM-11:45AM ET'},
    {'direction': 'DOWN', 'entry_price': 0.445, 'size': 11.24, 'market': '11:45AM-12:00PM ET'},
    # Smart trades
    {'direction': 'DOWN', 'entry_price': 0.235, 'size': 21.28, 'market': '11:45AM-12:00PM ET'},
    {'direction': 'DOWN', 'entry_price': 0.515, 'size': 9.71, 'market': '12:00PM-12:15PM ET'},
    {'direction': 'UP', 'entry_price': 0.515, 'size': 9.71, 'market': '12:15PM-12:30PM ET'},
    {'direction': 'DOWN', 'entry_price': 0.475, 'size': 10.53, 'market': '12:30PM-12:45PM ET'},
]

PROFIT_TARGET = 0.15  # 15% profit target
STOP_LOSS = -0.25     # 25% stop loss

def get_market_slug(market_time: str) -> str:
    """Convert market time to slug for API lookup"""
    # This is a simplified version - in production, calculate from timestamp
    now = datetime.now(timezone.utc)
    # Parse the market time and find the matching slug
    # For now, search recent markets
    return None

def check_market_resolution(slug: str) -> dict:
    """Check if a market has resolved and get final prices"""
    try:
        response = requests.get(f'https://gamma-api.polymarket.com/events/slug/{slug}', timeout=10)
        if response.status_code == 200:
            event = response.json()
            market = event.get('markets', [{}])[0]
            prices = json.loads(market.get('outcomePrices', '[]'))
            outcomes = json.loads(market.get('outcomes', '[]'))
            closed = event.get('closed', False)

            return {
                'closed': closed,
                'prices': prices,
                'outcomes': outcomes,
                'up_price': float(prices[0]) if prices else 0.5,
                'down_price': float(prices[1]) if len(prices) > 1 else 0.5
            }
    except Exception as e:
        print(f'Error checking market: {e}')
    return None

def calculate_pnl(entry_price: float, exit_price: float, size: float) -> dict:
    """Calculate profit/loss"""
    pnl = (exit_price - entry_price) * size
    pnl_pct = (exit_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
    return {
        'pnl': pnl,
        'pnl_pct': pnl_pct,
        'entry': entry_price,
        'exit': exit_price
    }

def main():
    print('='*60, flush=True)
    print('BTC 15-MINUTE POSITION MONITOR', flush=True)
    print(f'Profit Target: {PROFIT_TARGET:.0%}', flush=True)
    print(f'Stop Loss: {STOP_LOSS:.0%}', flush=True)
    print('='*60, flush=True)

    # For 15-minute markets, they resolve automatically
    # We track the outcomes and calculate P&L

    print('\nNote: 15-minute markets resolve automatically.', flush=True)
    print('Checking recent market resolutions...', flush=True)

    # Check recent market resolutions
    now = datetime.now(timezone.utc)
    minute = (now.minute // 15) * 15
    slot_time = now.replace(minute=minute, second=0, microsecond=0)
    current_ts = int(slot_time.timestamp())

    total_pnl = 0

    # Check last 4 15-minute slots
    for i in range(8):
        ts = current_ts - (i * 15 * 60)
        slug = f'btc-updown-15m-{ts}'

        try:
            response = requests.get(f'https://gamma-api.polymarket.com/events/slug/{slug}', timeout=10)
            if response.status_code != 200:
                continue

            event = response.json()
            title = event.get('title', '')
            closed = event.get('closed', False)

            if not closed:
                print(f'\n{title}: Still OPEN', flush=True)
                continue

            market = event.get('markets', [{}])[0]
            prices = json.loads(market.get('outcomePrices', '[]'))

            # Determine winner (price = 1 means that outcome won)
            if len(prices) >= 2:
                up_won = float(prices[0]) > 0.5
                winner = 'UP' if up_won else 'DOWN'

                print(f'\n{title}', flush=True)
                print(f'  Result: {winner} won', flush=True)

                # Check if we had a position in this market
                for pos in POSITIONS:
                    if pos['market'] in title:
                        direction = pos['direction']
                        entry = pos['entry_price']
                        size = pos['size']

                        if direction == winner:
                            # We won - payout is size * 1.0
                            exit_price = 1.0
                            pnl = (exit_price - entry) * size
                        else:
                            # We lost - position worth 0
                            exit_price = 0.0
                            pnl = -entry * size

                        total_pnl += pnl
                        print(f'  Our bet: {direction} at {entry:.1%}', flush=True)
                        print(f'  P&L: ${pnl:+.2f} ({"WIN" if direction == winner else "LOSS"})', flush=True)

        except Exception as e:
            continue

    print(f'\n{"="*60}', flush=True)
    print(f'TOTAL P&L: ${total_pnl:+.2f}', flush=True)
    print('='*60, flush=True)

if __name__ == '__main__':
    main()
