#!/usr/bin/env python
"""Check hourly trade results"""
import requests
import json

TRADES = [
    {'slug': 'bitcoin-up-or-down-february-4-2pm-et', 'name': '2PM', 'direction': 'DOWN', 'entry': 0.495, 'size': 10.25},
    {'slug': 'bitcoin-up-or-down-february-4-3pm-et', 'name': '3PM', 'direction': 'DOWN', 'entry': 0.495, 'size': 10.25},
    {'slug': 'bitcoin-up-or-down-february-4-4pm-et', 'name': '4PM', 'direction': 'DOWN', 'entry': 0.500, 'size': 10.10},
]

print('HOURLY TRADE RESULTS')
print('='*60)

total_pnl = 0

for trade in TRADES:
    slug = trade['slug']
    try:
        # Try markets endpoint first
        response = requests.get(f'https://gamma-api.polymarket.com/markets?slug={slug}', timeout=10)

        if response.status_code == 200:
            markets = response.json()
            market = markets[0] if markets else None
        else:
            market = None

        if not market:
            # Try finding by question
            response = requests.get(
                'https://gamma-api.polymarket.com/markets',
                params={'active': 'true', 'limit': 100},
                timeout=10
            )
            if response.status_code == 200:
                all_markets = response.json()
                for m in all_markets:
                    if trade['name'] in m.get('question', '') and 'February 4' in m.get('question', ''):
                        market = m
                        break

        if market:
            prices = market.get('outcomePrices', '[]')
            if isinstance(prices, str):
                prices = json.loads(prices)

            closed = market.get('closed', False)

            if prices and len(prices) >= 2:
                up_price = float(prices[0])
                down_price = float(prices[1])

                # Check if resolved
                if up_price in [0, 1] or down_price in [0, 1]:
                    winner = 'UP' if up_price > 0.5 else 'DOWN'

                    if trade['direction'] == winner:
                        pnl = (1.0 - trade['entry']) * trade['size']
                        result = 'WIN'
                    else:
                        pnl = -trade['entry'] * trade['size']
                        result = 'LOSS'

                    total_pnl += pnl
                    print(f"{trade['name']} ET: {winner} won | Our: {trade['direction']} | ${pnl:+.2f} ({result})")
                else:
                    print(f"{trade['name']} ET: PENDING (UP: {up_price:.1%}, DOWN: {down_price:.1%})")
            else:
                print(f"{trade['name']} ET: No price data")
        else:
            print(f"{trade['name']} ET: Market not found")

    except Exception as e:
        print(f"{trade['name']} ET: Error - {e}")

print('='*60)
print(f'TOTAL P&L: ${total_pnl:+.2f}')
