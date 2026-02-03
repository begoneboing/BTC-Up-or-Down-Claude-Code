"""Explore Bitcoin 15-minute markets on Polymarket"""
import requests
import json
from datetime import datetime

# Search for Bitcoin markets
url = 'https://gamma-api.polymarket.com/markets'
params = {'limit': 200, 'active': 'true', 'closed': 'false'}
response = requests.get(url, params=params, timeout=30)
markets = response.json()

# Filter for Bitcoin markets
btc_markets = []
for m in markets:
    question = m.get('question', '').lower()
    if 'bitcoin' in question or 'btc' in question:
        btc_markets.append(m)

print(f'Found {len(btc_markets)} Bitcoin markets')
print()

# Look for 15-minute / short-term markets
short_term = []
for m in btc_markets:
    question = m.get('question', '')
    end_date = m.get('endDate', '')
    volume = float(m.get('volume', 0) or 0)
    liquidity = float(m.get('liquidity', 0) or 0)

    # Check if it's a short-term market (15min, hourly, etc)
    q_lower = question.lower()
    is_short_term = any(x in q_lower for x in ['15', 'minute', 'hour', 'up or down', 'price'])

    clob_tokens = m.get('clobTokenIds', '[]')
    if isinstance(clob_tokens, str):
        clob_tokens = json.loads(clob_tokens)

    print(f'Q: {question[:80]}')
    print(f'   End: {end_date[:19] if end_date else "N/A"}')
    print(f'   Vol: ${volume:,.0f} | Liq: ${liquidity:,.0f} | Tokens: {len(clob_tokens)}')
    if is_short_term:
        print(f'   ** SHORT-TERM MARKET **')
        short_term.append(m)
    print()

print('=' * 80)
print(f'Short-term markets identified: {len(short_term)}')

# Also search for events related to Bitcoin
print()
print('=' * 80)
print('Searching events for Bitcoin...')
print('=' * 80)

events_url = 'https://gamma-api.polymarket.com/events'
params = {'limit': 100, 'active': 'true', 'closed': 'false'}
response = requests.get(events_url, params=params, timeout=30)
events = response.json()

btc_events = []
for e in events:
    title = e.get('title', '').lower()
    if 'bitcoin' in title or 'btc' in title:
        btc_events.append(e)

print(f'Found {len(btc_events)} Bitcoin events')
for e in btc_events[:10]:
    title = e.get('title', '')
    markets = e.get('markets', [])
    print(f'\nEvent: {title}')
    print(f'Markets: {len(markets)}')
    for m in markets[:5]:
        q = m.get('question', '')[:60]
        print(f'  - {q}')
