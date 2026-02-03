"""Explore short-term price/volatility markets on Polymarket"""
import requests
import json
from datetime import datetime, timedelta

# Search for all markets
url = 'https://gamma-api.polymarket.com/markets'
params = {'limit': 500, 'active': 'true', 'closed': 'false'}
response = requests.get(url, params=params, timeout=30)
markets = response.json()

print(f'Total active markets: {len(markets)}')
print()

# Look for short-term / volatility markets
keywords = ['minute', '15', 'hour', 'price', 'up or down', 'volatility', 'crypto', 'eth', 'sol']
short_term = []

for m in markets:
    question = m.get('question', '').lower()
    end_date_str = m.get('endDate', '')

    # Check end date - if within 24 hours it's short-term
    is_soon = False
    if end_date_str:
        try:
            # Parse ISO date
            end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
            now = datetime.now(end_date.tzinfo) if end_date.tzinfo else datetime.now()
            hours_until_end = (end_date - now).total_seconds() / 3600
            is_soon = hours_until_end < 24
        except:
            pass

    # Check keywords
    has_keyword = any(kw in question for kw in keywords)

    if is_soon or has_keyword:
        volume = float(m.get('volume', 0) or 0)
        liquidity = float(m.get('liquidity', 0) or 0)

        short_term.append({
            'question': m.get('question', ''),
            'end_date': end_date_str,
            'volume': volume,
            'liquidity': liquidity,
            'market': m,
            'is_soon': is_soon
        })

print(f'Potential short-term/volatility markets: {len(short_term)}')
print('=' * 80)

# Sort by end date (soonest first)
for item in sorted(short_term, key=lambda x: x['end_date'] if x['end_date'] else 'z')[:30]:
    print(f"\nQ: {item['question'][:75]}")
    print(f"   End: {item['end_date'][:19] if item['end_date'] else 'N/A'}")
    print(f"   Vol: ${item['volume']:,.0f} | Liq: ${item['liquidity']:,.0f}")
    if item['is_soon']:
        print(f"   ** ENDS WITHIN 24 HOURS **")

# Also look at events
print()
print('=' * 80)
print('Searching all events...')
print('=' * 80)

events_url = 'https://gamma-api.polymarket.com/events'
params = {'limit': 200, 'active': 'true', 'closed': 'false'}
response = requests.get(events_url, params=params, timeout=30)
events = response.json()

print(f'Total active events: {len(events)}')

# Look for crypto/price events
for e in events:
    title = e.get('title', '').lower()
    if any(kw in title for kw in ['price', 'crypto', 'bitcoin', 'btc', 'eth', 'minute', 'hour']):
        print(f"\nEvent: {e.get('title', '')}")
        markets = e.get('markets', [])
        print(f"Markets: {len(markets)}")
        for m in markets[:3]:
            print(f"  - {m.get('question', '')[:60]}")
