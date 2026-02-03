"""Find Bitcoin 15-minute Up/Down markets on Polymarket"""
import requests
import json
from datetime import datetime

GAMMA_URL = "https://gamma-api.polymarket.com"

print("=" * 70)
print("Searching Polymarket for Bitcoin 15-minute Markets")
print("=" * 70)

# Get all events
print("\nFetching all events...")
response = requests.get(f"{GAMMA_URL}/events", params={"limit": 500, "active": "true"}, timeout=60)
events = response.json()
print(f"Found {len(events)} events")

# Search for Bitcoin/BTC/15 minute/up down related events
keywords = ['bitcoin', 'btc', '15', 'minute', 'up', 'down', 'volatility', 'price']
btc_events = []

for e in events:
    title = (e.get('title', '') or '').lower()
    description = (e.get('description', '') or '').lower()

    # Check if any keyword matches
    if any(kw in title or kw in description for kw in keywords):
        btc_events.append(e)

print(f"\nFound {len(btc_events)} potentially relevant events")

for e in btc_events[:20]:
    title = e.get('title', 'N/A')
    slug = e.get('slug', 'N/A')
    markets = e.get('markets', [])

    print(f"\n  Event: {title[:60]}")
    print(f"  Slug: {slug}")
    print(f"  Markets: {len(markets)}")

    for m in markets[:3]:
        q = m.get('question', 'N/A')[:60]
        end = m.get('endDate', 'N/A')[:19] if m.get('endDate') else 'N/A'
        print(f"    - {q}")
        print(f"      End: {end}")

# Also search markets directly
print("\n" + "=" * 70)
print("Searching markets directly...")
print("=" * 70)

response = requests.get(f"{GAMMA_URL}/markets", params={"limit": 500, "active": "true"}, timeout=60)
markets = response.json()
print(f"Found {len(markets)} active markets")

# Look for short-term bitcoin markets
btc_markets = []
for m in markets:
    q = (m.get('question', '') or '').lower()

    if any(kw in q for kw in ['bitcoin', 'btc']) and any(kw in q for kw in ['up', 'down', 'minute', '15', 'price']):
        btc_markets.append(m)

if btc_markets:
    print(f"\nFound {len(btc_markets)} Bitcoin up/down markets:")
    for m in btc_markets:
        print(f"\n  Question: {m.get('question', 'N/A')}")
        print(f"  End: {m.get('endDate', 'N/A')[:19] if m.get('endDate') else 'N/A'}")
        print(f"  Volume: ${float(m.get('volume', 0) or 0):,.0f}")
else:
    print("\nNo specific Bitcoin 15-minute up/down markets found")
    print("These markets may not be currently active on Polymarket")

# Check if there are any markets ending very soon (within next hour)
print("\n" + "=" * 70)
print("Markets ending within next 2 hours...")
print("=" * 70)

now = datetime.utcnow()
soon_markets = []

for m in markets:
    end_str = m.get('endDate', '')
    if end_str:
        try:
            end_date = datetime.fromisoformat(end_str.replace('Z', ''))
            hours_left = (end_date - now).total_seconds() / 3600
            if 0 < hours_left < 2:
                soon_markets.append({
                    'question': m.get('question', ''),
                    'end': end_str,
                    'hours_left': hours_left,
                    'market': m
                })
        except:
            pass

soon_markets.sort(key=lambda x: x['hours_left'])

if soon_markets:
    print(f"Found {len(soon_markets)} markets ending soon:")
    for sm in soon_markets[:10]:
        print(f"\n  Q: {sm['question'][:60]}")
        print(f"  Ends: {sm['end'][:19]} ({sm['hours_left']:.1f} hours)")
else:
    print("No markets ending within 2 hours")
