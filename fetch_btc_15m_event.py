"""Fetch specific BTC 15-minute Up/Down market from Polymarket"""
import requests
import json
from datetime import datetime

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"

# The event slug from the URL
event_slug = "btc-updown-15m-1770081300"
event_id = "1770081300"

print("=" * 70)
print("Fetching BTC 15-minute Up/Down Market")
print("=" * 70)
print(f"Event slug: {event_slug}")
print()

# Try different approaches to fetch the market

# 1. Try fetching by slug
print("1. Fetching by slug...")
try:
    response = requests.get(f"{GAMMA_URL}/events/{event_slug}", timeout=30)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Data: {json.dumps(data, indent=2)[:500]}")
except Exception as e:
    print(f"   Error: {e}")

# 2. Try searching events with the slug pattern
print("\n2. Searching events for 'btc-updown'...")
try:
    response = requests.get(
        f"{GAMMA_URL}/events",
        params={"limit": 200, "active": "true"},
        timeout=60
    )
    events = response.json()

    btc_events = []
    for e in events:
        slug = (e.get('slug', '') or '').lower()
        title = (e.get('title', '') or '').lower()
        if 'btc' in slug or 'btc' in title or 'updown' in slug or '15m' in slug:
            btc_events.append(e)

    print(f"   Found {len(btc_events)} matching events")
    for e in btc_events[:10]:
        print(f"\n   Title: {e.get('title', 'N/A')}")
        print(f"   Slug: {e.get('slug', 'N/A')}")
        markets = e.get('markets', [])
        print(f"   Markets: {len(markets)}")
except Exception as e:
    print(f"   Error: {e}")

# 3. Try markets search with bitcoin/btc
print("\n3. Searching markets for 'btc' and '15'...")
try:
    response = requests.get(
        f"{GAMMA_URL}/markets",
        params={"limit": 500, "active": "true", "closed": "false"},
        timeout=60
    )
    markets = response.json()

    btc_markets = []
    for m in markets:
        q = (m.get('question', '') or '').lower()
        cond_id = m.get('conditionId', '')
        if ('btc' in q and '15' in q) or 'up or down' in q or 'updown' in q:
            btc_markets.append(m)

    print(f"   Found {len(btc_markets)} potential matches")
    for m in btc_markets[:10]:
        print(f"\n   Question: {m.get('question', 'N/A')}")
        print(f"   Condition ID: {m.get('conditionId', 'N/A')}")
        print(f"   End: {m.get('endDate', 'N/A')[:25] if m.get('endDate') else 'N/A'}")
except Exception as e:
    print(f"   Error: {e}")

# 4. Try CLOB sampling markets for BTC
print("\n4. Checking CLOB sampling markets...")
try:
    response = requests.get(f"{CLOB_URL}/sampling-markets", timeout=30)
    if response.status_code == 200:
        data = response.json()
        markets = data.get('data', [])
        print(f"   Found {len(markets)} sampling markets")

        btc_sampling = [m for m in markets if 'btc' in str(m).lower() or 'bitcoin' in str(m).lower()]
        print(f"   BTC-related: {len(btc_sampling)}")
        for m in btc_sampling[:5]:
            print(f"   {m}")
except Exception as e:
    print(f"   Error: {e}")

# 5. Try to get the condition/market directly
print("\n5. Trying direct market lookup...")
try:
    # Try with the event ID from URL
    response = requests.get(f"{GAMMA_URL}/markets/{event_id}", timeout=30)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Data: {json.dumps(data, indent=2)[:500]}")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "=" * 70)
print("Note: The BTC 15-minute markets may be on a different series or")
print("scheduled to start at specific times. The timestamp 1770081300")
print("converts to:", datetime.fromtimestamp(1770081300))
print("=" * 70)
