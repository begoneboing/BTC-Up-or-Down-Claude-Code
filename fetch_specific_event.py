"""Fetch the specific BTC 15-minute event from Polymarket"""
import requests
import json
from datetime import datetime

# Different API endpoints to try
endpoints = [
    "https://gamma-api.polymarket.com",
    "https://strapi-matic.poly.market",
    "https://clob.polymarket.com",
]

event_slug = "btc-updown-15m-1770081300"
timestamps = [
    1770081300,  # 7:15 PM
    1770082200,  # 7:30 PM
    1770083100,  # 7:45 PM
]

print("=" * 70)
print("Fetching BTC 15-minute Up/Down Markets")
print("=" * 70)

# Try the strapi endpoint which might have different data
print("\n1. Trying Strapi API...")
try:
    response = requests.get(
        "https://strapi-matic.poly.market/events",
        params={"slug_contains": "btc-updown"},
        timeout=30
    )
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Found: {len(data) if isinstance(data, list) else 'dict'}")
        if data:
            print(f"   Data: {json.dumps(data[:2] if isinstance(data, list) else data, indent=2)[:500]}")
except Exception as e:
    print(f"   Error: {e}")

# Try to access the market via clob
print("\n2. Trying CLOB markets endpoint...")
try:
    response = requests.get(
        "https://clob.polymarket.com/markets",
        timeout=30
    )
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        markets = data if isinstance(data, list) else data.get('markets', data.get('data', []))
        print(f"   Total markets: {len(markets)}")

        # Search for BTC or updown
        btc_markets = [m for m in markets if 'btc' in str(m).lower() or 'updown' in str(m).lower() or '15m' in str(m).lower()]
        print(f"   BTC/updown markets: {len(btc_markets)}")
        for m in btc_markets[:5]:
            print(f"   {json.dumps(m, indent=2)[:300]}")
except Exception as e:
    print(f"   Error: {e}")

# Try negative risk markets endpoint
print("\n3. Checking neg-risk markets...")
try:
    response = requests.get(
        "https://clob.polymarket.com/neg-risk",
        timeout=30
    )
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Data type: {type(data)}")
        if isinstance(data, list):
            btc_neg = [m for m in data if 'btc' in str(m).lower()]
            print(f"   BTC neg-risk: {len(btc_neg)}")
except Exception as e:
    print(f"   Error: {e}")

# Try the gamma events with different params
print("\n4. Gamma events with cursor...")
try:
    # Get more events using pagination
    all_events = []
    cursor = ""
    for _ in range(3):
        params = {"limit": 200, "active": "true"}
        if cursor:
            params["next_cursor"] = cursor

        response = requests.get(
            "https://gamma-api.polymarket.com/events",
            params=params,
            timeout=60
        )
        if response.status_code == 200:
            data = response.json()
            events = data if isinstance(data, list) else data.get('events', [])
            all_events.extend(events)
            cursor = data.get('next_cursor', '') if isinstance(data, dict) else ''
            if not cursor:
                break

    print(f"   Total events fetched: {len(all_events)}")

    # Search for BTC updown
    btc_events = [e for e in all_events if 'btc' in str(e.get('slug', '')).lower() and 'updown' in str(e.get('slug', '')).lower()]
    print(f"   BTC updown events: {len(btc_events)}")

    if not btc_events:
        # Try broader search
        btc_events = [e for e in all_events if 'btc' in str(e.get('title', '')).lower() or 'bitcoin' in str(e.get('title', '')).lower()]
        print(f"   BTC-related events (broader): {len(btc_events)}")
        for e in btc_events[:5]:
            print(f"\n   Title: {e.get('title', 'N/A')[:50]}")
            print(f"   Slug: {e.get('slug', 'N/A')[:50]}")

except Exception as e:
    print(f"   Error: {e}")

# Try fetching by condition ID patterns
print("\n5. Searching CLOB sampling markets for specific patterns...")
try:
    response = requests.get(
        "https://clob.polymarket.com/sampling-markets",
        params={"limit": 2000},
        timeout=60
    )
    if response.status_code == 200:
        data = response.json()
        markets = data.get('data', [])
        print(f"   Sampling markets: {len(markets)}")

        # Search for patterns
        patterns = ['btc', 'bitcoin', 'updown', 'up-down', '15m', '15 min']
        for pattern in patterns:
            matches = [m for m in markets if pattern in str(m.get('question', '')).lower() or pattern in str(m.get('market_slug', '')).lower()]
            if matches:
                print(f"\n   Pattern '{pattern}': {len(matches)} matches")
                for m in matches[:2]:
                    print(f"      Q: {m.get('question', 'N/A')[:50]}")
                    print(f"      Slug: {m.get('market_slug', 'N/A')[:50]}")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "=" * 70)
print("If the markets aren't found, they may:")
print("1. Not be active yet (markets open at specific times)")
print("2. Be on a different API endpoint or series")
print("3. Require authentication to access")
print("=" * 70)
