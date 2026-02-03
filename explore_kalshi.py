"""Explore Kalshi API for Bitcoin 15-minute markets"""
import requests
import json
from datetime import datetime

# Kalshi public API base URL
BASE_URL = "https://api.elections.kalshi.com"
TRADING_URL = "https://trading-api.kalshi.com"

def get_events():
    """Get events from Kalshi - trying different endpoints"""

    # Try public events endpoint
    endpoints = [
        f"{BASE_URL}/trade-api/v2/events",
        f"{TRADING_URL}/trade-api/v2/events",
        f"{BASE_URL}/v1/events",
    ]

    for url in endpoints:
        try:
            print(f"Trying: {url}")
            response = requests.get(url, params={'status': 'open', 'limit': 200}, timeout=30)
            print(f"  Status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                return data
            elif response.status_code == 401:
                print("  Auth required")
        except Exception as e:
            print(f"  Error: {e}")

    return None

def get_markets():
    """Try to get markets directly"""

    endpoints = [
        f"{BASE_URL}/trade-api/v2/markets",
        f"{TRADING_URL}/trade-api/v2/markets",
    ]

    for url in endpoints:
        try:
            print(f"Trying markets: {url}")
            # Search for Bitcoin-related markets
            params = {
                'limit': 200,
                'status': 'open',
            }
            response = requests.get(url, params=params, timeout=30)
            print(f"  Status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                return data
        except Exception as e:
            print(f"  Error: {e}")

    return None

# Try to get data
print("=" * 60)
print("Exploring Kalshi API")
print("=" * 60)

events_data = get_events()
if events_data:
    print(f"\nEvents data keys: {events_data.keys() if isinstance(events_data, dict) else 'list'}")

    events = events_data.get('events', events_data) if isinstance(events_data, dict) else events_data
    if isinstance(events, list):
        print(f"Found {len(events)} events")

        # Look for Bitcoin/crypto events
        btc_events = []
        for e in events:
            title = str(e.get('title', '') or e.get('event_ticker', '')).lower()
            if any(kw in title for kw in ['bitcoin', 'btc', 'crypto']):
                btc_events.append(e)

        print(f"Bitcoin events: {len(btc_events)}")
        for e in btc_events[:10]:
            print(f"\n  Title: {e.get('title', e.get('event_ticker', 'Unknown'))}")
            print(f"  Ticker: {e.get('event_ticker', 'N/A')}")
            print(f"  Category: {e.get('category', 'N/A')}")

markets_data = get_markets()
if markets_data:
    print(f"\nMarkets data keys: {markets_data.keys() if isinstance(markets_data, dict) else 'list'}")

    markets = markets_data.get('markets', markets_data) if isinstance(markets_data, dict) else markets_data
    if isinstance(markets, list):
        print(f"Found {len(markets)} markets")

        # Look for Bitcoin/crypto markets
        btc_markets = []
        for m in markets:
            ticker = str(m.get('ticker', '') or m.get('title', '')).lower()
            title = str(m.get('title', '')).lower()
            if any(kw in ticker or kw in title for kw in ['bitcoin', 'btc', 'crypto', 'inxd']):
                btc_markets.append(m)

        print(f"Bitcoin markets: {len(btc_markets)}")
        for m in btc_markets[:20]:
            print(f"\n  Ticker: {m.get('ticker', 'N/A')}")
            print(f"  Title: {m.get('title', 'N/A')}")
            print(f"  Close: {m.get('close_time', m.get('expected_expiration_time', 'N/A'))}")

print("\n" + "=" * 60)
print("Note: Kalshi may require authentication for full API access")
print("=" * 60)
