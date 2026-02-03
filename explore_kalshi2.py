"""Explore Kalshi API more thoroughly for Bitcoin markets"""
import requests
import json
from datetime import datetime

BASE_URL = "https://api.elections.kalshi.com"

# Get series - Kalshi organizes markets into series
print("=" * 60)
print("Fetching Kalshi Series")
print("=" * 60)

try:
    response = requests.get(f"{BASE_URL}/trade-api/v2/series", timeout=30)
    if response.status_code == 200:
        data = response.json()
        series_list = data.get('series', [])
        print(f"Found {len(series_list)} series")

        # Look for crypto/bitcoin series
        crypto_series = []
        for s in series_list:
            ticker = str(s.get('ticker', '')).lower()
            title = str(s.get('title', '')).lower()
            category = str(s.get('category', '')).lower()

            if any(kw in ticker or kw in title or kw in category for kw in ['bitcoin', 'btc', 'crypto', 'inxd', 'kxbtc', 'btcusd']):
                crypto_series.append(s)
                print(f"\n  Series: {s.get('ticker', 'N/A')}")
                print(f"  Title: {s.get('title', 'N/A')}")
                print(f"  Category: {s.get('category', 'N/A')}")

        print(f"\nFound {len(crypto_series)} crypto-related series")

        # If no crypto found, print all series categories to understand structure
        if not crypto_series:
            print("\nAll series categories:")
            categories = set()
            for s in series_list:
                cat = s.get('category', 'Unknown')
                categories.add(cat)
            for cat in sorted(categories):
                print(f"  - {cat}")

            print("\nSample series:")
            for s in series_list[:10]:
                print(f"  {s.get('ticker', 'N/A')}: {s.get('title', 'N/A')[:50]}")

except Exception as e:
    print(f"Error: {e}")

# Try fetching markets with different parameters
print("\n" + "=" * 60)
print("Searching for Crypto Markets")
print("=" * 60)

# Common Kalshi crypto tickers
crypto_tickers = ['KXBTC', 'KXETH', 'INXD', 'BTCUSD', 'BTC']

for ticker in crypto_tickers:
    try:
        print(f"\nSearching series: {ticker}")
        response = requests.get(
            f"{BASE_URL}/trade-api/v2/markets",
            params={'series_ticker': ticker, 'limit': 50},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            markets = data.get('markets', [])
            print(f"  Found {len(markets)} markets")
            for m in markets[:5]:
                print(f"    - {m.get('ticker', 'N/A')}: {m.get('title', 'N/A')[:50]}")
    except Exception as e:
        print(f"  Error: {e}")

# Search for event ticker patterns
print("\n" + "=" * 60)
print("Searching by event ticker pattern")
print("=" * 60)

patterns = ['KXBTC', 'BTCUSD', 'BTC-']
for pattern in patterns:
    try:
        print(f"\nEvent ticker: {pattern}")
        response = requests.get(
            f"{BASE_URL}/trade-api/v2/events",
            params={'series_ticker': pattern, 'status': 'open', 'limit': 50},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            events = data.get('events', [])
            print(f"  Found {len(events)} events")
            for e in events[:5]:
                print(f"    - {e.get('event_ticker', 'N/A')}: {e.get('title', 'N/A')[:50]}")
    except Exception as e:
        print(f"  Error: {e}")

# Check if there's a specific category
print("\n" + "=" * 60)
print("Checking categories")
print("=" * 60)

try:
    response = requests.get(
        f"{BASE_URL}/trade-api/v2/events",
        params={'status': 'open', 'limit': 500},
        timeout=30
    )
    if response.status_code == 200:
        data = response.json()
        events = data.get('events', [])

        categories = {}
        for e in events:
            cat = e.get('category', 'Unknown')
            categories[cat] = categories.get(cat, 0) + 1

        print("Event categories:")
        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            print(f"  {cat}: {count}")
except Exception as e:
    print(f"Error: {e}")
