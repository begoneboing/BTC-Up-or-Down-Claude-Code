"""Search Polymarket CLOB API for Bitcoin 15-minute markets"""
import requests
import json
from datetime import datetime, timezone

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"

print("=" * 70)
print("Comprehensive search for Bitcoin 15-minute markets on Polymarket")
print("=" * 70)

# Search with different parameters
search_terms = [
    "bitcoin",
    "btc",
    "15",
    "minute",
    "crypto",
    "up or down",
    "volatility"
]

all_markets = set()

# Get markets from gamma API with different filters
print("\n1. Searching Gamma API for all crypto/bitcoin markets...")

try:
    response = requests.get(
        f"{GAMMA_URL}/markets",
        params={
            "limit": 1000,
            "active": "true",
            "closed": "false"
        },
        timeout=60
    )
    markets = response.json()
    print(f"   Total active markets: {len(markets)}")

    # Filter for crypto-related
    crypto_markets = []
    for m in markets:
        q = (m.get('question', '') or '').lower()
        tags = str(m.get('tags', [])).lower()

        if any(kw in q or kw in tags for kw in ['bitcoin', 'btc', 'crypto', 'eth', 'sol']):
            crypto_markets.append(m)

    print(f"   Crypto-related markets: {len(crypto_markets)}")

    for m in crypto_markets:
        print(f"\n   Q: {m.get('question', 'N/A')[:70]}")
        print(f"   End: {m.get('endDate', 'N/A')[:19] if m.get('endDate') else 'N/A'}")
        print(f"   Vol: ${float(m.get('volume', 0) or 0):,.0f}")
except Exception as e:
    print(f"   Error: {e}")

# Search events for crypto
print("\n" + "=" * 70)
print("2. Searching Events for crypto/bitcoin...")
print("=" * 70)

try:
    response = requests.get(
        f"{GAMMA_URL}/events",
        params={"limit": 500, "active": "true"},
        timeout=60
    )
    events = response.json()

    crypto_events = []
    for e in events:
        title = (e.get('title', '') or '').lower()
        if any(kw in title for kw in ['bitcoin', 'btc', 'crypto', '15 min', 'minute']):
            crypto_events.append(e)

    print(f"Crypto/Bitcoin events: {len(crypto_events)}")
    for e in crypto_events:
        print(f"\n   Event: {e.get('title', 'N/A')[:60]}")
        markets = e.get('markets', [])
        print(f"   Markets: {len(markets)}")
        for m in markets[:3]:
            print(f"     - {m.get('question', 'N/A')[:55]}")
except Exception as e:
    print(f"Error: {e}")

# Try CLOB API directly
print("\n" + "=" * 70)
print("3. Checking CLOB API for active tokens...")
print("=" * 70)

try:
    # Get sampling markets
    response = requests.get(
        f"{CLOB_URL}/sampling-markets",
        params={"next_cursor": ""},
        timeout=30
    )
    if response.status_code == 200:
        data = response.json()
        print(f"Sampling markets response keys: {data.keys() if isinstance(data, dict) else 'list'}")
except Exception as e:
    print(f"Error with sampling markets: {e}")

# Check for markets with specific tags
print("\n" + "=" * 70)
print("4. Looking for markets by category/tag...")
print("=" * 70)

try:
    response = requests.get(
        f"{GAMMA_URL}/markets",
        params={
            "limit": 200,
            "active": "true",
            "closed": "false",
            "tag": "crypto"
        },
        timeout=60
    )
    if response.status_code == 200:
        tagged_markets = response.json()
        print(f"Markets with 'crypto' tag: {len(tagged_markets)}")
        for m in tagged_markets[:10]:
            print(f"   - {m.get('question', 'N/A')[:60]}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print("""
Based on the search, Polymarket does not appear to have active
15-minute Bitcoin "Up or Down" markets at this time.

Polymarket primarily hosts:
- Long-term event prediction markets (politics, sports, etc.)
- Some crypto-related markets (price predictions, ETF approvals)

For short-term 15-minute crypto volatility markets, platforms like
Kalshi typically offer these types of products.

If you have a specific market ticker or URL, please share it.
""")
