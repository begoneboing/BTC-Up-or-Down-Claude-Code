"""
Polymarket API Client
Fetches market data from Polymarket Gamma API
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

# API Configuration
GAMMA_API_URL = "https://gamma-api.polymarket.com"
CLOB_API_URL = "https://clob.polymarket.com"

# Credentials from environment
PRIVATE_KEY = os.environ.get("POLYMARKET_PRIVATE_KEY", "")
FUNDER_ADDRESS = os.environ.get("POLYMARKET_FUNDER_ADDRESS", "")
TRADING_MODE = os.environ.get("TRADING_MODE", "paper")


def get_active_markets(limit=100):
    """Fetch active markets from Polymarket"""
    url = f"{GAMMA_API_URL}/markets"
    params = {
        "limit": limit,
        "active": "true",
        "closed": "false"
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def get_events(limit=100):
    """Fetch active events from Polymarket"""
    url = f"{GAMMA_API_URL}/events"
    params = {
        "limit": limit,
        "active": "true",
        "closed": "false"
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def get_market_by_id(market_id):
    """Fetch a specific market by ID"""
    url = f"{GAMMA_API_URL}/markets/{market_id}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def get_prices(token_id):
    """Get current prices for a token"""
    url = f"{CLOB_API_URL}/price"
    params = {"token_id": token_id}
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def search_markets(query, limit=50):
    """Search markets by keyword"""
    markets = get_active_markets(limit=500)
    results = []

    query_lower = query.lower()
    for market in markets:
        title = market.get("question", "").lower()
        description = market.get("description", "").lower()

        if query_lower in title or query_lower in description:
            results.append(market)
            if len(results) >= limit:
                break

    return results


def test_connection():
    """Test the API connection"""
    print("Testing Polymarket API connection...")
    print()

    # Check credentials
    print("Credentials:")
    print(f"  PRIVATE_KEY: {'SET' if PRIVATE_KEY else 'NOT SET'}")
    print(f"  FUNDER_ADDRESS: {'SET' if FUNDER_ADDRESS else 'NOT SET'}")
    print(f"  TRADING_MODE: {TRADING_MODE}")
    print()

    # Test API
    try:
        events = get_events(limit=5)
        print(f"API Status: SUCCESS - Fetched {len(events)} events")
        print()
        print("Sample events:")
        for i, event in enumerate(events[:5], 1):
            title = event.get("title", "N/A")[:60]
            print(f"  {i}. {title}...")
    except Exception as e:
        print(f"API Status: FAILED - {e}")


if __name__ == "__main__":
    test_connection()
