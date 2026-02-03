"""Find Bitcoin 15-minute markets on Kalshi"""
import requests
import json
from datetime import datetime

BASE_URL = "https://api.elections.kalshi.com"

print("=" * 70)
print("BITCOIN 15-MINUTE MARKETS ON KALSHI")
print("=" * 70)

# Search for KXBTC15M series
series_tickers = ['KXBTC15M', 'KXETH15M', 'KXSOL15M']

for series in series_tickers:
    print(f"\n{'='*70}")
    print(f"Series: {series}")
    print("=" * 70)

    # Get events for this series
    try:
        response = requests.get(
            f"{BASE_URL}/trade-api/v2/events",
            params={'series_ticker': series, 'status': 'open', 'limit': 50},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            events = data.get('events', [])
            print(f"Found {len(events)} events")

            for e in events:
                print(f"\n  Event: {e.get('event_ticker', 'N/A')}")
                print(f"  Title: {e.get('title', 'N/A')}")
                print(f"  Category: {e.get('category', 'N/A')}")
                print(f"  Expiration: {e.get('expected_expiration_time', 'N/A')}")

                # Get markets for this event
                event_ticker = e.get('event_ticker', '')
                if event_ticker:
                    markets_resp = requests.get(
                        f"{BASE_URL}/trade-api/v2/markets",
                        params={'event_ticker': event_ticker, 'limit': 20},
                        timeout=30
                    )
                    if markets_resp.status_code == 200:
                        markets_data = markets_resp.json()
                        markets = markets_data.get('markets', [])
                        print(f"  Markets: {len(markets)}")
                        for m in markets[:5]:
                            ticker = m.get('ticker', 'N/A')
                            title = m.get('title', 'N/A')
                            yes_bid = m.get('yes_bid', 0)
                            yes_ask = m.get('yes_ask', 0)
                            no_bid = m.get('no_bid', 0)
                            no_ask = m.get('no_ask', 0)
                            close_time = m.get('close_time', m.get('expiration_time', 'N/A'))

                            print(f"    - {ticker}")
                            print(f"      Title: {title[:60]}")
                            print(f"      Yes: {yes_bid}-{yes_ask} | No: {no_bid}-{no_ask}")
                            print(f"      Closes: {close_time}")
    except Exception as ex:
        print(f"  Error: {ex}")

# Also try to get markets directly
print("\n" + "=" * 70)
print("Direct market search for BTC15M")
print("=" * 70)

try:
    response = requests.get(
        f"{BASE_URL}/trade-api/v2/markets",
        params={'series_ticker': 'KXBTC15M', 'status': 'open', 'limit': 100},
        timeout=30
    )
    if response.status_code == 200:
        data = response.json()
        markets = data.get('markets', [])
        print(f"Found {len(markets)} markets")

        # Group by close time
        by_close = {}
        for m in markets:
            close = m.get('close_time', m.get('expiration_time', ''))[:16]
            if close not in by_close:
                by_close[close] = []
            by_close[close].append(m)

        # Show upcoming markets
        for close_time in sorted(by_close.keys())[:10]:
            print(f"\n  Close time: {close_time}")
            for m in by_close[close_time][:3]:
                ticker = m.get('ticker', 'N/A')
                subtitle = m.get('subtitle', 'N/A')
                yes_bid = m.get('yes_bid', 0)
                yes_ask = m.get('yes_ask', 0)
                print(f"    {ticker}: {subtitle} (Yes: {yes_bid}-{yes_ask})")
except Exception as ex:
    print(f"Error: {ex}")
