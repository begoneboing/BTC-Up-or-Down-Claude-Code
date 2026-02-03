"""Find Bitcoin Up or Down markets on Polymarket"""
import requests
import json

print('Searching for Bitcoin Up/Down markets...')
print('=' * 70)

# Search by tag/category
for endpoint in ['markets', 'events']:
    print(f"\nChecking Gamma API /{endpoint} endpoint...")
    try:
        response = requests.get(f'https://gamma-api.polymarket.com/{endpoint}',
            params={'active': 'true', 'limit': 500, 'order': 'volume24hr', 'ascending': 'false'},
            timeout=60)

        if response.status_code == 200:
            items = response.json()
            print(f"Found {len(items)} items")

            updown = []
            for item in items:
                # For events, markets are nested
                if endpoint == 'events':
                    title = item.get('title', '').lower()
                    if 'up or down' in title or 'updown' in title:
                        updown.append(item)
                        print(f"  Event: {item.get('title')}")
                        print(f"  Slug: {item.get('slug')}")
                        print(f"  Markets: {len(item.get('markets', []))}")
                        for m in item.get('markets', [])[:3]:
                            print(f"    - {m.get('question', 'N/A')[:60]}")
                            print(f"      Tokens: {m.get('clobTokenIds', [])}")
                        print()
                else:
                    question = item.get('question', '').lower()
                    if 'up or down' in question or 'updown' in question:
                        updown.append(item)
                        print(f"  {item.get('question')}")

            print(f"Found {len(updown)} Up/Down markets/events")

    except Exception as e:
        print(f"Error: {e}")

# Also try CLOB markets endpoint with different params
print("\nChecking CLOB /markets endpoint...")
try:
    response = requests.get('https://clob.polymarket.com/markets',
        params={'next_cursor': 'MA=='},  # Start from beginning
        timeout=60)

    if response.status_code == 200:
        data = response.json()
        markets = data if isinstance(data, list) else data.get('data', data.get('markets', []))
        print(f"Found {len(markets)} CLOB markets")

        # Look for updown markets
        for m in markets[:50]:
            question = m.get('question', '').lower()
            slug = m.get('market_slug', '').lower()
            if 'up' in question and 'down' in question:
                print(f"  Slug: {slug}")
                print(f"  Question: {m.get('question')}")
                tokens = m.get('tokens', [])
                for t in tokens:
                    print(f"    {t.get('outcome')}: {t.get('token_id')[:40]}... price={t.get('price')}")
                print()
except Exception as e:
    print(f"Error: {e}")
