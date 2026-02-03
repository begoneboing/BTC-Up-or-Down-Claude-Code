"""Check for available crypto markets on Polymarket"""
import requests

# Check Gamma API for crypto markets
print('Checking Gamma API for active crypto markets...')
response = requests.get('https://gamma-api.polymarket.com/markets',
    params={'active': 'true', 'limit': 200, 'order': 'volume24hr', 'ascending': 'false'},
    timeout=60)

if response.status_code == 200:
    markets = response.json()

    # Filter for crypto/btc related markets
    crypto_keywords = ['btc', 'bitcoin', 'crypto', 'eth', 'ethereum', 'price']
    crypto_markets = []

    for m in markets:
        question = m.get('question', '').lower()
        slug = m.get('slug', '').lower()
        if any(kw in question or kw in slug for kw in crypto_keywords):
            crypto_markets.append({
                'question': m.get('question', '')[:100],
                'volume': m.get('volume24hr', 0),
                'liquidity': m.get('liquidity', 0),
                'slug': slug
            })

    print(f'Found {len(crypto_markets)} crypto-related markets:')
    for m in sorted(crypto_markets, key=lambda x: float(x['volume'] or 0), reverse=True)[:15]:
        print(f"  - {m['question']}")
        print(f"    Volume 24h: ${float(m['volume'] or 0):,.0f} | Liquidity: ${float(m['liquidity'] or 0):,.0f}")
        print()
else:
    print(f'Error: {response.status_code}')
