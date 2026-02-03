"""Find BTC 15-minute Up/Down markets on Polymarket"""
import requests
import json
from datetime import datetime, timezone

print('Searching for BTC 15-minute Up/Down markets...')
print('=' * 70)

# Try sampling-markets endpoint (usually has more current markets)
print("\nChecking CLOB sampling-markets endpoint...")
response = requests.get('https://clob.polymarket.com/sampling-markets',
    params={'limit': 500},
    timeout=60)

if response.status_code == 200:
    data = response.json()
    sampling_markets = data if isinstance(data, list) else data.get('data', [])
    print(f"Found {len(sampling_markets)} sampling markets")

    btc_sampling = [m for m in sampling_markets if 'btc' in str(m).lower() or 'bitcoin' in str(m).lower()]
    print(f"BTC-related sampling markets: {len(btc_sampling)}")
    for m in btc_sampling[:5]:
        print(f"  Slug: {m.get('market_slug', 'N/A')}")
        print(f"  Question: {m.get('question', 'N/A')}")
        tokens = m.get('tokens', [])
        if tokens:
            print(f"  Tokens: {len(tokens)}")
            for t in tokens[:2]:
                print(f"    - {t.get('outcome', 'N/A')}: {t.get('token_id', 'N/A')[:30]}...")
        print()

print("\nChecking Gamma API events endpoint...")
response = requests.get('https://gamma-api.polymarket.com/events',
    params={'active': 'true', 'limit': 100, 'order': 'volume24hr', 'ascending': 'false'},
    timeout=60)

if response.status_code == 200:
    markets = response.json()

    updown_markets = []
    for m in markets:
        question = m.get('question', '')
        slug = m.get('slug', '')
        # Look for Bitcoin Up or Down markets (any time frame)
        if 'Bitcoin Up or Down' in question or 'up or down' in question.lower():
            updown_markets.append(m)

    # Also show all questions for debugging
    print(f"Total markets: {len(markets)}")
    btc_questions = [m.get('question', '') for m in markets if 'btc' in m.get('question', '').lower() or 'bitcoin' in m.get('question', '').lower()]
    print(f"BTC-related questions found: {len(btc_questions)}")
    for q in btc_questions[:10]:
        print(f"  - {q}")

    print(f'Found {len(updown_markets)} potential 15-minute markets:\n')

    for m in sorted(updown_markets, key=lambda x: x.get('question', ''))[:20]:
        print(f"Question: {m.get('question')}")
        print(f"Slug: {m.get('slug')}")
        print(f"Condition ID: {m.get('conditionId')}")
        print(f"CLOB Token IDs: {m.get('clobTokenIds')}")
        print(f"Outcomes: {m.get('outcomes')}")
        print(f"End Date: {m.get('endDate')}")
        print(f"Volume 24h: ${float(m.get('volume24hr', 0) or 0):,.0f}")
        print()

else:
    print(f'Error: {response.status_code}')
