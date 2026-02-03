"""Find any BTC market still open"""
import requests
import json
from datetime import datetime, timezone

print('Finding any open BTC market...')
print('=' * 70)

now = datetime.now(timezone.utc)
print(f"Current UTC time: {now.isoformat()}")

response = requests.get('https://gamma-api.polymarket.com/markets',
    params={'active': 'true', 'limit': 500, 'order': 'volume24hr', 'ascending': 'false'},
    timeout=60)

if response.status_code == 200:
    markets = response.json()
    print(f"Total active markets: {len(markets)}")

    # Find any BTC/Bitcoin market that's accepting orders and not resolved
    btc_markets = []
    for market in markets:
        question = market.get('question', '').lower()
        if 'btc' in question or 'bitcoin' in question:
            accepting = market.get('acceptingOrders', False)
            closed = market.get('closed', False)

            outcome_prices = market.get('outcomePrices', '[]')
            if isinstance(outcome_prices, str):
                outcome_prices = json.loads(outcome_prices)

            # Check if resolved
            prices_set = set(str(p) for p in outcome_prices)
            is_resolved = prices_set == {'0', '1'} or prices_set == {'1', '0'}

            if accepting and not closed and not is_resolved:
                btc_markets.append(market)

    print(f"\nOpen BTC markets (accepting orders, not resolved): {len(btc_markets)}")

    for market in btc_markets[:10]:
        question = market.get('question', '')
        slug = market.get('slug', '')
        outcome_prices = market.get('outcomePrices', '[]')
        if isinstance(outcome_prices, str):
            outcome_prices = json.loads(outcome_prices)
        outcomes = market.get('outcomes', '[]')
        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)

        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"Slug: {slug}")
        for outcome, price in zip(outcomes, outcome_prices):
            print(f"  {outcome}: {float(price):.2%}")
        print(f"Volume 24h: ${float(market.get('volume24hr', 0) or 0):,.0f}")
        print(f"End Date: {market.get('endDate')}")
