"""Debug BTC Up/Down market data"""
import requests
import json
from datetime import datetime, timezone

print('Debugging BTC Up/Down market data...')
print('=' * 70)

now = datetime.now(timezone.utc)
print(f"Current UTC time: {now.isoformat()}")

response = requests.get('https://gamma-api.polymarket.com/markets',
    params={'active': 'true', 'limit': 500, 'order': 'volume24hr', 'ascending': 'false'},
    timeout=60)

if response.status_code == 200:
    markets = response.json()

    print(f"\nTotal markets: {len(markets)}")

    # Find unresolved BTC Up/Down markets (any timeframe)
    unresolved = []
    for market in markets:
        question = market.get('question', '')
        if 'Bitcoin Up or Down' in question or ('bitcoin' in question.lower() and 'up' in question.lower()):
            outcome_prices = market.get('outcomePrices', '[]')
            if isinstance(outcome_prices, str):
                outcome_prices = json.loads(outcome_prices)

            # Check if resolved (prices are 0 and 1)
            prices_set = set(str(p) for p in outcome_prices)
            is_resolved = prices_set == {'0', '1'} or prices_set == {'1', '0'}

            closed = market.get('closed', False)
            accepting = market.get('acceptingOrders', False)

            if not is_resolved and not closed and accepting:
                unresolved.append(market)

    print(f"Unresolved, open BTC Up/Down 15-minute markets: {len(unresolved)}")

    for market in unresolved[:5]:
        question = market.get('question', '')
        slug = market.get('slug', '')
        outcome_prices = market.get('outcomePrices', '[]')
        if isinstance(outcome_prices, str):
            outcome_prices = json.loads(outcome_prices)
        outcomes = market.get('outcomes', '[]')
        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        clob_tokens = market.get('clobTokenIds', '[]')
        if isinstance(clob_tokens, str):
            clob_tokens = json.loads(clob_tokens)

        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"Slug: {slug}")
        print(f"Outcomes: {outcomes}")
        print(f"Prices: {outcome_prices}")
        print(f"CLOB Tokens: {len(clob_tokens)} tokens")
        for i, (outcome, price, token) in enumerate(zip(outcomes, outcome_prices, clob_tokens)):
            print(f"  {outcome}: price={float(price):.2%}, token={token[:40]}...")
        print(f"End Date: {market.get('endDate')}")
        print(f"Accepting Orders: {market.get('acceptingOrders')}")
        print(f"Closed: {market.get('closed')}")
