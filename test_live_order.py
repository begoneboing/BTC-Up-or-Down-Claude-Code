"""Test live order placement"""
import os
import json
import requests

os.environ['TRADING_MODE'] = 'live'

from trade_executor import TradeExecutor, OrderSide

# Get a market to test with
print('Fetching active market...')
url = 'https://gamma-api.polymarket.com/markets'
params = {'limit': 10, 'active': 'true', 'closed': 'false'}
response = requests.get(url, params=params, timeout=30)
markets = response.json()

# Find one with good liquidity and valid token
test_market = None
for market in markets:
    clob_tokens = market.get('clobTokenIds', '[]')
    if isinstance(clob_tokens, str):
        clob_tokens = json.loads(clob_tokens)

    if clob_tokens:
        question = market.get('question', 'Unknown')[:60]
        volume = float(market.get('volume', 0) or 0)
        liquidity = float(market.get('liquidity', 0) or 0)

        if volume > 10000 and liquidity > 1000:
            test_market = {
                'question': question,
                'token_id': clob_tokens[0],
                'volume': volume,
                'liquidity': liquidity
            }
            break

if not test_market:
    print('No suitable market found')
    exit(1)

print(f"Market: {test_market['question']}")
print(f"Token ID: {test_market['token_id']}")
print(f"Volume: ${test_market['volume']:,.0f} | Liquidity: ${test_market['liquidity']:,.0f}")

# Get current price
book_url = 'https://clob.polymarket.com/book'
book_resp = requests.get(book_url, params={'token_id': test_market['token_id']}, timeout=30)
book = book_resp.json()

bids = book.get('bids', [])
asks = book.get('asks', [])

if not asks:
    print('No asks in orderbook')
    exit(1)

best_ask = float(asks[0]['price'])
best_bid = float(bids[0]['price']) if bids else 0
print(f"Best Bid: {best_bid:.4f} | Best Ask: {best_ask:.4f}")

print()
print('Initializing executor...')
executor = TradeExecutor()

# Place a small limit buy order below market
price = max(best_bid + 0.001, 0.01)  # Slightly above best bid (likely won't fill immediately)
size = min(1.0 / price, 5)  # ~$1 worth, max 5 shares

print()
print(f'Attempting order: BUY {size:.2f} shares @ {price:.4f} (${size * price:.2f})')
print('=' * 60)

order = executor.place_order(
    token_id=test_market['token_id'],
    side=OrderSide.BUY,
    price=price,
    size=size,
    market_question=test_market['question']
)

print('=' * 60)
if order:
    print(f'Order ID: {order.order_id}')
    print(f'Status: {order.status.value}')
else:
    print('Order failed')
