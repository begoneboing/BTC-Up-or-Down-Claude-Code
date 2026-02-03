"""
Trade a single BTC 15-minute event
"""
import os
import time
from datetime import datetime, timezone, timedelta

os.environ['TRADING_MODE'] = 'live'

from trade_executor import TradeExecutor, OrderSide

# Current market - Feb 3, 2026 5:15-5:30 PM ET (22:15-22:30 UTC)
MARKET = {
    "timestamp": 1770156900,
    "slug": "btc-updown-15m-1770156900",
    "up_token_id": "110970696200714091282179546377028274771306544419441362487821344906646930486603",
    "down_token_id": "34970941700440940246313909927249016673955624859981701672408396495889883833983",
    "up_price": 0.545,
    "down_price": 0.455
}

MAX_POSITION = 5.0  # $5 max


def get_orderbook(executor, token_id):
    """Get orderbook data"""
    import requests
    try:
        response = requests.get(
            "https://clob.polymarket.com/book",
            params={"token_id": token_id},
            timeout=30
        )
        if response.status_code == 200:
            book = response.json()
            bids = book.get('bids', [])
            asks = book.get('asks', [])
            return {
                'best_bid': float(bids[0]['price']) if bids else 0,
                'best_ask': float(asks[0]['price']) if asks else 1,
                'bid_size': float(bids[0]['size']) if bids else 0,
                'ask_size': float(asks[0]['size']) if asks else 0
            }
    except Exception as e:
        print(f"Orderbook error: {e}")
    return {'best_bid': 0, 'best_ask': 1}


def main():
    print("=" * 70)
    print("BTC 15-MINUTE SINGLE EVENT TRADE")
    print("=" * 70)

    now = datetime.now(timezone.utc)
    slot_end = datetime.fromtimestamp(MARKET["timestamp"] + 15*60, tz=timezone.utc)

    print(f"Current time: {now.strftime('%H:%M:%S')} UTC")
    print(f"Market: {MARKET['slug']}")
    print(f"Slot ends: {slot_end.strftime('%H:%M:%S')} UTC")
    print(f"Max position: ${MAX_POSITION:.2f}")
    print("=" * 70)

    executor = TradeExecutor()
    print(f"Mode: {'LIVE' if executor.is_live_mode() else 'PAPER'}")

    # Get orderbook prices
    print("\nFetching orderbook...")
    up_book = get_orderbook(executor, MARKET["up_token_id"])
    down_book = get_orderbook(executor, MARKET["down_token_id"])

    print(f"\nUP:   Bid {up_book['best_bid']:.2%} | Ask {up_book['best_ask']:.2%}")
    print(f"DOWN: Bid {down_book['best_bid']:.2%} | Ask {down_book['best_ask']:.2%}")

    # Use implied prices from the website
    up_price = MARKET["up_price"]
    down_price = MARKET["down_price"]

    print(f"\nImplied prices - UP: {up_price:.2%} | DOWN: {down_price:.2%}")

    # Decide direction - buy the cheaper side
    if down_price < up_price:
        direction = "DOWN"
        token_id = MARKET["down_token_id"]
        entry_price = down_price
    else:
        direction = "UP"
        token_id = MARKET["up_token_id"]
        entry_price = up_price

    # Calculate size
    size = MAX_POSITION / entry_price
    size = round(size, 2)

    # Place order slightly above market for better fill
    order_price = min(entry_price + 0.03, 0.99)
    total_cost = size * order_price

    print(f"\n{'='*60}")
    print(f"TRADING DECISION: {direction}")
    print(f"{'='*60}")
    print(f"Entry price: {entry_price:.2%}")
    print(f"Order price: {order_price:.2%} (limit)")
    print(f"Size: {size:.2f} shares")
    print(f"Total cost: ${total_cost:.2f}")

    # Place order
    print(f"\nPlacing BUY order...")

    order = executor.place_order(
        token_id=token_id,
        side=OrderSide.BUY,
        price=order_price,
        size=size,
        market_question=f"BTC 15M - {direction}"
    )

    if order:
        status = order.status.value.lower()
        print(f"\nOrder placed!")
        print(f"  ID: {order.order_id[:50]}...")
        print(f"  Status: {status.upper()}")

        if status in ['matched', 'filled']:
            print(f"  ORDER FILLED IMMEDIATELY!")
        else:
            print(f"  Order is resting in orderbook")
            print(f"  Will fill when matched by counterparty")

        # Monitor for a bit
        print(f"\nMonitoring order for 30 seconds...")
        time.sleep(30)

        # Check order status
        try:
            orders = executor.client.get_orders()
            our_order = next((o for o in orders if o.get('id') == order.order_id), None)
            if our_order:
                matched = float(our_order.get('size_matched', 0))
                original = float(our_order.get('original_size', size))
                print(f"\nOrder update: {matched:.2f}/{original:.2f} filled")

                if matched < original:
                    print("Order partially filled or still pending")
            else:
                print("\nOrder no longer in open orders (likely filled)")
        except Exception as e:
            print(f"Status check error: {e}")
    else:
        print("Order placement failed!")

    print("\n" + "=" * 70)
    print("TRADE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
