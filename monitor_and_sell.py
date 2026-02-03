"""
Monitor position and sell when profitable
"""
import os
import time
import requests
from datetime import datetime, timezone, timedelta

os.environ['TRADING_MODE'] = 'live'

from trade_executor import TradeExecutor, OrderSide

# Position details
POSITION = {
    "token_id": "62609370894063599815349376967571381428519616349450706969797045952167467332970",
    "side": "UP",
    "entry_price": 0.465,
    "size": 11.49,
    "market_end": 1770155100 + 15*60  # 22:00 UTC
}

CLOB_URL = "https://clob.polymarket.com"

# Trading parameters
PROFIT_TARGET = 0.03  # Sell if price rises 3% above entry
STOP_LOSS = 0.10      # Sell if price drops 10% below entry
CHECK_INTERVAL = 5    # Seconds between checks


def get_orderbook(token_id):
    """Get current orderbook"""
    try:
        response = requests.get(
            f"{CLOB_URL}/book",
            params={"token_id": token_id},
            timeout=10
        )
        if response.status_code == 200:
            book = response.json()
            bids = book.get('bids', [])
            asks = book.get('asks', [])

            best_bid = float(bids[0]['price']) if bids else 0
            best_ask = float(asks[0]['price']) if asks else 1
            bid_size = float(bids[0]['size']) if bids else 0

            return {
                'best_bid': best_bid,
                'best_ask': best_ask,
                'mid': (best_bid + best_ask) / 2 if best_bid > 0 else best_ask,
                'bid_size': bid_size
            }
    except Exception as e:
        print(f"Error: {e}")
    return None


def main():
    print("=" * 70)
    print("POSITION MONITOR - SELL ON PROFIT")
    print("=" * 70)

    entry_price = POSITION["entry_price"]
    size = POSITION["size"]
    token_id = POSITION["token_id"]
    market_end = datetime.fromtimestamp(POSITION["market_end"], tz=timezone.utc)

    print(f"Position: {POSITION['side']}")
    print(f"Entry: {entry_price:.2%}")
    print(f"Size: {size:.2f} shares")
    print(f"Cost basis: ${entry_price * size:.2f}")
    print(f"Market ends: {market_end.strftime('%H:%M:%S')} UTC")
    print(f"Profit target: +{PROFIT_TARGET:.0%} ({entry_price + PROFIT_TARGET:.2%})")
    print(f"Stop loss: -{STOP_LOSS:.0%} ({entry_price - STOP_LOSS:.2%})")
    print("=" * 70)

    executor = TradeExecutor()

    sell_placed = False
    iteration = 0

    while True:
        now = datetime.now(timezone.utc)
        time_left = (market_end - now).total_seconds()

        if time_left < 60:  # Less than 1 minute to resolution
            print(f"\n[{now.strftime('%H:%M:%S')}] Market closing soon - stopping monitor")
            break

        # Get current prices
        book = get_orderbook(token_id)

        if book:
            current_bid = book['best_bid']
            current_ask = book['best_ask']

            # Calculate P&L based on bid (what we could sell at)
            if current_bid > 0.01:  # Valid bid
                pnl_pct = (current_bid - entry_price) / entry_price
                pnl_dollars = (current_bid - entry_price) * size

                status = "PROFIT" if pnl_dollars > 0 else "LOSS"

                print(f"[{now.strftime('%H:%M:%S')}] Bid: {current_bid:.2%} | P&L: ${pnl_dollars:+.2f} ({pnl_pct:+.1%}) | {status} | {time_left/60:.1f}m left")

                # Check profit target
                if current_bid >= entry_price + PROFIT_TARGET and not sell_placed:
                    print(f"\n*** PROFIT TARGET HIT - SELLING ***")
                    sell_price = current_bid - 0.01  # Slightly below bid for fill

                    order = executor.place_order(
                        token_id=token_id,
                        side=OrderSide.SELL,
                        price=sell_price,
                        size=size,
                        market_question=f"BTC 15M - {POSITION['side']} TAKE PROFIT"
                    )

                    if order:
                        print(f"SELL order placed: {size:.2f} @ {sell_price:.2%}")
                        print(f"Order ID: {order.order_id[:40]}...")
                        print(f"Status: {order.status.value}")
                        sell_placed = True

                        if order.status.value.lower() in ['matched', 'filled']:
                            realized_pnl = (sell_price - entry_price) * size
                            print(f"\n*** SOLD! Realized P&L: ${realized_pnl:+.2f} ***")
                            break

                # Check stop loss
                elif current_bid <= entry_price - STOP_LOSS and not sell_placed:
                    print(f"\n*** STOP LOSS HIT - SELLING ***")
                    sell_price = current_bid - 0.01

                    order = executor.place_order(
                        token_id=token_id,
                        side=OrderSide.SELL,
                        price=sell_price,
                        size=size,
                        market_question=f"BTC 15M - {POSITION['side']} STOP LOSS"
                    )

                    if order:
                        print(f"SELL order placed: {size:.2f} @ {sell_price:.2%}")
                        sell_placed = True

                        if order.status.value.lower() in ['matched', 'filled']:
                            realized_pnl = (sell_price - entry_price) * size
                            print(f"\n*** SOLD! Realized P&L: ${realized_pnl:+.2f} ***")
                            break
            else:
                print(f"[{now.strftime('%H:%M:%S')}] Wide spread - Bid: {current_bid:.2%} Ask: {current_ask:.2%} | {time_left/60:.1f}m left")

        iteration += 1
        time.sleep(CHECK_INTERVAL)

    # Final status
    print("\n" + "=" * 70)
    if sell_placed:
        print("MONITORING COMPLETE - Position sold")
    else:
        print("MONITORING COMPLETE - Position held to resolution")
        print(f"Market resolves at {market_end.strftime('%H:%M:%S')} UTC")
        print("Check Polymarket for resolution outcome")
    print("=" * 70)


if __name__ == "__main__":
    main()
