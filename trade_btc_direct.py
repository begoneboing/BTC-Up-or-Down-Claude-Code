"""
Direct BTC 15-Minute Trading with Known Token IDs
Trades consecutive 15-minute Bitcoin Up/Down markets on Polymarket
"""

import os
import time
import requests
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import Optional, List, Dict

# Set live trading mode
os.environ['TRADING_MODE'] = 'live'

from trade_executor import TradeExecutor, OrderSide

CLOB_URL = "https://clob.polymarket.com"


@dataclass
class BTCMarketSlot:
    """A single 15-minute BTC market slot"""
    timestamp: int
    up_token_id: str
    down_token_id: str
    condition_id: str
    up_price: float
    down_price: float
    end_time: str


def fetch_market_from_polymarket(timestamp: int) -> Optional[BTCMarketSlot]:
    """
    Fetch market data for a specific 15-minute slot

    The BTC 15M markets follow the pattern:
    - URL: polymarket.com/event/btc-updown-15m-{timestamp}
    - Markets resolve based on Chainlink BTC/USD price
    """
    import subprocess
    import json

    # Use curl to fetch page data (more reliable than requests for some pages)
    url = f"https://polymarket.com/event/btc-updown-15m-{timestamp}"

    try:
        # Try to get the market via the API
        response = requests.get(
            f"{CLOB_URL}/markets",
            params={"limit": 500},
            timeout=60
        )

        if response.status_code == 200:
            markets = response.json()

            # Search for BTC updown market
            for m in markets:
                slug = str(m.get('market_slug', '')).lower()
                question = str(m.get('question', '')).lower()

                # Check if this is a BTC up/down market
                if ('btc' in slug or 'bitcoin' in question) and ('up' in question or 'down' in question):
                    tokens = m.get('tokens', [])
                    if len(tokens) >= 2:
                        up_token = None
                        down_token = None

                        for t in tokens:
                            outcome = t.get('outcome', '').lower()
                            if 'up' in outcome:
                                up_token = t
                            elif 'down' in outcome:
                                down_token = t

                        if up_token and down_token:
                            return BTCMarketSlot(
                                timestamp=timestamp,
                                up_token_id=up_token.get('token_id', ''),
                                down_token_id=down_token.get('token_id', ''),
                                condition_id=m.get('condition_id', ''),
                                up_price=float(up_token.get('price', 0.5)),
                                down_price=float(down_token.get('price', 0.5)),
                                end_time=m.get('end_date_iso', '')
                            )

    except Exception as e:
        print(f"API fetch error: {e}")

    return None


def get_current_slot_timestamp() -> int:
    """Get the timestamp for the current 15-minute slot"""
    now = datetime.now(timezone.utc)
    minute_slot = (now.minute // 15) * 15
    slot_time = now.replace(minute=minute_slot, second=0, microsecond=0)
    return int(slot_time.timestamp())


def get_next_slots(count: int = 3) -> List[int]:
    """Get timestamps for the next N 15-minute slots"""
    base = get_current_slot_timestamp()
    return [base + (i * 15 * 60) for i in range(count)]


def get_live_prices(executor: TradeExecutor, market: BTCMarketSlot) -> tuple:
    """Get live orderbook prices for both outcomes"""
    up_data = executor.get_market_price(market.up_token_id)
    down_data = executor.get_market_price(market.down_token_id)

    up_price = up_data.get('mid_price', market.up_price)
    down_price = down_data.get('mid_price', market.down_price)
    up_spread = up_data.get('spread', 0.1)
    down_spread = down_data.get('spread', 0.1)

    return up_price, down_price, up_spread, down_spread


def decide_trade(up_price: float, down_price: float) -> tuple:
    """
    Decide trading direction based on market prices

    Volatility trading strategy:
    - In efficient markets, 15-minute BTC movements are nearly random
    - Trade the underpriced side (lower price = higher potential return)
    - Prefer the side with more edge (further from 50%)
    """
    # Calculate edge for each side
    # Edge = how far from fair value (50%)
    up_edge = 0.5 - up_price if up_price < 0.5 else 0
    down_edge = 0.5 - down_price if down_price < 0.5 else 0

    if up_edge > down_edge and up_price < 0.5:
        return "UP", up_price
    elif down_edge > up_edge and down_price < 0.5:
        return "DOWN", down_price
    elif up_price < down_price:
        return "UP", up_price
    else:
        return "DOWN", down_price


def trade_slot(
    executor: TradeExecutor,
    market: BTCMarketSlot,
    max_position: float,
    slot_num: int
) -> Optional[Dict]:
    """
    Trade a single 15-minute slot

    Args:
        executor: Trade executor
        market: Market data for this slot
        max_position: Max $ to invest
        slot_num: Slot number for logging
    """
    print(f"\n{'='*60}")
    print(f"SLOT {slot_num}: Timestamp {market.timestamp}")
    print(f"{'='*60}")

    # Get live prices
    up_price, down_price, up_spread, down_spread = get_live_prices(executor, market)

    print(f"UP:   {up_price:.2%} (spread: {up_spread:.2%})")
    print(f"DOWN: {down_price:.2%} (spread: {down_spread:.2%})")

    # Decide direction
    direction, price = decide_trade(up_price, down_price)
    print(f"\nDecision: {direction} @ {price:.2%}")

    # Select token
    token_id = market.up_token_id if direction == "UP" else market.down_token_id

    # Calculate size
    if price < 0.01:
        print("Price too low, skipping")
        return None

    size = max_position / price
    size = round(size, 2)

    if size < 1:
        print(f"Size {size} too small, need at least 1 share")
        return None

    total_cost = size * price
    print(f"Size: {size:.2f} shares = ${total_cost:.2f}")

    # Place order slightly above market for better fill
    order_price = min(price + 0.02, 0.99)

    print(f"\nPlacing order: BUY {size:.2f} @ {order_price:.2%}")

    order = executor.place_order(
        token_id=token_id,
        side=OrderSide.BUY,
        price=order_price,
        size=size,
        market_question=f"BTC 15M Slot {slot_num} - {direction}"
    )

    if order:
        result = {
            "slot": slot_num,
            "timestamp": market.timestamp,
            "direction": direction,
            "price": price,
            "order_price": order_price,
            "size": size,
            "cost": total_cost,
            "order_id": order.order_id,
            "status": order.status.value
        }
        print(f"Order placed: {order.order_id[:30]}... Status: {order.status.value}")
        return result

    print("Order failed")
    return None


def run_consecutive_trades(
    num_events: int = 3,
    max_position: float = 5.0,
    known_market: Optional[BTCMarketSlot] = None
):
    """
    Trade consecutive 15-minute BTC slots

    Args:
        num_events: Number of slots to trade
        max_position: Max $ per trade
        known_market: Optional pre-fetched market data
    """
    print("=" * 70)
    print("BTC 15-MINUTE VOLATILITY TRADING")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Max Position: ${max_position:.2f}")
    print(f"  Events to Trade: {num_events}")
    print(f"  Wait Between: 15 minutes")
    print("=" * 70)

    executor = TradeExecutor()
    print(f"\nTrading Mode: {'LIVE' if executor.is_live_mode() else 'PAPER'}")

    trades = []
    slots = get_next_slots(num_events)

    for i, timestamp in enumerate(slots):
        slot_num = i + 1
        print(f"\n{'#'*70}")
        print(f"# TRADING SLOT {slot_num} of {num_events}")
        print(f"# Timestamp: {timestamp}")
        print(f"{'#'*70}")

        # Use known market if this is the first slot and we have data
        if i == 0 and known_market:
            market = known_market
        else:
            # Try to fetch market data
            market = fetch_market_from_polymarket(timestamp)

        if not market:
            print("Could not fetch market data")
            print("Using hardcoded current slot data...")

            # Fallback to current known market (from WebFetch)
            # These are the token IDs from the 8:30-8:45 PM slot
            market = BTCMarketSlot(
                timestamp=timestamp,
                up_token_id="2101956270576127954433938149262890418245500877709093189920908410411794301091",
                down_token_id="61335021091704818946128859302313422733343804902719048213575666327525223828848",
                condition_id="0xa378aa50db31280559124afcaf6e17dd7b3e488f90a29f0c5ab40fa6583188fc",
                up_price=0.415,
                down_price=0.585,
                end_time=""
            )

        # Trade this slot
        result = trade_slot(executor, market, max_position, slot_num)

        if result:
            trades.append(result)

        # Wait for next slot if not the last one
        if slot_num < num_events:
            wait_seconds = 15 * 60
            print(f"\nWaiting {wait_seconds // 60} minutes for next slot...")
            time.sleep(wait_seconds)

    # Summary
    print("\n" + "=" * 70)
    print("TRADING SESSION COMPLETE")
    print("=" * 70)
    print(f"Slots Attempted: {num_events}")
    print(f"Trades Executed: {len(trades)}")

    if trades:
        total_invested = sum(t['cost'] for t in trades)
        print(f"Total Invested: ${total_invested:.2f}")

        print("\nTrade Details:")
        for t in trades:
            print(f"  Slot {t['slot']}: {t['direction']:4} {t['size']:.1f} @ {t['price']:.2%} = ${t['cost']:.2f} [{t['status']}]")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="BTC 15-Minute Volatility Trading")
    parser.add_argument("--events", type=int, default=3, help="Number of 15-minute events to trade")
    parser.add_argument("--max-size", type=float, default=5.0, help="Maximum position size in dollars")

    args = parser.parse_args()

    # Run trading
    run_consecutive_trades(
        num_events=args.events,
        max_position=args.max_size
    )


if __name__ == "__main__":
    main()
