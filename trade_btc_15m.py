"""
Direct BTC 15-Minute Trading Script
Uses WebFetch to get current market data and trades consecutive events
"""

import os
import re
import json
import time
import requests
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

# Set live trading mode
os.environ['TRADING_MODE'] = 'live'

from trade_executor import TradeExecutor, OrderSide

CLOB_URL = "https://clob.polymarket.com"


@dataclass
class BTCMarket:
    """BTC 15-minute market data"""
    slug: str
    up_token_id: str
    down_token_id: str
    up_price: float
    down_price: float
    condition_id: str
    volume: float
    liquidity: float
    end_time: str


def get_btc_15m_market_url() -> str:
    """Generate the URL for current BTC 15-minute market"""
    now = datetime.now(timezone.utc)
    # Round to current/next 15-minute slot
    minute_slot = (now.minute // 15) * 15
    slot_time = now.replace(minute=minute_slot, second=0, microsecond=0)

    # If we're past the slot, get next one
    if now.minute % 15 > 12:  # Within 3 minutes of end
        slot_time += timedelta(minutes=15)

    timestamp = int(slot_time.timestamp())
    return f"https://polymarket.com/event/btc-updown-15m-{timestamp}"


def fetch_market_data_manual(timestamp: int) -> Optional[BTCMarket]:
    """
    Fetch market data by constructing the expected slug and
    searching the CLOB API
    """
    event_slug = f"btc-updown-15m-{timestamp}"
    print(f"Searching for market: {event_slug}")

    try:
        # Search in markets endpoint
        response = requests.get(
            f"{CLOB_URL}/markets",
            timeout=60
        )
        if response.status_code == 200:
            markets = response.json()
            if not isinstance(markets, list):
                markets = markets.get('data', markets.get('markets', []))

            for m in markets:
                slug = m.get('market_slug', '').lower()
                if 'btc' in slug and ('updown' in slug or 'up-down' in slug or '15m' in slug):
                    tokens = m.get('tokens', [])
                    if len(tokens) >= 2:
                        # Find UP and DOWN tokens
                        up_token = None
                        down_token = None
                        for t in tokens:
                            outcome = t.get('outcome', '').lower()
                            if outcome == 'up' or outcome == 'yes':
                                up_token = t
                            elif outcome == 'down' or outcome == 'no':
                                down_token = t

                        if not up_token:
                            up_token = tokens[0]
                        if not down_token:
                            down_token = tokens[1]

                        return BTCMarket(
                            slug=m.get('market_slug', ''),
                            up_token_id=up_token.get('token_id', ''),
                            down_token_id=down_token.get('token_id', ''),
                            up_price=float(up_token.get('price', 0.5)),
                            down_price=float(down_token.get('price', 0.5)),
                            condition_id=m.get('condition_id', ''),
                            volume=0,
                            liquidity=0,
                            end_time=m.get('end_date_iso', '')
                        )

        # Also check sampling-markets
        response = requests.get(
            f"{CLOB_URL}/sampling-markets",
            params={"limit": 2000},
            timeout=60
        )
        if response.status_code == 200:
            data = response.json()
            markets = data.get('data', [])

            for m in markets:
                slug = m.get('market_slug', '').lower()
                question = m.get('question', '').lower()

                if ('btc' in slug or 'bitcoin' in question) and ('up' in question and 'down' in question):
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
                            return BTCMarket(
                                slug=m.get('market_slug', ''),
                                up_token_id=up_token.get('token_id', ''),
                                down_token_id=down_token.get('token_id', ''),
                                up_price=float(up_token.get('price', 0.5)),
                                down_price=float(down_token.get('price', 0.5)),
                                condition_id=m.get('condition_id', ''),
                                volume=float(m.get('volume', 0) or 0),
                                liquidity=float(m.get('liquidity', 0) or 0),
                                end_time=m.get('end_date_iso', '')
                            )

    except Exception as e:
        print(f"Error searching markets: {e}")

    return None


def get_live_prices(executor: TradeExecutor, market: BTCMarket) -> Tuple[float, float]:
    """Get live orderbook prices"""
    up_data = executor.get_market_price(market.up_token_id)
    down_data = executor.get_market_price(market.down_token_id)

    up_price = up_data.get('mid_price', market.up_price)
    down_price = down_data.get('mid_price', market.down_price)

    return up_price, down_price


def decide_trade(up_price: float, down_price: float) -> Tuple[str, float, str]:
    """
    Decide which direction to trade based on prices

    For volatility trading on 15-minute markets:
    - Trade the undervalued side
    - In a fair market, both should be ~50%
    - If one side is significantly cheaper, it may be undervalued

    Returns: (direction, price, token_to_buy)
    """
    # Simple strategy: buy the cheaper side
    if up_price < down_price:
        return "UP", up_price, "up"
    else:
        return "DOWN", down_price, "down"


def trade_event(
    executor: TradeExecutor,
    market: BTCMarket,
    max_position: float = 5.0,
    event_num: int = 1
) -> Optional[Dict]:
    """
    Trade a single BTC 15-minute event

    Args:
        executor: Trade executor instance
        market: Market data
        max_position: Maximum $ to risk
        event_num: Event number for logging
    """
    print(f"\n{'='*60}")
    print(f"EVENT {event_num}: {market.slug}")
    print(f"{'='*60}")

    # Get live prices
    up_price, down_price = get_live_prices(executor, market)
    print(f"Live Prices - UP: {up_price:.2%} | DOWN: {down_price:.2%}")

    # Decide trade
    direction, price, side = decide_trade(up_price, down_price)
    print(f"Decision: {direction} @ {price:.2%}")

    # Calculate size
    token_id = market.up_token_id if side == "up" else market.down_token_id
    size = max_position / price if price > 0.01 else 0
    size = round(size, 2)

    if size < 1:
        print(f"Size too small: {size}")
        return None

    total = size * price
    print(f"Order: BUY {size:.2f} shares @ {price:.2%} = ${total:.2f}")

    # Place order (slightly above market for better fill)
    order_price = min(price + 0.02, 0.99)

    order = executor.place_order(
        token_id=token_id,
        side=OrderSide.BUY,
        price=order_price,
        size=size,
        market_question=f"BTC 15M - {direction}"
    )

    if order:
        return {
            "event": event_num,
            "market": market.slug,
            "direction": direction,
            "price": price,
            "size": size,
            "total": total,
            "order_id": order.order_id,
            "status": order.status.value
        }

    return None


def run_btc_15m_trading(
    num_events: int = 3,
    max_position: float = 5.0,
    wait_minutes: int = 15
):
    """
    Run BTC 15-minute volatility trading

    Args:
        num_events: Number of consecutive events to trade
        max_position: Maximum $ per position
        wait_minutes: Minutes to wait between events
    """
    print("=" * 70)
    print("BTC 15-MINUTE VOLATILITY TRADING")
    print("=" * 70)
    print(f"Max Position: ${max_position:.2f}")
    print(f"Events to Trade: {num_events}")
    print(f"Wait Between Events: {wait_minutes} minutes")
    print("=" * 70)

    executor = TradeExecutor()
    print(f"\nMode: {'LIVE' if executor.is_live_mode() else 'PAPER'}")

    trades = []

    for event_num in range(1, num_events + 1):
        print(f"\n{'#'*70}")
        print(f"# Looking for Event {event_num} of {num_events}")
        print(f"{'#'*70}")

        # Get current timestamp for market
        now = datetime.now(timezone.utc)
        minute_slot = (now.minute // 15) * 15
        slot_time = now.replace(minute=minute_slot, second=0, microsecond=0)
        timestamp = int(slot_time.timestamp())

        # Try to find market
        market = fetch_market_data_manual(timestamp)

        if not market:
            print("No active BTC 15M market found via API")
            print("Markets may be listed differently or require direct access")

            # Wait and try again if more events remaining
            if event_num < num_events:
                print(f"\nWaiting {wait_minutes} minutes for next slot...")
                time.sleep(wait_minutes * 60)
            continue

        # Trade the event
        trade = trade_event(executor, market, max_position, event_num)

        if trade:
            trades.append(trade)
            print(f"\nTrade completed: {trade['direction']} ${trade['total']:.2f}")

        # Wait for next event
        if event_num < num_events:
            print(f"\nWaiting {wait_minutes} minutes for next event...")
            time.sleep(wait_minutes * 60)

    # Summary
    print("\n" + "=" * 70)
    print("TRADING SESSION COMPLETE")
    print("=" * 70)
    print(f"Events Attempted: {num_events}")
    print(f"Trades Executed: {len(trades)}")

    if trades:
        total_invested = sum(t['total'] for t in trades)
        print(f"Total Invested: ${total_invested:.2f}")

        print("\nTrade Details:")
        for t in trades:
            print(f"  Event {t['event']}: {t['direction']} ${t['total']:.2f} - {t['status']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BTC 15-Minute Trading")
    parser.add_argument("--events", type=int, default=3, help="Number of events")
    parser.add_argument("--max-size", type=float, default=5.0, help="Max position $")
    parser.add_argument("--wait", type=int, default=15, help="Wait minutes")

    args = parser.parse_args()

    run_btc_15m_trading(
        num_events=args.events,
        max_position=args.max_size,
        wait_minutes=args.wait
    )
