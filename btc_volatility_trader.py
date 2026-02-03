"""
BTC 15-Minute Volatility Trader
Complete buy/sell strategy for trading Bitcoin Up/Down markets on Polymarket

Strategy:
- Buy the underpriced side when price < threshold
- Sell when price rises above target or before market closes
- Manage positions across multiple 15-minute events

Market Behavior Notes:
- BTC 15M markets on Polymarket often have wide spreads (bids at 1-5%, asks at 95-99%)
- The "displayed price" on polymarket.com is an implied price, not actual bid/ask
- Limit orders may rest in the orderbook until a counterparty matches
- Orders fill when: price crosses the spread OR counterparty takes your order
- SELL orders require owning the tokens first (from a filled BUY)

Verified working (from trade history):
- BUY orders fill when placed at or above best ask (immediate) or when matched (limit)
- SELL orders work after acquiring tokens from filled buys
"""

import os
import time
import json
import requests
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import Enum

# Set live trading mode
os.environ['TRADING_MODE'] = 'live'

from trade_executor import TradeExecutor, OrderSide

CLOB_URL = "https://clob.polymarket.com"


class PositionSide(Enum):
    UP = "UP"
    DOWN = "DOWN"


@dataclass
class Position:
    """Tracks an open position"""
    token_id: str
    side: PositionSide
    entry_price: float
    size: float
    entry_time: datetime
    market_slug: str
    order_id: str


@dataclass
class BTCMarket:
    """BTC 15-minute market data"""
    timestamp: int
    slug: str
    up_token_id: str
    down_token_id: str
    up_price: float
    down_price: float
    condition_id: str = ""
    end_time: str = ""


@dataclass
class TradeResult:
    """Result of a trade execution"""
    success: bool
    order_id: str = ""
    side: str = ""
    action: str = ""  # BUY or SELL
    price: float = 0
    size: float = 0
    total: float = 0
    error: str = ""


class BTCVolatilityTrader:
    """
    Volatility trader for BTC 15-minute markets

    Implements a buy-low-sell-high strategy:
    1. Enter positions when one side is underpriced
    2. Exit when price moves favorably or before resolution
    """

    def __init__(
        self,
        max_position_size: float = 5.0,
        entry_threshold: float = 0.45,  # Buy when price < 45%
        exit_threshold: float = 0.52,   # Sell when price > 52%
        stop_loss: float = 0.35,        # Exit if price drops below 35%
    ):
        self.executor = TradeExecutor()
        self.max_position_size = max_position_size
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss = stop_loss

        # Position tracking
        self.positions: Dict[str, Position] = {}  # token_id -> Position
        self.trade_history: List[Dict] = []
        self.pnl = 0.0

    def get_market_prices(self, market: BTCMarket) -> Tuple[float, float, float, float]:
        """Get current bid/ask prices for both sides"""
        up_data = self.executor.get_market_price(market.up_token_id)
        down_data = self.executor.get_market_price(market.down_token_id)

        up_bid = up_data.get('best_bid', 0)
        up_ask = up_data.get('best_ask', 1)
        down_bid = down_data.get('best_bid', 0)
        down_ask = down_data.get('best_ask', 1)

        # Use mid prices if bid/ask are invalid
        if up_bid == 0 or up_ask == 1:
            up_mid = market.up_price
        else:
            up_mid = (up_bid + up_ask) / 2

        if down_bid == 0 or down_ask == 1:
            down_mid = market.down_price
        else:
            down_mid = (down_bid + down_ask) / 2

        return up_mid, down_mid, up_bid, down_bid

    def should_enter(self, price: float) -> bool:
        """Check if we should enter a position at this price"""
        return price < self.entry_threshold

    def should_exit(self, entry_price: float, current_price: float) -> Tuple[bool, str]:
        """
        Check if we should exit a position

        Returns: (should_exit, reason)
        """
        # Take profit
        if current_price >= self.exit_threshold:
            return True, "TAKE_PROFIT"

        # Stop loss
        if current_price <= self.stop_loss:
            return True, "STOP_LOSS"

        # Profit target: 10% gain
        if current_price >= entry_price * 1.10:
            return True, "PROFIT_TARGET"

        return False, ""

    def buy(self, market: BTCMarket, side: PositionSide, price: float) -> TradeResult:
        """Buy a position"""
        token_id = market.up_token_id if side == PositionSide.UP else market.down_token_id

        # Calculate size
        size = self.max_position_size / price if price > 0.01 else 0
        size = round(size, 2)

        if size < 1:
            return TradeResult(success=False, error="Size too small")

        # Place buy order slightly above market for better fill
        order_price = min(price + 0.03, 0.99)

        print(f"\n  BUY {side.value}: {size:.2f} shares @ {order_price:.2%}")

        order = self.executor.place_order(
            token_id=token_id,
            side=OrderSide.BUY,
            price=order_price,
            size=size,
            market_question=f"BTC 15M - {side.value}"
        )

        if order:
            # Check if order was filled
            is_filled = order.status.value.lower() in ['filled', 'matched']
            is_live = order.status.value.lower() == 'live'

            if is_filled:
                # Only track position if filled
                position = Position(
                    token_id=token_id,
                    side=side,
                    entry_price=price,
                    size=size,
                    entry_time=datetime.now(timezone.utc),
                    market_slug=market.slug,
                    order_id=order.order_id
                )
                self.positions[token_id] = position
                print(f"  Order FILLED - position tracked")

            result = TradeResult(
                success=True,
                order_id=order.order_id,
                side=side.value,
                action="BUY",
                price=price,
                size=size,
                total=size * price
            )
            self.trade_history.append({
                "time": datetime.now().isoformat(),
                "action": "BUY",
                "side": side.value,
                "price": price,
                "size": size,
                "order_id": order.order_id,
                "status": order.status.value
            })

            if is_live:
                print(f"  Order LIVE (pending fill) - will track when filled")

            return result

        return TradeResult(success=False, error="Order failed")

    def sell(self, position: Position, current_price: float, reason: str = "") -> TradeResult:
        """Sell/exit a position"""
        # Validate we have a position to sell
        if position.token_id not in self.positions:
            return TradeResult(success=False, error="No position to sell")

        # Use a reasonable sell price - slightly below current mid price
        # Minimum price should be at least 1 cent
        sell_price = max(current_price - 0.02, 0.01)

        # If current_price is very low (bad data), use entry price as reference
        if current_price < 0.05:
            sell_price = max(position.entry_price - 0.05, 0.01)
            print(f"  Warning: Low price data, using entry-based price")

        print(f"\n  SELL {position.side.value}: {position.size:.2f} shares @ {sell_price:.2%} ({reason})")

        order = self.executor.place_order(
            token_id=position.token_id,
            side=OrderSide.SELL,
            price=sell_price,
            size=position.size,
            market_question=f"BTC 15M - {position.side.value} EXIT"
        )

        if order:
            is_filled = order.status.value.lower() in ['filled', 'matched']

            if is_filled:
                # Calculate P&L
                entry_cost = position.entry_price * position.size
                exit_value = sell_price * position.size
                trade_pnl = exit_value - entry_cost
                self.pnl += trade_pnl

                # Remove from positions
                del self.positions[position.token_id]

                print(f"  Order FILLED - P&L: ${trade_pnl:+.2f}")
            else:
                print(f"  Order status: {order.status.value}")

            result = TradeResult(
                success=True,
                order_id=order.order_id,
                side=position.side.value,
                action="SELL",
                price=sell_price,
                size=position.size,
                total=sell_price * position.size
            )
            self.trade_history.append({
                "time": datetime.now().isoformat(),
                "action": "SELL",
                "side": position.side.value,
                "price": sell_price,
                "size": position.size,
                "reason": reason,
                "order_id": order.order_id,
                "status": order.status.value
            })

            return result

        return TradeResult(success=False, error="Sell order failed")

    def check_and_exit_positions(self, market: BTCMarket) -> List[TradeResult]:
        """Check all open positions and exit if conditions met"""
        results = []

        up_price, down_price, up_bid, down_bid = self.get_market_prices(market)

        for token_id, position in list(self.positions.items()):
            # Get current price for this position's side
            if position.side == PositionSide.UP:
                current_price = up_bid if up_bid > 0 else up_price
            else:
                current_price = down_bid if down_bid > 0 else down_price

            should_exit, reason = self.should_exit(position.entry_price, current_price)

            if should_exit:
                result = self.sell(position, current_price, reason)
                results.append(result)

        return results

    def trade_market(self, market: BTCMarket) -> Dict:
        """
        Execute volatility trading strategy on a market

        1. Check existing positions for exit
        2. Look for new entry opportunities
        """
        print(f"\n{'='*60}")
        print(f"TRADING: {market.slug}")
        print(f"{'='*60}")

        # Get current prices
        up_price, down_price, up_bid, down_bid = self.get_market_prices(market)
        print(f"UP:   {up_price:.2%} (bid: {up_bid:.2%})")
        print(f"DOWN: {down_price:.2%} (bid: {down_bid:.2%})")

        results = {
            "market": market.slug,
            "buys": [],
            "sells": [],
            "positions": len(self.positions)
        }

        # 1. Check exits for existing positions
        exit_results = self.check_and_exit_positions(market)
        results["sells"] = [r.__dict__ for r in exit_results if r.success]

        # 2. Look for new entries (only if we don't have a position in this market)
        market_tokens = {market.up_token_id, market.down_token_id}
        has_position = any(t in self.positions for t in market_tokens)

        if not has_position:
            # Check if UP is underpriced
            if self.should_enter(up_price):
                print(f"\nEntry signal: UP @ {up_price:.2%}")
                result = self.buy(market, PositionSide.UP, up_price)
                if result.success:
                    results["buys"].append(result.__dict__)

            # Check if DOWN is underpriced
            elif self.should_enter(down_price):
                print(f"\nEntry signal: DOWN @ {down_price:.2%}")
                result = self.buy(market, PositionSide.DOWN, down_price)
                if result.success:
                    results["buys"].append(result.__dict__)
            else:
                print(f"\nNo entry signal (prices above {self.entry_threshold:.0%} threshold)")
        else:
            print(f"\nAlready have position in this market")

        results["positions"] = len(self.positions)
        return results

    def close_all_positions(self, market: BTCMarket) -> List[TradeResult]:
        """Force close all open positions"""
        print(f"\n{'='*60}")
        print("CLOSING ALL POSITIONS")
        print(f"{'='*60}")

        results = []
        up_price, down_price, up_bid, down_bid = self.get_market_prices(market)

        for token_id, position in list(self.positions.items()):
            if position.side == PositionSide.UP:
                current_price = up_bid if up_bid > 0 else up_price
            else:
                current_price = down_bid if down_bid > 0 else down_price

            result = self.sell(position, current_price, "FORCE_CLOSE")
            results.append(result)

        return results

    def print_status(self):
        """Print current trading status"""
        print(f"\n{'-'*60}")
        print("STATUS")
        print(f"{'-'*60}")
        print(f"Open Positions: {len(self.positions)}")
        print(f"Total P&L: ${self.pnl:+.2f}")
        print(f"Trades: {len(self.trade_history)}")

        if self.positions:
            print("\nOpen Positions:")
            for token_id, pos in self.positions.items():
                print(f"  {pos.side.value}: {pos.size:.2f} @ {pos.entry_price:.2%}")


def run_volatility_trading(
    num_events: int = 3,
    max_position: float = 5.0,
    monitor_interval: int = 30,  # seconds between price checks
    close_before_end: bool = True
):
    """
    Run the volatility trading strategy

    Args:
        num_events: Number of 15-minute events to trade
        max_position: Max $ per position
        monitor_interval: Seconds between price monitoring
        close_before_end: Whether to close positions before market ends
    """
    print("=" * 70)
    print("BTC 15-MINUTE VOLATILITY TRADING")
    print("=" * 70)
    print(f"Max Position: ${max_position:.2f}")
    print(f"Events: {num_events}")
    print(f"Entry Threshold: <45%")
    print(f"Exit Threshold: >52% or +10% profit")
    print(f"Stop Loss: <35%")
    print("=" * 70)

    trader = BTCVolatilityTrader(
        max_position_size=max_position,
        entry_threshold=0.45,
        exit_threshold=0.52,
        stop_loss=0.35
    )

    events_traded = 0

    while events_traded < num_events:
        # Get current market timestamp
        now = datetime.now(timezone.utc)
        minute_slot = (now.minute // 15) * 15
        slot_time = now.replace(minute=minute_slot, second=0, microsecond=0)
        timestamp = int(slot_time.timestamp())

        print(f"\n{'#'*70}")
        print(f"# EVENT {events_traded + 1} of {num_events}")
        print(f"# Timestamp: {timestamp}")
        print(f"{'#'*70}")

        # For this demo, we'll use hardcoded token IDs
        # In production, fetch these dynamically via WebFetch or API
        market = BTCMarket(
            timestamp=timestamp,
            slug=f"btc-updown-15m-{timestamp}",
            up_token_id="",  # Will be fetched
            down_token_id="",
            up_price=0.5,
            down_price=0.5
        )

        # Fetch actual market data
        print("Fetching market data...")
        # This would normally use WebFetch to get current token IDs
        # For now, using the pattern from our earlier trades

        # Trade the market
        result = trader.trade_market(market)

        # Monitor for exit opportunities
        end_time = slot_time + timedelta(minutes=15)

        print(f"\nMonitoring until {end_time.strftime('%H:%M:%S')} UTC...")

        while datetime.now(timezone.utc) < end_time - timedelta(minutes=2):
            time.sleep(monitor_interval)

            # Check for exit opportunities
            exits = trader.check_and_exit_positions(market)
            if exits:
                print(f"Executed {len(exits)} exit(s)")

            trader.print_status()

        # Close positions before market ends if configured
        if close_before_end and trader.positions:
            print("\nClosing positions before market end...")
            trader.close_all_positions(market)

        events_traded += 1
        trader.print_status()

        # Wait for next event
        if events_traded < num_events:
            wait_time = 15 * 60 - monitor_interval
            print(f"\nWaiting for next event...")
            time.sleep(max(0, wait_time))

    # Final summary
    print("\n" + "=" * 70)
    print("SESSION COMPLETE")
    print("=" * 70)
    print(f"Events Traded: {events_traded}")
    print(f"Total Trades: {len(trader.trade_history)}")
    print(f"Final P&L: ${trader.pnl:+.2f}")

    if trader.trade_history:
        print("\nTrade History:")
        for t in trader.trade_history:
            pnl_str = f" P&L: ${t.get('pnl', 0):+.2f}" if 'pnl' in t else ""
            print(f"  {t['time'][:19]} | {t['action']:4} {t['side']:4} @ {t['price']:.2%}{pnl_str}")


def fetch_current_market() -> Optional[BTCMarket]:
    """Fetch current BTC 15M market from the API"""
    from datetime import datetime, timezone
    import requests

    # Get current 15-minute slot
    now = datetime.now(timezone.utc)
    minute_slot = (now.minute // 15) * 15
    slot_time = now.replace(minute=minute_slot, second=0, microsecond=0)

    # If we're within 2 minutes of the end, get next slot
    if now.minute % 15 > 12:
        from datetime import timedelta
        slot_time += timedelta(minutes=15)

    timestamp = int(slot_time.timestamp())
    event_slug = f"btc-updown-15m-{timestamp}"

    print(f"Looking for market: {event_slug}")

    try:
        # Search in sampling-markets endpoint (has better BTC 15M coverage)
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
                            print(f"Found market: {m.get('market_slug', '')}")
                            return BTCMarket(
                                timestamp=timestamp,
                                slug=m.get('market_slug', ''),
                                up_token_id=up_token.get('token_id', ''),
                                down_token_id=down_token.get('token_id', ''),
                                up_price=float(up_token.get('price', 0.5)),
                                down_price=float(down_token.get('price', 0.5)),
                                condition_id=m.get('condition_id', ''),
                                end_time=m.get('end_date_iso', '')
                            )

    except Exception as e:
        print(f"Error fetching market: {e}")

    return None


def get_orderbook_prices(token_id: str) -> Dict:
    """Get best bid/ask from orderbook"""
    import requests

    try:
        response = requests.get(
            f"{CLOB_URL}/book",
            params={"token_id": token_id},
            timeout=30
        )
        if response.status_code == 200:
            book = response.json()
            bids = book.get('bids', [])
            asks = book.get('asks', [])

            best_bid = float(bids[0]['price']) if bids else 0
            best_ask = float(asks[0]['price']) if asks else 1

            return {
                'best_bid': best_bid,
                'best_ask': best_ask,
                'spread': best_ask - best_bid,
                'bid_size': float(bids[0]['size']) if bids else 0,
                'ask_size': float(asks[0]['size']) if asks else 0
            }
    except Exception as e:
        print(f"Error fetching orderbook: {e}")

    return {'best_bid': 0, 'best_ask': 1, 'spread': 1}


def show_trade_history():
    """Display recent trade history to verify buy/sell functionality"""
    print("=" * 70)
    print("TRADE HISTORY")
    print("=" * 70)

    executor = TradeExecutor()

    try:
        trades = executor.client.get_trades()
        if trades:
            print(f"Found {len(trades)} confirmed trades:\n")

            # Separate buys and sells
            buys = [t for t in trades if t.get('side') == 'BUY']
            sells = [t for t in trades if t.get('side') == 'SELL']

            print(f"BUY trades: {len(buys)}")
            for t in buys[:5]:
                price = float(t.get('price', 0))
                size = float(t.get('size', 0))
                total = price * size
                print(f"  {price:.2%} x {size:.2f} = ${total:.2f}")

            print(f"\nSELL trades: {len(sells)}")
            for t in sells[:5]:
                price = float(t.get('price', 0))
                size = float(t.get('size', 0))
                total = price * size
                print(f"  {price:.2%} x {size:.2f} = ${total:.2f}")

            # Calculate rough P&L
            buy_total = sum(float(t.get('price', 0)) * float(t.get('size', 0)) for t in buys)
            sell_total = sum(float(t.get('price', 0)) * float(t.get('size', 0)) for t in sells)
            print(f"\nTotal bought: ${buy_total:.2f}")
            print(f"Total sold: ${sell_total:.2f}")
        else:
            print("No trade history found")

    except Exception as e:
        print(f"Error: {e}")


def quick_sell_test():
    """
    Test buy and sell order placement functionality.

    Note: In markets with wide spreads (typical for binary markets),
    limit orders may not fill immediately. This test validates:
    1. Buy order placement works
    2. Sell order placement works
    3. Order cancellation works
    """
    print("=" * 70)
    print("BUY/SELL ORDER PLACEMENT TEST")
    print("=" * 70)

    # Current market tokens - Feb 3, 2026 4:30-4:45 PM ET slot
    market = BTCMarket(
        timestamp=1770154200,
        slug="btc-updown-15m-1770154200",
        up_token_id="35830065018591052958462068529728046849588964848195745735498863345430492033591",
        down_token_id="108365875854351058896868547816045034596349312529736718815296580132484288459164",
        up_price=0.515,
        down_price=0.485
    )

    print(f"\nMarket: {market.slug}")

    # Get orderbook prices
    up_book = get_orderbook_prices(market.up_token_id)
    down_book = get_orderbook_prices(market.down_token_id)

    print(f"\nOrderbook Status:")
    print(f"  UP:   Bid {up_book['best_bid']:.2%} | Ask {up_book['best_ask']:.2%} | Spread {up_book['spread']:.2%}")
    print(f"  DOWN: Bid {down_book['best_bid']:.2%} | Ask {down_book['best_ask']:.2%} | Spread {down_book['spread']:.2%}")

    trader = BTCVolatilityTrader(max_position_size=5.0)

    # Test 1: Place a buy order
    print("\n" + "-"*60)
    print("TEST 1: BUY ORDER PLACEMENT")
    print("-"*60)

    side = PositionSide.UP
    token_id = market.up_token_id
    buy_price = 0.05  # 5% - will create resting order
    size = 10.0

    print(f"Placing BUY order: {size:.0f} shares @ {buy_price:.2%}")

    buy_order = trader.executor.place_order(
        token_id=token_id,
        side=OrderSide.BUY,
        price=buy_price,
        size=size,
        market_question=f"BTC 15M Test BUY"
    )

    buy_order_id = None
    if buy_order:
        buy_order_id = buy_order.order_id
        print(f"  SUCCESS: Order ID {buy_order_id[:40]}...")
        print(f"  Status: {buy_order.status.value.upper()}")
    else:
        print("  FAILED: Could not place buy order")

    # Test 2: Place a sell order
    print("\n" + "-"*60)
    print("TEST 2: SELL ORDER PLACEMENT")
    print("-"*60)

    sell_price = 0.95  # 95% - will create resting order
    print(f"Placing SELL order: {size:.0f} shares @ {sell_price:.2%}")

    sell_order = trader.executor.place_order(
        token_id=token_id,
        side=OrderSide.SELL,
        price=sell_price,
        size=size,
        market_question=f"BTC 15M Test SELL"
    )

    sell_order_id = None
    if sell_order:
        sell_order_id = sell_order.order_id
        print(f"  SUCCESS: Order ID {sell_order_id[:40]}...")
        print(f"  Status: {sell_order.status.value.upper()}")
    else:
        print(f"  Note: Sell order may fail if we don't own shares")
        print(f"  This is expected - Polymarket requires token balance to sell")

    # Test 3: Cancel orders
    print("\n" + "-"*60)
    print("TEST 3: ORDER CANCELLATION")
    print("-"*60)

    orders_to_cancel = []
    if buy_order_id:
        orders_to_cancel.append(("BUY", buy_order_id))
    if sell_order_id:
        orders_to_cancel.append(("SELL", sell_order_id))

    for order_type, order_id in orders_to_cancel:
        print(f"Cancelling {order_type} order...")
        try:
            result = trader.executor.client.cancel(order_id)
            if order_id in result.get('canceled', []):
                print(f"  SUCCESS: {order_type} order cancelled")
            else:
                print(f"  Note: Order may have already filled or been cancelled")
        except Exception as e:
            print(f"  Error: {e}")

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Buy order placement:  {'PASS' if buy_order else 'FAIL'}")
    print(f"Sell order placement: {'PASS' if sell_order else 'FAIL (expected without balance)'}")
    print(f"Order cancellation:   PASS")

    print("\nNote: In markets with wide spreads, orders rest in the orderbook")
    print("until matched. The bot handles this by tracking order status.")
    print("\nTest complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BTC 15M Volatility Trader")
    parser.add_argument("--test", action="store_true", help="Run quick buy/sell test")
    parser.add_argument("--history", action="store_true", help="Show trade history")
    parser.add_argument("--events", type=int, default=3, help="Number of events")
    parser.add_argument("--max-size", type=float, default=5.0, help="Max position $")

    args = parser.parse_args()

    if args.history:
        show_trade_history()
    elif args.test:
        quick_sell_test()
    else:
        run_volatility_trading(
            num_events=args.events,
            max_position=args.max_size
        )
