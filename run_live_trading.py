"""
Run live trading with specified constraints
Max trade amount: $1
Max trades: 3
"""
import os
os.environ['TRADING_MODE'] = 'live'

from trading_bot import TradingBot, TradeConfig

# Configure for small trades
config = TradeConfig(
    max_position_size=1.0,      # Max $1 per position
    risk_per_trade=100.0,       # Not used in live mode
    max_positions=3,            # Maximum 3 positions
    max_daily_trades=3,         # Maximum 3 trades
    min_confidence=60.0,        # Minimum confidence to trade
    min_volume_24h=5000.0,      # Minimum 24h volume
    min_liquidity=1000.0,       # Minimum liquidity
    use_limit_orders=True,
    limit_offset=0.005,
    trade_strong_buy=True,
    trade_buy=True,
    trade_strong_sell=False,    # Don't short for now
    trade_sell=False
)

print("=" * 60)
print("LIVE TRADING BOT")
print("=" * 60)
print(f"Max trade amount: ${config.max_position_size:.2f}")
print(f"Max trades: {config.max_daily_trades}")
print(f"Min confidence: {config.min_confidence}%")
print("=" * 60)

bot = TradingBot(config)

# Run single scan with execution
print("\nRunning market scan and executing trades...")
orders = bot.run_once(limit=30, auto_execute=True)

print("\n" + "=" * 60)
print("TRADING COMPLETE")
print("=" * 60)
print(f"Orders placed: {len(orders)}")

for order in orders:
    print(f"  - {order.order_id[:20]}... | {order.side.value} | ${order.price * order.size:.2f}")

bot.print_summary()
