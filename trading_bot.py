"""
Automated Trading Bot for Polymarket
Combines prediction signals with trade execution
"""

import os
import json
import time
import argparse
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

from predictor import PolymarketPredictor, Signal, Prediction
from trade_executor import TradeExecutor, OrderSide, Order, DrawdownConfig, PortfolioHeatConfig
from logging_config import setup_logging, get_logger

# Initialize logger for this module
logger = get_logger(__name__)

GAMMA_API_URL = "https://gamma-api.polymarket.com"


@dataclass
class TradeConfig:
    """Trading configuration"""
    # Risk management
    max_position_size: float = 500.0  # Max $ per position
    risk_per_trade: float = 2.0  # % of portfolio to risk per trade
    max_positions: int = 10  # Maximum concurrent positions
    max_daily_trades: int = 20  # Maximum trades per day

    # Drawdown protection
    max_drawdown_percent: float = 15.0  # Maximum allowed drawdown before halting
    drawdown_warning_percent: float = 10.0  # Warning threshold
    drawdown_recovery_percent: float = 5.0  # Resume trading when recovered to this level

    # Portfolio heat (concurrent risk) limits
    max_portfolio_heat_percent: float = 20.0  # Max % of portfolio at risk across all positions
    heat_warning_percent: float = 15.0  # Warning threshold
    per_position_max_risk_percent: float = 5.0  # Max risk per individual position

    # Signal thresholds
    min_confidence: float = 60.0  # Minimum confidence to trade
    min_volume_24h: float = 1000.0  # Minimum 24h volume
    min_liquidity: float = 500.0  # Minimum liquidity

    # Entry/Exit
    use_limit_orders: bool = True  # Use limit orders vs market
    limit_offset: float = 0.005  # How far from mid price for limits

    # Signals to trade
    trade_strong_buy: bool = True
    trade_buy: bool = True
    trade_strong_sell: bool = True
    trade_sell: bool = False  # Conservative: only strong sell signals


class TradingBot:
    """
    Automated trading bot that:
    1. Scans markets for signals
    2. Filters based on risk parameters
    3. Executes trades automatically
    4. Manages positions
    """

    def __init__(self, config: TradeConfig = None):
        self.config = config or TradeConfig()
        self.predictor = PolymarketPredictor()

        # Initialize executor with drawdown and heat configs
        drawdown_config = DrawdownConfig(
            max_drawdown_percent=self.config.max_drawdown_percent,
            warning_threshold_percent=self.config.drawdown_warning_percent,
            recovery_threshold_percent=self.config.drawdown_recovery_percent
        )
        heat_config = PortfolioHeatConfig(
            max_portfolio_heat_percent=self.config.max_portfolio_heat_percent,
            warning_heat_percent=self.config.heat_warning_percent,
            per_position_max_risk_percent=self.config.per_position_max_risk_percent
        )
        self.executor = TradeExecutor(drawdown_config=drawdown_config, heat_config=heat_config)

        # Track daily stats
        self.daily_trades = 0
        self.last_trade_date = None

        # Trade history
        self.trade_history: List[Dict] = []

    def reset_daily_stats(self):
        """Reset daily trading stats"""
        today = datetime.now().date()
        if self.last_trade_date != today:
            self.daily_trades = 0
            self.last_trade_date = today

    def can_trade(self) -> tuple[bool, str]:
        """
        Check if we can make more trades today.
        Returns (can_trade, reason) tuple.
        """
        self.reset_daily_stats()

        # Check daily trade limit
        if self.daily_trades >= self.config.max_daily_trades:
            return False, f"Daily trade limit reached ({self.config.max_daily_trades})"

        # Check drawdown limit
        can_trade_dd, dd_reason = self.executor.can_trade_drawdown()
        if not can_trade_dd:
            return False, dd_reason

        # Check portfolio heat limit
        can_trade_heat, heat_reason = self.executor.can_trade_heat()
        if not can_trade_heat:
            return False, heat_reason

        return True, ""

    def should_trade_signal(self, prediction: Prediction) -> bool:
        """Determine if we should trade based on signal"""
        # Check confidence threshold
        if prediction.confidence < self.config.min_confidence:
            return False

        # Check volume
        if prediction.metrics.get("volume_24h", 0) < self.config.min_volume_24h:
            return False

        # Check liquidity
        if prediction.metrics.get("liquidity", 0) < self.config.min_liquidity:
            return False

        # Check signal type
        if prediction.signal == Signal.STRONG_BUY and self.config.trade_strong_buy:
            return True
        if prediction.signal == Signal.BUY and self.config.trade_buy:
            return True
        if prediction.signal == Signal.STRONG_SELL and self.config.trade_strong_sell:
            return True
        if prediction.signal == Signal.SELL and self.config.trade_sell:
            return True

        return False

    def get_token_id(self, prediction: Prediction) -> Optional[str]:
        """Get token ID from prediction"""
        # Token ID is now included in the prediction object
        return prediction.token_id if prediction.token_id else None

    def calculate_entry_price(self, token_id: str, side: OrderSide, prediction_price: float = None) -> float:
        """Calculate entry price based on orderbook or prediction price"""
        prices = self.executor.get_market_price(token_id)

        # If orderbook data is invalid, use prediction price
        if prices["best_bid"] == 0 or prices["best_ask"] == 1 or prices["spread"] > 0.5:
            if prediction_price:
                if side == OrderSide.BUY:
                    return min(prediction_price + self.config.limit_offset, 0.99)
                else:
                    return max(prediction_price - self.config.limit_offset, 0.01)
            return 0.5  # Default mid price

        if self.config.use_limit_orders:
            if side == OrderSide.BUY:
                # Place limit below best ask
                return max(prices["best_ask"] - self.config.limit_offset, prices["mid_price"])
            else:
                # Place limit above best bid
                return min(prices["best_bid"] + self.config.limit_offset, prices["mid_price"])
        else:
            # Market order: use best available price
            return prices["best_ask"] if side == OrderSide.BUY else prices["best_bid"]

    def calculate_position_size(self, price: float, confidence: float = 50.0, volatility: float = 0.5) -> float:
        """
        Calculate position size based on risk management with confidence and volatility scaling.

        Position size scales with:
        1. Signal confidence: Higher confidence = larger positions
        2. Market volatility: Higher volatility = smaller positions (risk management)

        Confidence scaling (0.5x to 1.5x):
        - Base risk is applied at 50% confidence
        - Higher confidence (>50%) increases position size up to 1.5x
        - Lower confidence (<50%) decreases position size down to 0.5x

        Volatility scaling (0.5x to 1.3x):
        - Low volatility (< 0.3): 1.3x multiplier (safer market)
        - Medium volatility (0.3-0.6): 1.0x multiplier (normal)
        - High volatility (0.6-0.8): 0.7x multiplier (risky)
        - Extreme volatility (> 0.8): 0.5x multiplier (very risky)

        Args:
            price: Current price per share
            confidence: Signal confidence from 0-100 (default 50%)
            volatility: Normalized volatility 0-1 where higher = more volatile (default 0.5)

        Returns:
            Position size in number of shares
        """
        # Clamp confidence to valid range
        confidence = max(0, min(100, confidence))

        # Clamp volatility to valid range (0.1 to 1.0)
        volatility = max(0.1, min(1.0, volatility))

        # Calculate confidence multiplier (0.5x to 1.5x)
        if confidence >= 50:
            confidence_multiplier = 1.0 + (confidence - 50) / 100
        else:
            confidence_multiplier = 0.5 + (confidence / 100)

        # Calculate volatility multiplier (0.5x to 1.3x)
        # Lower volatility = higher multiplier (can take larger positions)
        # Higher volatility = lower multiplier (reduce position size for safety)
        if volatility < 0.3:
            volatility_multiplier = 1.3  # Low volatility - can size up slightly
        elif volatility < 0.6:
            volatility_multiplier = 1.0  # Normal volatility - standard size
        elif volatility < 0.8:
            volatility_multiplier = 0.7  # High volatility - reduce size
        else:
            volatility_multiplier = 0.5  # Extreme volatility - minimum size

        # Combined multiplier: confidence * volatility
        combined_multiplier = confidence_multiplier * volatility_multiplier

        # For live mode, use max_position_size directly as dollar amount
        # For paper mode, use portfolio-based sizing
        if self.executor.is_live_mode():
            # In live mode, use max_position_size scaled by combined multiplier
            adjusted_max = self.config.max_position_size * combined_multiplier
            max_size = adjusted_max / price if price > 0 else 0
            return max(round(max_size, 2), 0)

        portfolio = self.executor.get_portfolio_summary()
        balance = portfolio["total_value"]

        # Risk-based sizing with combined adjustment
        adjusted_risk_percent = self.config.risk_per_trade * combined_multiplier
        risk_amount = balance * (adjusted_risk_percent / 100)
        size_by_risk = risk_amount / price if price > 0 else 0

        # Cap by max position size (also combined-adjusted)
        adjusted_max_position = self.config.max_position_size * combined_multiplier
        max_size = adjusted_max_position / price if price > 0 else 0

        # Use minimum of risk-based and max allowed
        size = min(size_by_risk, max_size)

        # Minimum size check
        return max(round(size, 2), 0)

    def calculate_stop_loss(self, entry_price: float, side: OrderSide, prediction: Prediction) -> tuple[float, float]:
        """
        Calculate stop-loss and take-profit prices based on prediction and risk parameters.

        Returns:
            tuple of (stop_loss_price, take_profit_price)
        """
        # Get levels from prediction if available
        stop_loss = prediction.stop_loss if hasattr(prediction, 'stop_loss') and prediction.stop_loss else None
        take_profit = prediction.take_profit if hasattr(prediction, 'take_profit') and prediction.take_profit else None

        # If not available from prediction, calculate based on risk parameters
        if stop_loss is None:
            if side == OrderSide.BUY:
                # For buys, stop loss below entry
                stop_loss = max(entry_price * (1 - self.config.risk_per_trade / 100 * 2), 0.01)
            else:
                # For sells, stop loss above entry
                stop_loss = min(entry_price * (1 + self.config.risk_per_trade / 100 * 2), 0.99)

        if take_profit is None:
            if side == OrderSide.BUY:
                # For buys, take profit above entry (2:1 reward:risk ratio)
                stop_distance = entry_price - stop_loss
                take_profit = min(entry_price + stop_distance * 2, 0.99)
            else:
                # For sells, take profit below entry
                stop_distance = stop_loss - entry_price
                take_profit = max(entry_price - stop_distance * 2, 0.01)

        return stop_loss, take_profit

    def execute_signal(self, prediction: Prediction) -> Optional[Order]:
        """Execute a trade based on prediction signal with stop-loss protection"""
        # Get token ID from prediction
        token_id = self.get_token_id(prediction)
        if not token_id:
            logger.warning(f"Could not get token ID", market=prediction.market_question[:30])
            return None

        # Determine side
        if prediction.signal in [Signal.STRONG_BUY, Signal.BUY]:
            side = OrderSide.BUY
        elif prediction.signal in [Signal.STRONG_SELL, Signal.SELL]:
            side = OrderSide.SELL
        else:
            return None

        # Calculate entry price (use prediction price as fallback)
        entry_price = self.calculate_entry_price(token_id, side, prediction.current_price)

        # Get volatility from prediction metrics (default 0.5 if not available)
        volatility = prediction.metrics.get("volatility_normalized", 0.5)

        # Calculate position size with confidence and volatility scaling
        size = self.calculate_position_size(entry_price, prediction.confidence, volatility)

        if size < 1:  # Minimum order size on Polymarket
            logger.debug(f"Position size too small: {size}")
            return None

        # Calculate stop-loss and take-profit
        stop_loss, take_profit = self.calculate_stop_loss(entry_price, side, prediction)

        # Check if adding this position would exceed portfolio heat limits
        stop_distance = abs(entry_price - stop_loss) if stop_loss else entry_price * 0.04
        proposed_risk = stop_distance * size
        can_add, heat_reason = self.executor.can_add_position_heat(proposed_risk_amount=proposed_risk)
        if not can_add:
            logger.risk(f"Cannot add position: {heat_reason}", proposed_risk=proposed_risk)
            return None

        # Place order with stop-loss protection
        order = self.executor.place_order(
            token_id=token_id,
            side=side,
            price=entry_price,
            size=size,
            market_question=prediction.market_question,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

        if order:
            self.daily_trades += 1
            self.trade_history.append({
                "timestamp": datetime.now().isoformat(),
                "market": prediction.market_question,
                "signal": prediction.signal.value,
                "confidence": prediction.confidence,
                "side": side.value,
                "price": entry_price,
                "size": size,
                "order_id": order.order_id,
                "status": order.status.value,
                "stop_loss": stop_loss,
                "take_profit": take_profit
            })
            logger.trade(
                f"Order placed: {order.status.value}",
                side=side.value,
                price=entry_price,
                size=size,
                order_id=order.order_id,
                stop_loss=stop_loss,
                take_profit=take_profit,
                market=prediction.market_question[:40]
            )

        return order

    def run_scan(self, limit: int = 30) -> List[Prediction]:
        """Run a scan and return tradeable predictions"""
        logger.info("=" * 60)
        logger.info(f"SCANNING MARKETS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)

        # Get predictions
        predictions = self.predictor.scan(
            limit=limit,
            min_volume=self.config.min_volume_24h,
            min_liquidity=self.config.min_liquidity
        )

        # Filter for tradeable signals
        tradeable = [p for p in predictions if self.should_trade_signal(p)]

        logger.info(f"Scan complete", markets_found=len(predictions), tradeable_signals=len(tradeable))

        return tradeable

    def run_once(self, limit: int = 30, auto_execute: bool = False) -> List[Order]:
        """Run one scan cycle and optionally execute trades"""
        can_trade, reason = self.can_trade()
        if not can_trade:
            logger.warning(f"Cannot trade: {reason}")
            return []

        tradeable = self.run_scan(limit)

        if not tradeable:
            logger.info("No tradeable signals found")
            return []

        # Check position limits
        portfolio = self.executor.get_portfolio_summary()
        current_positions = portfolio["num_positions"]

        orders = []

        for prediction in tradeable:
            if current_positions >= self.config.max_positions:
                logger.warning(f"Max positions reached", limit=self.config.max_positions)
                break

            can_trade, reason = self.can_trade()
            if not can_trade:
                logger.warning(f"Cannot trade: {reason}")
                break

            volatility_norm = prediction.metrics.get("volatility_normalized", 0.5)
            vol_level = "LOW" if volatility_norm < 0.3 else "NORMAL" if volatility_norm < 0.6 else "HIGH" if volatility_norm < 0.8 else "EXTREME"

            logger.signal(
                f"{prediction.signal.value} detected",
                signal=prediction.signal.value,
                confidence=prediction.confidence,
                market=prediction.market_question[:50],
                price=prediction.current_price,
                volatility=vol_level,
                reasons=prediction.reasons[:2]
            )

            if auto_execute:
                # Calculate proposed risk for this position to check against heat limits
                token_id = self.get_token_id(prediction)
                if token_id:
                    side = OrderSide.BUY if prediction.signal in [Signal.STRONG_BUY, Signal.BUY] else OrderSide.SELL
                    entry_price = self.calculate_entry_price(token_id, side, prediction.current_price)
                    volatility = prediction.metrics.get("volatility_normalized", 0.5)
                    size = self.calculate_position_size(entry_price, prediction.confidence, volatility)
                    stop_loss, _ = self.calculate_stop_loss(entry_price, side, prediction)
                    proposed_risk = abs(entry_price - stop_loss) * size

                    # Check if adding this position would exceed heat limits
                    can_add_heat, heat_reason = self.executor.can_add_position_heat(proposed_risk_amount=proposed_risk)
                    if not can_add_heat:
                        logger.risk(f"Heat limit: {heat_reason}", proposed_risk=proposed_risk)
                        continue

                order = self.execute_signal(prediction)
                if order:
                    orders.append(order)
                    current_positions += 1
            else:
                logger.debug("[DRY RUN] Would execute trade")

        return orders

    def run_continuous(self, interval_minutes: int = 5, auto_execute: bool = False):
        """Run continuous trading loop with stop-loss and drawdown monitoring"""
        logger.info("=" * 60)
        logger.info("STARTING TRADING BOT")
        logger.info("=" * 60)
        logger.info(f"Bot configuration",
            mode="LIVE" if self.executor.is_live_mode() else "PAPER",
            auto_execute=auto_execute,
            interval_minutes=interval_minutes,
            max_positions=self.config.max_positions,
            max_daily_trades=self.config.max_daily_trades,
            min_confidence=self.config.min_confidence,
            max_drawdown=self.config.max_drawdown_percent,
            max_heat=self.config.max_portfolio_heat_percent
        )
        logger.info("Press Ctrl+C to stop")

        while True:
            try:
                # First, update drawdown status
                self.executor.update_drawdown()

                # Check stop-loss conditions on existing positions
                stop_exits = self.executor.execute_stop_exits(auto_execute=auto_execute)
                if stop_exits:
                    logger.info(f"Stop-loss/take-profit exits triggered", exit_count=len(stop_exits))

                # Check if we can trade (includes drawdown check)
                can_trade, reason = self.can_trade()
                if can_trade:
                    # Scan for new trades
                    orders = self.run_once(auto_execute=auto_execute)
                    if orders:
                        logger.info(f"Trades executed", trade_count=len(orders))
                else:
                    logger.warning(f"Trading paused: {reason}")

                # Print portfolio, stop status, heat status, and drawdown status
                self.executor.print_portfolio()
                self.executor.print_stops_status()
                self.executor.print_heat_status()
                self.executor.print_drawdown_status()

                logger.debug(f"Next scan in {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)

            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                break
            except Exception as e:
                logger.error(f"Error during scan: {e}", exc_info=True)
                time.sleep(60)

        # Final summary
        self.print_summary()

    def print_summary(self):
        """Print trading session summary"""
        print(f"\n{'='*60}")
        print("TRADING SESSION SUMMARY")
        print(f"{'='*60}")
        print(f"Total Trades: {len(self.trade_history)}")

        if self.trade_history:
            buys = sum(1 for t in self.trade_history if t["side"] == "BUY")
            sells = sum(1 for t in self.trade_history if t["side"] == "SELL")
            print(f"Buys: {buys} | Sells: {sells}")

            print(f"\nRecent Trades:")
            for trade in self.trade_history[-5:]:
                print(f"  {trade['timestamp'][:19]} | {trade['side']} | {trade['market'][:30]}...")

        self.executor.print_portfolio()


def main():
    parser = argparse.ArgumentParser(description="Polymarket Trading Bot")
    parser.add_argument("--scan", action="store_true", help="Run single scan")
    parser.add_argument("--execute", action="store_true", help="Execute trades (otherwise dry run)")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=5, help="Scan interval in minutes")
    parser.add_argument("--limit", type=int, default=30, help="Markets to scan")
    parser.add_argument("--min-confidence", type=float, default=60, help="Minimum confidence percent")
    parser.add_argument("--max-positions", type=int, default=10, help="Maximum positions")
    parser.add_argument("--max-drawdown", type=float, default=15, help="Maximum drawdown percent before halt")
    parser.add_argument("--portfolio", action="store_true", help="Show portfolio only")
    parser.add_argument("--drawdown", action="store_true", help="Show drawdown status only")
    parser.add_argument("--heat", action="store_true", help="Show portfolio heat status only")
    parser.add_argument("--max-heat", type=float, default=20, help="Maximum portfolio heat percent (concurrent risk)")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    parser.add_argument("--log-dir", type=str, default=None, help="Directory for log files")

    args = parser.parse_args()

    # Initialize structured logging
    setup_logging(
        level=args.log_level,
        log_dir=args.log_dir,
        console_output=True
    )

    # Create config
    config = TradeConfig(
        min_confidence=args.min_confidence,
        max_positions=args.max_positions,
        max_drawdown_percent=args.max_drawdown,
        max_portfolio_heat_percent=args.max_heat
    )

    bot = TradingBot(config)

    if args.portfolio:
        bot.executor.print_portfolio()
        return

    if args.drawdown:
        bot.executor.update_drawdown()
        bot.executor.print_drawdown_status()
        return

    if args.heat:
        bot.executor.update_portfolio_heat()
        bot.executor.print_heat_status()
        return

    if args.continuous:
        bot.run_continuous(
            interval_minutes=args.interval,
            auto_execute=args.execute
        )
    else:
        bot.run_once(limit=args.limit, auto_execute=args.execute)
        bot.executor.print_portfolio()


if __name__ == "__main__":
    main()
