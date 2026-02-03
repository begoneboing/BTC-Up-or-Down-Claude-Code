"""
Trade Executor for Polymarket
Executes trades based on prediction signals using the Polymarket CLOB API
"""

import os
import json
import time
import hmac
import hashlib
import requests
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

# Import logging - handle case where logging_config might not be available yet
try:
    from logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# Try to import web3 for signing
try:
    from eth_account import Account
    from eth_account.messages import encode_defunct
    HAS_WEB3 = True
except ImportError:
    HAS_WEB3 = False
    logger.warning("eth_account not installed. Install with: pip install eth-account")

# Try to import py_clob_client
try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, OrderType
    HAS_CLOB_CLIENT = True
except ImportError:
    HAS_CLOB_CLIENT = False

# Try to import performance tracker
try:
    from performance_metrics import PerformanceTracker
    HAS_PERFORMANCE_TRACKER = True
except ImportError:
    HAS_PERFORMANCE_TRACKER = False


CLOB_API_URL = "https://clob.polymarket.com"
GAMMA_API_URL = "https://gamma-api.polymarket.com"

# Signature types for Polymarket
SIGNATURE_TYPE_EOA = 0  # Direct wallet
SIGNATURE_TYPE_POLY_PROXY = 1  # Magic Link / Proxy wallet with funder


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    PENDING = "PENDING"
    LIVE = "LIVE"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


@dataclass
class Order:
    order_id: str
    token_id: str
    side: OrderSide
    price: float
    size: float
    status: OrderStatus
    filled_size: float = 0
    timestamp: str = ""


@dataclass
class Position:
    token_id: str
    market_question: str
    outcome: str
    size: float
    avg_price: float
    current_price: float
    pnl: float
    pnl_percent: float


@dataclass
class StopLossConfig:
    """Configuration for stop-loss and take-profit levels"""
    stop_loss_price: Optional[float] = None  # Exit if price falls below this
    take_profit_price: Optional[float] = None  # Exit if price rises above this
    trailing_stop_percent: Optional[float] = None  # Trailing stop as % (e.g., 5.0 = 5%)
    trailing_stop_activated: bool = False  # Track if trailing stop is active
    highest_price_seen: float = 0.0  # For trailing stop calculation


@dataclass
class PositionWithStops:
    """Position with stop-loss and take-profit tracking"""
    token_id: str
    market_question: str
    side: str  # "BUY" or "SELL"
    size: float
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop_percent: Optional[float] = None
    highest_price_seen: float = 0.0  # For trailing stop
    created_at: str = ""

    def should_exit(self, current_price: float) -> tuple[bool, str]:
        """
        Check if position should be exited based on stops.
        Returns (should_exit, reason)
        """
        # Update highest price seen for trailing stop
        if current_price > self.highest_price_seen:
            self.highest_price_seen = current_price

        # Check stop-loss (for BUY positions, exit if price drops)
        if self.side == "BUY" and self.stop_loss is not None:
            if current_price <= self.stop_loss:
                return True, f"Stop-loss hit: {current_price:.2%} <= {self.stop_loss:.2%}"

        # Check take-profit (for BUY positions, exit if price rises enough)
        if self.side == "BUY" and self.take_profit is not None:
            if current_price >= self.take_profit:
                return True, f"Take-profit hit: {current_price:.2%} >= {self.take_profit:.2%}"

        # Check trailing stop (only for profitable positions)
        if self.trailing_stop_percent is not None and self.highest_price_seen > self.entry_price:
            trailing_stop_price = self.highest_price_seen * (1 - self.trailing_stop_percent / 100)
            if current_price <= trailing_stop_price:
                return True, f"Trailing stop hit: {current_price:.2%} <= {trailing_stop_price:.2%} (high: {self.highest_price_seen:.2%})"

        return False, ""

    def get_unrealized_pnl(self, current_price: float) -> tuple[float, float]:
        """
        Calculate unrealized P&L.
        Returns (pnl_dollars, pnl_percent)
        """
        if self.side == "BUY":
            pnl = (current_price - self.entry_price) * self.size
            pnl_pct = ((current_price / self.entry_price) - 1) * 100 if self.entry_price > 0 else 0
        else:
            pnl = (self.entry_price - current_price) * self.size
            pnl_pct = ((self.entry_price / current_price) - 1) * 100 if current_price > 0 else 0
        return pnl, pnl_pct


@dataclass
class DrawdownConfig:
    """Configuration for maximum drawdown protection"""
    max_drawdown_percent: float = 15.0  # Maximum allowed drawdown (e.g., 15%)
    warning_threshold_percent: float = 10.0  # Warning when drawdown reaches this level
    recovery_threshold_percent: float = 5.0  # Resume trading when drawdown recovers to this level
    track_session_only: bool = False  # If True, only track drawdown for current session


@dataclass
class PortfolioHeatConfig:
    """Configuration for portfolio heat (total concurrent risk) limits"""
    max_portfolio_heat_percent: float = 20.0  # Max % of portfolio at risk across all positions
    warning_heat_percent: float = 15.0  # Warning threshold
    per_position_max_risk_percent: float = 5.0  # Max risk per individual position
    default_stop_distance_percent: float = 4.0  # Default stop distance if no stop-loss set


@dataclass
class PortfolioHeatState:
    """Tracks current portfolio heat (total risk across all positions)"""
    total_risk_amount: float = 0.0  # Total $ at risk
    total_risk_percent: float = 0.0  # Total risk as % of portfolio
    num_positions_at_risk: int = 0  # Number of positions contributing to risk
    position_risks: Dict = None  # Risk breakdown by position
    last_updated: str = ""
    heat_warning: bool = False  # True if approaching max heat
    heat_exceeded: bool = False  # True if max heat exceeded

    def __post_init__(self):
        if self.position_risks is None:
            self.position_risks = {}


@dataclass
class DrawdownState:
    """Tracks portfolio drawdown state"""
    high_water_mark: float = 0.0  # Peak portfolio value
    current_value: float = 0.0  # Current portfolio value
    current_drawdown_percent: float = 0.0  # Current drawdown as percentage
    max_drawdown_seen: float = 0.0  # Maximum drawdown seen during session
    trading_halted: bool = False  # Whether trading is halted due to drawdown
    halt_reason: str = ""  # Reason for halt
    last_updated: str = ""  # Timestamp of last update

    def update(self, portfolio_value: float) -> None:
        """Update drawdown state with new portfolio value"""
        self.current_value = portfolio_value
        self.last_updated = datetime.now().isoformat()

        # Update high water mark if we have a new peak
        if portfolio_value > self.high_water_mark:
            self.high_water_mark = portfolio_value

        # Calculate current drawdown
        if self.high_water_mark > 0:
            self.current_drawdown_percent = ((self.high_water_mark - portfolio_value) / self.high_water_mark) * 100
        else:
            self.current_drawdown_percent = 0.0

        # Track maximum drawdown seen
        if self.current_drawdown_percent > self.max_drawdown_seen:
            self.max_drawdown_seen = self.current_drawdown_percent


class TradeExecutor:
    """
    Executes trades on Polymarket using the CLOB API
    """

    def __init__(self, private_key: str = None, funder_address: str = None, drawdown_config: DrawdownConfig = None, heat_config: PortfolioHeatConfig = None):
        self.private_key = private_key or os.environ.get("POLYMARKET_PRIVATE_KEY", "")
        self.funder_address = funder_address or os.environ.get("POLYMARKET_FUNDER_ADDRESS", "")
        self.trading_mode = os.environ.get("TRADING_MODE", "paper")

        # Initialize client
        self.client = None
        self.api_creds = None

        if HAS_CLOB_CLIENT and self.private_key:
            try:
                # Determine signature type based on funder address
                # Use POLY_PROXY (1) if funder address exists, EOA (0) otherwise
                signature_type = SIGNATURE_TYPE_POLY_PROXY if self.funder_address else SIGNATURE_TYPE_EOA
                logger.info(f"Using signature type: {'POLY_PROXY' if signature_type == 1 else 'EOA'}")

                # First create client without creds to derive them
                self.client = ClobClient(
                    host=CLOB_API_URL,
                    key=self.private_key,
                    chain_id=137,  # Polygon mainnet
                    signature_type=signature_type,
                    funder=self.funder_address if self.funder_address else None
                )

                # Derive API credentials from private key
                self.api_creds = self.client.derive_api_key()
                logger.info("Derived API credentials successfully")

                # Reinitialize with credentials
                self.client = ClobClient(
                    host=CLOB_API_URL,
                    key=self.private_key,
                    chain_id=137,
                    signature_type=signature_type,
                    funder=self.funder_address if self.funder_address else None,
                    creds=self.api_creds
                )

                # Verify connection by getting API keys
                try:
                    api_keys = self.client.get_api_keys()
                    logger.info(f"Verified API connection", api_keys_found=len(api_keys))
                except Exception as verify_error:
                    logger.warning(f"API key verification failed: {verify_error}")

                logger.info("Initialized Polymarket CLOB client with API credentials")
            except Exception as e:
                logger.warning(f"Could not initialize CLOB client: {e}")
                # Try without derived creds
                try:
                    signature_type = SIGNATURE_TYPE_POLY_PROXY if self.funder_address else SIGNATURE_TYPE_EOA
                    self.client = ClobClient(
                        host=CLOB_API_URL,
                        key=self.private_key,
                        chain_id=137,
                        signature_type=signature_type,
                        funder=self.funder_address if self.funder_address else None
                    )
                    logger.info("Initialized CLOB client (without derived creds)")
                except Exception as e2:
                    logger.error(f"Failed to initialize client: {e2}")

        # Order tracking
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}

        # Paper trading balance
        self.paper_balance = 10000.0  # $10,000 starting balance
        self.paper_positions: Dict[str, Dict] = {}

        # Stop-loss tracking for positions
        self.positions_with_stops: Dict[str, PositionWithStops] = {}

        # Drawdown protection
        self.drawdown_config = drawdown_config or DrawdownConfig()
        self.drawdown_state = DrawdownState(high_water_mark=self.paper_balance)
        self.drawdown_state.update(self.paper_balance)

        # Portfolio heat (concurrent risk) tracking
        self.heat_config = heat_config or PortfolioHeatConfig()
        self.heat_state = PortfolioHeatState()

        # Performance metrics tracking
        self.performance_tracker = None
        if HAS_PERFORMANCE_TRACKER:
            self.performance_tracker = PerformanceTracker()
            self.performance_tracker.set_initial_equity(self.paper_balance)

    def is_live_mode(self) -> bool:
        return self.trading_mode.lower() == "live"

    def get_balance(self) -> float:
        """Get current balance"""
        if not self.is_live_mode():
            return self.paper_balance

        # For live mode, query CLOB API for balance
        if self.client:
            try:
                balance_info = self.client.get_balance_allowance()
                if balance_info:
                    balance = float(balance_info.get("balance", 0) or 0)
                    print(f"[LIVE] Balance: ${balance:.2f}")
                    return balance
            except Exception as e:
                print(f"[LIVE] Error getting balance: {e}")

        return 0.0

    def get_market_price(self, token_id: str) -> Dict:
        """Get current market price for a token"""
        url = f"{CLOB_API_URL}/book"
        params = {"token_id": token_id}

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            book = response.json()

            bids = book.get("bids", [])
            asks = book.get("asks", [])

            best_bid = float(bids[0]["price"]) if bids else 0
            best_ask = float(asks[0]["price"]) if asks else 1
            mid_price = (best_bid + best_ask) / 2 if bids and asks else 0.5

            return {
                "best_bid": best_bid,
                "best_ask": best_ask,
                "mid_price": mid_price,
                "spread": best_ask - best_bid
            }
        except Exception as e:
            logger.warning(f"Error getting market price", token_id=token_id[:20], error=str(e))
            return {"best_bid": 0, "best_ask": 1, "mid_price": 0.5, "spread": 1}

    def calculate_position_size(
        self,
        balance: float,
        price: float,
        risk_percent: float = 2.0,
        confidence: float = 50.0
    ) -> float:
        """
        Calculate position size based on risk management with confidence scaling.

        Position size scales with signal confidence:
        - Base risk is applied at 50% confidence
        - Higher confidence (>50%) increases position size up to 1.5x
        - Lower confidence (<50%) decreases position size down to 0.5x

        Args:
            balance: Available trading balance
            price: Current price per share
            risk_percent: Base percentage of balance to risk (default 2%)
            confidence: Signal confidence from 0-100 (default 50%)

        Returns:
            Position size in number of shares, rounded to 2 decimal places
        """
        # Clamp confidence to valid range
        confidence = max(0, min(100, confidence))

        # Calculate confidence multiplier (0.5x to 1.5x)
        # At 50% confidence: multiplier = 1.0
        # At 0% confidence: multiplier = 0.5
        # At 100% confidence: multiplier = 1.5
        if confidence >= 50:
            # Scale from 1.0 to 1.5 as confidence goes from 50 to 100
            confidence_multiplier = 1.0 + (confidence - 50) / 100
        else:
            # Scale from 0.5 to 1.0 as confidence goes from 0 to 50
            confidence_multiplier = 0.5 + (confidence / 100)

        # Apply confidence-adjusted risk
        adjusted_risk_percent = risk_percent * confidence_multiplier
        risk_amount = balance * (adjusted_risk_percent / 100)

        # Position size = risk amount / price
        size = risk_amount / price if price > 0 else 0
        return round(size, 2)

    def place_order(
        self,
        token_id: str,
        side: OrderSide,
        price: float,
        size: float,
        market_question: str = "",
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trailing_stop_percent: Optional[float] = None
    ) -> Optional[Order]:
        """
        Place an order on Polymarket with optional stop-loss and take-profit.

        Args:
            token_id: The token to trade
            side: BUY or SELL
            price: Limit price (0-1)
            size: Number of shares
            market_question: Market description for logging
            stop_loss: Price at which to exit to limit losses
            take_profit: Price at which to exit to lock in profits
            trailing_stop_percent: Trailing stop as percentage (e.g., 5.0 = 5%)
        """
        if size < 1:
            logger.debug(f"Order size too small: {size}")
            return None

        order_id = f"order_{int(time.time() * 1000)}"

        logger.info("=" * 50)
        logger.info("PLACING ORDER",
            mode="LIVE" if self.is_live_mode() else "PAPER",
            market=market_question[:50],
            side=side.value,
            price=price,
            size=size,
            total=price * size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop=trailing_stop_percent
        )

        if not self.is_live_mode():
            # Paper trading
            order = self._execute_paper_order(order_id, token_id, side, price, size, market_question)
        else:
            # Live trading
            order = self._execute_live_order(order_id, token_id, side, price, size, market_question)

        # Register stop-loss tracking if order succeeded and stops are specified
        if order and order.status in [OrderStatus.FILLED, OrderStatus.LIVE]:
            if stop_loss or take_profit or trailing_stop_percent:
                self.register_stops(
                    token_id=token_id,
                    market_question=market_question,
                    side=side.value,
                    size=size,
                    entry_price=price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    trailing_stop_percent=trailing_stop_percent
                )

        return order

    def _execute_paper_order(
        self,
        order_id: str,
        token_id: str,
        side: OrderSide,
        price: float,
        size: float,
        market_question: str
    ) -> Order:
        """Execute paper trade"""
        cost = price * size

        if side == OrderSide.BUY:
            if cost > self.paper_balance:
                logger.warning(f"Insufficient balance", balance=self.paper_balance, required=cost)
                return Order(
                    order_id=order_id,
                    token_id=token_id,
                    side=side,
                    price=price,
                    size=size,
                    status=OrderStatus.FAILED,
                    timestamp=datetime.now().isoformat()
                )

            self.paper_balance -= cost

            # Update position
            if token_id in self.paper_positions:
                pos = self.paper_positions[token_id]
                total_size = pos["size"] + size
                pos["avg_price"] = (pos["avg_price"] * pos["size"] + price * size) / total_size
                pos["size"] = total_size
            else:
                self.paper_positions[token_id] = {
                    "size": size,
                    "avg_price": price,
                    "market_question": market_question
                }

            logger.info(f"[PAPER] BUY executed", size=size, price=price, new_balance=self.paper_balance)

            # Record trade entry for performance tracking
            if self.performance_tracker:
                self.performance_tracker.record_entry(
                    trade_id=order_id,
                    token_id=token_id,
                    market_question=market_question,
                    side="BUY",
                    entry_price=price,
                    entry_size=size
                )

        else:  # SELL
            if token_id not in self.paper_positions or self.paper_positions[token_id]["size"] < size:
                logger.warning("Insufficient position to sell")
                return Order(
                    order_id=order_id,
                    token_id=token_id,
                    side=side,
                    price=price,
                    size=size,
                    status=OrderStatus.FAILED,
                    timestamp=datetime.now().isoformat()
                )

            self.paper_balance += cost
            self.paper_positions[token_id]["size"] -= size

            if self.paper_positions[token_id]["size"] <= 0:
                del self.paper_positions[token_id]

            logger.info(f"[PAPER] SELL executed", size=size, price=price, new_balance=self.paper_balance)

            # Record trade exit for performance tracking
            # Find matching open trade by token_id
            if self.performance_tracker:
                # Look for open trades with this token_id
                matching_trade_id = None
                for trade_id, trade in self.performance_tracker.trades.items():
                    if trade.token_id == token_id and trade.side == "BUY":
                        matching_trade_id = trade_id
                        break
                if matching_trade_id:
                    self.performance_tracker.record_exit(
                        trade_id=matching_trade_id,
                        exit_price=price,
                        exit_reason="signal"  # Default, will be overwritten by stop/TP exits
                    )
                    # Record equity point
                    self.performance_tracker.record_equity_point(self.paper_balance)

        order = Order(
            order_id=order_id,
            token_id=token_id,
            side=side,
            price=price,
            size=size,
            status=OrderStatus.FILLED,
            filled_size=size,
            timestamp=datetime.now().isoformat()
        )

        self.orders[order_id] = order
        return order

    def _execute_live_order(
        self,
        order_id: str,
        token_id: str,
        side: OrderSide,
        price: float,
        size: float,
        market_question: str
    ) -> Optional[Order]:
        """Execute live trade via CLOB API"""
        if not self.client:
            logger.error("CLOB client not initialized. Install py-clob-client: pip install py-clob-client")
            return None

        try:
            from py_clob_client.clob_types import OrderArgs, OrderType
            from py_clob_client.order_builder.constants import BUY, SELL

            # Map side to py-clob-client constants
            clob_side = BUY if side == OrderSide.BUY else SELL

            logger.info(f"[LIVE] Creating order", token_id=token_id[:20], side=clob_side, price=price, size=size)

            # Create order args with proper format
            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side=clob_side
            )

            # Create and post order
            signed_order = self.client.create_order(order_args)
            logger.debug("[LIVE] Order signed, submitting...")

            response = self.client.post_order(signed_order, OrderType.GTC)

            if response:
                actual_order_id = response.get("orderID", order_id)
                status = response.get("status", "unknown")
                logger.info(f"[LIVE] Order response", order_id=actual_order_id, status=status)

                # Check if order was accepted
                if status in ["matched", "live", "LIVE", "MATCHED"]:
                    order = Order(
                        order_id=actual_order_id,
                        token_id=token_id,
                        side=side,
                        price=price,
                        size=size,
                        status=OrderStatus.LIVE if status.lower() == "live" else OrderStatus.FILLED,
                        timestamp=datetime.now().isoformat()
                    )
                    self.orders[actual_order_id] = order
                    logger.info("[LIVE] Order placed successfully!")
                    return order
                else:
                    error_msg = response.get("errorMsg", response.get("error", ""))
                    logger.warning(f"[LIVE] Order status: {status}", error=error_msg)
                    return None
            else:
                logger.warning("[LIVE] Order failed: empty response")
                return None

        except Exception as e:
            logger.error(f"[LIVE] Error placing order: {e}", exc_info=True)
            return None

    def register_stops(
        self,
        token_id: str,
        market_question: str,
        side: str,
        size: float,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trailing_stop_percent: Optional[float] = None
    ) -> None:
        """
        Register stop-loss and take-profit levels for a position.
        """
        position = PositionWithStops(
            token_id=token_id,
            market_question=market_question,
            side=side,
            size=size,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop_percent=trailing_stop_percent,
            highest_price_seen=entry_price,
            created_at=datetime.now().isoformat()
        )
        self.positions_with_stops[token_id] = position
        logger.info("[STOPS] Registered", stop_loss=stop_loss, take_profit=take_profit, trailing_stop=trailing_stop_percent)

    def check_stops(self) -> List[Dict]:
        """
        Check all positions against their stop-loss/take-profit levels.
        Returns list of positions that should be exited with reasons.
        """
        exits_needed = []

        for token_id, pos in list(self.positions_with_stops.items()):
            # Get current market price
            prices = self.get_market_price(token_id)
            current_price = prices["mid_price"]

            # Check if exit condition met
            should_exit, reason = pos.should_exit(current_price)

            if should_exit:
                pnl, pnl_pct = pos.get_unrealized_pnl(current_price)
                exits_needed.append({
                    "token_id": token_id,
                    "market_question": pos.market_question,
                    "side": pos.side,
                    "size": pos.size,
                    "entry_price": pos.entry_price,
                    "current_price": current_price,
                    "reason": reason,
                    "pnl": pnl,
                    "pnl_percent": pnl_pct
                })

        return exits_needed

    def execute_stop_exits(self, auto_execute: bool = False) -> List[Order]:
        """
        Check stops and execute exit orders for positions that hit their levels.

        Args:
            auto_execute: If True, actually place sell orders. If False, just log.

        Returns:
            List of executed orders
        """
        exits = self.check_stops()
        orders = []

        for exit_info in exits:
            token_id = exit_info["token_id"]
            size = exit_info["size"]
            current_price = exit_info["current_price"]

            logger.warning("STOP TRIGGERED",
                market=exit_info['market_question'][:50],
                reason=exit_info['reason'],
                entry_price=exit_info['entry_price'],
                current_price=current_price,
                pnl=exit_info['pnl'],
                pnl_percent=exit_info['pnl_percent']
            )

            if auto_execute:
                # Determine exit reason from the reason string
                exit_reason = "signal"
                reason_str = exit_info["reason"].lower()
                if "stop-loss" in reason_str or "stop loss" in reason_str:
                    exit_reason = "stop_loss"
                elif "take-profit" in reason_str or "take profit" in reason_str:
                    exit_reason = "take_profit"
                elif "trailing" in reason_str:
                    exit_reason = "trailing_stop"

                # Record exit with proper reason for performance tracking
                self.record_trade_exit_with_reason(token_id, current_price, exit_reason)

                # Place sell order to exit position
                order = self.place_order(
                    token_id=token_id,
                    side=OrderSide.SELL,
                    price=current_price,
                    size=size,
                    market_question=exit_info["market_question"]
                )

                if order and order.status in [OrderStatus.FILLED, OrderStatus.LIVE]:
                    # Remove from stop tracking
                    if token_id in self.positions_with_stops:
                        del self.positions_with_stops[token_id]
                    orders.append(order)
                    logger.info("[STOPS] Exit order placed successfully")
            else:
                logger.debug("[STOPS] Would exit position (auto_execute=False)")

        return orders

    def get_stops_summary(self) -> List[Dict]:
        """
        Get summary of all positions with stops.
        """
        summary = []

        for token_id, pos in self.positions_with_stops.items():
            prices = self.get_market_price(token_id)
            current_price = prices["mid_price"]
            pnl, pnl_pct = pos.get_unrealized_pnl(current_price)

            # Calculate distance to stops
            sl_distance = None
            tp_distance = None

            if pos.stop_loss:
                sl_distance = ((current_price - pos.stop_loss) / current_price) * 100

            if pos.take_profit:
                tp_distance = ((pos.take_profit - current_price) / current_price) * 100

            summary.append({
                "token_id": token_id,
                "market_question": pos.market_question,
                "side": pos.side,
                "size": pos.size,
                "entry_price": pos.entry_price,
                "current_price": current_price,
                "stop_loss": pos.stop_loss,
                "take_profit": pos.take_profit,
                "trailing_stop_percent": pos.trailing_stop_percent,
                "highest_price_seen": pos.highest_price_seen,
                "pnl": pnl,
                "pnl_percent": pnl_pct,
                "sl_distance_pct": sl_distance,
                "tp_distance_pct": tp_distance
            })

        return summary

    def print_stops_status(self):
        """Print status of all positions with stops."""
        summary = self.get_stops_summary()

        if not summary:
            print("\nNo positions with stop-loss/take-profit tracking.")
            return

        print(f"\n{'='*60}")
        print("STOP-LOSS/TAKE-PROFIT STATUS")
        print(f"{'='*60}")

        for pos in summary:
            print(f"\n{pos['market_question'][:50]}...")
            print(f"  Entry: {pos['entry_price']:.2%} | Current: {pos['current_price']:.2%}")
            print(f"  Size: {pos['size']:.2f} | P&L: ${pos['pnl']:+.2f} ({pos['pnl_percent']:+.1f}%)")

            if pos['stop_loss']:
                print(f"  Stop-Loss: {pos['stop_loss']:.2%} ({pos['sl_distance_pct']:+.1f}% away)")

            if pos['take_profit']:
                print(f"  Take-Profit: {pos['take_profit']:.2%} ({pos['tp_distance_pct']:+.1f}% away)")

            if pos['trailing_stop_percent']:
                trailing_price = pos['highest_price_seen'] * (1 - pos['trailing_stop_percent'] / 100)
                print(f"  Trailing Stop: {pos['trailing_stop_percent']:.1f}% (trigger at {trailing_price:.2%}, high: {pos['highest_price_seen']:.2%})")

    def update_drawdown(self) -> DrawdownState:
        """
        Update drawdown state based on current portfolio value.
        Returns the updated drawdown state.
        """
        summary = self.get_portfolio_summary()
        portfolio_value = summary["total_value"]

        # Update drawdown state
        self.drawdown_state.update(portfolio_value)

        # Check if drawdown exceeds maximum threshold
        if self.drawdown_state.current_drawdown_percent >= self.drawdown_config.max_drawdown_percent:
            if not self.drawdown_state.trading_halted:
                self.drawdown_state.trading_halted = True
                self.drawdown_state.halt_reason = (
                    f"Max drawdown exceeded: {self.drawdown_state.current_drawdown_percent:.1f}% "
                    f"(limit: {self.drawdown_config.max_drawdown_percent:.1f}%)"
                )
                logger.critical("TRADING HALTED - MAXIMUM DRAWDOWN EXCEEDED",
                    current_drawdown=self.drawdown_state.current_drawdown_percent,
                    max_allowed=self.drawdown_config.max_drawdown_percent,
                    high_water_mark=self.drawdown_state.high_water_mark,
                    current_value=portfolio_value,
                    recovery_threshold=self.drawdown_config.recovery_threshold_percent
                )

        # Check if we can resume trading after recovery
        elif self.drawdown_state.trading_halted:
            if self.drawdown_state.current_drawdown_percent <= self.drawdown_config.recovery_threshold_percent:
                self.drawdown_state.trading_halted = False
                self.drawdown_state.halt_reason = ""
                logger.info("TRADING RESUMED - DRAWDOWN RECOVERED",
                    current_drawdown=self.drawdown_state.current_drawdown_percent,
                    recovery_threshold=self.drawdown_config.recovery_threshold_percent
                )

        # Issue warning if approaching max drawdown
        elif self.drawdown_state.current_drawdown_percent >= self.drawdown_config.warning_threshold_percent:
            logger.warning("DRAWDOWN WARNING",
                current_drawdown=self.drawdown_state.current_drawdown_percent,
                max_allowed=self.drawdown_config.max_drawdown_percent
            )

        return self.drawdown_state

    def can_trade_drawdown(self) -> tuple[bool, str]:
        """
        Check if trading is allowed based on drawdown limits.
        Returns (can_trade, reason)
        """
        # Update drawdown state first
        self.update_drawdown()

        if self.drawdown_state.trading_halted:
            return False, self.drawdown_state.halt_reason

        return True, ""

    def get_drawdown_summary(self) -> Dict:
        """
        Get summary of drawdown state.
        """
        return {
            "high_water_mark": self.drawdown_state.high_water_mark,
            "current_value": self.drawdown_state.current_value,
            "current_drawdown_percent": self.drawdown_state.current_drawdown_percent,
            "max_drawdown_seen": self.drawdown_state.max_drawdown_seen,
            "max_drawdown_limit": self.drawdown_config.max_drawdown_percent,
            "warning_threshold": self.drawdown_config.warning_threshold_percent,
            "recovery_threshold": self.drawdown_config.recovery_threshold_percent,
            "trading_halted": self.drawdown_state.trading_halted,
            "halt_reason": self.drawdown_state.halt_reason,
            "last_updated": self.drawdown_state.last_updated
        }

    def print_drawdown_status(self):
        """Print current drawdown status."""
        summary = self.get_drawdown_summary()

        print(f"\n{'='*60}")
        print("DRAWDOWN STATUS")
        print(f"{'='*60}")
        print(f"High Water Mark: ${summary['high_water_mark']:,.2f}")
        print(f"Current Value: ${summary['current_value']:,.2f}")
        print(f"Current Drawdown: {summary['current_drawdown_percent']:.1f}%")
        print(f"Max Drawdown Seen: {summary['max_drawdown_seen']:.1f}%")
        print(f"Max Allowed: {summary['max_drawdown_limit']:.1f}%")

        if summary['trading_halted']:
            print(f"\n[!] TRADING HALTED: {summary['halt_reason']}")
        elif summary['current_drawdown_percent'] >= summary['warning_threshold']:
            print(f"\n[!] WARNING: Approaching max drawdown limit")
        else:
            remaining = summary['max_drawdown_limit'] - summary['current_drawdown_percent']
            print(f"\nDrawdown Buffer: {remaining:.1f}% remaining")

    # ========== PORTFOLIO HEAT (CONCURRENT RISK) METHODS ==========

    def calculate_position_risk(self, token_id: str, pos_with_stops: PositionWithStops = None) -> Dict:
        """
        Calculate the risk for a single position.
        Risk = (entry_price - stop_loss) * size

        If no stop-loss is set, uses default_stop_distance_percent from config.
        """
        # Get position data
        if pos_with_stops:
            pos = pos_with_stops
            entry_price = pos.entry_price
            stop_loss = pos.stop_loss
            size = pos.size
        elif token_id in self.positions_with_stops:
            pos = self.positions_with_stops[token_id]
            entry_price = pos.entry_price
            stop_loss = pos.stop_loss
            size = pos.size
        elif token_id in self.paper_positions:
            pos_data = self.paper_positions[token_id]
            entry_price = pos_data["avg_price"]
            stop_loss = None
            size = pos_data["size"]
        else:
            return {"risk_amount": 0, "risk_percent": 0, "stop_distance": 0}

        # Calculate stop distance
        if stop_loss and stop_loss > 0:
            stop_distance = abs(entry_price - stop_loss)
        else:
            # Use default stop distance if no stop-loss set
            stop_distance = entry_price * (self.heat_config.default_stop_distance_percent / 100)

        # Risk amount = potential loss if stop-loss is hit
        risk_amount = stop_distance * size

        # Get portfolio value for percentage calculation
        portfolio = self.get_portfolio_summary()
        portfolio_value = portfolio["total_value"]

        risk_percent = (risk_amount / portfolio_value * 100) if portfolio_value > 0 else 0

        return {
            "token_id": token_id,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "stop_distance": stop_distance,
            "size": size,
            "risk_amount": risk_amount,
            "risk_percent": risk_percent
        }

    def update_portfolio_heat(self) -> PortfolioHeatState:
        """
        Calculate and update total portfolio heat (concurrent risk).
        Returns the updated heat state.
        """
        portfolio = self.get_portfolio_summary()
        portfolio_value = portfolio["total_value"]

        total_risk = 0.0
        position_risks = {}

        # Calculate risk for each position with stops
        for token_id in self.positions_with_stops:
            risk_info = self.calculate_position_risk(token_id)
            position_risks[token_id] = risk_info
            total_risk += risk_info["risk_amount"]

        # Also include paper positions without stops (using default stop distance)
        for token_id in self.paper_positions:
            if token_id not in self.positions_with_stops:
                risk_info = self.calculate_position_risk(token_id)
                position_risks[token_id] = risk_info
                total_risk += risk_info["risk_amount"]

        # Calculate total risk percentage
        total_risk_percent = (total_risk / portfolio_value * 100) if portfolio_value > 0 else 0

        # Update state
        self.heat_state.total_risk_amount = total_risk
        self.heat_state.total_risk_percent = total_risk_percent
        self.heat_state.num_positions_at_risk = len(position_risks)
        self.heat_state.position_risks = position_risks
        self.heat_state.last_updated = datetime.now().isoformat()

        # Check warning/exceeded thresholds
        self.heat_state.heat_warning = total_risk_percent >= self.heat_config.warning_heat_percent
        self.heat_state.heat_exceeded = total_risk_percent >= self.heat_config.max_portfolio_heat_percent

        return self.heat_state

    def can_add_position_heat(self, proposed_risk_amount: float = 0, proposed_risk_percent: float = None) -> tuple[bool, str]:
        """
        Check if a new position can be added without exceeding heat limits.

        Args:
            proposed_risk_amount: Dollar amount of risk for the new position
            proposed_risk_percent: Risk as percentage of portfolio (alternative to dollar amount)

        Returns:
            (can_add, reason) tuple
        """
        # Update current heat
        self.update_portfolio_heat()

        portfolio = self.get_portfolio_summary()
        portfolio_value = portfolio["total_value"]

        # Convert to percent if dollar amount given
        if proposed_risk_percent is None and proposed_risk_amount > 0:
            proposed_risk_percent = (proposed_risk_amount / portfolio_value * 100) if portfolio_value > 0 else 0

        proposed_risk_percent = proposed_risk_percent or 0

        # Check per-position risk limit
        if proposed_risk_percent > self.heat_config.per_position_max_risk_percent:
            return False, (
                f"Position risk {proposed_risk_percent:.1f}% exceeds per-position limit "
                f"({self.heat_config.per_position_max_risk_percent:.1f}%)"
            )

        # Check total portfolio heat with proposed position
        new_total_heat = self.heat_state.total_risk_percent + proposed_risk_percent

        if new_total_heat > self.heat_config.max_portfolio_heat_percent:
            available_heat = self.heat_config.max_portfolio_heat_percent - self.heat_state.total_risk_percent
            return False, (
                f"Adding position would exceed max heat: {new_total_heat:.1f}% "
                f"(limit: {self.heat_config.max_portfolio_heat_percent:.1f}%, "
                f"available: {available_heat:.1f}%)"
            )

        return True, ""

    def can_trade_heat(self) -> tuple[bool, str]:
        """
        Check if trading is allowed based on portfolio heat limits.
        Returns (can_trade, reason)
        """
        self.update_portfolio_heat()

        if self.heat_state.heat_exceeded:
            return False, (
                f"Portfolio heat limit exceeded: {self.heat_state.total_risk_percent:.1f}% "
                f"(limit: {self.heat_config.max_portfolio_heat_percent:.1f}%)"
            )

        return True, ""

    def get_heat_summary(self) -> Dict:
        """Get summary of portfolio heat state."""
        self.update_portfolio_heat()

        portfolio = self.get_portfolio_summary()
        available_heat = self.heat_config.max_portfolio_heat_percent - self.heat_state.total_risk_percent

        return {
            "total_risk_amount": self.heat_state.total_risk_amount,
            "total_risk_percent": self.heat_state.total_risk_percent,
            "num_positions_at_risk": self.heat_state.num_positions_at_risk,
            "max_heat_percent": self.heat_config.max_portfolio_heat_percent,
            "warning_threshold": self.heat_config.warning_heat_percent,
            "per_position_limit": self.heat_config.per_position_max_risk_percent,
            "available_heat_percent": max(0, available_heat),
            "available_heat_amount": portfolio["total_value"] * max(0, available_heat) / 100,
            "heat_warning": self.heat_state.heat_warning,
            "heat_exceeded": self.heat_state.heat_exceeded,
            "position_risks": self.heat_state.position_risks,
            "last_updated": self.heat_state.last_updated
        }

    def print_heat_status(self):
        """Print current portfolio heat status."""
        summary = self.get_heat_summary()

        print(f"\n{'='*60}")
        print("PORTFOLIO HEAT STATUS")
        print(f"{'='*60}")
        print(f"Total Risk: ${summary['total_risk_amount']:,.2f} ({summary['total_risk_percent']:.1f}%)")
        print(f"Positions at Risk: {summary['num_positions_at_risk']}")
        print(f"Max Heat Limit: {summary['max_heat_percent']:.1f}%")
        print(f"Available Heat: {summary['available_heat_percent']:.1f}% (${summary['available_heat_amount']:,.2f})")

        if summary['heat_exceeded']:
            print(f"\n[!] HEAT LIMIT EXCEEDED - No new positions allowed")
        elif summary['heat_warning']:
            print(f"\n[!] WARNING: Approaching max heat limit")

        # Show individual position risks if any
        if summary['position_risks']:
            print(f"\n{'-'*40}")
            print("POSITION RISK BREAKDOWN")
            print(f"{'-'*40}")
            for token_id, risk in summary['position_risks'].items():
                market_name = "Unknown"
                if token_id in self.positions_with_stops:
                    market_name = self.positions_with_stops[token_id].market_question[:40]
                elif token_id in self.paper_positions:
                    market_name = self.paper_positions[token_id].get("market_question", "Unknown")[:40]
                print(f"  {market_name}...")
                print(f"    Risk: ${risk['risk_amount']:.2f} ({risk['risk_percent']:.1f}%)")
                print(f"    Stop Distance: {risk['stop_distance']:.2%}")

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order"""
        if order_id not in self.orders:
            return False

        order = self.orders[order_id]

        if not self.is_live_mode():
            order.status = OrderStatus.CANCELLED
            print(f"[PAPER] Order {order_id} cancelled")
            return True

        if self.client:
            try:
                response = self.client.cancel(order_id)
                if response.get("success"):
                    order.status = OrderStatus.CANCELLED
                    return True
            except Exception as e:
                print(f"Error cancelling order: {e}")

        return False

    def get_positions(self) -> List[Position]:
        """Get current positions"""
        positions = []

        if not self.is_live_mode():
            for token_id, pos in self.paper_positions.items():
                market_price = self.get_market_price(token_id)
                current_price = market_price["mid_price"]
                pnl = (current_price - pos["avg_price"]) * pos["size"]
                pnl_percent = ((current_price / pos["avg_price"]) - 1) * 100 if pos["avg_price"] > 0 else 0

                positions.append(Position(
                    token_id=token_id,
                    market_question=pos.get("market_question", "Unknown"),
                    outcome="Yes",
                    size=pos["size"],
                    avg_price=pos["avg_price"],
                    current_price=current_price,
                    pnl=pnl,
                    pnl_percent=pnl_percent
                ))

        return positions

    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary"""
        positions = self.get_positions()

        total_value = self.paper_balance if not self.is_live_mode() else 0
        total_pnl = 0
        position_value = 0

        for pos in positions:
            position_value += pos.current_price * pos.size
            total_pnl += pos.pnl

        total_value += position_value

        return {
            "cash_balance": self.paper_balance if not self.is_live_mode() else 0,
            "position_value": position_value,
            "total_value": total_value,
            "total_pnl": total_pnl,
            "num_positions": len(positions),
            "positions": positions
        }

    def print_portfolio(self):
        """Print portfolio summary"""
        summary = self.get_portfolio_summary()

        print(f"\n{'='*60}")
        print("PORTFOLIO SUMMARY")
        print(f"{'='*60}")
        print(f"Mode: {'LIVE' if self.is_live_mode() else 'PAPER'}")
        print(f"Cash Balance: ${summary['cash_balance']:,.2f}")
        print(f"Position Value: ${summary['position_value']:,.2f}")
        print(f"Total Value: ${summary['total_value']:,.2f}")
        print(f"Total P&L: ${summary['total_pnl']:+,.2f}")
        print(f"Positions: {summary['num_positions']}")

        if summary['positions']:
            print(f"\n{'-'*60}")
            print("OPEN POSITIONS")
            print(f"{'-'*60}")
            for pos in summary['positions']:
                print(f"\n{pos.market_question[:50]}...")
                print(f"  Size: {pos.size:.2f} | Avg: {pos.avg_price:.2%} | Current: {pos.current_price:.2%}")
                print(f"  P&L: ${pos.pnl:+.2f} ({pos.pnl_percent:+.1f}%)")

    # ========== PERFORMANCE METRICS METHODS ==========

    def get_performance_metrics(self, period_days: int = None) -> Optional[Dict]:
        """
        Get trading performance metrics.

        Args:
            period_days: Only include trades from the last N days (None = all time)

        Returns:
            Dictionary of performance metrics, or None if tracker not available
        """
        if not self.performance_tracker:
            return None

        metrics = self.performance_tracker.get_metrics(period_days)
        return metrics.to_dict()

    def print_performance_summary(self, period_days: int = None) -> None:
        """
        Print trading performance summary.

        Args:
            period_days: Only include trades from the last N days (None = all time)
        """
        if not self.performance_tracker:
            print("\nPerformance tracking not available. Install performance_metrics module.")
            return

        self.performance_tracker.print_summary(period_days)

    def get_trade_history(self, limit: int = None) -> List[Dict]:
        """
        Get trade history.

        Args:
            limit: Maximum number of trades to return (most recent first)

        Returns:
            List of trade records as dictionaries
        """
        if not self.performance_tracker:
            return []

        return self.performance_tracker.get_trade_history(limit=limit)

    def record_trade_entry_with_signal(
        self,
        trade_id: str,
        token_id: str,
        market_question: str,
        side: str,
        entry_price: float,
        entry_size: float,
        signal_type: str = None,
        signal_confidence: float = None,
        volatility: float = None
    ) -> None:
        """
        Record a trade entry with full signal information for analysis.

        This method should be called by the trading bot when executing a signal
        to capture the signal type and confidence for later analysis.
        """
        if self.performance_tracker:
            self.performance_tracker.record_entry(
                trade_id=trade_id,
                token_id=token_id,
                market_question=market_question,
                side=side,
                entry_price=entry_price,
                entry_size=entry_size,
                signal_type=signal_type,
                signal_confidence=signal_confidence,
                volatility=volatility
            )

    def record_trade_exit_with_reason(
        self,
        token_id: str,
        exit_price: float,
        exit_reason: str = "signal"
    ) -> None:
        """
        Record a trade exit with the exit reason.

        Args:
            token_id: Token ID to find the matching trade
            exit_price: Exit price
            exit_reason: Why the trade was exited (signal, stop_loss, take_profit, trailing_stop, manual)
        """
        if self.performance_tracker:
            # Find matching open trade by token_id
            matching_trade_id = None
            for trade_id, trade in self.performance_tracker.trades.items():
                if trade.token_id == token_id:
                    matching_trade_id = trade_id
                    break
            if matching_trade_id:
                self.performance_tracker.record_exit(
                    trade_id=matching_trade_id,
                    exit_price=exit_price,
                    exit_reason=exit_reason
                )
                # Record equity point
                portfolio = self.get_portfolio_summary()
                self.performance_tracker.record_equity_point(portfolio["total_value"])


def test_paper_trading():
    """Test paper trading functionality"""
    print("Testing Paper Trading")
    print("=" * 60)

    executor = TradeExecutor()
    executor.trading_mode = "paper"

    # Get a sample market
    url = f"{GAMMA_API_URL}/markets"
    params = {"limit": 1, "active": "true"}
    response = requests.get(url, params=params, timeout=30)
    markets = response.json()

    if not markets:
        print("No markets found")
        return

    market = markets[0]
    clob_tokens = market.get("clobTokenIds", "[]")
    if isinstance(clob_tokens, str):
        clob_tokens = json.loads(clob_tokens)

    if not clob_tokens:
        print("No tokens found")
        return

    token_id = clob_tokens[0]
    question = market.get("question", "Unknown")

    # Get current price
    prices = executor.get_market_price(token_id)
    print(f"\nMarket: {question[:50]}...")
    print(f"Best Bid: {prices['best_bid']:.2%}")
    print(f"Best Ask: {prices['best_ask']:.2%}")
    print(f"Mid Price: {prices['mid_price']:.2%}")

    # Test buy order
    print("\n--- Test BUY Order ---")
    buy_price = prices['best_ask']
    buy_size = 100

    order = executor.place_order(
        token_id=token_id,
        side=OrderSide.BUY,
        price=buy_price,
        size=buy_size,
        market_question=question
    )

    if order:
        print(f"Order Status: {order.status.value}")

    # Print portfolio
    executor.print_portfolio()

    # Test sell order
    print("\n--- Test SELL Order ---")
    sell_price = prices['best_bid']
    sell_size = 50

    order = executor.place_order(
        token_id=token_id,
        side=OrderSide.SELL,
        price=sell_price,
        size=sell_size,
        market_question=question
    )

    if order:
        print(f"Order Status: {order.status.value}")

    # Final portfolio
    executor.print_portfolio()


if __name__ == "__main__":
    test_paper_trading()
