"""
Performance Metrics Tracking for Polymarket Trading Bot

Tracks and calculates key trading performance metrics:
- Win rate and trade statistics
- Profit factor and average win/loss
- Sharpe ratio and risk-adjusted returns
- Maximum drawdown tracking
- Trade history with detailed records
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from enum import Enum
import math


class TradeOutcome(Enum):
    WIN = "WIN"
    LOSS = "LOSS"
    BREAKEVEN = "BREAKEVEN"
    OPEN = "OPEN"


@dataclass
class TradeRecord:
    """Record of a single trade for performance tracking"""
    trade_id: str
    token_id: str
    market_question: str
    side: str  # "BUY" or "SELL"
    entry_price: float
    entry_time: str
    entry_size: float
    entry_cost: float  # entry_price * entry_size

    # Exit info (filled when trade is closed)
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    exit_size: Optional[float] = None
    exit_proceeds: Optional[float] = None  # exit_price * exit_size

    # P&L
    realized_pnl: Optional[float] = None
    realized_pnl_percent: Optional[float] = None
    outcome: TradeOutcome = TradeOutcome.OPEN

    # Signal info for analysis
    signal_type: Optional[str] = None
    signal_confidence: Optional[float] = None
    volatility_at_entry: Optional[float] = None

    # Exit reason
    exit_reason: Optional[str] = None  # "signal", "stop_loss", "take_profit", "trailing_stop", "manual"

    # Duration
    hold_time_seconds: Optional[int] = None

    def close(self, exit_price: float, exit_time: str = None, exit_reason: str = "signal") -> None:
        """Close the trade and calculate P&L"""
        self.exit_price = exit_price
        self.exit_time = exit_time or datetime.now().isoformat()
        self.exit_size = self.entry_size
        self.exit_proceeds = exit_price * self.entry_size
        self.exit_reason = exit_reason

        # Calculate P&L
        if self.side == "BUY":
            self.realized_pnl = self.exit_proceeds - self.entry_cost
        else:
            self.realized_pnl = self.entry_cost - self.exit_proceeds

        self.realized_pnl_percent = (self.realized_pnl / self.entry_cost) * 100 if self.entry_cost > 0 else 0

        # Determine outcome
        if self.realized_pnl > 0.01:  # Small threshold for "win"
            self.outcome = TradeOutcome.WIN
        elif self.realized_pnl < -0.01:
            self.outcome = TradeOutcome.LOSS
        else:
            self.outcome = TradeOutcome.BREAKEVEN

        # Calculate hold time
        try:
            entry_dt = datetime.fromisoformat(self.entry_time.replace('Z', '+00:00'))
            exit_dt = datetime.fromisoformat(self.exit_time.replace('Z', '+00:00'))
            self.hold_time_seconds = int((exit_dt - entry_dt).total_seconds())
        except:
            self.hold_time_seconds = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        d = asdict(self)
        d['outcome'] = self.outcome.value
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> 'TradeRecord':
        """Create from dictionary"""
        data = data.copy()
        data['outcome'] = TradeOutcome(data.get('outcome', 'OPEN'))
        return cls(**data)


@dataclass
class PerformanceMetrics:
    """Aggregated performance metrics"""
    # Basic counts
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0
    open_trades: int = 0

    # Win rate
    win_rate: float = 0.0  # percentage

    # P&L metrics
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # Ratios
    profit_factor: float = 0.0  # gross_profit / gross_loss
    average_rr_ratio: float = 0.0  # average win / average loss
    expectancy: float = 0.0  # (win_rate * avg_win) - (loss_rate * avg_loss)

    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0

    # Trade analysis
    avg_hold_time_seconds: float = 0.0
    avg_pnl_per_trade: float = 0.0

    # By exit reason
    stop_loss_exits: int = 0
    take_profit_exits: int = 0
    trailing_stop_exits: int = 0
    signal_exits: int = 0

    # Time period
    period_start: str = ""
    period_end: str = ""
    trading_days: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


class PerformanceTracker:
    """
    Tracks and calculates trading performance metrics.

    Usage:
        tracker = PerformanceTracker()

        # Record a trade entry
        tracker.record_entry(
            trade_id="trade_123",
            token_id="token_abc",
            market_question="Will BTC go up?",
            side="BUY",
            entry_price=0.55,
            entry_size=100,
            signal_type="STRONG_BUY",
            signal_confidence=85.0
        )

        # Record trade exit
        tracker.record_exit(
            trade_id="trade_123",
            exit_price=0.62,
            exit_reason="take_profit"
        )

        # Get metrics
        metrics = tracker.get_metrics()
    """

    def __init__(self, data_dir: str = None, auto_save: bool = True):
        """
        Initialize performance tracker.

        Args:
            data_dir: Directory to save trade history (default: ./performance_data)
            auto_save: Whether to automatically save after each trade
        """
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), "performance_data")
        self.auto_save = auto_save

        # Trade records
        self.trades: Dict[str, TradeRecord] = {}  # trade_id -> TradeRecord
        self.closed_trades: List[TradeRecord] = []

        # Equity curve for Sharpe/Sortino calculation
        self.equity_curve: List[Tuple[str, float]] = []  # [(timestamp, equity), ...]
        self.initial_equity: float = 10000.0  # Default starting equity

        # Running stats
        self._high_water_mark: float = 0.0
        self._max_drawdown: float = 0.0

        # Create data directory
        os.makedirs(self.data_dir, exist_ok=True)

        # Load existing data
        self._load_history()

    def set_initial_equity(self, equity: float) -> None:
        """Set the initial equity for calculations"""
        self.initial_equity = equity
        self._high_water_mark = equity

    def record_entry(
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
    ) -> TradeRecord:
        """
        Record a new trade entry.

        Returns the created TradeRecord.
        """
        record = TradeRecord(
            trade_id=trade_id,
            token_id=token_id,
            market_question=market_question,
            side=side,
            entry_price=entry_price,
            entry_time=datetime.now().isoformat(),
            entry_size=entry_size,
            entry_cost=entry_price * entry_size,
            signal_type=signal_type,
            signal_confidence=signal_confidence,
            volatility_at_entry=volatility
        )

        self.trades[trade_id] = record

        if self.auto_save:
            self._save_history()

        return record

    def record_exit(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str = "signal"
    ) -> Optional[TradeRecord]:
        """
        Record a trade exit.

        Args:
            trade_id: The trade ID to close
            exit_price: Exit price
            exit_reason: Why the trade was exited (signal, stop_loss, take_profit, trailing_stop, manual)

        Returns the closed TradeRecord, or None if trade not found.
        """
        if trade_id not in self.trades:
            return None

        record = self.trades[trade_id]
        record.close(exit_price, exit_reason=exit_reason)

        # Move to closed trades
        self.closed_trades.append(record)
        del self.trades[trade_id]

        if self.auto_save:
            self._save_history()

        return record

    def record_equity_point(self, equity: float, timestamp: str = None) -> None:
        """
        Record an equity curve point for Sharpe/Sortino calculation.
        Should be called periodically (e.g., daily or after each trade).
        """
        timestamp = timestamp or datetime.now().isoformat()
        self.equity_curve.append((timestamp, equity))

        # Update high water mark and drawdown
        if equity > self._high_water_mark:
            self._high_water_mark = equity

        current_drawdown = (self._high_water_mark - equity) / self._high_water_mark if self._high_water_mark > 0 else 0
        if current_drawdown > self._max_drawdown:
            self._max_drawdown = current_drawdown

    def get_metrics(self, period_days: int = None) -> PerformanceMetrics:
        """
        Calculate and return performance metrics.

        Args:
            period_days: Only include trades from the last N days (None = all time)

        Returns:
            PerformanceMetrics with calculated values
        """
        # Filter trades by period if specified
        if period_days:
            cutoff = datetime.now() - timedelta(days=period_days)
            cutoff_str = cutoff.isoformat()
            trades = [t for t in self.closed_trades if t.exit_time and t.exit_time >= cutoff_str]
        else:
            trades = self.closed_trades

        metrics = PerformanceMetrics()

        if not trades:
            return metrics

        # Basic counts
        metrics.total_trades = len(trades)
        metrics.winning_trades = len([t for t in trades if t.outcome == TradeOutcome.WIN])
        metrics.losing_trades = len([t for t in trades if t.outcome == TradeOutcome.LOSS])
        metrics.breakeven_trades = len([t for t in trades if t.outcome == TradeOutcome.BREAKEVEN])
        metrics.open_trades = len(self.trades)

        # Win rate
        completed_trades = metrics.winning_trades + metrics.losing_trades + metrics.breakeven_trades
        metrics.win_rate = (metrics.winning_trades / completed_trades * 100) if completed_trades > 0 else 0

        # P&L calculations
        pnls = [t.realized_pnl for t in trades if t.realized_pnl is not None]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        metrics.total_pnl = sum(pnls)
        metrics.gross_profit = sum(wins)
        metrics.gross_loss = abs(sum(losses))
        metrics.average_win = sum(wins) / len(wins) if wins else 0
        metrics.average_loss = abs(sum(losses)) / len(losses) if losses else 0
        metrics.largest_win = max(wins) if wins else 0
        metrics.largest_loss = abs(min(losses)) if losses else 0

        # Ratios
        metrics.profit_factor = metrics.gross_profit / metrics.gross_loss if metrics.gross_loss > 0 else float('inf') if metrics.gross_profit > 0 else 0
        metrics.average_rr_ratio = metrics.average_win / metrics.average_loss if metrics.average_loss > 0 else float('inf') if metrics.average_win > 0 else 0

        # Expectancy: (win_rate * avg_win) - (loss_rate * avg_loss)
        win_rate_decimal = metrics.win_rate / 100
        loss_rate_decimal = (100 - metrics.win_rate) / 100
        metrics.expectancy = (win_rate_decimal * metrics.average_win) - (loss_rate_decimal * metrics.average_loss)

        # Average P&L per trade
        metrics.avg_pnl_per_trade = metrics.total_pnl / metrics.total_trades if metrics.total_trades > 0 else 0

        # Hold time
        hold_times = [t.hold_time_seconds for t in trades if t.hold_time_seconds is not None]
        metrics.avg_hold_time_seconds = sum(hold_times) / len(hold_times) if hold_times else 0

        # Exit reasons
        metrics.stop_loss_exits = len([t for t in trades if t.exit_reason == "stop_loss"])
        metrics.take_profit_exits = len([t for t in trades if t.exit_reason == "take_profit"])
        metrics.trailing_stop_exits = len([t for t in trades if t.exit_reason == "trailing_stop"])
        metrics.signal_exits = len([t for t in trades if t.exit_reason == "signal"])

        # Max drawdown
        metrics.max_drawdown = self._max_drawdown * self.initial_equity
        metrics.max_drawdown_percent = self._max_drawdown * 100

        # Sharpe and Sortino ratios (if we have equity curve data)
        if len(self.equity_curve) >= 2:
            metrics.sharpe_ratio = self._calculate_sharpe_ratio()
            metrics.sortino_ratio = self._calculate_sortino_ratio()

        # Time period
        if trades:
            entry_times = [t.entry_time for t in trades if t.entry_time]
            exit_times = [t.exit_time for t in trades if t.exit_time]
            if entry_times:
                metrics.period_start = min(entry_times)
            if exit_times:
                metrics.period_end = max(exit_times)

            # Calculate trading days
            if metrics.period_start and metrics.period_end:
                try:
                    start = datetime.fromisoformat(metrics.period_start.replace('Z', '+00:00'))
                    end = datetime.fromisoformat(metrics.period_end.replace('Z', '+00:00'))
                    metrics.trading_days = max(1, (end - start).days)
                except:
                    metrics.trading_days = 1

        return metrics

    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe ratio from equity curve.

        Sharpe = (Mean Return - Risk Free Rate) / Std Dev of Returns
        """
        if len(self.equity_curve) < 2:
            return 0.0

        # Calculate returns
        returns = []
        for i in range(1, len(self.equity_curve)):
            prev_equity = self.equity_curve[i-1][1]
            curr_equity = self.equity_curve[i][1]
            if prev_equity > 0:
                ret = (curr_equity - prev_equity) / prev_equity
                returns.append(ret)

        if not returns:
            return 0.0

        mean_return = sum(returns) / len(returns)

        # Standard deviation
        if len(returns) < 2:
            return 0.0
        variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
        std_dev = math.sqrt(variance) if variance > 0 else 0

        if std_dev == 0:
            return 0.0

        # Annualize (assuming daily returns)
        sharpe = (mean_return - risk_free_rate) / std_dev * math.sqrt(252)
        return round(sharpe, 2)

    def _calculate_sortino_ratio(self, risk_free_rate: float = 0.0, target_return: float = 0.0) -> float:
        """
        Calculate Sortino ratio from equity curve.

        Sortino = (Mean Return - Target Return) / Downside Deviation
        Uses only negative returns for volatility calculation.
        """
        if len(self.equity_curve) < 2:
            return 0.0

        # Calculate returns
        returns = []
        for i in range(1, len(self.equity_curve)):
            prev_equity = self.equity_curve[i-1][1]
            curr_equity = self.equity_curve[i][1]
            if prev_equity > 0:
                ret = (curr_equity - prev_equity) / prev_equity
                returns.append(ret)

        if not returns:
            return 0.0

        mean_return = sum(returns) / len(returns)

        # Downside deviation (only negative returns below target)
        downside_returns = [r for r in returns if r < target_return]
        if not downside_returns:
            return float('inf') if mean_return > target_return else 0.0

        downside_variance = sum((r - target_return) ** 2 for r in downside_returns) / len(downside_returns)
        downside_dev = math.sqrt(downside_variance) if downside_variance > 0 else 0

        if downside_dev == 0:
            return 0.0

        # Annualize
        sortino = (mean_return - target_return) / downside_dev * math.sqrt(252)
        return round(sortino, 2)

    def get_trade_history(self, limit: int = None, include_open: bool = True) -> List[Dict]:
        """
        Get trade history as list of dictionaries.

        Args:
            limit: Maximum number of trades to return (most recent first)
            include_open: Whether to include open trades
        """
        all_trades = list(self.closed_trades)
        if include_open:
            all_trades.extend(self.trades.values())

        # Sort by entry time descending
        all_trades.sort(key=lambda t: t.entry_time, reverse=True)

        if limit:
            all_trades = all_trades[:limit]

        return [t.to_dict() for t in all_trades]

    def print_summary(self, period_days: int = None) -> None:
        """Print a summary of performance metrics"""
        metrics = self.get_metrics(period_days)

        period_str = f"Last {period_days} days" if period_days else "All time"

        print(f"\n{'='*60}")
        print(f"PERFORMANCE SUMMARY ({period_str})")
        print(f"{'='*60}")

        print(f"\nTrade Statistics:")
        print(f"  Total Trades: {metrics.total_trades}")
        print(f"  Wins: {metrics.winning_trades} | Losses: {metrics.losing_trades} | Breakeven: {metrics.breakeven_trades}")
        print(f"  Win Rate: {metrics.win_rate:.1f}%")
        print(f"  Open Positions: {metrics.open_trades}")

        print(f"\nP&L Metrics:")
        print(f"  Total P&L: ${metrics.total_pnl:+,.2f}")
        print(f"  Gross Profit: ${metrics.gross_profit:,.2f}")
        print(f"  Gross Loss: ${metrics.gross_loss:,.2f}")
        print(f"  Average Win: ${metrics.average_win:,.2f}")
        print(f"  Average Loss: ${metrics.average_loss:,.2f}")
        print(f"  Largest Win: ${metrics.largest_win:,.2f}")
        print(f"  Largest Loss: ${metrics.largest_loss:,.2f}")

        print(f"\nRatios:")
        pf_str = f"{metrics.profit_factor:.2f}" if metrics.profit_factor != float('inf') else "inf"
        rr_str = f"{metrics.average_rr_ratio:.2f}" if metrics.average_rr_ratio != float('inf') else "inf"
        print(f"  Profit Factor: {pf_str}")
        print(f"  Avg R:R Ratio: {rr_str}")
        print(f"  Expectancy: ${metrics.expectancy:+,.2f} per trade")

        print(f"\nRisk Metrics:")
        print(f"  Max Drawdown: ${metrics.max_drawdown:,.2f} ({metrics.max_drawdown_percent:.1f}%)")
        print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        print(f"  Sortino Ratio: {metrics.sortino_ratio:.2f}")

        print(f"\nExit Analysis:")
        print(f"  Stop-Loss Exits: {metrics.stop_loss_exits}")
        print(f"  Take-Profit Exits: {metrics.take_profit_exits}")
        print(f"  Trailing Stop Exits: {metrics.trailing_stop_exits}")
        print(f"  Signal Exits: {metrics.signal_exits}")

        if metrics.avg_hold_time_seconds > 0:
            hours = metrics.avg_hold_time_seconds / 3600
            print(f"\nAvg Hold Time: {hours:.1f} hours")

        if metrics.period_start and metrics.period_end:
            print(f"\nPeriod: {metrics.period_start[:10]} to {metrics.period_end[:10]} ({metrics.trading_days} days)")

    def _save_history(self) -> None:
        """Save trade history to disk"""
        data = {
            "version": "1.0",
            "initial_equity": self.initial_equity,
            "high_water_mark": self._high_water_mark,
            "max_drawdown": self._max_drawdown,
            "open_trades": {tid: t.to_dict() for tid, t in self.trades.items()},
            "closed_trades": [t.to_dict() for t in self.closed_trades],
            "equity_curve": self.equity_curve[-1000:],  # Keep last 1000 points
            "last_updated": datetime.now().isoformat()
        }

        filepath = os.path.join(self.data_dir, "trade_history.json")
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save trade history: {e}")

    def _load_history(self) -> None:
        """Load trade history from disk"""
        filepath = os.path.join(self.data_dir, "trade_history.json")

        if not os.path.exists(filepath):
            return

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            self.initial_equity = data.get("initial_equity", 10000.0)
            self._high_water_mark = data.get("high_water_mark", self.initial_equity)
            self._max_drawdown = data.get("max_drawdown", 0.0)

            # Load open trades
            open_trades = data.get("open_trades", {})
            self.trades = {tid: TradeRecord.from_dict(t) for tid, t in open_trades.items()}

            # Load closed trades
            closed_trades = data.get("closed_trades", [])
            self.closed_trades = [TradeRecord.from_dict(t) for t in closed_trades]

            # Load equity curve
            self.equity_curve = data.get("equity_curve", [])

        except Exception as e:
            print(f"Warning: Could not load trade history: {e}")

    def reset(self, confirm: bool = False) -> None:
        """
        Reset all performance data.

        Args:
            confirm: Must be True to actually reset
        """
        if not confirm:
            print("Pass confirm=True to actually reset performance data")
            return

        self.trades = {}
        self.closed_trades = []
        self.equity_curve = []
        self._high_water_mark = self.initial_equity
        self._max_drawdown = 0.0

        # Remove data file
        filepath = os.path.join(self.data_dir, "trade_history.json")
        if os.path.exists(filepath):
            os.remove(filepath)

        print("Performance data reset")


# Convenience function for quick access
def get_tracker(data_dir: str = None) -> PerformanceTracker:
    """Get a PerformanceTracker instance"""
    return PerformanceTracker(data_dir=data_dir)


if __name__ == "__main__":
    # Test the performance tracker
    print("Testing Performance Metrics Tracker")
    print("=" * 60)

    tracker = PerformanceTracker(data_dir="./test_performance_data")
    tracker.set_initial_equity(10000.0)

    # Simulate some trades
    print("\nSimulating trades...")

    # Trade 1: Winning trade
    tracker.record_entry(
        trade_id="test_001",
        token_id="token_a",
        market_question="Will BTC go up?",
        side="BUY",
        entry_price=0.50,
        entry_size=100,
        signal_type="STRONG_BUY",
        signal_confidence=85.0
    )
    tracker.record_exit("test_001", exit_price=0.65, exit_reason="take_profit")
    tracker.record_equity_point(10015.0)  # +$15

    # Trade 2: Losing trade
    tracker.record_entry(
        trade_id="test_002",
        token_id="token_b",
        market_question="Will ETH go up?",
        side="BUY",
        entry_price=0.55,
        entry_size=50,
        signal_type="BUY",
        signal_confidence=65.0
    )
    tracker.record_exit("test_002", exit_price=0.48, exit_reason="stop_loss")
    tracker.record_equity_point(10011.5)  # -$3.50

    # Trade 3: Another win
    tracker.record_entry(
        trade_id="test_003",
        token_id="token_c",
        market_question="Will SOL go up?",
        side="BUY",
        entry_price=0.40,
        entry_size=200,
        signal_type="STRONG_BUY",
        signal_confidence=90.0
    )
    tracker.record_exit("test_003", exit_price=0.52, exit_reason="signal")
    tracker.record_equity_point(10035.5)  # +$24

    # Print summary
    tracker.print_summary()

    # Show trade history
    print("\n" + "-" * 60)
    print("TRADE HISTORY (last 5)")
    print("-" * 60)
    for trade in tracker.get_trade_history(limit=5):
        outcome = trade.get('outcome', 'OPEN')
        pnl = trade.get('realized_pnl', 0) or 0
        print(f"  {trade['trade_id']}: {trade['market_question'][:30]}... "
              f"| {outcome} | P&L: ${pnl:+.2f}")

    # Cleanup test data
    import shutil
    if os.path.exists("./test_performance_data"):
        shutil.rmtree("./test_performance_data")

    print("\n" + "=" * 60)
    print("Performance Metrics Tracker Test Complete")
    print("=" * 60)
