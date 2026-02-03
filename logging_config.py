"""
Structured Logging Configuration for Polymarket Trading Bot

Provides centralized logging configuration with:
- Structured log format with timestamps and context
- Multiple output handlers (console, file)
- Log level configuration
- Trade-specific logging fields
"""

import os
import sys
import logging
import json
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict


# Log levels for different message types
LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

# Default log directory
DEFAULT_LOG_DIR = "logs"


@dataclass
class TradeLogContext:
    """Context fields for trade-related log entries"""
    market_id: Optional[str] = None
    market_question: Optional[str] = None
    token_id: Optional[str] = None
    signal: Optional[str] = None
    side: Optional[str] = None
    price: Optional[float] = None
    size: Optional[float] = None
    confidence: Optional[float] = None
    pnl: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values"""
        return {k: v for k, v in asdict(self).items() if v is not None}


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that outputs structured log messages.

    Format: TIMESTAMP | LEVEL | MODULE | MESSAGE | {context}
    """

    def __init__(self, include_context: bool = True):
        super().__init__()
        self.include_context = include_context

    def format(self, record: logging.LogRecord) -> str:
        # Timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # Level with padding for alignment
        level = f"{record.levelname:<8}"

        # Module/logger name
        module = f"{record.name:<20}"

        # Base message
        message = record.getMessage()

        # Build log line
        log_line = f"{timestamp} | {level} | {module} | {message}"

        # Add context if present
        if self.include_context and hasattr(record, 'trade_context'):
            context = record.trade_context
            if isinstance(context, TradeLogContext):
                context = context.to_dict()
            if context:
                log_line += f" | {json.dumps(context)}"

        # Add exception info if present
        if record.exc_info:
            log_line += f"\n{self.formatException(record.exc_info)}"

        return log_line


class JSONFormatter(logging.Formatter):
    """
    Formatter that outputs logs as JSON lines for machine parsing.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        # Add trade context if present
        if hasattr(record, 'trade_context'):
            context = record.trade_context
            if isinstance(context, TradeLogContext):
                context = context.to_dict()
            if context:
                log_data["context"] = context

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class TradingLogger:
    """
    Enhanced logger for trading operations with structured context support.
    """

    def __init__(self, name: str, logger: logging.Logger):
        self._name = name
        self._logger = logger
        self._default_context: Dict[str, Any] = {}

    def set_context(self, **kwargs):
        """Set default context that will be included in all log messages"""
        self._default_context.update(kwargs)

    def clear_context(self):
        """Clear default context"""
        self._default_context = {}

    def _log(self, level: int, message: str, context: Optional[Dict] = None, exc_info: bool = False):
        """Internal logging method with context support"""
        # Merge default context with provided context
        merged_context = {**self._default_context}
        if context:
            merged_context.update(context)

        # Create log record with extra context
        extra = {"trade_context": merged_context} if merged_context else {}
        self._logger.log(level, message, exc_info=exc_info, extra=extra)

    def debug(self, message: str, **context):
        """Log debug message with optional context"""
        self._log(logging.DEBUG, message, context)

    def info(self, message: str, **context):
        """Log info message with optional context"""
        self._log(logging.INFO, message, context)

    def warning(self, message: str, **context):
        """Log warning message with optional context"""
        self._log(logging.WARNING, message, context)

    def error(self, message: str, exc_info: bool = False, **context):
        """Log error message with optional context and exception info"""
        self._log(logging.ERROR, message, context, exc_info=exc_info)

    def critical(self, message: str, exc_info: bool = False, **context):
        """Log critical message with optional context and exception info"""
        self._log(logging.CRITICAL, message, context, exc_info=exc_info)

    # Trade-specific convenience methods
    def trade(self, message: str, side: str = None, price: float = None, size: float = None, **context):
        """Log trade execution with standard fields"""
        ctx = {k: v for k, v in {"side": side, "price": price, "size": size}.items() if v is not None}
        ctx.update(context)
        self._log(logging.INFO, f"[TRADE] {message}", ctx)

    def signal(self, message: str, signal: str = None, confidence: float = None, **context):
        """Log signal generation with standard fields"""
        ctx = {k: v for k, v in {"signal": signal, "confidence": confidence}.items() if v is not None}
        ctx.update(context)
        self._log(logging.INFO, f"[SIGNAL] {message}", ctx)

    def risk(self, message: str, **context):
        """Log risk management event"""
        self._log(logging.WARNING, f"[RISK] {message}", context)

    def position(self, message: str, pnl: float = None, **context):
        """Log position update"""
        ctx = {k: v for k, v in {"pnl": pnl}.items() if v is not None}
        ctx.update(context)
        self._log(logging.INFO, f"[POSITION] {message}", ctx)


def setup_logging(
    level: str = "INFO",
    log_dir: str = None,
    log_file: str = None,
    console_output: bool = True,
    json_output: bool = False,
    include_context: bool = True
) -> None:
    """
    Configure logging for the trading bot.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (default: ./logs)
        log_file: Log filename (default: trading_bot_YYYYMMDD.log)
        console_output: Whether to output to console
        json_output: Whether to use JSON format for file output
        include_context: Whether to include trade context in logs
    """
    # Get log level
    log_level = LOG_LEVEL_MAP.get(level.upper(), logging.INFO)

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(StructuredFormatter(include_context=include_context))
        root_logger.addHandler(console_handler)

    # File handler
    if log_dir or log_file:
        log_dir = log_dir or DEFAULT_LOG_DIR
        os.makedirs(log_dir, exist_ok=True)

        if not log_file:
            log_file = f"trading_bot_{datetime.now().strftime('%Y%m%d')}.log"

        log_path = os.path.join(log_dir, log_file)

        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(log_level)

        if json_output:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(StructuredFormatter(include_context=include_context))

        root_logger.addHandler(file_handler)

    # Reduce noise from external libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def get_logger(name: str) -> TradingLogger:
    """
    Get a TradingLogger instance for a module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        TradingLogger instance with trade-specific logging methods
    """
    return TradingLogger(name, logging.getLogger(name))


# Module-level loggers for common components
def get_trading_bot_logger() -> TradingLogger:
    """Get logger for trading_bot module"""
    return get_logger("trading_bot")


def get_executor_logger() -> TradingLogger:
    """Get logger for trade_executor module"""
    return get_logger("trade_executor")


def get_data_collector_logger() -> TradingLogger:
    """Get logger for data_collector module"""
    return get_logger("data_collector")


def get_predictor_logger() -> TradingLogger:
    """Get logger for predictor module"""
    return get_logger("predictor")


# Initialize logging with defaults when module is imported
# Can be reconfigured by calling setup_logging() with different options
_initialized = False


def init_default_logging():
    """Initialize logging with default settings if not already done"""
    global _initialized
    if not _initialized:
        setup_logging(
            level=os.environ.get("LOG_LEVEL", "INFO"),
            log_dir=os.environ.get("LOG_DIR", None),
            console_output=True
        )
        _initialized = True


if __name__ == "__main__":
    # Test the logging configuration
    setup_logging(level="DEBUG", log_dir="logs", console_output=True, include_context=True)

    logger = get_logger("test")

    print("\n--- Testing Structured Logging ---\n")

    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")

    print("\n--- Testing Trade-Specific Logging ---\n")

    logger.trade("Order executed", side="BUY", price=0.65, size=100, market="BTC up?")
    logger.signal("Signal generated", signal="STRONG_BUY", confidence=85.5, market="ETH up?")
    logger.risk("Drawdown warning", drawdown_pct=12.5, limit=15.0)
    logger.position("Position closed", pnl=25.50, market="SOL up?")

    print("\n--- Testing Context ---\n")

    logger.set_context(session_id="sess_123", mode="PAPER")
    logger.info("Message with default context")
    logger.trade("Trade with context", side="SELL", price=0.70)
    logger.clear_context()

    print("\n--- Logging Tests Complete ---")
