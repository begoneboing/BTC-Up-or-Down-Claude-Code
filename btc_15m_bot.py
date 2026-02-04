"""
Bitcoin 15-Minute Volatility Trading Bot for Polymarket
Trades the BTC Up/Down 15-minute markets based on volatility signals
"""

import os
import re
import json
import time
import requests
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

# Set trading mode (paper or live)
os.environ['TRADING_MODE'] = os.environ.get('TRADING_MODE', 'paper')

from trade_executor import TradeExecutor, OrderSide
from prediction_engine import PredictionEngine, Direction as PredDirection, MarketRegime

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"
BINANCE_URL = "https://api.binance.com/api/v3"


class Direction(Enum):
    UP = "up"
    DOWN = "down"


@dataclass
class BTC15MMarket:
    """Represents a BTC 15-minute market"""
    event_slug: str
    condition_id: str
    up_token_id: str
    down_token_id: str
    up_price: float
    down_price: float
    start_time: datetime
    end_time: datetime
    volume: float
    liquidity: float


@dataclass
class TradeConfig:
    """Trading configuration for BTC 15M bot"""
    max_position_size: float = 5.0  # Max $ per position
    max_events: int = 3  # Max consecutive events to trade
    min_edge: float = 0.05  # Minimum edge to trade (5%)
    volatility_lookback: int = 5  # Candles to analyze for volatility
    use_limit_orders: bool = True
    limit_offset: float = 0.01  # 1% offset from mid


class BTC15MBot:
    """
    Trading bot for Polymarket BTC 15-minute Up/Down markets

    Strategy: Analyzes short-term BTC volatility to predict
    likely direction in the next 15-minute window.
    """

    def __init__(self, config: TradeConfig = None):
        self.config = config or TradeConfig()
        self.executor = TradeExecutor()
        self.prediction_engine = PredictionEngine()

        # Trading state
        self.events_traded = 0
        self.trade_history: List[Dict] = []
        self.current_market: Optional[BTC15MMarket] = None
        self.btc_prices: List[float] = []  # Cache for BTC price history

    def fetch_btc_price(self) -> float:
        """Fetch current BTC price from Binance"""
        try:
            response = requests.get(
                f"{BINANCE_URL}/ticker/price",
                params={"symbol": "BTCUSDT"},
                timeout=10
            )
            if response.status_code == 200:
                return float(response.json().get("price", 0))
        except Exception as e:
            print(f"Error fetching BTC price: {e}")
        return 0

    def fetch_btc_price_history(self, interval: str = "1m", limit: int = 60) -> List[float]:
        """
        Fetch BTC price history from Binance

        Args:
            interval: Candle interval (1m, 5m, 15m, etc.)
            limit: Number of candles to fetch (max 1000)

        Returns:
            List of closing prices (oldest to newest)
        """
        try:
            response = requests.get(
                f"{BINANCE_URL}/klines",
                params={
                    "symbol": "BTCUSDT",
                    "interval": interval,
                    "limit": limit
                },
                timeout=30
            )
            if response.status_code == 200:
                klines = response.json()
                # Klines format: [open_time, open, high, low, close, volume, ...]
                # Index 4 is the closing price
                prices = [float(k[4]) for k in klines]
                self.btc_prices = prices
                return prices
        except Exception as e:
            print(f"Error fetching BTC price history: {e}")
        return []

    def get_btc_prediction(self) -> Tuple[Optional[Direction], float, List[str]]:
        """
        Get BTC direction prediction using the prediction engine

        Returns:
            Tuple of (direction, confidence, reasons)
        """
        # Fetch recent BTC prices (1-minute candles for last hour)
        prices = self.fetch_btc_price_history(interval="1m", limit=60)

        if len(prices) < 20:
            print(f"Insufficient price data: {len(prices)} candles (need 20+)")
            return None, 0, ["Insufficient price data"]

        # Run prediction engine
        prediction = self.prediction_engine.analyze(prices)

        if not prediction:
            return None, 0, ["Prediction engine returned no result"]

        # Map prediction direction to our Direction enum
        direction_map = {
            PredDirection.STRONG_UP: Direction.UP,
            PredDirection.UP: Direction.UP,
            PredDirection.STRONG_DOWN: Direction.DOWN,
            PredDirection.DOWN: Direction.DOWN,
            PredDirection.NEUTRAL: None
        }

        direction = direction_map.get(prediction.direction)
        confidence = prediction.confidence
        reasons = prediction.reasoning.copy()

        # Add market regime info
        if prediction.market_regime:
            reasons.insert(0, f"Market Regime: {prediction.market_regime.value}")

        # Print prediction details
        print(f"\n{'='*50}")
        print("BTC PRICE PREDICTION")
        print(f"{'='*50}")
        print(f"Direction: {prediction.direction.value}")
        print(f"Confidence: {confidence:.0f}%")
        print(f"Market Regime: {prediction.market_regime.value if prediction.market_regime else 'Unknown'}")
        print(f"Current BTC: ${prices[-1]:,.2f}")

        # Show key signals
        signals = prediction.signals
        if signals.get("rsi") and signals["rsi"] != "neutral":
            print(f"RSI Signal: {signals['rsi']}")
        if signals.get("momentum"):
            print(f"Momentum: {signals['momentum']}")
        if signals.get("trend"):
            print(f"Trend: {signals['trend']}")
        if signals.get("macd"):
            print(f"MACD: {signals['macd']}")

        return direction, confidence, reasons

    def fetch_btc_15m_market(self) -> Optional[BTC15MMarket]:
        """
        Fetch the current active BTC 15-minute Up/Down market from Polymarket

        Uses timestamp-based slug lookup: btc-updown-15m-{timestamp}
        """
        now = datetime.now(timezone.utc)
        print(f"Looking for BTC 15-minute Up/Down markets... (current UTC: {now.strftime('%H:%M')})")

        try:
            # Calculate current 15-minute slot timestamp
            minute = (now.minute // 15) * 15
            slot_time = now.replace(minute=minute, second=0, microsecond=0)
            current_ts = int(slot_time.timestamp())

            # Try current and next few slots to find an open market
            for offset in range(0, 4):
                ts = current_ts + (offset * 15 * 60)
                slug = f"btc-updown-15m-{ts}"

                response = requests.get(
                    f"{GAMMA_URL}/events/slug/{slug}",
                    timeout=30
                )

                if response.status_code != 200:
                    continue

                event = response.json()

                # Check if closed
                if event.get('closed', True):
                    continue

                markets = event.get('markets', [])
                if not markets:
                    continue

                market = markets[0]

                # Check if accepting orders
                if not market.get('acceptingOrders', False):
                    continue

                # Check if already resolved
                outcome_prices = market.get('outcomePrices', '[]')
                if isinstance(outcome_prices, str):
                    outcome_prices = json.loads(outcome_prices)

                prices_set = set(str(p) for p in outcome_prices)
                if prices_set == {'0', '1'} or prices_set == {'1', '0'}:
                    continue  # Already resolved

                title = event.get('title', '')
                print(f"Found market: {title}")
                print(f"  Slug: {slug}")
                print(f"  Liquidity: ${float(event.get('liquidity', 0) or 0):,.0f}")

                # Get token IDs
                clob_tokens = market.get('clobTokenIds', '[]')
                if isinstance(clob_tokens, str):
                    clob_tokens = json.loads(clob_tokens)

                outcomes = market.get('outcomes', '[]')
                if isinstance(outcomes, str):
                    outcomes = json.loads(outcomes)

                if len(clob_tokens) >= 2 and len(outcomes) >= 2:
                    # Match tokens to outcomes (Up/Down)
                    up_idx = next((i for i, o in enumerate(outcomes) if o.lower() == 'up'), 0)
                    down_idx = next((i for i, o in enumerate(outcomes) if o.lower() == 'down'), 1 if up_idx == 0 else 0)

                    up_token_id = clob_tokens[up_idx]
                    down_token_id = clob_tokens[down_idx]

                    up_price = float(outcome_prices[up_idx]) if up_idx < len(outcome_prices) else 0.5
                    down_price = float(outcome_prices[down_idx]) if down_idx < len(outcome_prices) else 0.5

                    print(f"  UP: {up_price:.1%} | DOWN: {down_price:.1%}")

                    return BTC15MMarket(
                        event_slug=slug,
                        condition_id=market.get('conditionId', ''),
                        up_token_id=up_token_id,
                        down_token_id=down_token_id,
                        up_price=up_price,
                        down_price=down_price,
                        start_time=now,
                        end_time=now,
                        volume=float(event.get('volume', 0) or 0),
                        liquidity=float(event.get('liquidity', 0) or 0)
                    )

            print("No active BTC 15-minute markets found")

        except Exception as e:
            print(f"Error fetching market: {e}")
            import traceback
            traceback.print_exc()

        return None

    def fetch_market_from_url(self, event_slug: str) -> Optional[BTC15MMarket]:
        """Fetch market data by scraping the condition ID from known data"""
        # Known market data from the URL provided by user
        # We'll use WebFetch to get the current market data

        try:
            # Extract timestamp from slug
            match = re.search(r'(\d{10})', event_slug)
            if not match:
                return None

            timestamp = int(match.group(1))
            slot_time = datetime.fromtimestamp(timestamp, tz=timezone.utc)

            # Get market data from CLOB book endpoint using condition ID
            # We need to find the condition ID first

            # Try to find in sampling markets
            response = requests.get(
                f"{CLOB_URL}/sampling-markets",
                params={"limit": 2000},
                timeout=60
            )
            if response.status_code == 200:
                data = response.json()
                markets = data.get('data', [])

                for m in markets:
                    slug = m.get('market_slug', '')
                    if 'btc' in slug.lower() and 'updown' in slug.lower():
                        tokens = m.get('tokens', [])
                        if len(tokens) >= 2:
                            up_token = next((t for t in tokens if 'up' in t.get('outcome', '').lower()), tokens[0])
                            down_token = next((t for t in tokens if 'down' in t.get('outcome', '').lower()), tokens[1])

                            return BTC15MMarket(
                                event_slug=slug,
                                condition_id=m.get('condition_id', ''),
                                up_token_id=up_token.get('token_id', ''),
                                down_token_id=down_token.get('token_id', ''),
                                up_price=float(up_token.get('price', 0.5)),
                                down_price=float(down_token.get('price', 0.5)),
                                start_time=slot_time,
                                end_time=slot_time,
                                volume=float(m.get('volume', 0) or 0),
                                liquidity=float(m.get('liquidity', 0) or 0)
                            )

        except Exception as e:
            print(f"Error: {e}")

        return None

    def get_market_prices(self, market: BTC15MMarket) -> Tuple[float, float]:
        """Get current bid/ask prices for the market"""
        # Get orderbook for UP token
        up_prices = self.executor.get_market_price(market.up_token_id)
        down_prices = self.executor.get_market_price(market.down_token_id)

        up_mid = up_prices.get('mid_price', market.up_price)
        down_mid = down_prices.get('mid_price', market.down_price)

        return up_mid, down_mid

    def analyze_volatility(self) -> Dict:
        """
        Analyze BTC volatility to determine trade direction

        Strategy:
        - If recent momentum is bullish, bet on UP
        - If recent momentum is bearish, bet on DOWN
        - Consider market implied probability
        """
        # Get BTC price data (simplified - in production use proper OHLCV)
        btc_price = self.fetch_btc_price()

        # For a simple volatility signal, we'll use market implied probability
        # If UP is cheap (<40%), there might be value in UP
        # If DOWN is cheap (<40%), there might be value in DOWN

        return {
            "btc_price": btc_price,
            "signal": None,
            "confidence": 0.5
        }

    def calculate_edge(self, market: BTC15MMarket) -> Tuple[Direction, float, List[str]]:
        """
        Calculate the trading edge based on prediction + market prices

        Combines:
        1. Technical analysis prediction (direction, confidence)
        2. Market pricing (mispriced outcomes = additional edge)

        Returns:
            Tuple of (direction, edge, reasons)
        """
        up_price, down_price = self.get_market_prices(market)
        reasons = []

        # Get prediction from technical analysis
        pred_direction, pred_confidence, pred_reasons = self.get_btc_prediction()
        reasons.extend(pred_reasons)

        # Base edge from prediction (0-0.5 based on confidence)
        prediction_edge = 0.0
        direction = None

        if pred_direction and pred_confidence >= 55:
            # Use prediction direction if confidence is above threshold
            direction = pred_direction
            prediction_edge = (pred_confidence - 50) / 100  # 55% conf = 0.05 edge, 80% = 0.30
            reasons.append(f"Prediction edge: {prediction_edge:.1%} ({pred_confidence:.0f}% confidence)")

        # Calculate market pricing edge
        market_edge = 0.0
        market_direction = None

        if up_price < 0.40:
            # UP is underpriced
            market_edge = (0.5 - up_price) / 0.5
            market_direction = Direction.UP
            reasons.append(f"UP underpriced at {up_price:.0%} (market edge: {market_edge:.1%})")
        elif down_price < 0.40:
            # DOWN is underpriced
            market_edge = (0.5 - down_price) / 0.5
            market_direction = Direction.DOWN
            reasons.append(f"DOWN underpriced at {down_price:.0%} (market edge: {market_edge:.1%})")
        else:
            reasons.append(f"Market fairly priced: UP={up_price:.0%}, DOWN={down_price:.0%}")

        # Combine prediction and market edge
        if direction and market_direction:
            if direction == market_direction:
                # Prediction and market agree - combine edges
                combined_edge = min(prediction_edge + market_edge * 0.5, 0.50)
                reasons.append(f"Prediction + market AGREE on {direction.value.upper()}")
                return direction, combined_edge, reasons
            else:
                # Prediction and market disagree - use stronger signal
                if prediction_edge > market_edge:
                    reasons.append(f"Prediction overrides market (pred={prediction_edge:.1%} > mkt={market_edge:.1%})")
                    return direction, prediction_edge * 0.8, reasons  # Reduce edge due to conflict
                else:
                    reasons.append(f"Market overrides prediction (mkt={market_edge:.1%} > pred={prediction_edge:.1%})")
                    return market_direction, market_edge * 0.8, reasons

        elif direction:
            # Only have prediction
            return direction, prediction_edge, reasons

        elif market_direction:
            # Only have market signal
            return market_direction, market_edge, reasons

        else:
            # No clear signal - default to UP with no edge
            reasons.append("No clear signal - skipping")
            return Direction.UP, 0.0, reasons

    def place_trade(self, market: BTC15MMarket, direction: Direction) -> Optional[Dict]:
        """Place a trade on the market"""
        # Determine which token to buy
        if direction == Direction.UP:
            token_id = market.up_token_id
            price = market.up_price
            outcome = "UP"
        else:
            token_id = market.down_token_id
            price = market.down_price
            outcome = "DOWN"

        # Calculate position size
        # For $5 max and the given price
        size = self.config.max_position_size / price if price > 0 else 0
        size = round(size, 2)

        if size < 1:
            print(f"Position size too small: {size}")
            return None

        # Place order
        print(f"\n{'='*60}")
        print(f"PLACING TRADE: {outcome}")
        print(f"Market: {market.event_slug}")
        print(f"Token: {token_id[:30]}...")
        print(f"Price: {price:.2%}")
        print(f"Size: {size:.2f} shares (${size * price:.2f})")
        print(f"{'='*60}")

        order = self.executor.place_order(
            token_id=token_id,
            side=OrderSide.BUY,
            price=price + self.config.limit_offset if self.config.use_limit_orders else price,
            size=size,
            market_question=f"BTC 15M - {outcome}"
        )

        if order:
            trade_record = {
                "timestamp": datetime.now().isoformat(),
                "market": market.event_slug,
                "direction": direction.value,
                "price": price,
                "size": size,
                "total": size * price,
                "order_id": order.order_id,
                "status": order.status.value
            }
            self.trade_history.append(trade_record)
            self.events_traded += 1
            return trade_record

        return None

    def run_single_event(self, market: BTC15MMarket) -> Optional[Dict]:
        """Trade a single 15-minute event"""
        print(f"\n{'='*70}")
        print(f"ANALYZING EVENT: {market.event_slug}")
        print(f"{'='*70}")

        # Get current prices
        up_price, down_price = self.get_market_prices(market)
        print(f"UP Price: {up_price:.2%} | DOWN Price: {down_price:.2%}")
        print(f"Volume: ${market.volume:,.0f} | Liquidity: ${market.liquidity:,.0f}")

        # Calculate edge (now returns 3 values including reasons)
        direction, edge, reasons = self.calculate_edge(market)

        print(f"\n{'='*50}")
        print("EDGE CALCULATION")
        print(f"{'='*50}")
        print(f"Direction: {direction.value.upper()}")
        print(f"Edge: {edge:.1%}")
        print(f"\nReasoning:")
        for reason in reasons[-5:]:  # Show last 5 reasons
            print(f"  - {reason}")

        # Check if edge meets threshold
        if edge < self.config.min_edge:
            print(f"\n[SKIP] Edge {edge:.1%} below threshold {self.config.min_edge:.1%}")
            return None

        # Place trade
        return self.place_trade(market, direction)

    def run_consecutive_events(self, num_events: int = 3):
        """
        Trade consecutive 15-minute events

        This will trade up to num_events consecutive BTC 15M markets
        """
        print(f"\n{'='*70}")
        print(f"BTC 15-MINUTE VOLATILITY BOT")
        print(f"{'='*70}")
        print(f"Max Position Size: ${self.config.max_position_size:.2f}")
        print(f"Events to Trade: {num_events}")
        print(f"Min Edge: {self.config.min_edge:.1%}")
        print(f"Mode: {'LIVE' if self.executor.is_live_mode() else 'PAPER'}")
        print(f"{'='*70}")

        trades_placed = 0

        while trades_placed < num_events:
            print(f"\n--- Event {trades_placed + 1} of {num_events} ---")

            # Fetch current market
            market = self.fetch_btc_15m_market()

            if not market:
                print("No active market found - checking for specific market...")
                # Try with known event slug pattern
                market = self.fetch_market_from_url("btc-updown-15m")

            if not market:
                print("Could not find active BTC 15M market")
                print("Waiting 60 seconds before retry...")
                time.sleep(60)
                continue

            self.current_market = market

            # Trade the event
            trade = self.run_single_event(market)

            if trade:
                trades_placed += 1
                print(f"\nTrade {trades_placed} completed: {trade['direction'].upper()} ${trade['total']:.2f}")
            else:
                print("No trade placed for this event")

            # Wait for next 15-minute slot
            if trades_placed < num_events:
                wait_time = 15 * 60  # 15 minutes
                print(f"\nWaiting {wait_time // 60} minutes for next event...")
                time.sleep(wait_time)

        self.print_summary()

    def fetch_btc_price_target_markets(self) -> List[Dict]:
        """
        Fetch BTC monthly price target markets (e.g., "Will Bitcoin reach $85,000 in February?")
        These are longer-term markets that use our prediction to determine likelihood.
        """
        print("Looking for BTC price target markets...")

        try:
            response = requests.get(
                f"{GAMMA_URL}/markets",
                params={"active": "true", "limit": 500, "order": "volume24hr", "ascending": "false"},
                timeout=60
            )
            if response.status_code != 200:
                return []

            markets = response.json()
            price_target_markets = []

            for market in markets:
                question = market.get('question', '').lower()
                accepting = market.get('acceptingOrders', False)
                closed = market.get('closed', False)

                # Look for BTC price target markets
                if ('bitcoin' in question or 'btc' in question) and \
                   ('reach' in question or 'dip' in question or 'above' in question) and \
                   '$' in question and accepting and not closed:

                    outcome_prices = market.get('outcomePrices', '[]')
                    if isinstance(outcome_prices, str):
                        outcome_prices = json.loads(outcome_prices)

                    prices_set = set(str(p) for p in outcome_prices)
                    is_resolved = prices_set == {'0', '1'} or prices_set == {'1', '0'}

                    if not is_resolved:
                        # Extract target price from question
                        import re
                        price_match = re.search(r'\$([0-9,]+)', market.get('question', ''))
                        if price_match:
                            target_price = int(price_match.group(1).replace(',', ''))
                            market['target_price'] = target_price
                            market['is_reach'] = 'reach' in question  # True for "reach", False for "dip"
                            price_target_markets.append(market)

            print(f"Found {len(price_target_markets)} BTC price target markets")
            return price_target_markets

        except Exception as e:
            print(f"Error fetching price target markets: {e}")
            return []

    def analyze_price_target(self, market: Dict) -> Tuple[str, float, List[str]]:
        """
        Analyze a price target market and determine whether to bet Yes or No

        Returns:
            Tuple of (side ('Yes' or 'No'), edge, reasons)
        """
        target_price = market.get('target_price', 0)
        is_reach = market.get('is_reach', True)
        question = market.get('question', '')

        # Get current BTC price and prediction
        prices = self.fetch_btc_price_history(interval="1h", limit=48)  # 48 hours of hourly data
        if not prices:
            return 'No', 0, ["Could not fetch BTC price data"]

        current_price = prices[-1]

        # Calculate distance to target
        distance_pct = (target_price - current_price) / current_price * 100

        # Get market prices
        outcome_prices = market.get('outcomePrices', '[]')
        if isinstance(outcome_prices, str):
            outcome_prices = json.loads(outcome_prices)

        yes_price = float(outcome_prices[0]) if outcome_prices else 0.5
        no_price = float(outcome_prices[1]) if len(outcome_prices) > 1 else 0.5

        reasons = [
            f"Current BTC: ${current_price:,.0f}",
            f"Target: ${target_price:,} ({'reach' if is_reach else 'dip'})",
            f"Distance: {distance_pct:+.1f}%",
            f"Market: Yes={yes_price:.1%}, No={no_price:.1%}"
        ]

        # Get momentum prediction
        pred_direction, pred_confidence, pred_reasons = self.get_btc_prediction()
        reasons.extend(pred_reasons[:3])

        # Decision logic for "reach" markets
        if is_reach:
            if distance_pct > 30:
                # Target is very far above - likely No
                side = 'No'
                edge = min((distance_pct - 20) / 100, 0.3) if no_price < 0.95 else 0
                reasons.append(f"Target {distance_pct:.0f}% above current - betting No")
            elif distance_pct > 15:
                # Target is moderately far - slight No bias
                if pred_direction == Direction.UP and pred_confidence > 60:
                    side = 'Yes'
                    edge = 0.05
                    reasons.append("Strong bullish prediction - small Yes bet")
                else:
                    side = 'No'
                    edge = 0.1
                    reasons.append("Target far above + weak momentum - betting No")
            elif distance_pct > 0:
                # Target is above but reachable
                if pred_direction == Direction.UP:
                    side = 'Yes'
                    edge = (pred_confidence - 50) / 200 if yes_price < 0.6 else 0
                    reasons.append("Bullish + reachable target - betting Yes")
                else:
                    side = 'No'
                    edge = 0.05
                    reasons.append("Bearish + target above - slight No bet")
            else:
                # Target is below current (already reached)
                side = 'Yes'
                edge = (1 - yes_price) * 0.5 if yes_price < 0.95 else 0
                reasons.append("Target below current - already reached, betting Yes")
        else:
            # "Dip to" market - betting on price dropping
            if distance_pct < -15:
                # Target is far below - likely No (won't dip that much)
                side = 'No'
                edge = 0.15 if no_price < 0.85 else 0
                reasons.append("Target far below current - betting No on dip")
            elif distance_pct < 0:
                # Target is below but close
                if pred_direction == Direction.DOWN:
                    side = 'Yes'
                    edge = (pred_confidence - 50) / 200 if yes_price < 0.7 else 0
                    reasons.append("Bearish + dip target close - betting Yes")
                else:
                    side = 'No'
                    edge = 0.05
                    reasons.append("Bullish momentum - betting No on dip")
            else:
                # Target is above current (would need to go up then dip)
                side = 'No'
                edge = 0.1
                reasons.append("Target above current - unlikely to dip there")

        return side, edge, reasons

    def trade_price_target_market(self, market: Dict) -> Optional[Dict]:
        """Trade a single price target market"""
        question = market.get('question', '')
        slug = market.get('slug', '')

        print(f"\n{'='*70}")
        print(f"ANALYZING: {question}")
        print(f"{'='*70}")

        side, edge, reasons = self.analyze_price_target(market)

        print(f"\nAnalysis:")
        for reason in reasons:
            print(f"  - {reason}")

        print(f"\nDecision: {side} (edge: {edge:.1%})")

        if edge < self.config.min_edge:
            print(f"Edge {edge:.1%} below threshold {self.config.min_edge:.1%} - skipping")
            return None

        # Get token IDs
        clob_tokens = market.get('clobTokenIds', '[]')
        if isinstance(clob_tokens, str):
            clob_tokens = json.loads(clob_tokens)

        outcome_prices = market.get('outcomePrices', '[]')
        if isinstance(outcome_prices, str):
            outcome_prices = json.loads(outcome_prices)

        if len(clob_tokens) < 2:
            print("Invalid market tokens")
            return None

        # Select token based on side
        token_idx = 0 if side == 'Yes' else 1
        token_id = clob_tokens[token_idx]
        price = float(outcome_prices[token_idx])

        # Calculate size
        size = self.config.max_position_size / price if price > 0 else 0
        size = round(size, 2)

        if size < 1:
            print(f"Position size too small: {size}")
            return None

        print(f"\n{'='*60}")
        print(f"PLACING TRADE: {side}")
        print(f"Market: {slug}")
        print(f"Price: {price:.2%}")
        print(f"Size: {size:.2f} shares (${size * price:.2f})")
        print(f"{'='*60}")

        order = self.executor.place_order(
            token_id=token_id,
            side=OrderSide.BUY,
            price=price + self.config.limit_offset,
            size=size,
            market_question=question[:50]
        )

        if order:
            trade_record = {
                "timestamp": datetime.now().isoformat(),
                "market": slug,
                "direction": side,
                "price": price,
                "size": size,
                "total": size * price,
                "order_id": order.order_id,
                "status": order.status.value
            }
            self.trade_history.append(trade_record)
            return trade_record

        return None

    def run_price_targets(self, max_trades: int = 3):
        """Run trading on price target markets"""
        print(f"\n{'='*70}")
        print("BTC PRICE TARGET TRADING")
        print(f"{'='*70}")
        print(f"Max Position Size: ${self.config.max_position_size:.2f}")
        print(f"Min Edge: {self.config.min_edge:.1%}")
        print(f"Mode: {'LIVE' if self.executor.is_live_mode() else 'PAPER'}")

        markets = self.fetch_btc_price_target_markets()

        if not markets:
            print("No price target markets found")
            return

        # Sort by volume
        markets.sort(key=lambda x: float(x.get('volume24hr', 0) or 0), reverse=True)

        trades_placed = 0
        for market in markets[:max_trades * 2]:  # Check more than needed in case some are skipped
            if trades_placed >= max_trades:
                break

            trade = self.trade_price_target_market(market)
            if trade:
                trades_placed += 1
                print(f"\nTrade {trades_placed} completed: {trade['direction']} ${trade['total']:.2f}")

        self.print_summary()

    def check_position_profit(self, token_id: str, entry_price: float, size: float) -> Dict:
        """
        Check current profit/loss on a position

        Returns:
            Dict with current_price, pnl, pnl_pct, should_exit
        """
        try:
            current_prices = self.executor.get_market_price(token_id)
            current_price = current_prices.get('mid_price', entry_price)

            pnl = (current_price - entry_price) * size
            pnl_pct = (current_price - entry_price) / entry_price * 100 if entry_price > 0 else 0

            return {
                'current_price': current_price,
                'entry_price': entry_price,
                'size': size,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'should_exit': pnl_pct >= 15  # Exit at 15% profit
            }
        except Exception as e:
            print(f"Error checking position: {e}")
            return {'current_price': entry_price, 'pnl': 0, 'pnl_pct': 0, 'should_exit': False}

    def exit_position(self, token_id: str, size: float, current_price: float, reason: str = "profit_target") -> bool:
        """
        Exit a position by selling

        Args:
            token_id: Token to sell
            size: Number of shares to sell
            current_price: Current market price
            reason: Reason for exit

        Returns:
            True if exit successful
        """
        try:
            print(f"\n[EXIT] Selling {size} shares at {current_price:.1%} ({reason})")

            order = self.executor.place_order(
                token_id=token_id,
                side=OrderSide.SELL,
                price=max(current_price - 0.02, 0.01),  # Sell slightly below market
                size=size,
                market_question=f"EXIT: {reason}"
            )

            if order:
                print(f"Exit order {order.status.value}: {order.order_id[:30]}...")
                return True
            return False
        except Exception as e:
            print(f"Exit error: {e}")
            return False

    def monitor_and_exit_positions(self, positions: List[Dict], profit_threshold: float = 0.15):
        """
        Monitor open positions and exit when profitable

        Args:
            positions: List of position dicts with token_id, entry_price, size, direction
            profit_threshold: Exit when profit exceeds this (default 15%)
        """
        print(f"\n{'='*60}")
        print("MONITORING POSITIONS FOR PROFIT EXIT")
        print(f"Profit threshold: {profit_threshold:.0%}")
        print(f"{'='*60}")

        for pos in positions:
            token_id = pos.get('token_id')
            entry_price = pos.get('entry_price')
            size = pos.get('size')
            direction = pos.get('direction', 'unknown')

            if not token_id or not entry_price:
                continue

            profit_info = self.check_position_profit(token_id, entry_price, size)

            print(f"\n{direction}: Entry={entry_price:.1%}, Current={profit_info['current_price']:.1%}, PnL={profit_info['pnl_pct']:+.1f}%")

            if profit_info['pnl_pct'] >= profit_threshold * 100:
                print(f"  -> Profit target hit! Exiting...")
                self.exit_position(token_id, size, profit_info['current_price'], "profit_target")
            elif profit_info['pnl_pct'] <= -20:  # Stop loss at -20%
                print(f"  -> Stop loss triggered! Exiting...")
                self.exit_position(token_id, size, profit_info['current_price'], "stop_loss")
            else:
                print(f"  -> Holding position")

    def print_summary(self):
        """Print trading session summary"""
        print(f"\n{'='*70}")
        print("SESSION SUMMARY")
        print(f"{'='*70}")
        print(f"Events Traded: {self.events_traded}")
        print(f"Total Trades: {len(self.trade_history)}")

        if self.trade_history:
            total_spent = sum(t['total'] for t in self.trade_history)
            print(f"Total Invested: ${total_spent:.2f}")

            print(f"\nTrade History:")
            for t in self.trade_history:
                print(f"  {t['timestamp'][:19]} | {t['direction'].upper():4} | ${t['total']:.2f} | {t['status']}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="BTC 15-Minute Volatility Bot")
    parser.add_argument("--events", type=int, default=3, help="Number of events to trade")
    parser.add_argument("--max-size", type=float, default=5.0, help="Max position size in $")
    parser.add_argument("--min-edge", type=float, default=0.05, help="Minimum edge to trade")
    parser.add_argument("--test", action="store_true", help="Run in test mode (fetch only)")
    parser.add_argument("--predict", action="store_true", help="Run prediction only (no market needed)")
    parser.add_argument("--paper", action="store_true", help="Force paper trading mode")
    parser.add_argument("--price-targets", action="store_true", help="Trade monthly price target markets instead of 15m")

    args = parser.parse_args()

    # Force paper mode if requested
    if args.paper:
        os.environ['TRADING_MODE'] = 'paper'

    config = TradeConfig(
        max_position_size=args.max_size,
        max_events=args.events,
        min_edge=args.min_edge
    )

    bot = BTC15MBot(config)

    if args.price_targets:
        print("=" * 60)
        print("PRICE TARGET TRADING MODE")
        print("=" * 60)
        bot.run_price_targets(max_trades=args.events)

    elif args.predict:
        print("=" * 60)
        print("PREDICTION MODE - Testing BTC Direction Prediction")
        print("=" * 60)
        print(f"Mode: {'LIVE' if bot.executor.is_live_mode() else 'PAPER'}")
        print()

        # Get prediction
        direction, confidence, reasons = bot.get_btc_prediction()

        print(f"\n{'='*60}")
        print("PREDICTION RESULT")
        print(f"{'='*60}")
        if direction:
            print(f"Direction: {direction.value.upper()}")
            print(f"Confidence: {confidence:.0f}%")
            print(f"\nTop Reasons:")
            for reason in reasons[:8]:
                print(f"  - {reason}")
        else:
            print("No clear direction signal")
            print(f"\nReasons:")
            for reason in reasons[:5]:
                print(f"  - {reason}")

    elif args.test:
        print("Test mode - fetching market data only")
        market = bot.fetch_btc_15m_market()
        if market:
            print(f"\nFound market: {market.event_slug}")
            print(f"UP: {market.up_price:.2%} | DOWN: {market.down_price:.2%}")
        else:
            print("No market found")
    else:
        bot.run_consecutive_events(args.events)


if __name__ == "__main__":
    main()
