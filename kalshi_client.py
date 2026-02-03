"""
Kalshi API Client for Python
Handles authentication and trading for Kalshi prediction markets
"""

import os
import time
import json
import base64
import requests
from datetime import datetime
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

# Try to import RSA signing
try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.backends import default_backend
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False
    print("Warning: cryptography not installed. Install with: pip install cryptography")


KALSHI_API_URL = "https://api.elections.kalshi.com"


class OrderSide(Enum):
    YES = "yes"
    NO = "no"


class OrderAction(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class KalshiMarket:
    ticker: str
    event_ticker: str
    title: str
    subtitle: str
    yes_bid: int  # in cents
    yes_ask: int
    no_bid: int
    no_ask: int
    close_time: str
    volume: int
    open_interest: int


@dataclass
class KalshiOrder:
    order_id: str
    ticker: str
    side: str
    action: str
    count: int
    price: int  # in cents
    status: str
    filled_count: int = 0


class KalshiClient:
    """
    Kalshi API client with RSA authentication
    """

    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key or os.environ.get("KALSHI_API_KEY", "")
        self.api_secret = api_secret or os.environ.get("KALSHI_API_SECRET", "")
        self.base_url = KALSHI_API_URL
        self.session = requests.Session()
        self.private_key = None

        # Load RSA private key if available
        if HAS_CRYPTO and self.api_secret and "BEGIN" in self.api_secret:
            try:
                self.private_key = serialization.load_pem_private_key(
                    self.api_secret.encode(),
                    password=None,
                    backend=default_backend()
                )
                print("Kalshi RSA key loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load RSA key: {e}")

    def _get_auth_headers(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        """Generate authentication headers with RSA signature"""
        timestamp = str(int(time.time()))

        headers = {
            "Content-Type": "application/json",
        }

        if self.api_key:
            headers["KALSHI-ACCESS-KEY"] = self.api_key

        # Sign request if we have a private key
        if self.private_key and HAS_CRYPTO:
            # Signature string format: timestamp + method + path + body
            message = f"{timestamp}{method.upper()}{path}{body}"

            signature = self.private_key.sign(
                message.encode(),
                padding.PKCS1v15(),
                hashes.SHA256()
            )
            signature_b64 = base64.standard_b64encode(signature).decode()

            headers["KALSHI-ACCESS-SIGNATURE"] = signature_b64
            headers["KALSHI-ACCESS-TIMESTAMP"] = timestamp

        return headers

    def _request(self, method: str, path: str, params: Dict = None, data: Dict = None) -> Dict:
        """Make authenticated request to Kalshi API"""
        url = f"{self.base_url}{path}"
        body = json.dumps(data) if data else ""

        headers = self._get_auth_headers(method, path, body)

        try:
            if method.upper() == "GET":
                response = self.session.get(url, headers=headers, params=params, timeout=30)
            elif method.upper() == "POST":
                response = self.session.post(url, headers=headers, json=data, timeout=30)
            elif method.upper() == "DELETE":
                response = self.session.delete(url, headers=headers, timeout=30)
            else:
                raise ValueError(f"Unsupported method: {method}")

            if response.status_code == 401:
                return {"error": "Authentication failed - check API key and secret"}

            if not response.ok:
                return {"error": f"API error {response.status_code}: {response.text}"}

            return response.json()

        except Exception as e:
            return {"error": str(e)}

    def get_balance(self) -> Optional[float]:
        """Get account balance in dollars"""
        result = self._request("GET", "/trade-api/v2/portfolio/balance")
        if "error" in result:
            print(f"Balance error: {result['error']}")
            return None

        # Balance is in cents
        balance_cents = result.get("balance", 0)
        return balance_cents / 100.0

    def get_btc_15m_events(self, limit: int = 10) -> List[Dict]:
        """Get Bitcoin 15-minute events"""
        result = self._request(
            "GET",
            "/trade-api/v2/events",
            params={"series_ticker": "KXBTC15M", "status": "open", "limit": limit}
        )
        if "error" in result:
            print(f"Events error: {result['error']}")
            return []

        return result.get("events", [])

    def get_markets_for_event(self, event_ticker: str) -> List[KalshiMarket]:
        """Get markets for a specific event"""
        result = self._request(
            "GET",
            "/trade-api/v2/markets",
            params={"event_ticker": event_ticker, "limit": 50}
        )
        if "error" in result:
            print(f"Markets error: {result['error']}")
            return []

        markets = []
        for m in result.get("markets", []):
            markets.append(KalshiMarket(
                ticker=m.get("ticker", ""),
                event_ticker=m.get("event_ticker", ""),
                title=m.get("title", ""),
                subtitle=m.get("subtitle", ""),
                yes_bid=m.get("yes_bid", 0),
                yes_ask=m.get("yes_ask", 0),
                no_bid=m.get("no_bid", 0),
                no_ask=m.get("no_ask", 0),
                close_time=m.get("close_time", m.get("expiration_time", "")),
                volume=m.get("volume", 0),
                open_interest=m.get("open_interest", 0)
            ))

        return markets

    def get_btc_15m_markets(self) -> List[KalshiMarket]:
        """Get all active BTC 15-minute markets"""
        events = self.get_btc_15m_events()
        all_markets = []

        for event in events:
            event_ticker = event.get("event_ticker", "")
            markets = self.get_markets_for_event(event_ticker)
            all_markets.extend(markets)

        return all_markets

    def place_order(
        self,
        ticker: str,
        side: OrderSide,
        action: OrderAction,
        count: int,
        price_cents: int
    ) -> Optional[KalshiOrder]:
        """
        Place an order on Kalshi

        Args:
            ticker: Market ticker
            side: YES or NO
            action: BUY or SELL
            count: Number of contracts
            price_cents: Price in cents (1-99)
        """
        if not self.api_key:
            print("Error: Kalshi API key not configured")
            return None

        order_data = {
            "ticker": ticker,
            "side": side.value,
            "action": action.value,
            "count": count,
            "type": "limit",
            "yes_price": price_cents if side == OrderSide.YES else None,
            "no_price": price_cents if side == OrderSide.NO else None,
        }

        # Remove None values
        order_data = {k: v for k, v in order_data.items() if v is not None}

        result = self._request("POST", "/trade-api/v2/portfolio/orders", data=order_data)

        if "error" in result:
            print(f"Order error: {result['error']}")
            return None

        order = result.get("order", {})
        return KalshiOrder(
            order_id=order.get("order_id", ""),
            ticker=ticker,
            side=side.value,
            action=action.value,
            count=count,
            price=price_cents,
            status=order.get("status", "unknown"),
            filled_count=order.get("filled_count", 0)
        )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order"""
        result = self._request("DELETE", f"/trade-api/v2/portfolio/orders/{order_id}")
        return "error" not in result

    def get_positions(self) -> List[Dict]:
        """Get current positions"""
        result = self._request("GET", "/trade-api/v2/portfolio/positions")
        if "error" in result:
            print(f"Positions error: {result['error']}")
            return []

        return result.get("market_positions", [])


def test_kalshi_connection():
    """Test Kalshi API connection"""
    print("=" * 60)
    print("Testing Kalshi API Connection")
    print("=" * 60)

    client = KalshiClient()

    # Test getting BTC 15M markets (public endpoint)
    print("\nFetching BTC 15-minute markets...")
    markets = client.get_btc_15m_markets()

    print(f"Found {len(markets)} markets")
    for m in markets[:5]:
        print(f"\n  Ticker: {m.ticker}")
        print(f"  Title: {m.title}")
        print(f"  Yes: {m.yes_bid}-{m.yes_ask}c | No: {m.no_bid}-{m.no_ask}c")
        print(f"  Closes: {m.close_time}")

    # Test authenticated endpoints if credentials exist
    if client.api_key:
        print("\n" + "-" * 60)
        print("Testing authenticated endpoints...")

        balance = client.get_balance()
        if balance is not None:
            print(f"Balance: ${balance:.2f}")
        else:
            print("Could not fetch balance (check credentials)")
    else:
        print("\nNo Kalshi API key configured - skipping authenticated tests")
        print("Set KALSHI_API_KEY and KALSHI_API_SECRET in .env file")


if __name__ == "__main__":
    test_kalshi_connection()
