"""
Data Collector for Polymarket
Fetches and stores historical price data for analysis
"""

import os
import json
import time
import random
import hashlib
import logging
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple, Callable, TypeVar
from dataclasses import dataclass, field
from functools import wraps
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for generic return type in retry decorator
T = TypeVar('T')


@dataclass
class RetryConfig:
    """Configuration for API retry behavior"""
    max_retries: int = 3  # Maximum number of retry attempts
    base_delay: float = 1.0  # Base delay between retries (seconds)
    max_delay: float = 30.0  # Maximum delay between retries
    exponential_base: float = 2.0  # Exponential backoff multiplier
    jitter: bool = True  # Add random jitter to prevent thundering herd
    retry_on_status: tuple = (429, 500, 502, 503, 504)  # HTTP status codes to retry
    retry_on_exceptions: tuple = (
        requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
        requests.exceptions.ChunkedEncodingError,
    )


def retry_with_backoff(config: RetryConfig = None):
    """
    Decorator that adds retry logic with exponential backoff to API calls.

    Features:
    - Exponential backoff: delay doubles with each retry
    - Jitter: random variance to prevent thundering herd
    - Configurable retry conditions (status codes, exceptions)
    - Logging of retry attempts

    Args:
        config: RetryConfig with retry parameters

    Returns:
        Decorated function with retry logic
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(config.max_retries + 1):
                try:
                    response = func(*args, **kwargs)

                    # Check if we got an HTTP response object with a status code to retry
                    if hasattr(response, 'status_code'):
                        if response.status_code in config.retry_on_status:
                            if attempt < config.max_retries:
                                delay = _calculate_delay(attempt, config)
                                logger.warning(
                                    f"Retry {attempt + 1}/{config.max_retries}: "
                                    f"{func.__name__} returned status {response.status_code}. "
                                    f"Retrying in {delay:.2f}s..."
                                )
                                time.sleep(delay)
                                continue
                            else:
                                # Last attempt, raise or return as-is
                                response.raise_for_status()

                    return response

                except config.retry_on_exceptions as e:
                    last_exception = e
                    if attempt < config.max_retries:
                        delay = _calculate_delay(attempt, config)
                        logger.warning(
                            f"Retry {attempt + 1}/{config.max_retries}: "
                            f"{func.__name__} failed with {type(e).__name__}: {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {config.max_retries} retries exhausted for {func.__name__}. "
                            f"Last error: {e}"
                        )
                        raise

                except requests.exceptions.HTTPError as e:
                    # Check if the HTTP error status code is retryable
                    if hasattr(e, 'response') and e.response is not None:
                        if e.response.status_code in config.retry_on_status:
                            last_exception = e
                            if attempt < config.max_retries:
                                delay = _calculate_delay(attempt, config)
                                logger.warning(
                                    f"Retry {attempt + 1}/{config.max_retries}: "
                                    f"{func.__name__} got HTTP {e.response.status_code}. "
                                    f"Retrying in {delay:.2f}s..."
                                )
                                time.sleep(delay)
                                continue
                    raise

            # If we exit the loop without returning, raise the last exception
            if last_exception:
                raise last_exception
            return None

        return wrapper
    return decorator


def _calculate_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate delay for a given retry attempt with exponential backoff and jitter"""
    delay = config.base_delay * (config.exponential_base ** attempt)
    delay = min(delay, config.max_delay)

    if config.jitter:
        # Add random jitter between 0% and 25% of the delay
        jitter_amount = delay * 0.25 * random.random()
        delay += jitter_amount

    return delay


class RetryableRequests:
    """
    Wrapper around requests library with built-in retry logic and connection pooling.

    This class provides resilient HTTP methods that automatically retry
    on transient failures with exponential backoff.

    Features:
    - Connection pooling via requests.Session for TCP connection reuse
    - Automatic retry with exponential backoff on transient failures
    - Configurable pool size for concurrent connections
    - Keep-alive connections for reduced latency
    """

    def __init__(self, config: RetryConfig = None, pool_connections: int = 10, pool_maxsize: int = 20):
        """
        Initialize the HTTP client with connection pooling.

        Args:
            config: RetryConfig for retry behavior
            pool_connections: Number of connection pools to cache (default 10)
            pool_maxsize: Maximum number of connections in each pool (default 20)
        """
        self.config = config or RetryConfig()

        # Create a session for connection pooling
        self._session = requests.Session()

        # Configure connection pooling via HTTPAdapter
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        # Create adapter with connection pooling settings
        adapter = HTTPAdapter(
            pool_connections=pool_connections,  # Number of host pools
            pool_maxsize=pool_maxsize,  # Max connections per pool
            max_retries=0,  # We handle retries ourselves with exponential backoff
            pool_block=False  # Don't block when pool is full, create new connection
        )

        # Mount adapter for both HTTP and HTTPS
        self._session.mount('http://', adapter)
        self._session.mount('https://', adapter)

        # Set default headers for keep-alive
        self._session.headers.update({
            'Connection': 'keep-alive',
            'Accept': 'application/json',
            'User-Agent': 'PolymarketBot/1.0'
        })

        # Track connection statistics
        self._stats = {
            'requests': 0,
            'retries': 0,
            'errors': 0
        }

    def get(self, url: str, **kwargs) -> requests.Response:
        """Make a GET request with retry logic and connection reuse"""
        @retry_with_backoff(self.config)
        def _get():
            self._stats['requests'] += 1
            response = self._session.get(url, **kwargs)
            return response

        response = _get()
        response.raise_for_status()
        return response

    def post(self, url: str, **kwargs) -> requests.Response:
        """Make a POST request with retry logic and connection reuse"""
        @retry_with_backoff(self.config)
        def _post():
            self._stats['requests'] += 1
            response = self._session.post(url, **kwargs)
            return response

        response = _post()
        response.raise_for_status()
        return response

    def close(self):
        """Close all connections in the pool"""
        self._session.close()

    def get_stats(self) -> Dict[str, int]:
        """Get request statistics"""
        return self._stats.copy()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close connections"""
        self.close()
        return False

GAMMA_API_URL = "https://gamma-api.polymarket.com"
CLOB_API_URL = "https://clob.polymarket.com"


@dataclass
class CacheEntry:
    """A cached API response with TTL (time-to-live) tracking"""
    data: Any
    timestamp: datetime
    ttl_seconds: float
    hits: int = 0

    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        age = (datetime.now() - self.timestamp).total_seconds()
        return age > self.ttl_seconds

    def age_seconds(self) -> float:
        """Get age of cache entry in seconds"""
        return (datetime.now() - self.timestamp).total_seconds()


@dataclass
class CacheConfig:
    """Configuration for API response caching"""
    enabled: bool = True
    # TTL settings (in seconds) for different data types
    markets_ttl: float = 60.0  # Active markets list: 1 minute
    orderbook_ttl: float = 5.0  # Orderbook data: 5 seconds (changes frequently)
    price_history_ttl: float = 30.0  # Price history: 30 seconds
    market_details_ttl: float = 300.0  # Market details: 5 minutes (rarely changes)
    # Memory management
    max_entries: int = 1000  # Maximum cache entries before cleanup
    cleanup_threshold: float = 0.8  # Trigger cleanup at 80% capacity


@dataclass
class CacheStats:
    """Statistics for cache performance monitoring"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_requests: int = 0
    bytes_saved_estimate: int = 0
    last_cleanup: Optional[datetime] = None

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate as percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.hits / self.total_requests) * 100


class ResponseCache:
    """
    In-memory cache for API responses with TTL support.

    Features:
    - Per-endpoint TTL configuration
    - Automatic cache expiration
    - Memory management with LRU-style cleanup
    - Cache statistics for monitoring
    """

    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self._cache: Dict[str, CacheEntry] = {}
        self.stats = CacheStats()

    def _make_key(self, endpoint: str, params: Dict = None) -> str:
        """Generate a unique cache key from endpoint and parameters"""
        key_data = endpoint
        if params:
            # Sort params for consistent key generation
            sorted_params = sorted(params.items())
            key_data += "?" + "&".join(f"{k}={v}" for k, v in sorted_params)
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, endpoint: str, params: Dict = None) -> Tuple[bool, Any]:
        """
        Get cached response if available and not expired.

        Returns:
            Tuple of (hit: bool, data: Any)
            If hit=True, data contains cached response
            If hit=False, data is None
        """
        if not self.config.enabled:
            return False, None

        self.stats.total_requests += 1
        key = self._make_key(endpoint, params)

        entry = self._cache.get(key)
        if entry is None:
            self.stats.misses += 1
            return False, None

        if entry.is_expired():
            # Remove expired entry
            del self._cache[key]
            self.stats.misses += 1
            return False, None

        # Cache hit
        entry.hits += 1
        self.stats.hits += 1
        return True, entry.data

    def set(self, endpoint: str, params: Dict, data: Any, ttl_seconds: float = None) -> None:
        """
        Store response in cache with TTL.

        Args:
            endpoint: API endpoint URL
            params: Request parameters
            data: Response data to cache
            ttl_seconds: Override default TTL (optional)
        """
        if not self.config.enabled:
            return

        # Check if cleanup needed
        if len(self._cache) >= self.config.max_entries * self.config.cleanup_threshold:
            self._cleanup()

        key = self._make_key(endpoint, params)

        # Determine TTL based on endpoint type
        if ttl_seconds is None:
            ttl_seconds = self._get_default_ttl(endpoint)

        self._cache[key] = CacheEntry(
            data=data,
            timestamp=datetime.now(),
            ttl_seconds=ttl_seconds
        )

        # Estimate bytes saved on future hits
        try:
            self.stats.bytes_saved_estimate += len(json.dumps(data))
        except (TypeError, ValueError):
            pass

    def _get_default_ttl(self, endpoint: str) -> float:
        """Get default TTL based on endpoint type"""
        endpoint_lower = endpoint.lower()

        if "book" in endpoint_lower:
            return self.config.orderbook_ttl
        elif "prices-history" in endpoint_lower or "history" in endpoint_lower:
            return self.config.price_history_ttl
        elif "events" in endpoint_lower or "markets" in endpoint_lower:
            if "/markets/" in endpoint_lower:  # Specific market details
                return self.config.market_details_ttl
            return self.config.markets_ttl
        else:
            return self.config.markets_ttl  # Default fallback

    def _cleanup(self) -> None:
        """Remove expired entries and oldest entries if over capacity"""
        now = datetime.now()
        self.stats.last_cleanup = now

        # First, remove all expired entries
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired()
        ]
        for key in expired_keys:
            del self._cache[key]
            self.stats.evictions += 1

        # If still over capacity, remove least recently used (oldest + least hits)
        if len(self._cache) >= self.config.max_entries:
            # Sort by (hits, -age) so low-hit old entries are removed first
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda x: (x[1].hits, -x[1].age_seconds())
            )

            # Remove bottom 20%
            remove_count = int(len(sorted_entries) * 0.2)
            for key, _ in sorted_entries[:remove_count]:
                del self._cache[key]
                self.stats.evictions += 1

    def clear(self) -> None:
        """Clear all cached entries"""
        self._cache.clear()

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            "enabled": self.config.enabled,
            "entries": len(self._cache),
            "max_entries": self.config.max_entries,
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "hit_rate": f"{self.stats.hit_rate:.1f}%",
            "evictions": self.stats.evictions,
            "bytes_saved_estimate": self.stats.bytes_saved_estimate,
            "last_cleanup": self.stats.last_cleanup.isoformat() if self.stats.last_cleanup else None
        }

    def invalidate(self, endpoint: str = None, params: Dict = None) -> int:
        """
        Invalidate cache entries.

        Args:
            endpoint: Specific endpoint to invalidate (or None for pattern match)
            params: Specific params to invalidate

        Returns:
            Number of entries invalidated
        """
        if endpoint and params:
            # Invalidate specific entry
            key = self._make_key(endpoint, params)
            if key in self._cache:
                del self._cache[key]
                return 1
            return 0
        elif endpoint:
            # Invalidate all entries matching endpoint pattern
            keys_to_remove = [
                key for key in self._cache.keys()
                if endpoint.lower() in key.lower()
            ]
            for key in keys_to_remove:
                del self._cache[key]
            return len(keys_to_remove)
        else:
            # Clear all
            count = len(self._cache)
            self.clear()
            return count


class DataCollector:
    def __init__(self, cache_dir="data", cache_config: CacheConfig = None, retry_config: RetryConfig = None,
                 pool_connections: int = 10, pool_maxsize: int = 20):
        """
        Initialize the data collector with caching and connection pooling.

        Args:
            cache_dir: Directory for file-based caching
            cache_config: Configuration for API response caching
            retry_config: Configuration for retry behavior
            pool_connections: Number of connection pools to cache (default 10)
            pool_maxsize: Maximum connections per pool (default 20)
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # Initialize response cache for API calls
        self.cache = ResponseCache(cache_config or CacheConfig())

        # Initialize retry-enabled requests client with connection pooling
        self.retry_config = retry_config or RetryConfig()
        self.http = RetryableRequests(
            self.retry_config,
            pool_connections=pool_connections,
            pool_maxsize=pool_maxsize
        )

    def get_active_markets(self, limit=100, use_cache=True):
        """Fetch active markets from Polymarket via events endpoint"""
        url = f"{GAMMA_API_URL}/events"
        params = {
            "limit": limit,
            "active": "true",
            "closed": "false"
        }

        # Check cache first
        if use_cache:
            hit, cached_data = self.cache.get(url, params)
            if hit:
                return cached_data

        response = self.http.get(url, params=params, timeout=30)
        events = response.json()

        # Extract markets from events with proper token structure
        markets = []
        for event in events:
            event_markets = event.get("markets", [])
            for market in event_markets:
                # Get token IDs from clobTokenIds - may be JSON string or list
                clob_tokens_raw = market.get("clobTokenIds", [])

                # Parse if it's a JSON string
                if isinstance(clob_tokens_raw, str):
                    try:
                        clob_tokens = json.loads(clob_tokens_raw)
                    except json.JSONDecodeError:
                        clob_tokens = []
                else:
                    clob_tokens = clob_tokens_raw

                # Get outcomes - may also be JSON string
                outcomes_raw = market.get("outcomes", '["Yes", "No"]')
                if isinstance(outcomes_raw, str):
                    try:
                        outcomes = json.loads(outcomes_raw)
                    except json.JSONDecodeError:
                        outcomes = ["Yes", "No"]
                else:
                    outcomes = outcomes_raw or ["Yes", "No"]

                # Build tokens list
                tokens = []
                for i, token_id in enumerate(clob_tokens):
                    if token_id:
                        tokens.append({
                            "token_id": str(token_id),
                            "outcome": outcomes[i] if i < len(outcomes) else f"Outcome {i+1}"
                        })

                market["tokens"] = tokens
                market["event_title"] = event.get("title", "")
                markets.append(market)

        # Cache the processed result
        if use_cache:
            self.cache.set(url, params, markets)

        return markets

    def get_market_details(self, condition_id, use_cache=True):
        """Get detailed market info including tokens"""
        url = f"{GAMMA_API_URL}/markets/{condition_id}"

        # Check cache first
        if use_cache:
            hit, cached_data = self.cache.get(url, None)
            if hit:
                return cached_data

        response = self.http.get(url, timeout=30)
        data = response.json()

        # Cache the result
        if use_cache:
            self.cache.set(url, None, data)

        return data

    def get_price_history(self, token_id, interval="1h", fidelity=60, use_cache=True):
        """
        Fetch price history for a token
        interval: 1m, 5m, 15m, 1h, 4h, 1d
        fidelity: number of data points
        """
        url = f"{CLOB_API_URL}/prices-history"
        params = {
            "market": token_id,
            "interval": interval,
            "fidelity": fidelity
        }

        # Check cache first
        if use_cache:
            hit, cached_data = self.cache.get(url, params)
            if hit:
                return cached_data

        try:
            response = self.http.get(url, params=params, timeout=30)
            data = response.json()
            history = data.get("history", [])

            # Cache the result
            if use_cache:
                self.cache.set(url, params, history)

            return history
        except Exception as e:
            logger.warning(f"Error fetching price history: {e}")
            return []

    def get_orderbook(self, token_id, use_cache=True):
        """Get current orderbook for a token"""
        url = f"{CLOB_API_URL}/book"
        params = {"token_id": token_id}

        # Check cache first (short TTL for orderbook - 5 seconds)
        if use_cache:
            hit, cached_data = self.cache.get(url, params)
            if hit:
                return cached_data

        try:
            response = self.http.get(url, params=params, timeout=30)
            data = response.json()

            # Cache the result (short TTL automatically applied for orderbook)
            if use_cache:
                self.cache.set(url, params, data)

            return data
        except Exception as e:
            logger.warning(f"Error fetching orderbook: {e}")
            return {"bids": [], "asks": []}

    def get_current_price(self, token_id):
        """Get current mid price for a token"""
        orderbook = self.get_orderbook(token_id)

        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])

        if bids and asks:
            best_bid = float(bids[0].get("price", 0))
            best_ask = float(asks[0].get("price", 1))
            return (best_bid + best_ask) / 2
        elif bids:
            return float(bids[0].get("price", 0.5))
        elif asks:
            return float(asks[0].get("price", 0.5))
        return 0.5

    def collect_market_data(self, market):
        """Collect comprehensive data for a single market"""
        tokens = market.get("tokens", [])
        if not tokens:
            return None

        market_data = {
            "condition_id": market.get("conditionId", ""),
            "question": market.get("question", ""),
            "volume": float(market.get("volume", 0) or 0),
            "liquidity": float(market.get("liquidity", 0) or 0),
            "outcomes": [],
            "timestamp": datetime.now().isoformat()
        }

        for token in tokens:
            token_id = token.get("token_id", "")
            outcome = token.get("outcome", "")

            if not token_id:
                continue

            # Get current price
            current_price = self.get_current_price(token_id)

            # Get price history
            history = self.get_price_history(token_id, interval="1h", fidelity=48)

            outcome_data = {
                "token_id": token_id,
                "outcome": outcome,
                "current_price": current_price,
                "price_history": history
            }

            market_data["outcomes"].append(outcome_data)
            time.sleep(0.1)  # Rate limiting

        return market_data

    def scan_markets(self, limit=50, min_volume=1000, min_liquidity=500):
        """Scan markets and collect data for analysis"""
        print(f"Scanning top {limit} markets...")

        markets = self.get_active_markets(limit=limit * 2)

        # Filter by volume and liquidity
        filtered = []
        for m in markets:
            vol = float(m.get("volume", 0) or 0)
            liq = float(m.get("liquidity", 0) or 0)
            if vol >= min_volume and liq >= min_liquidity:
                filtered.append(m)
                if len(filtered) >= limit:
                    break

        print(f"Found {len(filtered)} markets meeting criteria")

        collected = []
        for i, market in enumerate(filtered):
            print(f"  [{i+1}/{len(filtered)}] {market.get('question', 'Unknown')[:50]}...")
            data = self.collect_market_data(market)
            if data and data["outcomes"]:
                collected.append(data)
            time.sleep(0.2)  # Rate limiting

        return collected

    def save_data(self, data, filename="market_data.json"):
        """Save collected data to file"""
        filepath = os.path.join(self.cache_dir, filename)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved data to {filepath}")

    def load_data(self, filename="market_data.json"):
        """Load data from file"""
        filepath = os.path.join(self.cache_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                return json.load(f)
        return None

    def get_cache_stats(self) -> Dict:
        """Get API cache statistics"""
        return self.cache.get_stats()

    def get_http_stats(self) -> Dict[str, int]:
        """Get HTTP request statistics (connection pool usage)"""
        return self.http.get_stats()

    def print_cache_stats(self):
        """Print cache and connection pool statistics"""
        cache_stats = self.get_cache_stats()
        http_stats = self.get_http_stats()

        print(f"\n{'='*50}")
        print("API CACHE STATISTICS")
        print(f"{'='*50}")
        print(f"Cache Enabled: {cache_stats['enabled']}")
        print(f"Cached Entries: {cache_stats['entries']} / {cache_stats['max_entries']}")
        print(f"Cache Hits: {cache_stats['hits']}")
        print(f"Cache Misses: {cache_stats['misses']}")
        print(f"Hit Rate: {cache_stats['hit_rate']}")
        print(f"Evictions: {cache_stats['evictions']}")
        if cache_stats['bytes_saved_estimate'] > 0:
            saved_kb = cache_stats['bytes_saved_estimate'] / 1024
            print(f"Est. Bytes Saved: {saved_kb:.1f} KB")

        print(f"\n{'='*50}")
        print("CONNECTION POOL STATISTICS")
        print(f"{'='*50}")
        print(f"Total HTTP Requests: {http_stats['requests']}")
        print(f"Retries: {http_stats['retries']}")
        print(f"Errors: {http_stats['errors']}")

    def clear_cache(self):
        """Clear the API response cache"""
        self.cache.clear()
        print("API cache cleared")

    def close(self):
        """Close all HTTP connections in the pool"""
        self.http.close()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close connections"""
        self.close()
        return False


if __name__ == "__main__":
    # Use context manager for proper connection cleanup
    with DataCollector() as collector:
        print("Testing data collection with caching and connection pooling...")
        print("=" * 60)

        # First request - cache miss, establishes connection pool
        print("\n[1] First request (cache miss, connection pool init):")
        start = time.time()
        markets = collector.get_active_markets(limit=5)
        elapsed1 = time.time() - start
        print(f"    Time: {elapsed1:.3f}s - Found {len(markets)} markets")

        # Second request - cache hit
        print("\n[2] Second request (cache hit expected):")
        start = time.time()
        markets = collector.get_active_markets(limit=5)
        elapsed2 = time.time() - start
        print(f"    Time: {elapsed2:.3f}s - Found {len(markets)} markets")

        # Calculate speedup
        if elapsed2 > 0:
            speedup = elapsed1 / elapsed2
            print(f"    Speedup: {speedup:.1f}x faster with cache")

        # Third request - cache bypass, tests connection reuse
        print("\n[3] Third request (no cache, tests connection reuse):")
        start = time.time()
        markets = collector.get_active_markets(limit=5, use_cache=False)
        elapsed3 = time.time() - start
        print(f"    Time: {elapsed3:.3f}s - Found {len(markets)} markets")

        # Compare with and without connection reuse
        if elapsed3 > 0 and elapsed1 > 0:
            conn_speedup = elapsed1 / elapsed3
            print(f"    Connection reuse benefit: {conn_speedup:.2f}x")
            print(f"    (First request includes TCP handshake, subsequent reuse existing connection)")

        # Test orderbook caching and connection pooling
        print("\n[4] Testing orderbook requests (connection pool reuse):")
        for m in markets[:3]:
            tokens = m.get("tokens", [])
            if tokens:
                token_id = tokens[0].get("token_id", "")
                if token_id:
                    # First call - cache miss
                    start = time.time()
                    price1 = collector.get_current_price(token_id)
                    t1 = time.time() - start

                    # Second call - cache hit
                    start = time.time()
                    price2 = collector.get_current_price(token_id)
                    t2 = time.time() - start

                    print(f"    {m.get('question', 'N/A')[:40]}...")
                    print(f"      Price: {price1:.2%} | API call: {t1:.3f}s | Cached: {t2:.3f}s")

        # Print cache and connection pool statistics
        collector.print_cache_stats()

        # Show markets
        print(f"\n{'='*60}")
        print("MARKETS FOUND")
        print(f"{'='*60}")
        for m in markets[:3]:
            print(f"  - {m.get('question', 'N/A')[:60]}")

        print(f"\n{'='*60}")
        print("CONNECTION POOLING BENEFITS")
        print(f"{'='*60}")
        print("- TCP connection reuse reduces latency by ~50-100ms per request")
        print("- Keep-alive headers maintain persistent connections")
        print("- Connection pool avoids handshake overhead for repeated API calls")
        print("- HTTPAdapter manages pool size (10 hosts, 20 connections each)")
