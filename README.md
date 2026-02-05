# BTC Up or Down Trading Bot

Automated trading bot for Polymarket's Bitcoin Up or Down prediction markets. Supports both 15-minute and hourly timeframes with technical analysis-based predictions.

## Features

- **Technical Analysis**: RSI, MACD, Bollinger Bands, SMA/EMA, momentum indicators
- **Multiple Timeframes**: 15-minute and hourly market support
- **Contrarian Strategy**: Bets against heavily skewed markets (>70%)
- **Kelly Criterion**: Position sizing based on calculated edge
- **Live Trading**: Direct integration with Polymarket CLOB API

## Files

### Core Trading Bots

| File | Description |
|------|-------------|
| `run_hourly_bot.py` | Hourly BTC trading with contrarian strategy and Kelly sizing |
| `run_smart_trades.py` | 15-minute trading with prediction engine integration |
| `run_4_events.py` | Trade 4 consecutive 15-minute markets |
| `btc_15m_bot.py` | Full-featured 15-minute trading bot |

### Analysis & Monitoring

| File | Description |
|------|-------------|
| `prediction_engine.py` | Technical analysis engine with market regime detection |
| `ensemble_predictor.py` | Combines multiple prediction signals |
| `indicators.py` | RSI, MACD, Bollinger Bands, momentum calculations |
| `monitor_positions.py` | Track open positions and P&L |
| `check_hourly.py` | Check hourly trade results |

### Infrastructure

| File | Description |
|------|-------------|
| `trade_executor.py` | Polymarket CLOB order execution |
| `trading_bot.py` | Base trading bot class |
| `data_collector.py` | Historical data collection |

## Configuration

### Environment Variables

```bash
export TRADING_MODE=live  # or 'paper' for simulation
export POLYMARKET_API_KEY=your_api_key
export POLYMARKET_API_SECRET=your_api_secret
export POLYMARKET_PASSPHRASE=your_passphrase
```

### Trading Parameters

In `run_hourly_bot.py`:
```python
TARGET_TRADES = 3           # Number of trades per session
BASE_POSITION_SIZE = 5.0    # Base $ per trade
MAX_POSITION_SIZE = 15.0    # Max with Kelly sizing
MIN_CONFIDENCE = 60         # Minimum prediction confidence %
CONTRARIAN_THRESHOLD = 0.70 # Bet against if one side > 70%
```

## Usage

### Run Hourly Bot
```bash
python run_hourly_bot.py
```

### Run 15-Minute Bot
```bash
python run_smart_trades.py
```

### Check Results
```bash
python check_hourly.py
```

## Trading Strategy

### Signal Priority

1. **Contrarian**: If market is >70% skewed, bet against the crowd
2. **Technical Prediction**: Use prediction engine if confidence >= threshold
3. **Value Betting**: Buy underpriced side (<45%)
4. **Default**: Slight bearish bias if no clear signal

### Position Sizing (Kelly Criterion)

```python
kelly_pct = (odds * win_prob - lose_prob) / odds
fractional_kelly = kelly_pct * 0.25  # 25% Kelly for safety
position_size = base_size * (1 + fractional_kelly * 2)
```

## Market Discovery

Markets are discovered using timestamp-based slugs:
- 15-minute: `btc-updown-15m-{unix_timestamp}`
- Hourly: Search for "Bitcoin Up or Down - [Date], [Time] ET"

## API Integration

- **Polymarket Gamma API**: Market discovery and prices
- **Polymarket CLOB API**: Order placement and execution
- **Binance API**: BTC price history for technical analysis

## Disclaimer

This bot is for educational purposes. Trading involves risk. Past performance does not guarantee future results. Use at your own risk.
