# Advanced Cryptocurrency Trading Bot 🚀

A comprehensive, professional-grade cryptocurrency futures trading bot with advanced strategies, risk management, and machine learning optimization.

## 🌟 Features

### Trading Strategies
- **Trend Following**: EMA crossover with RSI filtering (50/200 EMA + RSI < 70/30)
- **Mean Reversion**: Bollinger Bands with volume spike confirmation
- **News Sentiment Analysis**: CryptoPanic API integration with FUD/FOMO detection
- **Multi-Strategy Fusion**: Weighted signal combination with conflict resolution

### Risk Management
- **Kelly Criterion Position Sizing**: Optimal position sizing based on win rate and risk-reward ratio
- **Portfolio Heat Monitoring**: Maximum 5% total portfolio risk exposure
- **Dynamic Drawdown Protection**: Automatic position reduction during drawdowns
- **Correlation Analysis**: Prevents over-exposure to correlated assets
- **Auto-Liquidation Protection**: 50% margin level threshold with emergency position closure

### Advanced Features
- **20x Leverage Trading**: Binance Futures integration with real-time WebSocket data
- **Machine Learning Optimization**: TensorFlow-based strategy weight adjustment
- **Genetic Algorithm Backtesting**: Parameter optimization using evolutionary algorithms
- **Real-time News Analysis**: High-impact news detection with sentiment scoring
- **SEC Investigation Blacklist**: Automatic asset filtering for regulatory risks

### Monitoring & Alerts
- **Telegram Notifications**: Trade alerts, performance reports, and risk warnings
- **Google Sheets Logging**: Comprehensive trade and performance tracking
- **24/7 VPS Deployment**: Auto-restart on disconnect with <5ms ping requirements
- **Performance Analytics**: Sharpe ratio, Calmar ratio, win rate tracking

## 📊 Performance Targets

- **Minimum Win Rate**: 55%
- **Target Sharpe Ratio**: > 1.5
- **Maximum Drawdown**: < 20%
- **Risk Per Trade**: 1% of capital
- **Diversification**: 3 uncorrelated pairs (BTC, ETH, SOL)

## 🛠️ Installation

### Prerequisites
- Python 3.9+
- Binance Futures API access
- VPS with <5ms ping to Binance servers (recommended)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd crypto-trading-bot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and settings
   ```

4. **Set up API keys**
   - **Binance Futures API**: Enable futures trading with read/write permissions
   - **CryptoPanic API**: Get free API key from cryptopanic.com
   - **Telegram Bot**: Create bot via @BotFather
   - **Google Sheets**: Set up service account credentials

### Environment Configuration

```bash
# Binance API Configuration
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET_KEY=your_binance_secret_key_here
BINANCE_TESTNET=True  # Set to False for live trading

# CryptoPanic API
CRYPTOPANIC_API_KEY=your_cryptopanic_api_key_here

# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here

# Trading Configuration
INITIAL_CAPITAL=10000  # USD
MAX_RISK_PER_TRADE=0.01  # 1%
LEVERAGE=20
```

## 🚀 Usage

### Backtesting (Recommended First Step)

Run comprehensive backtests to validate strategies:

```bash
# Basic backtest
python run_backtest.py --symbol BTC/USDT --start 2021-01-01 --end 2023-12-31

# Parameter optimization
python run_backtest.py --optimize --symbol BTC/USDT --start 2022-01-01 --end 2023-12-31

# Generate detailed report with plots
python run_backtest.py --symbol BTC/USDT --start 2021-01-01 --end 2023-12-31 --plot --report
```

### Live Trading

After successful backtesting:

```bash
# Start the trading bot
python crypto_trading_bot.py
```

### Strategy Testing

Test individual strategies:

```bash
# Test trend following strategy only
python run_backtest.py --strategy trend --symbol BTC/USDT

# Test mean reversion strategy only
python run_backtest.py --strategy mean_reversion --symbol ETH/USDT
```

## 📈 Trading Strategies

### 1. Trend Following Strategy

**Entry Rules:**
- Buy: 50 EMA crosses above 200 EMA AND RSI(14) < 70
- Sell: 50 EMA crosses below 200 EMA AND RSI(14) > 30

**Exit Rules:**
- Stop Loss: 2%
- Trailing Take Profit: 5%

**Based on:** "Trend Following" by Michael Covel, "Technical Analysis of Financial Markets" by John Murphy

### 2. Mean Reversion Strategy

**Entry Rules:**
- Buy: Price touches lower Bollinger Band (2σ) AND volume spikes 50%+
- Sell: Price touches upper Bollinger Band (2σ)

**Exit Rules:**
- Stop Loss: 1.5%
- Fixed Take Profit: 3%

**Based on:** "Bollinger on Bollinger Bands" by John Bollinger, "Mean Reversion Trading Systems" by Howard Bandy

### 3. News Sentiment Strategy

**Entry Rules:**
- Buy: Sentiment score > +0.5 AND volume increases 50%+
- Sell: Sentiment score < -0.5 AND confirmed FUD detection

**Features:**
- Real-time news monitoring from 50+ crypto sources
- High-impact keyword detection (ETF approval, regulation, partnerships)
- Social media sentiment aggregation
- FUD/FOMO pattern recognition

## ⚠️ Risk Management

### Position Sizing (Kelly Criterion)
```python
# Optimal fraction calculation
f = (bp - q) / b
# Where: b = win/loss ratio, p = win probability, q = loss probability

# Applied with 25% safety margin
safe_position_size = kelly_fraction * 0.25 * account_balance
```

### Portfolio Risk Controls
- **Maximum Heat**: 5% of total portfolio at risk
- **Correlation Limits**: Max 70% correlation between positions
- **Drawdown Protection**: Position reduction during 10%+ drawdowns
- **Margin Monitoring**: Auto-liquidation at 50% margin level

### Asset Blacklisting
Automatic filtering of assets under:
- SEC investigation
- Regulatory scrutiny
- Exchange delisting notices
- Major security breaches

## 📊 Performance Monitoring

### Real-time Metrics
- **Live P&L**: Real-time profit/loss tracking
- **Win Rate**: Rolling 30-trade average
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Peak-to-trough decline
- **Portfolio Heat**: Current risk exposure

### Alerts & Notifications
- **Trade Executions**: Entry/exit notifications
- **Risk Warnings**: Margin level alerts
- **Performance Reports**: Daily/weekly summaries
- **System Status**: Connection and health monitoring

## 🔧 Advanced Configuration

### Strategy Weights
Adjust strategy allocation in `crypto_trading_bot.py`:
```python
self.strategy_weights = {
    'trend_following': 0.6,  # 60% allocation
    'mean_reversion': 0.4    # 40% allocation
}
```

### Risk Parameters
Modify risk settings in `config.py`:
```python
MAX_RISK_PER_TRADE = 0.01      # 1% per trade
MAX_PORTFOLIO_HEAT = 0.05      # 5% total exposure
MAX_DRAWDOWN = 0.20            # 20% maximum drawdown
LIQUIDATION_THRESHOLD = 0.5    # 50% margin level
```

### Technical Indicators
Customize indicator parameters:
```python
EMA_FAST = 50                  # Fast EMA period
EMA_SLOW = 200                 # Slow EMA period
RSI_PERIOD = 14                # RSI calculation period
BOLLINGER_PERIOD = 20          # Bollinger Bands period
BOLLINGER_STD = 2              # Standard deviation multiplier
```

## 🏗️ Architecture

### Core Components
```
crypto_trading_bot.py          # Main orchestrator
├── exchanges/
│   └── binance_client.py      # Binance Futures integration
├── strategies/
│   ├── trend_following.py     # EMA + RSI strategy
│   └── mean_reversion.py      # Bollinger Bands strategy
├── analysis/
│   ├── technical_indicators.py # TA-Lib indicators
│   └── news_sentiment.py      # News analysis engine
├── risk_management/
│   └── risk_manager.py        # Portfolio risk controls
├── backtesting/
│   └── backtest_engine.py     # Strategy backtesting
└── utils/
    └── logger.py              # Comprehensive logging
```

### Data Flow
1. **Market Data**: Real-time WebSocket feeds from Binance
2. **Technical Analysis**: Calculate indicators (EMA, RSI, Bollinger Bands)
3. **News Analysis**: Fetch and analyze sentiment from CryptoPanic
4. **Signal Generation**: Combine technical and fundamental signals
5. **Risk Assessment**: Apply portfolio risk controls
6. **Order Execution**: Place trades via Binance Futures API
7. **Position Management**: Monitor and adjust positions
8. **Performance Tracking**: Log and analyze results

## 📚 Theoretical Foundation

### Technical Analysis
- **"Technical Analysis of the Financial Markets"** by John J. Murphy
- **"New Concepts in Technical Trading Systems"** by J. Welles Wilder
- **"Bollinger on Bollinger Bands"** by John Bollinger

### Quantitative Finance
- **"Quantitative Trading"** by Ernest Chan
- **"The Mathematics of Money Management"** by Ralph Vince
- **"Evidence-Based Technical Analysis"** by David Aronson

### Risk Management
- **"Risk Management and Financial Institutions"** by John Hull
- **"The Black Swan"** by Nassim Nicholas Taleb
- **"When Genius Failed"** by Roger Lowenstein

### Behavioral Finance
- **"Behavioral Finance"** by Hersh Shefrin
- **"The Wisdom of Crowds"** by James Surowiecki
- **"Sentiment Analysis and Opinion Mining"** by Bing Liu

## 🚨 Important Disclaimers

### Trading Risks
- **High Risk**: Cryptocurrency trading involves substantial risk of loss
- **Leverage Warning**: 20x leverage amplifies both gains and losses
- **Market Volatility**: Crypto markets are highly volatile and unpredictable
- **No Guarantees**: Past performance does not guarantee future results

### Regulatory Compliance
- **Local Laws**: Ensure compliance with local cryptocurrency regulations
- **Tax Obligations**: Consult tax professionals for trading tax implications
- **KYC/AML**: Maintain proper know-your-customer documentation

### Technical Considerations
- **API Limits**: Respect exchange rate limits and API terms of service
- **System Reliability**: Ensure stable internet and VPS infrastructure
- **Security**: Protect API keys and use IP restrictions
- **Monitoring**: Never leave the bot completely unattended

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd crypto-trading-bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/ # (when tests are implemented)
```

### Contribution Guidelines
1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Test** your changes thoroughly
4. **Commit** your changes (`git commit -m 'Add amazing feature'`)
5. **Push** to the branch (`git push origin feature/amazing-feature`)
6. **Open** a Pull Request

## 📞 Support

### Documentation
- **Strategy Guides**: Detailed strategy explanations
- **API Reference**: Complete function documentation
- **Configuration Guide**: Parameter optimization tips
- **Troubleshooting**: Common issues and solutions

### Community
- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Strategy discussions and improvements

## 📄 License

This project is licensed under the MIT License.

## ⚡ Quick Start Checklist

- [ ] Install Python 3.9+ and dependencies
- [ ] Set up Binance Futures API keys
- [ ] Configure environment variables in `.env`
- [ ] Run backtests to validate strategies
- [ ] Start with testnet/paper trading
- [ ] Monitor performance and adjust parameters
- [ ] Deploy on VPS for 24/7 operation
- [ ] Set up monitoring and alerts
- [ ] Implement proper security measures
- [ ] Maintain trading logs and compliance

---

**⚠️ Remember: This is experimental software. Start with small amounts, monitor closely, and never risk more than you can afford to lose.**

**🎯 Target Audience: Experienced traders, quantitative analysts, and developers with understanding of financial markets and cryptocurrency trading.**