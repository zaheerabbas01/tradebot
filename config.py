import os
from dotenv import load_dotenv
from typing import Dict, List

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the crypto trading bot"""
    
    # Binance API Configuration
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
    BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')
    BINANCE_TESTNET = os.getenv('BINANCE_TESTNET', 'True').lower() == 'true'
    
    # Bybit API Configuration
    BYBIT_API_KEY = os.getenv('BYBIT_API_KEY')
    BYBIT_SECRET_KEY = os.getenv('BYBIT_SECRET_KEY')
    
    # CryptoPanic API
    CRYPTOPANIC_API_KEY = os.getenv('CRYPTOPANIC_API_KEY')
    
    # Telegram Configuration
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
    
    # Google Sheets Configuration
    GOOGLE_SHEETS_CREDENTIALS_FILE = os.getenv('GOOGLE_SHEETS_CREDENTIALS_FILE', 'credentials.json')
    GOOGLE_SHEETS_ID = os.getenv('GOOGLE_SHEETS_ID')
    
    # Trading Configuration
    INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', 10000))
    MAX_RISK_PER_TRADE = float(os.getenv('MAX_RISK_PER_TRADE', 0.01))
    LEVERAGE = int(os.getenv('LEVERAGE', 20))
    LIQUIDATION_THRESHOLD = float(os.getenv('LIQUIDATION_THRESHOLD', 0.5))
    
    # Strategy Parameters
    EMA_FAST = int(os.getenv('EMA_FAST', 50))
    EMA_SLOW = int(os.getenv('EMA_SLOW', 200))
    RSI_PERIOD = int(os.getenv('RSI_PERIOD', 14))
    RSI_OVERBOUGHT = float(os.getenv('RSI_OVERBOUGHT', 70))
    RSI_OVERSOLD = float(os.getenv('RSI_OVERSOLD', 30))
    BOLLINGER_PERIOD = int(os.getenv('BOLLINGER_PERIOD', 20))
    BOLLINGER_STD = float(os.getenv('BOLLINGER_STD', 2))
    STOP_LOSS_PERCENT = float(os.getenv('STOP_LOSS_PERCENT', 0.02))
    TRAILING_TAKE_PROFIT = float(os.getenv('TRAILING_TAKE_PROFIT', 0.05))
    MEAN_REVERSION_STOP_LOSS = float(os.getenv('MEAN_REVERSION_STOP_LOSS', 0.015))
    MEAN_REVERSION_TAKE_PROFIT = float(os.getenv('MEAN_REVERSION_TAKE_PROFIT', 0.03))
    
    # News Sentiment Thresholds
    BULLISH_SENTIMENT_THRESHOLD = float(os.getenv('BULLISH_SENTIMENT_THRESHOLD', 0.5))
    BEARISH_SENTIMENT_THRESHOLD = float(os.getenv('BEARISH_SENTIMENT_THRESHOLD', -0.5))
    VOLUME_SPIKE_THRESHOLD = float(os.getenv('VOLUME_SPIKE_THRESHOLD', 1.5))
    
    # Arbitrage Configuration
    MIN_ARBITRAGE_SPREAD = float(os.getenv('MIN_ARBITRAGE_SPREAD', 0.003))
    
    # Backtesting Configuration
    BACKTEST_START_DATE = os.getenv('BACKTEST_START_DATE', '2021-01-01')
    BACKTEST_END_DATE = os.getenv('BACKTEST_END_DATE', '2025-01-01')
    MIN_WIN_RATE = float(os.getenv('MIN_WIN_RATE', 0.55))
    
    # VPS Configuration
    MAX_PING_MS = int(os.getenv('MAX_PING_MS', 5))
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # Trading Pairs
    TRADING_PAIRS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    
    # Timeframes
    TIMEFRAME = '15m'  # 15-minute charts
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that all required configuration is present"""
        required_vars = [
            'BINANCE_API_KEY',
            'BINANCE_SECRET_KEY',
            'TELEGRAM_BOT_TOKEN',
            'TELEGRAM_CHAT_ID'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not getattr(cls, var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        
        return True