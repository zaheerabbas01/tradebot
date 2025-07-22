BINANCE_API_KEY = 'your_binance_api_key'
BINANCE_API_SECRET = 'your_binance_api_secret'
BYBIT_API_KEY = 'your_bybit_api_key'
BYBIT_API_SECRET = 'your_bybit_api_secret'
CRYPTOPANIC_API_KEY = 'your_cryptopanic_api_key'
GOOGLE_SHEETS_CREDENTIALS = 'path_to_google_sheets_credentials.json'
TELEGRAM_BOT_TOKEN = 'your_telegram_bot_token'
TELEGRAM_CHAT_ID = 'your_telegram_chat_id'

# Trading settings
LEVERAGE = 20
RISK_PER_TRADE = 0.01  # 1% of capital
LIQUIDATION_MARGIN = 0.5  # 50%
DIVERSIFY_PAIRS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
BLACKLISTED_ASSETS = ['XRP', 'LTC']  # Example, update as needed

# Arbitrage settings
ARBITRAGE_SPREAD_THRESHOLD = 0.003  # 0.3%

# Backtest settings
BACKTEST_START = '2021-01-01'
BACKTEST_END = '2025-01-01'
MIN_WIN_RATE = 0.55