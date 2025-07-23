import logging
import os
from datetime import datetime
from typing import Optional
from config import Config

class TradingLogger:
    """Enhanced logging utility for the crypto trading bot"""
    
    def __init__(self, name: str = "CryptoTradingBot"):
        self.logger = logging.getLogger(name)
        self.setup_logger()
    
    def setup_logger(self):
        """Setup logger with file and console handlers"""
        self.logger.setLevel(getattr(logging, Config.LOG_LEVEL.upper()))
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # File handler for all logs
        file_handler = logging.FileHandler(
            f'logs/trading_bot_{datetime.now().strftime("%Y%m%d")}.log'
        )
        file_handler.setLevel(logging.DEBUG)
        
        # File handler for trades only
        trade_handler = logging.FileHandler(
            f'logs/trades_{datetime.now().strftime("%Y%m%d")}.log'
        )
        trade_handler.setLevel(logging.INFO)
        trade_handler.addFilter(TradeFilter())
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        file_handler.setFormatter(detailed_formatter)
        trade_handler.setFormatter(detailed_formatter)
        console_handler.setFormatter(simple_formatter)
        
        # Add handlers if not already added
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(trade_handler)
            self.logger.addHandler(console_handler)
    
    def trade(self, message: str, extra_data: Optional[dict] = None):
        """Log trading-specific messages"""
        if extra_data:
            message = f"{message} | Data: {extra_data}"
        self.logger.info(f"[TRADE] {message}")
    
    def strategy(self, message: str, strategy_name: str):
        """Log strategy-specific messages"""
        self.logger.info(f"[STRATEGY-{strategy_name.upper()}] {message}")
    
    def risk(self, message: str):
        """Log risk management messages"""
        self.logger.warning(f"[RISK] {message}")
    
    def error(self, message: str, exception: Optional[Exception] = None):
        """Log error messages"""
        if exception:
            self.logger.error(f"[ERROR] {message} | Exception: {str(exception)}")
        else:
            self.logger.error(f"[ERROR] {message}")
    
    def performance(self, message: str, metrics: Optional[dict] = None):
        """Log performance metrics"""
        if metrics:
            message = f"{message} | Metrics: {metrics}"
        self.logger.info(f"[PERFORMANCE] {message}")
    
    def arbitrage(self, message: str, spread_data: Optional[dict] = None):
        """Log arbitrage opportunities"""
        if spread_data:
            message = f"{message} | Spread: {spread_data}"
        self.logger.info(f"[ARBITRAGE] {message}")
    
    def news(self, message: str, sentiment_data: Optional[dict] = None):
        """Log news sentiment analysis"""
        if sentiment_data:
            message = f"{message} | Sentiment: {sentiment_data}"
        self.logger.info(f"[NEWS] {message}")

class TradeFilter(logging.Filter):
    """Filter to capture only trade-related log messages"""
    
    def filter(self, record):
        return '[TRADE]' in record.getMessage()

# Global logger instance
logger = TradingLogger()