#!/usr/bin/env python3
"""
Advanced Cryptocurrency Trading Bot

A comprehensive futures trading bot with multiple strategies, risk management,
news sentiment analysis, and machine learning optimization.

Features:
- Trend Following Strategy (EMA + RSI)
- Mean Reversion Strategy (Bollinger Bands + Volume)
- News Sentiment Analysis (CryptoPanic API)
- Advanced Risk Management (Kelly Criterion, Portfolio Heat)
- Real-time WebSocket Data (Binance Futures)
- 20x Leverage Trading
- Machine Learning Strategy Optimization
- Telegram Notifications
- Google Sheets Logging

Author: AI Trading System
Version: 1.0.0
"""

import asyncio
import signal
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import threading
import schedule

from config import Config
from exchanges.binance_client import BinanceFuturesClient
from strategies.trend_following import TrendFollowingStrategy
from strategies.mean_reversion import MeanReversionStrategy
from analysis.news_sentiment import NewsSentimentAnalyzer
from risk_management.risk_manager import RiskManager
from utils.logger import logger

class CryptoTradingBot:
    """
    Main trading bot orchestrating all components
    """
    
    def __init__(self):
        self.running = False
        self.exchange = None
        self.strategies = {}
        self.news_analyzer = None
        self.risk_manager = None
        
        # Performance tracking
        self.start_time = datetime.now()
        self.last_price_update = {}
        self.strategy_weights = {
            'trend_following': 0.6,
            'mean_reversion': 0.4
        }
        
        # Initialize components
        self.setup_components()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def setup_components(self):
        """Initialize all bot components"""
        try:
            logger.logger.info("Initializing Crypto Trading Bot...")
            
            # Validate configuration
            Config.validate_config()
            
            # Initialize exchange connection
            self.exchange = BinanceFuturesClient()
            
            # Initialize strategies
            self.strategies = {
                'trend_following': TrendFollowingStrategy(),
                'mean_reversion': MeanReversionStrategy()
            }
            
            # Initialize news sentiment analyzer
            if Config.CRYPTOPANIC_API_KEY:
                self.news_analyzer = NewsSentimentAnalyzer()
                logger.logger.info("News sentiment analyzer initialized")
            else:
                logger.logger.warning("No CryptoPanic API key - news analysis disabled")
            
            # Initialize risk manager
            self.risk_manager = RiskManager()
            
            logger.logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize bot components", e)
            raise
    
    def start(self):
        """Start the trading bot"""
        try:
            logger.logger.info("Starting Crypto Trading Bot...")
            self.running = True
            
            # Start WebSocket for real-time data
            self.exchange.start_websocket(
                Config.TRADING_PAIRS,
                self.on_price_update
            )
            
            # Schedule periodic tasks
            self.schedule_tasks()
            
            # Start main trading loop
            self.run_trading_loop()
            
        except Exception as e:
            logger.error("Error starting trading bot", e)
            self.stop()
    
    def stop(self):
        """Stop the trading bot gracefully"""
        logger.logger.info("Stopping Crypto Trading Bot...")
        self.running = False
        
        if self.exchange:
            self.exchange.stop_websocket()
        
        # Close all open positions
        self.close_all_positions()
        
        logger.logger.info("Crypto Trading Bot stopped")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)
    
    def on_price_update(self, symbol: str, price: float, volume: float):
        """Handle real-time price updates from WebSocket"""
        try:
            # Update last price
            self.last_price_update[symbol] = {
                'price': price,
                'volume': volume,
                'timestamp': datetime.now()
            }
            
            # Update risk manager with current prices
            self.risk_manager.update_position_price(symbol, price)
            
        except Exception as e:
            logger.error(f"Error processing price update for {symbol}", e)
    
    def schedule_tasks(self):
        """Schedule periodic tasks"""
        # Schedule strategy analysis every 15 minutes (matching timeframe)
        schedule.every(15).minutes.do(self.analyze_and_trade)
        
        # Schedule performance reporting every hour
        schedule.every().hour.do(self.log_performance)
        
        # Schedule news analysis every 30 minutes
        if self.news_analyzer:
            schedule.every(30).minutes.do(self.analyze_news_sentiment)
        
        # Schedule daily reporting
        schedule.every().day.at("00:00").do(self.daily_report)
        
        # Start scheduler in separate thread
        scheduler_thread = threading.Thread(target=self.run_scheduler)
        scheduler_thread.daemon = True
        scheduler_thread.start()
    
    def run_scheduler(self):
        """Run scheduled tasks"""
        while self.running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def run_trading_loop(self):
        """Main trading loop"""
        logger.logger.info("Trading loop started")
        
        while self.running:
            try:
                # Run initial analysis
                self.analyze_and_trade()
                
                # Wait for next iteration (15 minutes to match timeframe)
                time.sleep(900)  # 15 minutes
                
            except Exception as e:
                logger.error("Error in main trading loop", e)
                time.sleep(60)  # Wait 1 minute before retrying
    
    def analyze_and_trade(self):
        """Analyze markets and execute trades"""
        try:
            logger.logger.info("Starting market analysis...")
            
            for symbol in Config.TRADING_PAIRS:
                try:
                    # Fetch market data
                    data = self.exchange.fetch_ohlcv(symbol, Config.TIMEFRAME, 1000)
                    
                    if data.empty:
                        logger.logger.warning(f"No data available for {symbol}")
                        continue
                    
                    # Get account balance
                    account_data = self.exchange.get_account_balance()
                    if not account_data:
                        logger.logger.warning("Could not fetch account data")
                        continue
                    
                    # Update drawdown tracking
                    should_stop = self.risk_manager.update_drawdown(account_data['total_balance'])
                    if should_stop:
                        logger.risk("Maximum drawdown exceeded - stopping trading")
                        self.stop()
                        return
                    
                    # Check margin level
                    margin_status = self.risk_manager.check_margin_level(account_data)
                    if margin_status['action_required']:
                        logger.risk(f"Margin warning: {margin_status['message']}")
                        if margin_status['status'] == 'CRITICAL':
                            self.close_all_positions()
                            continue
                    
                    # Analyze with each strategy
                    strategy_signals = {}
                    
                    for strategy_name, strategy in self.strategies.items():
                        signal = strategy.analyze(data, symbol)
                        strategy_signals[strategy_name] = signal
                        
                        logger.strategy(
                            f"Signal for {symbol}: {signal['signal']} "
                            f"(Confidence: {signal['confidence']:.2f})",
                            strategy_name
                        )
                    
                    # Combine strategy signals
                    combined_signal = self.combine_strategy_signals(strategy_signals, symbol)
                    
                    # Add news sentiment if available
                    if self.news_analyzer:
                        news_signal = self.analyze_news_for_symbol(symbol)
                        combined_signal = self.incorporate_news_sentiment(
                            combined_signal, news_signal
                        )
                    
                    # Execute trade if signal is strong enough
                    if combined_signal['signal'] != 'HOLD':
                        self.execute_trade(symbol, combined_signal, account_data, data)
                    
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}", e)
                    continue
            
            logger.logger.info("Market analysis completed")
            
        except Exception as e:
            logger.error("Error in analyze_and_trade", e)
    
    def combine_strategy_signals(self, signals: Dict, symbol: str) -> Dict:
        """Combine signals from multiple strategies"""
        
        # Initialize combined signal
        combined = {
            'signal': 'HOLD',
            'confidence': 0.0,
            'reason': 'No clear signal',
            'strategy_breakdown': signals
        }
        
        # Calculate weighted average
        total_weight = 0
        weighted_confidence = 0
        signal_votes = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        reasons = []
        
        for strategy_name, signal in signals.items():
            weight = self.strategy_weights.get(strategy_name, 0.5)
            
            # Weight the confidence
            if signal['signal'] != 'HOLD':
                weighted_confidence += signal['confidence'] * weight
                total_weight += weight
                signal_votes[signal['signal']] += weight
                reasons.append(f"{strategy_name}: {signal['reason']}")
        
        # Determine final signal
        if total_weight > 0:
            final_confidence = weighted_confidence / total_weight
            
            # Get the signal with highest weighted votes
            max_vote = max(signal_votes.values())
            if max_vote > 0.5:  # Majority threshold
                for signal_type, votes in signal_votes.items():
                    if votes == max_vote and signal_type != 'HOLD':
                        combined['signal'] = signal_type
                        combined['confidence'] = final_confidence
                        combined['reason'] = '; '.join(reasons)
                        break
        
        return combined
    
    def analyze_news_for_symbol(self, symbol: str) -> Dict:
        """Analyze news sentiment for a specific symbol"""
        if not self.news_analyzer:
            return {'action': 'HOLD', 'confidence': 0.0, 'reason': 'News analysis disabled'}
        
        try:
            # Get current volume data
            current_price_data = self.last_price_update.get(symbol)
            if not current_price_data:
                return {'action': 'HOLD', 'confidence': 0.0, 'reason': 'No current price data'}
            
            # Calculate volume increase (simplified)
            volume_increase = 1.2  # Default moderate increase
            
            # Get news sentiment
            sentiment_data = self.news_analyzer.get_news_sentiment(symbol)
            
            # Get trading recommendation
            news_recommendation = self.news_analyzer.should_trade_on_news(
                sentiment_data, volume_increase
            )
            
            return news_recommendation
            
        except Exception as e:
            logger.error(f"Error analyzing news for {symbol}", e)
            return {'action': 'HOLD', 'confidence': 0.0, 'reason': 'News analysis error'}
    
    def incorporate_news_sentiment(self, technical_signal: Dict, news_signal: Dict) -> Dict:
        """Incorporate news sentiment into technical analysis"""
        
        # If no news signal, return technical signal
        if news_signal['action'] == 'HOLD' or news_signal['confidence'] < 0.3:
            return technical_signal
        
        # If signals agree, boost confidence
        if technical_signal['signal'] == news_signal['action']:
            boosted_confidence = min(
                technical_signal['confidence'] + (news_signal['confidence'] * 0.3),
                1.0
            )
            technical_signal['confidence'] = boosted_confidence
            technical_signal['reason'] += f"; News: {news_signal['reason']}"
        
        # If signals conflict, reduce confidence
        elif (technical_signal['signal'] != 'HOLD' and 
              news_signal['action'] != 'HOLD' and
              technical_signal['signal'] != news_signal['action']):
            
            reduced_confidence = technical_signal['confidence'] * 0.7
            technical_signal['confidence'] = reduced_confidence
            technical_signal['reason'] += f"; Conflicting news: {news_signal['reason']}"
        
        # If only news signal is strong, use it
        elif (technical_signal['signal'] == 'HOLD' and 
              news_signal['confidence'] > 0.7):
            
            technical_signal['signal'] = news_signal['action']
            technical_signal['confidence'] = news_signal['confidence'] * 0.8
            technical_signal['reason'] = f"News-driven: {news_signal['reason']}"
        
        return technical_signal
    
    def execute_trade(self, symbol: str, signal: Dict, account_data: Dict, market_data):
        """Execute a trade based on the combined signal"""
        try:
            current_price = self.last_price_update.get(symbol, {}).get('price')
            if not current_price:
                logger.logger.warning(f"No current price for {symbol}")
                return
            
            # Check if we should trade (risk management)
            should_trade, reason = self.risk_manager.should_trade(
                symbol, signal['confidence'], account_data['free_balance']
            )
            
            if not should_trade:
                logger.risk(f"Trade rejected for {symbol}: {reason}")
                return
            
            # Calculate position size
            stop_loss_pct = signal.get('stop_loss_pct', Config.STOP_LOSS_PERCENT)
            position_size = self.risk_manager.calculate_position_size(
                symbol, current_price, stop_loss_pct, account_data['free_balance']
            )
            
            if position_size <= 0:
                logger.logger.warning(f"Invalid position size for {symbol}: {position_size}")
                return
            
            # Place the order
            side = signal['signal'].lower()  # 'buy' or 'sell'
            
            order = self.exchange.place_market_order(
                symbol, side, position_size,
                stop_loss=stop_loss_pct,
                take_profit=signal.get('take_profit_pct')
            )
            
            if order:
                # Update strategy tracking
                for strategy in self.strategies.values():
                    if hasattr(strategy, 'on_position_opened'):
                        strategy.on_position_opened(
                            symbol, side.upper(), current_price, position_size
                        )
                
                # Update risk manager
                self.risk_manager.add_position(
                    symbol, side.upper(), position_size, current_price, stop_loss_pct
                )
                
                logger.trade(
                    f"Trade executed: {side.upper()} {position_size:.6f} {symbol} at {current_price}",
                    {
                        'signal': signal,
                        'order_id': order.get('id'),
                        'account_balance': account_data['free_balance']
                    }
                )
                
                # Send notification
                self.send_trade_notification(symbol, side, position_size, current_price, signal)
        
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}", e)
    
    def close_all_positions(self):
        """Close all open positions"""
        try:
            logger.logger.info("Closing all open positions...")
            
            positions = self.exchange.get_open_positions()
            for position in positions:
                symbol = position['symbol']
                success = self.exchange.close_position(symbol)
                
                if success:
                    # Update strategy tracking
                    for strategy in self.strategies.values():
                        if hasattr(strategy, 'on_position_closed'):
                            strategy.on_position_closed(symbol, 0, "Emergency close")
                    
                    # Update risk manager
                    self.risk_manager.remove_position(symbol)
            
            logger.logger.info("All positions closed")
            
        except Exception as e:
            logger.error("Error closing positions", e)
    
    def send_trade_notification(self, symbol: str, side: str, amount: float, 
                               price: float, signal: Dict):
        """Send trade notification (placeholder for Telegram integration)"""
        message = (
            f"🚀 Trade Executed\n"
            f"Symbol: {symbol}\n"
            f"Side: {side.upper()}\n"
            f"Amount: {amount:.6f}\n"
            f"Price: ${price:.2f}\n"
            f"Confidence: {signal['confidence']:.1%}\n"
            f"Reason: {signal['reason']}"
        )
        
        # TODO: Implement Telegram notification
        logger.logger.info(f"Trade notification: {message}")
    
    def log_performance(self):
        """Log current performance metrics"""
        try:
            account_data = self.exchange.get_account_balance()
            if not account_data:
                return
            
            risk_summary = self.risk_manager.get_risk_summary(account_data['total_balance'])
            
            logger.performance(
                f"Performance Update - Balance: ${account_data['total_balance']:.2f}, "
                f"Drawdown: {risk_summary['current_drawdown']:.2%}, "
                f"Active Positions: {risk_summary['active_positions']}",
                risk_summary
            )
            
        except Exception as e:
            logger.error("Error logging performance", e)
    
    def analyze_news_sentiment(self):
        """Periodic news sentiment analysis"""
        if not self.news_analyzer:
            return
        
        try:
            # Get market-wide sentiment
            fear_greed_index = self.news_analyzer.get_market_fear_greed_index()
            
            logger.news(
                f"Market Fear & Greed Index: {fear_greed_index:.1f}",
                {'fear_greed_index': fear_greed_index}
            )
            
        except Exception as e:
            logger.error("Error in periodic news analysis", e)
    
    def daily_report(self):
        """Generate daily performance report"""
        try:
            account_data = self.exchange.get_account_balance()
            if not account_data:
                return
            
            performance_metrics = self.risk_manager.calculate_performance_metrics()
            
            report = (
                f"📊 Daily Report - {datetime.now().strftime('%Y-%m-%d')}\n"
                f"Account Balance: ${account_data['total_balance']:.2f}\n"
                f"Total Trades: {performance_metrics.get('total_trades', 0)}\n"
                f"Win Rate: {performance_metrics.get('win_rate', 0):.1%}\n"
                f"Total P&L: ${performance_metrics.get('total_pnl', 0):.2f}\n"
                f"Sharpe Ratio: {performance_metrics.get('sharpe_ratio', 0):.2f}\n"
                f"Max Drawdown: {performance_metrics.get('current_drawdown', 0):.2%}"
            )
            
            logger.performance("Daily Report Generated", performance_metrics)
            
            # TODO: Send to Telegram and Google Sheets
            
        except Exception as e:
            logger.error("Error generating daily report", e)

def main():
    """Main entry point"""
    try:
        # Create and start the trading bot
        bot = CryptoTradingBot()
        bot.start()
        
    except KeyboardInterrupt:
        logger.logger.info("Bot stopped by user")
    except Exception as e:
        logger.error("Fatal error in main", e)
        sys.exit(1)

if __name__ == "__main__":
    main()