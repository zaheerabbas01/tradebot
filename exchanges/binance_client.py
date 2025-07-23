import ccxt
import asyncio
import websocket
import json
import threading
import time
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from config import Config
from utils.logger import logger

class BinanceFuturesClient:
    """Binance Futures client with WebSocket support for real-time data"""
    
    def __init__(self):
        self.exchange = None
        self.ws_connection = None
        self.price_data = {}
        self.order_book = {}
        self.callbacks = {}
        self.running = False
        self.setup_exchange()
    
    def setup_exchange(self):
        """Initialize Binance Futures exchange connection"""
        try:
            sandbox_mode = Config.BINANCE_TESTNET
            
            self.exchange = ccxt.binance({
                'apiKey': Config.BINANCE_API_KEY,
                'secret': Config.BINANCE_SECRET_KEY,
                'sandbox': sandbox_mode,
                'options': {
                    'defaultType': 'future',  # Use futures
                    'adjustForTimeDifference': True,
                },
                'enableRateLimit': True,
            })
            
            # Test connection
            balance = self.exchange.fetch_balance()
            logger.logger.info(f"Binance Futures connection established. Testnet: {sandbox_mode}")
            
            # Set leverage for trading pairs
            self.setup_leverage()
            
        except Exception as e:
            logger.error(f"Failed to setup Binance exchange", e)
            raise
    
    def setup_leverage(self):
        """Set leverage for trading pairs"""
        try:
            for symbol in Config.TRADING_PAIRS:
                self.exchange.set_leverage(Config.LEVERAGE, symbol)
                logger.logger.info(f"Set leverage {Config.LEVERAGE}x for {symbol}")
        except Exception as e:
            logger.error(f"Failed to set leverage", e)
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '15m', limit: int = 1000) -> pd.DataFrame:
        """Fetch OHLCV data for technical analysis"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"Failed to fetch OHLCV data for {symbol}", e)
            return pd.DataFrame()
    
    def get_account_balance(self) -> Dict:
        """Get account balance and margin information"""
        try:
            balance = self.exchange.fetch_balance()
            return {
                'total_balance': balance['USDT']['total'],
                'free_balance': balance['USDT']['free'],
                'used_balance': balance['USDT']['used'],
                'margin_level': self.calculate_margin_level()
            }
        except Exception as e:
            logger.error("Failed to fetch account balance", e)
            return {}
    
    def calculate_margin_level(self) -> float:
        """Calculate current margin level"""
        try:
            account = self.exchange.fapiPrivateV2GetAccount()
            total_wallet_balance = float(account['totalWalletBalance'])
            total_unrealized_pnl = float(account['totalUnrealizedPnL'])
            total_margin_balance = total_wallet_balance + total_unrealized_pnl
            total_maintenance_margin = float(account['totalMaintMargin'])
            
            if total_maintenance_margin == 0:
                return float('inf')
            
            margin_level = total_margin_balance / total_maintenance_margin
            return margin_level
        except Exception as e:
            logger.error("Failed to calculate margin level", e)
            return 0.0
    
    def place_market_order(self, symbol: str, side: str, amount: float, 
                          stop_loss: Optional[float] = None, 
                          take_profit: Optional[float] = None) -> Dict:
        """Place a market order with optional stop loss and take profit"""
        try:
            # Calculate position size based on risk management
            balance = self.get_account_balance()
            risk_amount = balance['free_balance'] * Config.MAX_RISK_PER_TRADE
            
            # Adjust amount based on risk
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            max_amount = (risk_amount * Config.LEVERAGE) / current_price
            
            final_amount = min(amount, max_amount)
            
            # Place main order
            order = self.exchange.create_market_order(symbol, side, final_amount)
            
            logger.trade(f"Market order placed", {
                'symbol': symbol,
                'side': side,
                'amount': final_amount,
                'price': current_price,
                'order_id': order['id']
            })
            
            # Place stop loss and take profit orders if specified
            if stop_loss or take_profit:
                self.place_stop_orders(symbol, side, final_amount, current_price, stop_loss, take_profit)
            
            return order
            
        except Exception as e:
            logger.error(f"Failed to place market order for {symbol}", e)
            return {}
    
    def place_stop_orders(self, symbol: str, side: str, amount: float, entry_price: float,
                         stop_loss: Optional[float] = None, take_profit: Optional[float] = None):
        """Place stop loss and take profit orders"""
        try:
            opposite_side = 'sell' if side == 'buy' else 'buy'
            
            if stop_loss:
                if side == 'buy':
                    stop_price = entry_price * (1 - stop_loss)
                else:
                    stop_price = entry_price * (1 + stop_loss)
                
                stop_order = self.exchange.create_order(
                    symbol, 'stop_market', opposite_side, amount,
                    None, None, {'stopPrice': stop_price}
                )
                logger.trade(f"Stop loss order placed at {stop_price}")
            
            if take_profit:
                if side == 'buy':
                    tp_price = entry_price * (1 + take_profit)
                else:
                    tp_price = entry_price * (1 - take_profit)
                
                tp_order = self.exchange.create_limit_order(
                    symbol, opposite_side, amount, tp_price
                )
                logger.trade(f"Take profit order placed at {tp_price}")
                
        except Exception as e:
            logger.error("Failed to place stop orders", e)
    
    def get_open_positions(self) -> List[Dict]:
        """Get all open positions"""
        try:
            positions = self.exchange.fetch_positions()
            open_positions = [pos for pos in positions if float(pos['contracts']) != 0]
            return open_positions
        except Exception as e:
            logger.error("Failed to fetch open positions", e)
            return []
    
    def close_position(self, symbol: str) -> bool:
        """Close all positions for a symbol"""
        try:
            positions = self.get_open_positions()
            for position in positions:
                if position['symbol'] == symbol:
                    side = 'sell' if position['side'] == 'long' else 'buy'
                    amount = abs(float(position['contracts']))
                    
                    order = self.exchange.create_market_order(symbol, side, amount)
                    logger.trade(f"Position closed for {symbol}", {'order_id': order['id']})
                    return True
            return False
        except Exception as e:
            logger.error(f"Failed to close position for {symbol}", e)
            return False
    
    def start_websocket(self, symbols: List[str], callback: Callable):
        """Start WebSocket connection for real-time price data"""
        def on_message(ws, message):
            try:
                data = json.loads(message)
                if 'stream' in data:
                    stream_data = data['data']
                    symbol = stream_data['s']
                    price = float(stream_data['c'])
                    volume = float(stream_data['v'])
                    
                    self.price_data[symbol] = {
                        'price': price,
                        'volume': volume,
                        'timestamp': datetime.now()
                    }
                    
                    callback(symbol, price, volume)
            except Exception as e:
                logger.error("WebSocket message processing error", e)
        
        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            logger.logger.info("WebSocket connection closed")
            if self.running:
                # Reconnect after 5 seconds
                time.sleep(5)
                self.start_websocket(symbols, callback)
        
        def on_open(ws):
            logger.logger.info("WebSocket connection opened")
            # Subscribe to price streams
            streams = [f"{symbol.lower().replace('/', '')}@ticker" for symbol in symbols]
            subscribe_msg = {
                "method": "SUBSCRIBE",
                "params": streams,
                "id": 1
            }
            ws.send(json.dumps(subscribe_msg))
        
        # Start WebSocket in a separate thread
        def run_websocket():
            websocket_url = "wss://fstream.binance.com/ws/"
            if Config.BINANCE_TESTNET:
                websocket_url = "wss://stream.binancefuture.com/ws/"
            
            self.ws_connection = websocket.WebSocketApp(
                websocket_url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            
            self.ws_connection.run_forever()
        
        self.running = True
        ws_thread = threading.Thread(target=run_websocket)
        ws_thread.daemon = True
        ws_thread.start()
    
    def stop_websocket(self):
        """Stop WebSocket connection"""
        self.running = False
        if self.ws_connection:
            self.ws_connection.close()
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price from WebSocket data or API fallback"""
        if symbol in self.price_data:
            price_info = self.price_data[symbol]
            # Check if data is recent (less than 30 seconds old)
            if (datetime.now() - price_info['timestamp']).seconds < 30:
                return price_info['price']
        
        # Fallback to API
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            logger.error(f"Failed to get current price for {symbol}", e)
            return 0.0