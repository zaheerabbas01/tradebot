import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime
from config import Config
from analysis.technical_indicators import TechnicalIndicators, TrendAnalysis
from utils.logger import logger

class TrendFollowingStrategy:
    """
    Trend-following strategy based on EMA crossover with RSI filter
    
    Entry Rules:
    - Buy: 50 EMA crosses above 200 EMA AND RSI < 70 (avoid overbought)
    - Sell: 50 EMA crosses below 200 EMA AND RSI > 30 (avoid oversold)
    
    Exit Rules:
    - Stop Loss: 2%
    - Take Profit: 5% trailing
    
    Based on:
    - "Trend Following" by Michael Covel
    - "The Complete TurtleTrader" by Michael Covel
    - "Technical Analysis of the Financial Markets" by John Murphy
    """
    
    def __init__(self):
        self.name = "TrendFollowing"
        self.active_positions = {}
        self.entry_prices = {}
        self.trailing_stops = {}
        
    def analyze(self, data: pd.DataFrame, symbol: str) -> Dict:
        """
        Analyze market data and generate trading signals
        
        Args:
            data: OHLCV DataFrame
            symbol: Trading pair symbol
            
        Returns:
            Dictionary with signal information
        """
        if len(data) < Config.EMA_SLOW:
            return {'signal': 'HOLD', 'confidence': 0, 'reason': 'Insufficient data'}
        
        try:
            # Calculate technical indicators
            close = data['close']
            high = data['high']
            low = data['low']
            volume = data['volume']
            
            # EMAs
            ema_fast = TechnicalIndicators.calculate_ema(close, Config.EMA_FAST)
            ema_slow = TechnicalIndicators.calculate_ema(close, Config.EMA_SLOW)
            
            # RSI
            rsi = TechnicalIndicators.calculate_rsi(close, Config.RSI_PERIOD)
            
            # Volume analysis
            volume_sma = TechnicalIndicators.calculate_volume_sma(volume)
            
            # ATR for volatility assessment
            atr = TechnicalIndicators.calculate_atr(high, low, close)
            
            # Detect EMA crossover
            crossover_signals = TrendAnalysis.detect_ema_crossover(ema_fast, ema_slow)
            
            # Get latest values
            current_ema_fast = ema_fast.iloc[-1]
            current_ema_slow = ema_slow.iloc[-1]
            current_rsi = rsi.iloc[-1]
            current_crossover = crossover_signals.iloc[-1]
            current_volume = volume.iloc[-1]
            avg_volume = volume_sma.iloc[-1]
            current_atr = atr.iloc[-1]
            current_price = close.iloc[-1]
            
            # Calculate trend strength
            trend_strength = TrendAnalysis.detect_trend_strength(ema_fast, ema_slow, atr).iloc[-1]
            
            # Generate signals
            signal_data = self._generate_signal(
                current_ema_fast, current_ema_slow, current_rsi, current_crossover,
                current_volume, avg_volume, trend_strength, current_price, symbol
            )
            
            # Add technical data for logging
            signal_data.update({
                'ema_fast': current_ema_fast,
                'ema_slow': current_ema_slow,
                'rsi': current_rsi,
                'volume_ratio': current_volume / avg_volume,
                'trend_strength': trend_strength,
                'atr': current_atr
            })
            
            logger.strategy(
                f"Analysis complete for {symbol}: {signal_data['signal']} "
                f"(Confidence: {signal_data['confidence']:.2f})",
                self.name
            )
            
            return signal_data
            
        except Exception as e:
            logger.error(f"Error in trend following analysis for {symbol}", e)
            return {'signal': 'HOLD', 'confidence': 0, 'reason': f'Analysis error: {str(e)}'}
    
    def _generate_signal(self, ema_fast: float, ema_slow: float, rsi: float, crossover: int,
                        volume: float, avg_volume: float, trend_strength: float, 
                        current_price: float, symbol: str) -> Dict:
        """Generate trading signal based on strategy rules"""
        
        signal = 'HOLD'
        confidence = 0
        reason = []
        
        # Check for bullish signal
        if crossover == 1:  # Bullish EMA crossover
            if rsi < Config.RSI_OVERBOUGHT:
                signal = 'BUY'
                confidence += 40  # Base confidence for crossover
                reason.append("EMA bullish crossover")
                
                # RSI filter bonus
                if rsi < 60:
                    confidence += 20
                    reason.append("RSI not overbought")
                
                # Volume confirmation
                if volume > avg_volume * 1.2:
                    confidence += 15
                    reason.append("Volume confirmation")
                
                # Trend strength bonus
                if trend_strength > 0.02:
                    confidence += 15
                    reason.append("Strong trend")
                
            else:
                reason.append("RSI overbought - signal filtered")
        
        # Check for bearish signal
        elif crossover == -1:  # Bearish EMA crossover
            if rsi > Config.RSI_OVERSOLD:
                signal = 'SELL'
                confidence += 40  # Base confidence for crossover
                reason.append("EMA bearish crossover")
                
                # RSI filter bonus
                if rsi > 40:
                    confidence += 20
                    reason.append("RSI not oversold")
                
                # Volume confirmation
                if volume > avg_volume * 1.2:
                    confidence += 15
                    reason.append("Volume confirmation")
                
                # Trend strength bonus
                if trend_strength > 0.02:
                    confidence += 15
                    reason.append("Strong trend")
                
            else:
                reason.append("RSI oversold - signal filtered")
        
        # Check for position management if we have an active position
        if symbol in self.active_positions:
            position_signal = self._check_position_management(symbol, current_price)
            if position_signal:
                return position_signal
        
        # Normalize confidence to 0-100 range
        confidence = min(confidence, 100)
        
        return {
            'signal': signal,
            'confidence': confidence / 100,  # Normalize to 0-1
            'reason': '; '.join(reason) if reason else 'No clear signal',
            'stop_loss_pct': Config.STOP_LOSS_PERCENT,
            'take_profit_pct': Config.TRAILING_TAKE_PROFIT
        }
    
    def _check_position_management(self, symbol: str, current_price: float) -> Optional[Dict]:
        """Check if we need to exit based on stop loss or trailing take profit"""
        
        if symbol not in self.active_positions:
            return None
        
        position = self.active_positions[symbol]
        entry_price = self.entry_prices[symbol]
        side = position['side']
        
        # Calculate current P&L
        if side == 'BUY':
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price
        
        # Check stop loss
        if pnl_pct <= -Config.STOP_LOSS_PERCENT:
            logger.risk(f"Stop loss triggered for {symbol}: {pnl_pct:.2%} loss")
            return {
                'signal': 'SELL' if side == 'BUY' else 'BUY',
                'confidence': 1.0,
                'reason': f'Stop loss triggered: {pnl_pct:.2%} loss',
                'exit_type': 'stop_loss'
            }
        
        # Update trailing stop
        if symbol not in self.trailing_stops:
            self.trailing_stops[symbol] = entry_price
        
        if side == 'BUY':
            # For long positions, trail stop upward
            if current_price > self.trailing_stops[symbol]:
                new_stop = current_price * (1 - Config.TRAILING_TAKE_PROFIT)
                if new_stop > self.trailing_stops[symbol]:
                    self.trailing_stops[symbol] = new_stop
            
            # Check if trailing stop is hit
            if current_price <= self.trailing_stops[symbol]:
                logger.trade(f"Trailing stop hit for {symbol}: {pnl_pct:.2%} profit")
                return {
                    'signal': 'SELL',
                    'confidence': 1.0,
                    'reason': f'Trailing stop hit: {pnl_pct:.2%} profit',
                    'exit_type': 'trailing_stop'
                }
        
        else:  # Short position
            # For short positions, trail stop downward
            if current_price < self.trailing_stops[symbol]:
                new_stop = current_price * (1 + Config.TRAILING_TAKE_PROFIT)
                if new_stop < self.trailing_stops[symbol]:
                    self.trailing_stops[symbol] = new_stop
            
            # Check if trailing stop is hit
            if current_price >= self.trailing_stops[symbol]:
                logger.trade(f"Trailing stop hit for {symbol}: {pnl_pct:.2%} profit")
                return {
                    'signal': 'BUY',
                    'confidence': 1.0,
                    'reason': f'Trailing stop hit: {pnl_pct:.2%} profit',
                    'exit_type': 'trailing_stop'
                }
        
        return None
    
    def on_position_opened(self, symbol: str, side: str, entry_price: float, amount: float):
        """Callback when a position is opened"""
        self.active_positions[symbol] = {
            'side': side,
            'amount': amount,
            'timestamp': datetime.now()
        }
        self.entry_prices[symbol] = entry_price
        self.trailing_stops[symbol] = entry_price
        
        logger.trade(f"Position opened: {side} {amount} {symbol} at {entry_price}")
    
    def on_position_closed(self, symbol: str, exit_price: float, reason: str = ""):
        """Callback when a position is closed"""
        if symbol in self.active_positions:
            position = self.active_positions[symbol]
            entry_price = self.entry_prices[symbol]
            
            # Calculate P&L
            if position['side'] == 'BUY':
                pnl_pct = (exit_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - exit_price) / entry_price
            
            logger.trade(
                f"Position closed: {symbol} at {exit_price}, "
                f"P&L: {pnl_pct:.2%}, Reason: {reason}"
            )
            
            # Clean up tracking
            del self.active_positions[symbol]
            del self.entry_prices[symbol]
            if symbol in self.trailing_stops:
                del self.trailing_stops[symbol]
    
    def get_position_size(self, symbol: str, current_price: float, account_balance: float) -> float:
        """
        Calculate position size based on risk management
        
        Risk per trade: 1% of account balance
        Position size = (Account Balance * Risk %) / (Stop Loss % * Price)
        """
        risk_amount = account_balance * Config.MAX_RISK_PER_TRADE
        stop_loss_amount = current_price * Config.STOP_LOSS_PERCENT
        
        # Calculate base position size
        position_size = risk_amount / stop_loss_amount
        
        # Apply leverage
        leveraged_size = position_size * Config.LEVERAGE
        
        return leveraged_size
    
    def get_strategy_info(self) -> Dict:
        """Get strategy information and current state"""
        return {
            'name': self.name,
            'active_positions': len(self.active_positions),
            'positions': list(self.active_positions.keys()),
            'parameters': {
                'ema_fast': Config.EMA_FAST,
                'ema_slow': Config.EMA_SLOW,
                'rsi_period': Config.RSI_PERIOD,
                'rsi_overbought': Config.RSI_OVERBOUGHT,
                'rsi_oversold': Config.RSI_OVERSOLD,
                'stop_loss': Config.STOP_LOSS_PERCENT,
                'trailing_tp': Config.TRAILING_TAKE_PROFIT
            }
        }