import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime
from config import Config
from analysis.technical_indicators import TechnicalIndicators
from utils.logger import logger

class MeanReversionStrategy:
    """
    Mean reversion strategy using Bollinger Bands and volume analysis
    
    Entry Rules:
    - Buy: Price touches lower Bollinger Band (2 std dev) AND volume spikes
    - Sell: Price touches upper Bollinger Band (2 std dev)
    
    Exit Rules:
    - Stop Loss: 1.5%
    - Take Profit: 3% fixed
    
    Based on:
    - "Bollinger on Bollinger Bands" by John Bollinger
    - "Mean Reversion Trading Systems" by Howard Bandy
    - "Quantitative Trading" by Ernest Chan
    """
    
    def __init__(self):
        self.name = "MeanReversion"
        self.active_positions = {}
        self.entry_prices = {}
        
    def analyze(self, data: pd.DataFrame, symbol: str) -> Dict:
        """
        Analyze market data and generate mean reversion signals
        
        Args:
            data: OHLCV DataFrame
            symbol: Trading pair symbol
            
        Returns:
            Dictionary with signal information
        """
        if len(data) < Config.BOLLINGER_PERIOD * 2:
            return {'signal': 'HOLD', 'confidence': 0, 'reason': 'Insufficient data'}
        
        try:
            # Calculate technical indicators
            close = data['close']
            high = data['high']
            low = data['low']
            volume = data['volume']
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = TechnicalIndicators.calculate_bollinger_bands(
                close, Config.BOLLINGER_PERIOD, Config.BOLLINGER_STD
            )
            
            # Volume analysis
            volume_sma = TechnicalIndicators.calculate_volume_sma(volume, 20)
            volume_spike = TechnicalIndicators.detect_volume_spike(
                volume, volume_sma, Config.VOLUME_SPIKE_THRESHOLD
            )
            
            # RSI for momentum confirmation
            rsi = TechnicalIndicators.calculate_rsi(close, Config.RSI_PERIOD)
            
            # ATR for volatility assessment
            atr = TechnicalIndicators.calculate_atr(high, low, close)
            
            # Get latest values
            current_price = close.iloc[-1]
            current_bb_upper = bb_upper.iloc[-1]
            current_bb_middle = bb_middle.iloc[-1]
            current_bb_lower = bb_lower.iloc[-1]
            current_volume = volume.iloc[-1]
            avg_volume = volume_sma.iloc[-1]
            current_volume_spike = volume_spike.iloc[-1]
            current_rsi = rsi.iloc[-1]
            current_atr = atr.iloc[-1]
            
            # Calculate Bollinger Band position
            bb_position = (current_price - current_bb_lower) / (current_bb_upper - current_bb_lower)
            
            # Calculate band width (volatility measure)
            band_width = (current_bb_upper - current_bb_lower) / current_bb_middle
            
            # Generate signals
            signal_data = self._generate_signal(
                current_price, current_bb_upper, current_bb_middle, current_bb_lower,
                bb_position, band_width, current_volume_spike, current_rsi,
                current_volume, avg_volume, symbol
            )
            
            # Add technical data for logging
            signal_data.update({
                'bb_upper': current_bb_upper,
                'bb_middle': current_bb_middle,
                'bb_lower': current_bb_lower,
                'bb_position': bb_position,
                'band_width': band_width,
                'volume_spike': current_volume_spike,
                'volume_ratio': current_volume / avg_volume,
                'rsi': current_rsi,
                'atr': current_atr
            })
            
            logger.strategy(
                f"Mean reversion analysis for {symbol}: {signal_data['signal']} "
                f"(Confidence: {signal_data['confidence']:.2f})",
                self.name
            )
            
            return signal_data
            
        except Exception as e:
            logger.error(f"Error in mean reversion analysis for {symbol}", e)
            return {'signal': 'HOLD', 'confidence': 0, 'reason': f'Analysis error: {str(e)}'}
    
    def _generate_signal(self, price: float, bb_upper: float, bb_middle: float, bb_lower: float,
                        bb_position: float, band_width: float, volume_spike: bool, rsi: float,
                        volume: float, avg_volume: float, symbol: str) -> Dict:
        """Generate trading signal based on mean reversion rules"""
        
        signal = 'HOLD'
        confidence = 0
        reason = []
        
        # Check for oversold conditions (buy signal)
        if price <= bb_lower * 1.01:  # Price at or near lower band (1% tolerance)
            if volume_spike:  # Volume confirmation
                signal = 'BUY'
                confidence += 50  # Base confidence for BB touch + volume
                reason.append("Price at lower Bollinger Band with volume spike")
                
                # Additional confirmations
                if rsi < 40:  # Oversold RSI
                    confidence += 20
                    reason.append("RSI oversold")
                
                if bb_position < 0.1:  # Very close to lower band
                    confidence += 15
                    reason.append("Very close to lower band")
                
                if band_width > 0.04:  # High volatility (good for mean reversion)
                    confidence += 10
                    reason.append("High volatility")
                
            else:
                reason.append("Price at lower band but no volume confirmation")
        
        # Check for overbought conditions (sell signal)
        elif price >= bb_upper * 0.99:  # Price at or near upper band (1% tolerance)
            signal = 'SELL'
            confidence += 40  # Base confidence for BB touch
            reason.append("Price at upper Bollinger Band")
            
            # Additional confirmations
            if rsi > 60:  # Overbought RSI
                confidence += 20
                reason.append("RSI overbought")
            
            if bb_position > 0.9:  # Very close to upper band
                confidence += 15
                reason.append("Very close to upper band")
            
            if volume > avg_volume * 1.2:  # Volume confirmation
                confidence += 15
                reason.append("Volume confirmation")
            
            if band_width > 0.04:  # High volatility
                confidence += 10
                reason.append("High volatility")
        
        # Check for mean reversion back to middle band
        elif signal == 'HOLD' and symbol in self.active_positions:
            position = self.active_positions[symbol]
            entry_price = self.entry_prices[symbol]
            
            # Check for profit taking at middle band
            if position['side'] == 'BUY' and price >= bb_middle * 0.99:
                pnl_pct = (price - entry_price) / entry_price
                if pnl_pct > 0.01:  # At least 1% profit
                    signal = 'SELL'
                    confidence = 0.8
                    reason.append(f"Mean reversion to middle band, profit: {pnl_pct:.2%}")
            
            elif position['side'] == 'SELL' and price <= bb_middle * 1.01:
                pnl_pct = (entry_price - price) / entry_price
                if pnl_pct > 0.01:  # At least 1% profit
                    signal = 'BUY'
                    confidence = 0.8
                    reason.append(f"Mean reversion to middle band, profit: {pnl_pct:.2%}")
        
        # Check for position management
        if symbol in self.active_positions:
            position_signal = self._check_position_management(symbol, price)
            if position_signal:
                return position_signal
        
        # Normalize confidence to 0-100 range
        confidence = min(confidence, 100)
        
        return {
            'signal': signal,
            'confidence': confidence / 100,  # Normalize to 0-1
            'reason': '; '.join(reason) if reason else 'No clear signal',
            'stop_loss_pct': Config.MEAN_REVERSION_STOP_LOSS,
            'take_profit_pct': Config.MEAN_REVERSION_TAKE_PROFIT
        }
    
    def _check_position_management(self, symbol: str, current_price: float) -> Optional[Dict]:
        """Check if we need to exit based on stop loss or take profit"""
        
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
        if pnl_pct <= -Config.MEAN_REVERSION_STOP_LOSS:
            logger.risk(f"Stop loss triggered for {symbol}: {pnl_pct:.2%} loss")
            return {
                'signal': 'SELL' if side == 'BUY' else 'BUY',
                'confidence': 1.0,
                'reason': f'Stop loss triggered: {pnl_pct:.2%} loss',
                'exit_type': 'stop_loss'
            }
        
        # Check take profit
        if pnl_pct >= Config.MEAN_REVERSION_TAKE_PROFIT:
            logger.trade(f"Take profit hit for {symbol}: {pnl_pct:.2%} profit")
            return {
                'signal': 'SELL' if side == 'BUY' else 'BUY',
                'confidence': 1.0,
                'reason': f'Take profit hit: {pnl_pct:.2%} profit',
                'exit_type': 'take_profit'
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
        
        logger.trade(f"Mean reversion position opened: {side} {amount} {symbol} at {entry_price}")
    
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
                f"Mean reversion position closed: {symbol} at {exit_price}, "
                f"P&L: {pnl_pct:.2%}, Reason: {reason}"
            )
            
            # Clean up tracking
            del self.active_positions[symbol]
            del self.entry_prices[symbol]
    
    def get_position_size(self, symbol: str, current_price: float, account_balance: float) -> float:
        """
        Calculate position size based on risk management
        Smaller position sizes for mean reversion due to higher frequency
        """
        risk_amount = account_balance * Config.MAX_RISK_PER_TRADE * 0.7  # 70% of normal risk
        stop_loss_amount = current_price * Config.MEAN_REVERSION_STOP_LOSS
        
        # Calculate base position size
        position_size = risk_amount / stop_loss_amount
        
        # Apply leverage
        leveraged_size = position_size * Config.LEVERAGE
        
        return leveraged_size
    
    def calculate_bollinger_squeeze(self, data: pd.DataFrame) -> bool:
        """
        Detect Bollinger Band squeeze (low volatility period)
        Good setup for mean reversion trades
        """
        close = data['close']
        
        # Calculate Bollinger Bands
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.calculate_bollinger_bands(
            close, Config.BOLLINGER_PERIOD, Config.BOLLINGER_STD
        )
        
        # Calculate band width
        band_width = (bb_upper - bb_lower) / bb_middle
        
        # Check if current band width is in lowest 20% of recent period
        recent_band_widths = band_width.tail(50)  # Last 50 periods
        current_width = band_width.iloc[-1]
        
        squeeze_threshold = recent_band_widths.quantile(0.2)
        
        return current_width <= squeeze_threshold
    
    def get_strategy_info(self) -> Dict:
        """Get strategy information and current state"""
        return {
            'name': self.name,
            'active_positions': len(self.active_positions),
            'positions': list(self.active_positions.keys()),
            'parameters': {
                'bollinger_period': Config.BOLLINGER_PERIOD,
                'bollinger_std': Config.BOLLINGER_STD,
                'volume_spike_threshold': Config.VOLUME_SPIKE_THRESHOLD,
                'stop_loss': Config.MEAN_REVERSION_STOP_LOSS,
                'take_profit': Config.MEAN_REVERSION_TAKE_PROFIT
            }
        }