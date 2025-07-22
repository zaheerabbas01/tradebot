import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from config import Config
from utils.logger import logger

class RiskManager:
    """
    Comprehensive risk management system for crypto trading
    
    Features:
    - Position sizing based on Kelly Criterion
    - Portfolio heat monitoring
    - Drawdown protection
    - Correlation analysis
    - Auto-liquidation protection
    - SEC investigation blacklist
    
    Based on:
    - "The Mathematics of Money Management" by Ralph Vince
    - "Risk Management and Financial Institutions" by John Hull
    - "Quantitative Risk Management" by Alexander McNeil
    """
    
    def __init__(self):
        self.max_risk_per_trade = Config.MAX_RISK_PER_TRADE
        self.liquidation_threshold = Config.LIQUIDATION_THRESHOLD
        self.max_portfolio_heat = 0.05  # 5% max total portfolio risk
        self.max_drawdown = 0.20  # 20% max drawdown before shutdown
        self.correlation_threshold = 0.7  # Max correlation between positions
        
        # Track trading statistics
        self.trade_history = []
        self.daily_pnl = []
        self.peak_equity = Config.INITIAL_CAPITAL
        self.current_drawdown = 0.0
        
        # Blacklisted assets (SEC investigation, etc.)
        self.blacklisted_assets = set()
        
        # Position tracking
        self.active_positions = {}
        self.position_correlations = {}
        
    def calculate_position_size(self, symbol: str, entry_price: float, 
                              stop_loss_pct: float, account_balance: float,
                              win_rate: float = 0.55, avg_win_loss_ratio: float = 1.5) -> float:
        """
        Calculate optimal position size using Kelly Criterion with risk adjustments
        
        Args:
            symbol: Trading pair symbol
            entry_price: Entry price for the position
            stop_loss_pct: Stop loss percentage
            account_balance: Current account balance
            win_rate: Historical win rate (default 55%)
            avg_win_loss_ratio: Average win/loss ratio
            
        Returns:
            Position size in base currency
        """
        try:
            # Kelly Criterion calculation
            # f = (bp - q) / b
            # where: f = fraction of capital to wager
            #        b = odds received (win/loss ratio)
            #        p = probability of winning
            #        q = probability of losing (1-p)
            
            p = win_rate
            q = 1 - p
            b = avg_win_loss_ratio
            
            kelly_fraction = (b * p - q) / b
            
            # Apply Kelly fraction with safety margin (25% of Kelly)
            safe_kelly_fraction = kelly_fraction * 0.25
            
            # Calculate risk-based position size
            risk_amount = account_balance * self.max_risk_per_trade
            stop_loss_amount = entry_price * stop_loss_pct
            
            # Position size based on risk
            risk_based_size = risk_amount / stop_loss_amount
            
            # Position size based on Kelly
            kelly_based_size = (account_balance * safe_kelly_fraction) / entry_price
            
            # Use the more conservative of the two
            base_position_size = min(risk_based_size, kelly_based_size)
            
            # Apply leverage
            leveraged_size = base_position_size * Config.LEVERAGE
            
            # Apply additional risk adjustments
            adjusted_size = self._apply_risk_adjustments(
                symbol, leveraged_size, account_balance
            )
            
            logger.risk(
                f"Position size calculated for {symbol}: {adjusted_size:.6f} "
                f"(Kelly: {kelly_based_size:.6f}, Risk: {risk_based_size:.6f})"
            )
            
            return adjusted_size
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}", e)
            # Fallback to basic risk calculation
            risk_amount = account_balance * self.max_risk_per_trade
            return (risk_amount / (entry_price * stop_loss_pct)) * Config.LEVERAGE
    
    def _apply_risk_adjustments(self, symbol: str, base_size: float, 
                               account_balance: float) -> float:
        """Apply additional risk adjustments to position size"""
        
        adjusted_size = base_size
        
        # Drawdown adjustment
        if self.current_drawdown > 0.10:  # 10% drawdown
            drawdown_factor = 1 - (self.current_drawdown * 2)  # Reduce size proportionally
            adjusted_size *= max(drawdown_factor, 0.1)  # Minimum 10% of original size
            logger.risk(f"Drawdown adjustment applied: {drawdown_factor:.2f}")
        
        # Portfolio heat adjustment
        current_heat = self.calculate_portfolio_heat(account_balance)
        if current_heat > self.max_portfolio_heat * 0.8:  # 80% of max heat
            heat_factor = (self.max_portfolio_heat - current_heat) / (self.max_portfolio_heat * 0.2)
            adjusted_size *= max(heat_factor, 0.1)
            logger.risk(f"Portfolio heat adjustment applied: {heat_factor:.2f}")
        
        # Correlation adjustment
        correlation_factor = self._calculate_correlation_adjustment(symbol)
        adjusted_size *= correlation_factor
        
        # Volatility adjustment
        volatility_factor = self._calculate_volatility_adjustment(symbol)
        adjusted_size *= volatility_factor
        
        return adjusted_size
    
    def calculate_portfolio_heat(self, account_balance: float) -> float:
        """
        Calculate total portfolio heat (risk exposure)
        
        Returns:
            Portfolio heat as percentage of account balance
        """
        total_risk = 0
        
        for symbol, position in self.active_positions.items():
            position_value = position['amount'] * position['current_price']
            stop_loss_risk = position_value * position['stop_loss_pct']
            total_risk += stop_loss_risk
        
        portfolio_heat = total_risk / account_balance
        return portfolio_heat
    
    def _calculate_correlation_adjustment(self, symbol: str) -> float:
        """Calculate position size adjustment based on correlation with existing positions"""
        
        if not self.active_positions:
            return 1.0
        
        # Simplified correlation check based on asset class
        base_asset = symbol.split('/')[0]
        
        # Check if we already have highly correlated positions
        correlated_exposure = 0
        for existing_symbol in self.active_positions.keys():
            existing_base = existing_symbol.split('/')[0]
            
            # Simple correlation rules
            if base_asset == existing_base:
                correlated_exposure += 1.0  # Same asset = 100% correlation
            elif base_asset in ['BTC', 'ETH'] and existing_base in ['BTC', 'ETH']:
                correlated_exposure += 0.8  # BTC/ETH high correlation
            elif base_asset in ['BNB', 'SOL', 'ADA'] and existing_base in ['BNB', 'SOL', 'ADA']:
                correlated_exposure += 0.6  # Altcoins moderate correlation
        
        # Reduce position size based on correlation
        if correlated_exposure > 1.5:
            correlation_factor = 1 / (1 + correlated_exposure * 0.3)
        else:
            correlation_factor = 1.0
        
        return max(correlation_factor, 0.2)  # Minimum 20% of original size
    
    def _calculate_volatility_adjustment(self, symbol: str) -> float:
        """Calculate position size adjustment based on asset volatility"""
        
        # Volatility-based adjustments (simplified)
        volatility_adjustments = {
            'BTC': 1.0,    # Base volatility
            'ETH': 1.1,    # Slightly more volatile
            'SOL': 1.3,    # Higher volatility
            'BNB': 1.2,    # Moderate volatility
        }
        
        base_asset = symbol.split('/')[0]
        volatility_multiplier = volatility_adjustments.get(base_asset, 1.5)  # Default high volatility
        
        # Inverse relationship: higher volatility = smaller position
        volatility_factor = 1 / volatility_multiplier
        
        return volatility_factor
    
    def check_margin_level(self, account_data: Dict) -> Dict:
        """
        Check margin level and return risk assessment
        
        Args:
            account_data: Account balance and margin information
            
        Returns:
            Risk assessment dictionary
        """
        margin_level = account_data.get('margin_level', float('inf'))
        
        risk_status = {
            'status': 'SAFE',
            'margin_level': margin_level,
            'action_required': False,
            'message': 'Margin level is healthy'
        }
        
        if margin_level < 1.5:  # Critical level
            risk_status.update({
                'status': 'CRITICAL',
                'action_required': True,
                'message': 'Critical margin level - close positions immediately'
            })
            logger.risk(f"CRITICAL: Margin level at {margin_level:.2f}")
            
        elif margin_level < 2.0:  # Warning level
            risk_status.update({
                'status': 'WARNING',
                'action_required': True,
                'message': 'Low margin level - reduce position sizes'
            })
            logger.risk(f"WARNING: Margin level at {margin_level:.2f}")
            
        elif margin_level < 3.0:  # Caution level
            risk_status.update({
                'status': 'CAUTION',
                'action_required': False,
                'message': 'Margin level requires monitoring'
            })
        
        return risk_status
    
    def update_drawdown(self, current_equity: float):
        """Update drawdown tracking"""
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
        
        # Check if maximum drawdown exceeded
        if self.current_drawdown > self.max_drawdown:
            logger.risk(f"CRITICAL: Maximum drawdown exceeded: {self.current_drawdown:.2%}")
            return True  # Signal to stop trading
        
        return False
    
    def should_trade(self, symbol: str, signal_confidence: float, 
                    account_balance: float) -> Tuple[bool, str]:
        """
        Determine if we should execute a trade based on risk parameters
        
        Args:
            symbol: Trading pair symbol
            signal_confidence: Strategy confidence (0-1)
            account_balance: Current account balance
            
        Returns:
            Tuple of (should_trade, reason)
        """
        
        # Check if asset is blacklisted
        base_asset = symbol.split('/')[0]
        if base_asset in self.blacklisted_assets:
            return False, f"{base_asset} is blacklisted"
        
        # Check portfolio heat
        current_heat = self.calculate_portfolio_heat(account_balance)
        if current_heat > self.max_portfolio_heat:
            return False, f"Portfolio heat too high: {current_heat:.2%}"
        
        # Check drawdown
        if self.current_drawdown > self.max_drawdown * 0.8:  # 80% of max drawdown
            return False, f"Approaching max drawdown: {self.current_drawdown:.2%}"
        
        # Check signal confidence
        min_confidence = 0.6 + (current_heat * 2)  # Higher confidence required with higher heat
        if signal_confidence < min_confidence:
            return False, f"Signal confidence too low: {signal_confidence:.2f} < {min_confidence:.2f}"
        
        # Check maximum positions
        if len(self.active_positions) >= 3:  # Max 3 positions per config
            return False, "Maximum number of positions reached"
        
        return True, "Risk checks passed"
    
    def add_position(self, symbol: str, side: str, amount: float, 
                    entry_price: float, stop_loss_pct: float):
        """Add a position to risk tracking"""
        self.active_positions[symbol] = {
            'side': side,
            'amount': amount,
            'entry_price': entry_price,
            'current_price': entry_price,
            'stop_loss_pct': stop_loss_pct,
            'timestamp': datetime.now()
        }
        
        logger.risk(f"Position added to risk tracking: {symbol}")
    
    def remove_position(self, symbol: str):
        """Remove a position from risk tracking"""
        if symbol in self.active_positions:
            del self.active_positions[symbol]
            logger.risk(f"Position removed from risk tracking: {symbol}")
    
    def update_position_price(self, symbol: str, current_price: float):
        """Update current price for position tracking"""
        if symbol in self.active_positions:
            self.active_positions[symbol]['current_price'] = current_price
    
    def add_to_blacklist(self, asset: str, reason: str = ""):
        """Add asset to blacklist"""
        self.blacklisted_assets.add(asset)
        logger.risk(f"Asset {asset} added to blacklist: {reason}")
    
    def remove_from_blacklist(self, asset: str):
        """Remove asset from blacklist"""
        if asset in self.blacklisted_assets:
            self.blacklisted_assets.remove(asset)
            logger.risk(f"Asset {asset} removed from blacklist")
    
    def record_trade(self, symbol: str, side: str, entry_price: float, 
                    exit_price: float, amount: float, pnl: float):
        """Record completed trade for statistics"""
        trade_record = {
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'amount': amount,
            'pnl': pnl,
            'pnl_pct': pnl / (entry_price * amount),
            'timestamp': datetime.now()
        }
        
        self.trade_history.append(trade_record)
        
        # Update daily P&L
        today = datetime.now().date()
        if not self.daily_pnl or self.daily_pnl[-1]['date'] != today:
            self.daily_pnl.append({'date': today, 'pnl': pnl})
        else:
            self.daily_pnl[-1]['pnl'] += pnl
    
    def calculate_performance_metrics(self) -> Dict:
        """Calculate trading performance metrics"""
        if not self.trade_history:
            return {}
        
        trades_df = pd.DataFrame(self.trade_history)
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = trades_df['pnl'].sum()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if losing_trades > 0 else 0
        
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        # Risk metrics
        returns = trades_df['pnl_pct'].values
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        max_consecutive_losses = self._calculate_max_consecutive_losses(trades_df)
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio,
            'sharpe_ratio': sharpe_ratio,
            'max_consecutive_losses': max_consecutive_losses,
            'current_drawdown': self.current_drawdown,
            'peak_equity': self.peak_equity
        }
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
    
    def _calculate_max_consecutive_losses(self, trades_df: pd.DataFrame) -> int:
        """Calculate maximum consecutive losses"""
        consecutive_losses = 0
        max_consecutive_losses = 0
        
        for _, trade in trades_df.iterrows():
            if trade['pnl'] < 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0
        
        return max_consecutive_losses
    
    def get_risk_summary(self, account_balance: float) -> Dict:
        """Get comprehensive risk summary"""
        portfolio_heat = self.calculate_portfolio_heat(account_balance)
        performance_metrics = self.calculate_performance_metrics()
        
        return {
            'portfolio_heat': portfolio_heat,
            'max_portfolio_heat': self.max_portfolio_heat,
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'active_positions': len(self.active_positions),
            'blacklisted_assets': list(self.blacklisted_assets),
            'performance_metrics': performance_metrics,
            'risk_status': 'HEALTHY' if portfolio_heat < self.max_portfolio_heat and 
                                      self.current_drawdown < self.max_drawdown else 'AT_RISK'
        }