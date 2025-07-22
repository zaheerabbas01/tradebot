import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from config import Config
from strategies.trend_following import TrendFollowingStrategy
from strategies.mean_reversion import MeanReversionStrategy
from analysis.technical_indicators import TechnicalIndicators
from utils.logger import logger

@dataclass
class BacktestResult:
    """Data class for backtest results"""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_trade_return: float
    volatility: float
    calmar_ratio: float
    sortino_ratio: float
    profit_factor: float
    trades: List[Dict]
    equity_curve: pd.Series

class BacktestEngine:
    """
    Comprehensive backtesting engine for crypto trading strategies
    
    Features:
    - Historical data simulation
    - Multiple strategy testing
    - Parameter optimization
    - Performance metrics calculation
    - Risk analysis
    - Genetic algorithm optimization
    
    Based on:
    - "Quantitative Trading" by Ernest Chan
    - "Algorithmic Trading" by Andreas Clenow
    - "Evidence-Based Technical Analysis" by David Aronson
    """
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.commission = 0.001  # 0.1% commission
        self.slippage = 0.0005   # 0.05% slippage
        
    def load_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load historical data for backtesting
        In production, this would fetch from exchange or data provider
        """
        # For now, generate synthetic data for demonstration
        # In real implementation, use exchange.fetch_ohlcv with historical dates
        
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Generate date range (15-minute intervals)
        date_range = pd.date_range(start=start, end=end, freq='15T')
        
        # Generate synthetic OHLCV data (this would be replaced with real data)
        np.random.seed(42)  # For reproducible results
        
        # Base price (e.g., BTC starting at $30,000)
        base_price = 30000 if 'BTC' in symbol else 2000
        
        # Generate price series with trend and volatility
        returns = np.random.normal(0.0001, 0.02, len(date_range))  # Small upward drift
        price_series = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC from price series
        data = []
        for i, price in enumerate(price_series):
            # Add some randomness for OHLC
            noise = np.random.normal(0, price * 0.005, 4)
            open_price = price + noise[0]
            high_price = price + abs(noise[1])
            low_price = price - abs(noise[2])
            close_price = price + noise[3]
            
            # Ensure OHLC logic
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            # Generate volume
            volume = np.random.normal(1000, 200)
            volume = max(volume, 100)  # Minimum volume
            
            data.append({
                'timestamp': date_range[i],
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def run_backtest(self, strategy, data: pd.DataFrame, symbol: str) -> BacktestResult:
        """
        Run backtest for a single strategy
        
        Args:
            strategy: Trading strategy instance
            data: Historical OHLCV data
            symbol: Trading pair symbol
            
        Returns:
            BacktestResult with performance metrics
        """
        
        # Initialize tracking variables
        capital = self.initial_capital
        position = 0  # Current position size
        entry_price = 0
        trades = []
        equity_curve = []
        
        # Track performance
        peak_capital = capital
        max_drawdown = 0
        
        logger.logger.info(f"Starting backtest for {strategy.name} on {symbol}")
        
        # Iterate through historical data
        for i in range(200, len(data)):  # Start after sufficient data for indicators
            current_data = data.iloc[:i+1]
            current_price = current_data['close'].iloc[-1]
            current_time = current_data.index[-1]
            
            # Get strategy signal
            signal = strategy.analyze(current_data, symbol)
            
            # Execute trades based on signals
            if signal['signal'] == 'BUY' and position == 0 and signal['confidence'] > 0.6:
                # Enter long position
                position_size = self._calculate_position_size(capital, current_price, signal)
                position = position_size
                entry_price = current_price * (1 + self.slippage)  # Apply slippage
                capital -= position * entry_price * (1 + self.commission)  # Apply commission
                
                logger.logger.debug(f"BUY: {position:.6f} at {entry_price:.2f}")
                
            elif signal['signal'] == 'SELL' and position == 0 and signal['confidence'] > 0.6:
                # Enter short position
                position_size = self._calculate_position_size(capital, current_price, signal)
                position = -position_size
                entry_price = current_price * (1 - self.slippage)  # Apply slippage
                capital += abs(position) * entry_price * (1 - self.commission)  # Apply commission
                
                logger.logger.debug(f"SELL: {position:.6f} at {entry_price:.2f}")
                
            elif position != 0:
                # Check for exit conditions
                should_exit = False
                exit_reason = ""
                exit_price = current_price
                
                # Check strategy-specific exit signals
                if hasattr(strategy, '_check_position_management'):
                    exit_signal = strategy._check_position_management(symbol, current_price)
                    if exit_signal:
                        should_exit = True
                        exit_reason = exit_signal.get('reason', 'Strategy exit')
                
                # Check opposite signal
                elif ((position > 0 and signal['signal'] == 'SELL') or 
                      (position < 0 and signal['signal'] == 'BUY')):
                    should_exit = True
                    exit_reason = "Opposite signal"
                
                if should_exit:
                    # Exit position
                    if position > 0:
                        exit_price = current_price * (1 - self.slippage)
                        pnl = position * (exit_price - entry_price)
                        capital += position * exit_price * (1 - self.commission)
                    else:
                        exit_price = current_price * (1 + self.slippage)
                        pnl = abs(position) * (entry_price - exit_price)
                        capital -= abs(position) * exit_price * (1 + self.commission)
                    
                    # Record trade
                    trade = {
                        'entry_time': current_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position_size': position,
                        'pnl': pnl,
                        'pnl_pct': pnl / (abs(position) * entry_price),
                        'reason': exit_reason
                    }
                    trades.append(trade)
                    
                    # Update strategy tracking
                    if hasattr(strategy, 'on_position_closed'):
                        strategy.on_position_closed(symbol, exit_price, exit_reason)
                    
                    logger.logger.debug(f"EXIT: {position:.6f} at {exit_price:.2f}, P&L: {pnl:.2f}")
                    
                    position = 0
                    entry_price = 0
            
            # Calculate current equity
            if position != 0:
                if position > 0:
                    unrealized_pnl = position * (current_price - entry_price)
                else:
                    unrealized_pnl = abs(position) * (entry_price - current_price)
                current_equity = capital + unrealized_pnl
            else:
                current_equity = capital
            
            equity_curve.append(current_equity)
            
            # Update drawdown
            if current_equity > peak_capital:
                peak_capital = current_equity
            
            current_drawdown = (peak_capital - current_equity) / peak_capital
            max_drawdown = max(max_drawdown, current_drawdown)
        
        # Close any remaining position
        if position != 0:
            final_price = data['close'].iloc[-1]
            if position > 0:
                final_price *= (1 - self.slippage)
                pnl = position * (final_price - entry_price)
                capital += position * final_price * (1 - self.commission)
            else:
                final_price *= (1 + self.slippage)
                pnl = abs(position) * (entry_price - final_price)
                capital -= abs(position) * final_price * (1 + self.commission)
            
            trade = {
                'entry_time': data.index[-1],
                'exit_time': data.index[-1],
                'entry_price': entry_price,
                'exit_price': final_price,
                'position_size': position,
                'pnl': pnl,
                'pnl_pct': pnl / (abs(position) * entry_price),
                'reason': 'Final close'
            }
            trades.append(trade)
        
        # Calculate performance metrics
        equity_series = pd.Series(equity_curve, index=data.index[200:])
        result = self._calculate_performance_metrics(trades, equity_series, self.initial_capital)
        
        logger.logger.info(f"Backtest completed: {len(trades)} trades, "
                          f"{result.win_rate:.1%} win rate, "
                          f"{result.total_return:.1%} total return")
        
        return result
    
    def _calculate_position_size(self, capital: float, price: float, signal: Dict) -> float:
        """Calculate position size based on available capital and risk"""
        # Use fixed percentage of capital for backtesting
        risk_per_trade = 0.02  # 2% risk per trade
        stop_loss_pct = signal.get('stop_loss_pct', 0.02)
        
        risk_amount = capital * risk_per_trade
        position_value = risk_amount / stop_loss_pct
        position_size = position_value / price
        
        # Apply leverage
        position_size *= Config.LEVERAGE
        
        return position_size
    
    def _calculate_performance_metrics(self, trades: List[Dict], equity_curve: pd.Series, 
                                     initial_capital: float) -> BacktestResult:
        """Calculate comprehensive performance metrics"""
        
        if not trades:
            return BacktestResult(
                total_return=0, sharpe_ratio=0, max_drawdown=0, win_rate=0,
                total_trades=0, avg_trade_return=0, volatility=0,
                calmar_ratio=0, sortino_ratio=0, profit_factor=0,
                trades=[], equity_curve=equity_curve
            )
        
        # Basic metrics
        final_capital = equity_curve.iloc[-1]
        total_return = (final_capital - initial_capital) / initial_capital
        
        # Trade analysis
        trade_returns = [trade['pnl_pct'] for trade in trades]
        winning_trades = [r for r in trade_returns if r > 0]
        losing_trades = [r for r in trade_returns if r < 0]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0
        avg_trade_return = np.mean(trade_returns) if trade_returns else 0
        
        # Risk metrics
        returns = equity_curve.pct_change().dropna()
        volatility = returns.std() * np.sqrt(365 * 24 * 4)  # Annualized (15-min data)
        
        # Sharpe ratio
        risk_free_rate = 0.02  # 2% annual
        excess_returns = returns - risk_free_rate / (365 * 24 * 4)
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(365 * 24 * 4) if excess_returns.std() > 0 else 0
        
        # Maximum drawdown
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = abs(drawdown.min())
        
        # Calmar ratio
        calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
        
        # Sortino ratio
        negative_returns = returns[returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(365 * 24 * 4)
        sortino_ratio = excess_returns.mean() / downside_deviation * np.sqrt(365 * 24 * 4) if downside_deviation > 0 else 0
        
        # Profit factor
        gross_profit = sum(winning_trades) if winning_trades else 0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=len(trades),
            avg_trade_return=avg_trade_return,
            volatility=volatility,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            profit_factor=profit_factor,
            trades=trades,
            equity_curve=equity_curve
        )
    
    def optimize_parameters(self, strategy_class, data: pd.DataFrame, symbol: str,
                           parameter_ranges: Dict) -> Dict:
        """
        Optimize strategy parameters using grid search
        
        Args:
            strategy_class: Strategy class to optimize
            data: Historical data
            symbol: Trading pair
            parameter_ranges: Dictionary of parameter ranges to test
            
        Returns:
            Best parameters and performance
        """
        
        best_result = None
        best_params = None
        best_score = -float('inf')
        
        # Generate parameter combinations
        param_combinations = self._generate_parameter_combinations(parameter_ranges)
        
        logger.logger.info(f"Testing {len(param_combinations)} parameter combinations...")
        
        for i, params in enumerate(param_combinations):
            try:
                # Create strategy instance with parameters
                strategy = strategy_class()
                
                # Apply parameters (this would need to be implemented in strategy classes)
                for param_name, param_value in params.items():
                    if hasattr(strategy, param_name):
                        setattr(strategy, param_name, param_value)
                
                # Run backtest
                result = self.run_backtest(strategy, data, symbol)
                
                # Calculate optimization score (weighted combination of metrics)
                score = self._calculate_optimization_score(result)
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    best_result = result
                
                if (i + 1) % 10 == 0:
                    logger.logger.info(f"Tested {i + 1}/{len(param_combinations)} combinations")
                
            except Exception as e:
                logger.error(f"Error testing parameters {params}", e)
                continue
        
        logger.logger.info(f"Optimization complete. Best score: {best_score:.4f}")
        
        return {
            'best_parameters': best_params,
            'best_result': best_result,
            'best_score': best_score
        }
    
    def _generate_parameter_combinations(self, parameter_ranges: Dict) -> List[Dict]:
        """Generate all combinations of parameters"""
        import itertools
        
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        
        combinations = []
        for values in itertools.product(*param_values):
            combination = dict(zip(param_names, values))
            combinations.append(combination)
        
        return combinations
    
    def _calculate_optimization_score(self, result: BacktestResult) -> float:
        """Calculate optimization score (higher is better)"""
        # Weighted combination of key metrics
        score = (
            result.total_return * 0.3 +
            result.sharpe_ratio * 0.25 +
            (1 - result.max_drawdown) * 0.2 +
            result.win_rate * 0.15 +
            min(result.profit_factor / 2, 1.0) * 0.1  # Cap profit factor contribution
        )
        
        # Penalty for insufficient trades
        if result.total_trades < 10:
            score *= 0.5
        
        return score
    
    def plot_results(self, results: List[Tuple[str, BacktestResult]], save_path: str = None):
        """Plot backtest results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Equity curves
        for name, result in results:
            axes[0, 0].plot(result.equity_curve.index, result.equity_curve.values, 
                           label=name, linewidth=2)
        axes[0, 0].set_title('Equity Curves')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Performance metrics comparison
        metrics = ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate']
        strategy_names = [name for name, _ in results]
        
        metric_values = []
        for metric in metrics:
            values = []
            for _, result in results:
                if metric == 'Total Return':
                    values.append(result.total_return * 100)
                elif metric == 'Sharpe Ratio':
                    values.append(result.sharpe_ratio)
                elif metric == 'Max Drawdown':
                    values.append(result.max_drawdown * 100)
                elif metric == 'Win Rate':
                    values.append(result.win_rate * 100)
            metric_values.append(values)
        
        x = np.arange(len(strategy_names))
        width = 0.2
        
        for i, (metric, values) in enumerate(zip(metrics, metric_values)):
            axes[0, 1].bar(x + i * width, values, width, label=metric)
        
        axes[0, 1].set_title('Performance Metrics Comparison')
        axes[0, 1].set_xlabel('Strategy')
        axes[0, 1].set_xticks(x + width * 1.5)
        axes[0, 1].set_xticklabels(strategy_names)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Drawdown analysis
        for name, result in results:
            peak = result.equity_curve.expanding().max()
            drawdown = (result.equity_curve - peak) / peak * 100
            axes[1, 0].fill_between(drawdown.index, drawdown.values, 0, 
                                   alpha=0.5, label=name)
        
        axes[1, 0].set_title('Drawdown Analysis')
        axes[1, 0].set_ylabel('Drawdown (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Monthly returns heatmap (for first strategy)
        if results:
            first_result = results[0][1]
            monthly_returns = first_result.equity_curve.resample('M').last().pct_change().dropna()
            
            # Create monthly returns matrix
            monthly_returns.index = pd.to_datetime(monthly_returns.index)
            monthly_returns_matrix = monthly_returns.groupby([
                monthly_returns.index.year,
                monthly_returns.index.month
            ]).first().unstack()
            
            if not monthly_returns_matrix.empty:
                sns.heatmap(monthly_returns_matrix * 100, annot=True, fmt='.1f',
                           cmap='RdYlGn', center=0, ax=axes[1, 1])
                axes[1, 1].set_title('Monthly Returns Heatmap (%)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_report(self, results: List[Tuple[str, BacktestResult]]) -> str:
        """Generate comprehensive backtest report"""
        
        report = "=" * 80 + "\n"
        report += "CRYPTOCURRENCY TRADING BOT - BACKTEST REPORT\n"
        report += "=" * 80 + "\n\n"
        
        for name, result in results:
            report += f"Strategy: {name}\n"
            report += "-" * 40 + "\n"
            report += f"Total Return:      {result.total_return:>10.1%}\n"
            report += f"Sharpe Ratio:      {result.sharpe_ratio:>10.2f}\n"
            report += f"Sortino Ratio:     {result.sortino_ratio:>10.2f}\n"
            report += f"Calmar Ratio:      {result.calmar_ratio:>10.2f}\n"
            report += f"Max Drawdown:      {result.max_drawdown:>10.1%}\n"
            report += f"Volatility:        {result.volatility:>10.1%}\n"
            report += f"Win Rate:          {result.win_rate:>10.1%}\n"
            report += f"Total Trades:      {result.total_trades:>10d}\n"
            report += f"Avg Trade Return:  {result.avg_trade_return:>10.2%}\n"
            report += f"Profit Factor:     {result.profit_factor:>10.2f}\n"
            report += "\n"
        
        return report