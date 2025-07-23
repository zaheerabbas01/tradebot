#!/usr/bin/env python3
"""
Backtest Runner for Crypto Trading Bot

This script runs backtests on historical data to evaluate and optimize
trading strategies before live deployment.

Usage:
    python run_backtest.py --symbol BTC/USDT --start 2021-01-01 --end 2023-12-31
    python run_backtest.py --optimize --symbol BTC/USDT --start 2022-01-01 --end 2023-12-31
"""

import argparse
import sys
from datetime import datetime
from backtesting.backtest_engine import BacktestEngine
from strategies.trend_following import TrendFollowingStrategy
from strategies.mean_reversion import MeanReversionStrategy
from config import Config
from utils.logger import logger

def main():
    parser = argparse.ArgumentParser(description='Cryptocurrency Trading Bot Backtester')
    parser.add_argument('--symbol', type=str, default='BTC/USDT',
                       help='Trading pair to backtest (default: BTC/USDT)')
    parser.add_argument('--start', type=str, default='2021-01-01',
                       help='Start date for backtesting (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2023-12-31',
                       help='End date for backtesting (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=10000,
                       help='Initial capital for backtesting (default: $10,000)')
    parser.add_argument('--optimize', action='store_true',
                       help='Run parameter optimization')
    parser.add_argument('--strategy', type=str, choices=['trend', 'mean_reversion', 'both'],
                       default='both', help='Strategy to backtest')
    parser.add_argument('--plot', action='store_true',
                       help='Generate performance plots')
    parser.add_argument('--report', action='store_true',
                       help='Generate detailed report')
    
    args = parser.parse_args()
    
    try:
        # Initialize backtest engine
        engine = BacktestEngine(initial_capital=args.capital)
        
        # Load historical data
        logger.logger.info(f"Loading historical data for {args.symbol} from {args.start} to {args.end}")
        data = engine.load_historical_data(args.symbol, args.start, args.end)
        
        if data.empty:
            logger.error(f"No data available for {args.symbol}")
            sys.exit(1)
        
        logger.logger.info(f"Loaded {len(data)} data points")
        
        results = []
        
        # Run backtests for selected strategies
        if args.strategy in ['trend', 'both']:
            logger.logger.info("Running Trend Following Strategy backtest...")
            
            if args.optimize:
                # Parameter optimization for trend following
                parameter_ranges = {
                    'ema_fast': [20, 30, 50, 70],
                    'ema_slow': [100, 150, 200, 250],
                    'rsi_period': [10, 14, 18, 21],
                    'rsi_overbought': [65, 70, 75, 80],
                    'rsi_oversold': [20, 25, 30, 35]
                }
                
                optimization_result = engine.optimize_parameters(
                    TrendFollowingStrategy, data, args.symbol, parameter_ranges
                )
                
                logger.logger.info("Trend Following Optimization Results:")
                logger.logger.info(f"Best Parameters: {optimization_result['best_parameters']}")
                logger.logger.info(f"Best Score: {optimization_result['best_score']:.4f}")
                
                results.append(("Trend Following (Optimized)", optimization_result['best_result']))
            else:
                trend_strategy = TrendFollowingStrategy()
                trend_result = engine.run_backtest(trend_strategy, data, args.symbol)
                results.append(("Trend Following", trend_result))
        
        if args.strategy in ['mean_reversion', 'both']:
            logger.logger.info("Running Mean Reversion Strategy backtest...")
            
            if args.optimize:
                # Parameter optimization for mean reversion
                parameter_ranges = {
                    'bollinger_period': [15, 20, 25, 30],
                    'bollinger_std': [1.5, 2.0, 2.5, 3.0],
                    'volume_spike_threshold': [1.2, 1.5, 1.8, 2.0]
                }
                
                optimization_result = engine.optimize_parameters(
                    MeanReversionStrategy, data, args.symbol, parameter_ranges
                )
                
                logger.logger.info("Mean Reversion Optimization Results:")
                logger.logger.info(f"Best Parameters: {optimization_result['best_parameters']}")
                logger.logger.info(f"Best Score: {optimization_result['best_score']:.4f}")
                
                results.append(("Mean Reversion (Optimized)", optimization_result['best_result']))
            else:
                mean_reversion_strategy = MeanReversionStrategy()
                mean_reversion_result = engine.run_backtest(mean_reversion_strategy, data, args.symbol)
                results.append(("Mean Reversion", mean_reversion_result))
        
        # Display results
        print("\n" + "="*80)
        print("BACKTEST RESULTS SUMMARY")
        print("="*80)
        
        for name, result in results:
            print(f"\n{name}:")
            print(f"  Total Return:     {result.total_return:>8.1%}")
            print(f"  Sharpe Ratio:     {result.sharpe_ratio:>8.2f}")
            print(f"  Max Drawdown:     {result.max_drawdown:>8.1%}")
            print(f"  Win Rate:         {result.win_rate:>8.1%}")
            print(f"  Total Trades:     {result.total_trades:>8d}")
            print(f"  Profit Factor:    {result.profit_factor:>8.2f}")
            
            # Check if strategy meets minimum requirements
            meets_requirements = (
                result.win_rate >= Config.MIN_WIN_RATE and
                result.total_trades >= 10 and
                result.max_drawdown < 0.3  # Max 30% drawdown
            )
            
            status = "✅ MEETS REQUIREMENTS" if meets_requirements else "❌ DOES NOT MEET REQUIREMENTS"
            print(f"  Status:           {status}")
        
        # Generate detailed report if requested
        if args.report:
            report = engine.generate_report(results)
            
            # Save report to file
            report_filename = f"backtest_report_{args.symbol.replace('/', '_')}_{args.start}_{args.end}.txt"
            with open(report_filename, 'w') as f:
                f.write(report)
            
            logger.logger.info(f"Detailed report saved to {report_filename}")
            print(f"\nDetailed report saved to: {report_filename}")
        
        # Generate plots if requested
        if args.plot:
            try:
                plot_filename = f"backtest_plots_{args.symbol.replace('/', '_')}_{args.start}_{args.end}.png"
                engine.plot_results(results, save_path=plot_filename)
                logger.logger.info(f"Performance plots saved to {plot_filename}")
                print(f"Performance plots saved to: {plot_filename}")
            except Exception as e:
                logger.error(f"Error generating plots: {e}")
                print("Note: Plotting requires matplotlib and seaborn packages")
        
        # Final recommendations
        print("\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80)
        
        if results:
            best_strategy = max(results, key=lambda x: x[1].total_return)
            print(f"Best performing strategy: {best_strategy[0]}")
            print(f"Total return: {best_strategy[1].total_return:.1%}")
            
            if best_strategy[1].win_rate >= Config.MIN_WIN_RATE:
                print("✅ Strategy is ready for live trading")
                print("Next steps:")
                print("1. Set up API keys in .env file")
                print("2. Start with small position sizes")
                print("3. Monitor performance closely")
            else:
                print("❌ Strategy needs further optimization")
                print("Consider:")
                print("1. Adjusting parameters")
                print("2. Adding additional filters")
                print("3. Testing on different time periods")
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()