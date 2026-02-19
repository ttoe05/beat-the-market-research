"""
Backtesting Engine

Simulates trading strategies using historical data and model predictions.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from datetime import datetime
from src.backtesting.portfolio import Portfolio
from src.backtesting.strategy import Strategy
from src.backtesting.risk_manager import RiskManager
from src.backtesting.performance_metrics import PerformanceMetrics
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class BacktestEngine:
    """
    Backtesting engine for trading strategies.

    Examples:
        >>> import pandas as pd
        >>> from src.backtesting.strategy import Strategy
        >>> engine = BacktestEngine(initial_capital=100000)
        >>> dates = pd.date_range('2020-01-01', periods=10)
        >>> data = pd.DataFrame({'close': range(100, 110), 'signal': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]}, index=dates)
        >>> results = engine.run(data, 'TEST')
        >>> 'portfolio_value' in results.columns
        True
    """

    def __init__(self,
                 initial_capital: float = 100000,
                 commission: float = 0.001,
                 slippage: float = 0.0005,
                 use_risk_manager: bool = True,
                 risk_config: Optional[Dict] = None):
        """
        Initialize BacktestEngine.

        Args:
            initial_capital: Starting capital
            commission: Commission rate (e.g., 0.001 = 0.1%)
            slippage: Slippage rate (e.g., 0.0005 = 0.05%)
            use_risk_manager: Whether to use risk manager
            risk_config: Configuration for risk manager
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.use_risk_manager = use_risk_manager

        # Initialize components
        self.portfolio = Portfolio(initial_capital=initial_capital)
        self.risk_manager = RiskManager(config=risk_config) if use_risk_manager else None

        logger.info(f"Initialized BacktestEngine (capital=${initial_capital:,.2f}, "
                   f"commission={commission:.4f}, slippage={slippage:.4f})")

    def calculate_position_size(self, signal: int, price: float,
                                portfolio_value: float) -> float:
        """
        Calculate position size based on signal and risk management.

        Args:
            signal: Trading signal (1=buy, 0=hold, -1=sell)
            price: Current price
            portfolio_value: Current portfolio value

        Returns:
            Number of shares to trade
        """
        if signal == 0:
            return 0

        # Default: use all available capital for buy signals
        if signal == 1:
            available_cash = self.portfolio.cash
            max_shares = available_cash / (price * (1 + self.commission + self.slippage))

            # Apply risk management if enabled
            if self.risk_manager:
                position_size = self.risk_manager.calculate_position_size(
                    price, portfolio_value, available_cash
                )
                max_shares = min(max_shares, position_size / price)

            return int(max_shares)

        elif signal == -1:
            # Sell all holdings
            return -self.portfolio.get_position('default')

        return 0

    def execute_trade(self, symbol: str, shares: float, price: float,
                     timestamp: datetime) -> Dict:
        """
        Execute a trade with commission and slippage.

        Args:
            symbol: Stock symbol
            shares: Number of shares (positive=buy, negative=sell)
            price: Execution price
            timestamp: Trade timestamp

        Returns:
            Dictionary with trade details
        """
        if shares == 0:
            return {'executed': False}

        # Apply slippage
        if shares > 0:  # Buy
            execution_price = price * (1 + self.slippage)
        else:  # Sell
            execution_price = price * (1 - self.slippage)

        # Calculate costs
        trade_value = abs(shares) * execution_price
        commission_cost = trade_value * self.commission
        total_cost = trade_value + commission_cost

        # Execute trade in portfolio
        if shares > 0:  # Buy
            if self.portfolio.cash >= total_cost:
                self.portfolio.update_position('default', shares, execution_price, commission_cost)
                success = True
            else:
                success = False
                logger.warning(f"Insufficient cash for trade: ${total_cost:,.2f} > ${self.portfolio.cash:,.2f}")
        else:  # Sell
            self.portfolio.update_position('default', shares, execution_price, commission_cost)
            success = True

        return {
            'executed': success,
            'timestamp': timestamp,
            'symbol': symbol,
            'shares': shares,
            'price': price,
            'execution_price': execution_price,
            'commission': commission_cost,
            'total_cost': total_cost if shares > 0 else -total_cost
        }

    def run(self, data: pd.DataFrame, symbol_or_signals,
            signal_column: str = 'signal') -> pd.DataFrame:
        """
        Run backtest on historical data.

        Args:
            data: DataFrame with price data (and optionally signals)
            symbol_or_signals: Either a stock symbol string, or array of trading signals
            signal_column: Name of column containing trading signals (used if symbol_or_signals is a string)

        Returns:
            DataFrame with backtest results
        """
        # Handle both symbol string and signal array
        if isinstance(symbol_or_signals, str):
            symbol = symbol_or_signals
            # Signals should already be in data[signal_column]
        else:
            # symbol_or_signals is an array of predictions/signals
            symbol = 'UNKNOWN'
            data = data.copy()
            data[signal_column] = symbol_or_signals

        logger.info(f"Starting backtest for {symbol} ({len(data)} periods)")

        # Reset portfolio
        self.portfolio = Portfolio(initial_capital=self.initial_capital)

        # Initialize results storage
        results = []
        trades = []

        for timestamp, row in data.iterrows():
            price = row['close']
            signal = row.get(signal_column, 0)

            # Update portfolio value with current price
            portfolio_value = self.portfolio.get_total_value(default_price=price)

            # Calculate position size
            shares = self.calculate_position_size(signal, price, portfolio_value)

            # Execute trade if shares > 0
            if shares != 0:
                trade_result = self.execute_trade(symbol, shares, price, timestamp)
                if trade_result['executed']:
                    trades.append(trade_result)

            # Record current state
            position = self.portfolio.get_position('default')
            results.append({
                'timestamp': timestamp,
                'close': price,
                'signal': signal,
                'position': position,
                'cash': self.portfolio.cash,
                'portfolio_value': self.portfolio.get_total_value(default_price=price),
                'returns': (self.portfolio.get_total_value(default_price=price) - self.initial_capital) / self.initial_capital
            })

        results_df = pd.DataFrame(results).set_index('timestamp')

        # Calculate performance metrics
        metrics = PerformanceMetrics(results_df)
        performance = metrics.calculate_all_metrics()

        logger.info(f"Backtest complete: {len(trades)} trades executed")
        logger.info(f"Final portfolio value: ${results_df['portfolio_value'].iloc[-1]:,.2f}")
        logger.info(f"Total return: {performance['total_return']*100:.2f}%")

        # Store trades and metrics
        self.trades = pd.DataFrame(trades)
        self.performance_metrics = performance

        return results_df

    def run_with_strategy(self, data: pd.DataFrame, symbol: str,
                         strategy: Strategy) -> pd.DataFrame:
        """
        Run backtest using a Strategy object.

        Args:
            data: DataFrame with price data
            symbol: Stock symbol
            strategy: Strategy instance

        Returns:
            DataFrame with backtest results
        """
        logger.info(f"Running backtest with strategy: {strategy.__class__.__name__}")

        # Generate signals using strategy
        signals = strategy.generate_signals(data)
        data_with_signals = data.copy()
        data_with_signals['signal'] = signals

        # Run backtest
        return self.run(data_with_signals, symbol, signal_column='signal')

    def get_trades(self) -> pd.DataFrame:
        """Get all executed trades."""
        if hasattr(self, 'trades'):
            return self.trades
        return pd.DataFrame()

    def get_performance_metrics(self) -> Dict:
        """Get performance metrics."""
        if hasattr(self, 'performance_metrics'):
            return self.performance_metrics
        return {}

    def get_trade_statistics(self) -> Dict:
        """
        Calculate trade statistics.

        Returns:
            Dictionary with trade statistics
        """
        if not hasattr(self, 'trades') or len(self.trades) == 0:
            return {'total_trades': 0}

        trades_df = self.trades

        # Separate buy and sell trades
        buy_trades = trades_df[trades_df['shares'] > 0]
        sell_trades = trades_df[trades_df['shares'] < 0]

        stats = {
            'total_trades': len(trades_df),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'total_commission': trades_df['commission'].sum(),
            'avg_trade_size': trades_df['shares'].abs().mean(),
            'largest_trade': trades_df['shares'].abs().max(),
            'smallest_trade': trades_df['shares'].abs().min()
        }

        return stats

    def print_summary(self, results: pd.DataFrame) -> None:
        """
        Print a summary of backtest results.

        Args:
            results: DataFrame with backtest results from run() method
        """
        print("=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)

        # Performance metrics
        metrics = self.get_performance_metrics()
        print("\nPerformance Metrics:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")

        # Trade statistics
        trade_stats = self.get_trade_statistics()
        print("\nTrade Statistics:")
        for stat, value in trade_stats.items():
            if isinstance(value, float):
                print(f"  {stat}: {value:.2f}")
            else:
                print(f"  {stat}: {value}")

        # Results summary
        print(f"\nFinal Results:")
        print(f"  Initial Capital: ${self.initial_capital:,.2f}")
        print(f"  Final Value: ${results['portfolio_value'].iloc[-1]:,.2f}")
        print(f"  Total Return: {metrics.get('total_return', 0)*100:.2f}%")


# if __name__ == "__main__":
#     # Example usage
#     import logging
#     from ..utils.logger import setup_logger
#
#     setup_logger(__name__, level=logging.INFO)
#
#     # Create sample data with signals
#     dates = pd.date_range('2020-01-01', periods=252)  # 1 year of trading days
#     np.random.seed(42)
#
#     # Simulate price data
#     returns = np.random.randn(252) * 0.02  # 2% daily volatility
#     prices = 100 * np.exp(np.cumsum(returns))
#
#     # Generate random signals
#     signals = np.random.choice([1, 0, -1], size=252, p=[0.1, 0.8, 0.1])
#
#     data = pd.DataFrame({
#         'close': prices,
#         'signal': signals
#     }, index=dates)
#
#     # Run backtest
#     engine = BacktestEngine(initial_capital=100000, commission=0.001, slippage=0.0005)
#     results = engine.run(data, 'TEST', signal_column='signal')
#
#     print("=" * 60)
#     print("BACKTEST RESULTS")
#     print("=" * 60)
#     engine.print_summary(results)
