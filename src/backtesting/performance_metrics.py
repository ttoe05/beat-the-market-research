"""
Performance Metrics Module

Calculates comprehensive performance metrics for backtesting results including:
- Returns (total, annualized, cumulative)
- Risk metrics (volatility, Sharpe, Sortino, max drawdown)
- Trade statistics
- Risk-adjusted returns
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class PerformanceMetrics:
    """
    Calculate performance metrics for trading backtests.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> dates = pd.date_range('2020-01-01', periods=252)
        >>> results = pd.DataFrame({
        ...     'portfolio_value': 100000 * (1 + np.cumsum(np.random.randn(252) * 0.01)),
        ...     'returns': np.random.randn(252) * 0.01,
        ...     'cash': 50000,
        ...     'position': 100
        ... }, index=dates)
        >>> metrics = PerformanceMetrics(results)
        >>> perf = metrics.calculate_all_metrics()
        >>> 'total_return' in perf
        True
    """

    def __init__(self, results: pd.DataFrame, initial_capital: float = 100000, risk_free_rate: float = 0.02):
        """
        Initialize PerformanceMetrics.

        Args:
            results: DataFrame with backtest results (must have 'portfolio_value' and 'returns' columns)
            initial_capital: Starting capital
            risk_free_rate: Annual risk-free rate for Sharpe/Sortino calculations
        """
        self.results = results
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate

        logger.info(f"Initialized PerformanceMetrics with {len(results)} periods")

    def calculate_total_return(self) -> float:
        """
        Calculate total return over the period.

        Returns:
            Total return as decimal (e.g., 0.25 = 25%)
        """
        if len(self.results) == 0:
            return 0.0

        final_value = self.results['portfolio_value'].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital

        return total_return

    def calculate_annualized_return(self, trading_days: int = 252) -> float:
        """
        Calculate annualized return.

        Args:
            trading_days: Number of trading days per year

        Returns:
            Annualized return as decimal
        """
        if len(self.results) == 0:
            return 0.0

        total_return = self.calculate_total_return()
        n_periods = len(self.results)
        years = n_periods / trading_days

        if years <= 0:
            return 0.0

        annualized_return = (1 + total_return) ** (1 / years) - 1

        return annualized_return

    def calculate_volatility(self, annualize: bool = True, trading_days: int = 252) -> float:
        """
        Calculate portfolio volatility (standard deviation of returns).

        Args:
            annualize: Whether to annualize the volatility
            trading_days: Number of trading days per year

        Returns:
            Volatility as decimal
        """
        if 'returns' not in self.results.columns or len(self.results) < 2:
            return 0.0

        volatility = self.results['returns'].std()

        if annualize:
            volatility = volatility * np.sqrt(trading_days)

        return volatility

    def calculate_sharpe_ratio(self, trading_days: int = 252) -> float:
        """
        Calculate Sharpe ratio.

        Sharpe = (Annual Return - Risk Free Rate) / Annual Volatility

        Args:
            trading_days: Number of trading days per year

        Returns:
            Sharpe ratio
        """
        annual_return = self.calculate_annualized_return(trading_days)
        annual_volatility = self.calculate_volatility(annualize=True, trading_days=trading_days)

        if annual_volatility == 0:
            return 0.0

        sharpe = (annual_return - self.risk_free_rate) / annual_volatility

        return sharpe

    def calculate_sortino_ratio(self, trading_days: int = 252) -> float:
        """
        Calculate Sortino ratio (uses downside deviation instead of total volatility).

        Args:
            trading_days: Number of trading days per year

        Returns:
            Sortino ratio
        """
        if 'returns' not in self.results.columns or len(self.results) < 2:
            return 0.0

        annual_return = self.calculate_annualized_return(trading_days)

        # Calculate downside deviation (only negative returns)
        negative_returns = self.results['returns'][self.results['returns'] < 0]

        if len(negative_returns) == 0:
            return np.inf

        downside_deviation = negative_returns.std() * np.sqrt(trading_days)

        if downside_deviation == 0:
            return 0.0

        sortino = (annual_return - self.risk_free_rate) / downside_deviation

        return sortino

    def calculate_max_drawdown(self) -> Dict[str, Any]:
        """
        Calculate maximum drawdown and related metrics.

        Returns:
            Dictionary with max_drawdown, max_drawdown_pct, drawdown_duration
        """
        if len(self.results) == 0:
            return {
                'max_drawdown': 0.0,
                'max_drawdown_pct': 0.0,
                'drawdown_duration': 0
            }

        portfolio_values = self.results['portfolio_value']

        # Calculate running maximum
        running_max = portfolio_values.expanding().max()

        # Calculate drawdown
        drawdown = portfolio_values - running_max
        drawdown_pct = drawdown / running_max

        # Find maximum drawdown
        max_dd = drawdown.min()
        max_dd_pct = drawdown_pct.min()

        # Find drawdown duration (periods from peak to recovery)
        if max_dd < 0:
            max_dd_idx = drawdown.idxmin()
            # Find the peak before the max drawdown
            peak_idx = running_max[:max_dd_idx].idxmax()
            # Find recovery (if any)
            recovery_idx = None
            peak_value = running_max[peak_idx]

            for idx in portfolio_values[max_dd_idx:].index:
                if portfolio_values[idx] >= peak_value:
                    recovery_idx = idx
                    break

            if recovery_idx is not None:
                duration = len(portfolio_values[peak_idx:recovery_idx])
            else:
                duration = len(portfolio_values[peak_idx:])
        else:
            duration = 0

        return {
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd_pct,
            'drawdown_duration': duration
        }

    def calculate_calmar_ratio(self, trading_days: int = 252) -> float:
        """
        Calculate Calmar ratio (Annual Return / Max Drawdown).

        Args:
            trading_days: Number of trading days per year

        Returns:
            Calmar ratio
        """
        annual_return = self.calculate_annualized_return(trading_days)
        max_dd = self.calculate_max_drawdown()['max_drawdown_pct']

        if max_dd == 0:
            return 0.0

        calmar = annual_return / abs(max_dd)

        return calmar

    def calculate_win_rate(self) -> float:
        """
        Calculate win rate (percentage of profitable periods).

        Returns:
            Win rate as decimal
        """
        if 'returns' not in self.results.columns or len(self.results) == 0:
            return 0.0

        winning_periods = (self.results['returns'] > 0).sum()
        total_periods = len(self.results)

        win_rate = winning_periods / total_periods

        return win_rate

    def calculate_profit_factor(self) -> float:
        """
        Calculate profit factor (gross profits / gross losses).

        Returns:
            Profit factor
        """
        if 'returns' not in self.results.columns or len(self.results) == 0:
            return 0.0

        gross_profits = self.results['returns'][self.results['returns'] > 0].sum()
        gross_losses = abs(self.results['returns'][self.results['returns'] < 0].sum())

        if gross_losses == 0:
            return np.inf if gross_profits > 0 else 0.0

        profit_factor = gross_profits / gross_losses

        return profit_factor

    def calculate_expectancy(self) -> float:
        """
        Calculate expectancy (average return per period).

        Returns:
            Expectancy as decimal
        """
        if 'returns' not in self.results.columns or len(self.results) == 0:
            return 0.0

        return self.results['returns'].mean()

    def calculate_consecutive_stats(self) -> Dict[str, int]:
        """
        Calculate consecutive wins and losses.

        Returns:
            Dictionary with max_consecutive_wins and max_consecutive_losses
        """
        if 'returns' not in self.results.columns or len(self.results) == 0:
            return {
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0
            }

        returns = self.results['returns']

        # Track consecutive wins
        max_wins = 0
        current_wins = 0

        # Track consecutive losses
        max_losses = 0
        current_losses = 0

        for ret in returns:
            if ret > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif ret < 0:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
            else:
                current_wins = 0
                current_losses = 0

        return {
            'max_consecutive_wins': max_wins,
            'max_consecutive_losses': max_losses
        }

    def calculate_recovery_factor(self) -> float:
        """
        Calculate recovery factor (Total Return / Max Drawdown).

        Returns:
            Recovery factor
        """
        total_return = self.calculate_total_return()
        max_dd = self.calculate_max_drawdown()['max_drawdown_pct']

        if max_dd == 0:
            return 0.0

        recovery_factor = total_return / abs(max_dd)

        return recovery_factor

    def calculate_value_at_risk(self, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR) at specified confidence level.

        Args:
            confidence: Confidence level (e.g., 0.95 for 95%)

        Returns:
            VaR as decimal
        """
        if 'returns' not in self.results.columns or len(self.results) == 0:
            return 0.0

        var = self.results['returns'].quantile(1 - confidence)

        return var

    def calculate_conditional_value_at_risk(self, confidence: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR/Expected Shortfall).

        Args:
            confidence: Confidence level

        Returns:
            CVaR as decimal
        """
        if 'returns' not in self.results.columns or len(self.results) == 0:
            return 0.0

        var = self.calculate_value_at_risk(confidence)
        cvar = self.results['returns'][self.results['returns'] <= var].mean()

        return cvar

    def calculate_all_metrics(self, trading_days: int = 252) -> Dict[str, Any]:
        """
        Calculate all performance metrics.

        Args:
            trading_days: Number of trading days per year

        Returns:
            Dictionary with all metrics
        """
        logger.info("Calculating all performance metrics")

        metrics = {}

        # Return metrics
        metrics['total_return'] = self.calculate_total_return()
        metrics['annualized_return'] = self.calculate_annualized_return(trading_days)
        metrics['expectancy'] = self.calculate_expectancy()

        # Risk metrics
        metrics['volatility'] = self.calculate_volatility(annualize=True, trading_days=trading_days)
        metrics['sharpe_ratio'] = self.calculate_sharpe_ratio(trading_days)
        metrics['sortino_ratio'] = self.calculate_sortino_ratio(trading_days)
        metrics['calmar_ratio'] = self.calculate_calmar_ratio(trading_days)

        # Drawdown metrics
        dd_metrics = self.calculate_max_drawdown()
        metrics.update(dd_metrics)

        # Trade statistics
        metrics['win_rate'] = self.calculate_win_rate()
        metrics['profit_factor'] = self.calculate_profit_factor()
        metrics['recovery_factor'] = self.calculate_recovery_factor()

        # Consecutive stats
        consecutive = self.calculate_consecutive_stats()
        metrics.update(consecutive)

        # Risk metrics
        metrics['value_at_risk_95'] = self.calculate_value_at_risk(0.95)
        metrics['cvar_95'] = self.calculate_conditional_value_at_risk(0.95)

        # Portfolio stats
        if len(self.results) > 0:
            metrics['final_value'] = self.results['portfolio_value'].iloc[-1]
            metrics['initial_value'] = self.initial_capital
            metrics['total_periods'] = len(self.results)

            if 'returns' in self.results.columns:
                metrics['best_day'] = self.results['returns'].max()
                metrics['worst_day'] = self.results['returns'].min()
                metrics['avg_daily_return'] = self.results['returns'].mean()
                metrics['positive_days'] = (self.results['returns'] > 0).sum()
                metrics['negative_days'] = (self.results['returns'] < 0).sum()

        logger.info(f"Calculated {len(metrics)} performance metrics")

        return metrics

    def print_summary(self, metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Print formatted performance summary.

        Args:
            metrics: Metrics dictionary (if None, calculates all metrics)
        """
        if metrics is None:
            metrics = self.calculate_all_metrics()

        print("\n" + "=" * 70)
        print("BACKTEST PERFORMANCE SUMMARY")
        print("=" * 70)

        print("\nReturn Metrics:")
        print(f"  Total Return:        {metrics.get('total_return', 0) * 100:>10.2f}%")
        print(f"  Annualized Return:   {metrics.get('annualized_return', 0) * 100:>10.2f}%")
        print(f"  Expectancy (daily):  {metrics.get('expectancy', 0) * 100:>10.4f}%")

        print("\nRisk Metrics:")
        print(f"  Annual Volatility:   {metrics.get('volatility', 0) * 100:>10.2f}%")
        print(f"  Sharpe Ratio:        {metrics.get('sharpe_ratio', 0):>10.3f}")
        print(f"  Sortino Ratio:       {metrics.get('sortino_ratio', 0):>10.3f}")
        print(f"  Calmar Ratio:        {metrics.get('calmar_ratio', 0):>10.3f}")

        print("\nDrawdown Metrics:")
        print(f"  Max Drawdown:        {metrics.get('max_drawdown_pct', 0) * 100:>10.2f}%")
        print(f"  Max DD Duration:     {metrics.get('drawdown_duration', 0):>10} periods")
        print(f"  Recovery Factor:     {metrics.get('recovery_factor', 0):>10.3f}")

        print("\nTrade Statistics:")
        print(f"  Win Rate:            {metrics.get('win_rate', 0) * 100:>10.2f}%")
        print(f"  Profit Factor:       {metrics.get('profit_factor', 0):>10.3f}")
        print(f"  Best Day:            {metrics.get('best_day', 0) * 100:>10.2f}%")
        print(f"  Worst Day:           {metrics.get('worst_day', 0) * 100:>10.2f}%")
        print(f"  Max Consecutive Wins: {metrics.get('max_consecutive_wins', 0):>9}")
        print(f"  Max Consecutive Loss: {metrics.get('max_consecutive_losses', 0):>9}")

        print("\nRisk Measures:")
        print(f"  VaR (95%):           {metrics.get('value_at_risk_95', 0) * 100:>10.2f}%")
        print(f"  CVaR (95%):          {metrics.get('cvar_95', 0) * 100:>10.2f}%")

        print("\nPortfolio Summary:")
        print(f"  Initial Value:       ${metrics.get('initial_value', 0):>10,.2f}")
        print(f"  Final Value:         ${metrics.get('final_value', 0):>10,.2f}")
        print(f"  Total Periods:       {metrics.get('total_periods', 0):>10}")
        print(f"  Positive Days:       {metrics.get('positive_days', 0):>10}")
        print(f"  Negative Days:       {metrics.get('negative_days', 0):>10}")

        print("=" * 70 + "\n")


if __name__ == "__main__":
    # Example usage
    import logging
    from ..utils.logger import setup_logger

    setup_logger(__name__, level=logging.INFO)

    # Create sample backtest results
    dates = pd.date_range('2020-01-01', periods=252)  # 1 year
    np.random.seed(42)

    # Simulate returns with some realistic characteristics
    daily_returns = np.random.randn(252) * 0.015 + 0.0003  # ~1.5% daily vol, slight positive drift
    cumulative_returns = np.cumprod(1 + daily_returns)

    initial_capital = 100000
    portfolio_values = initial_capital * cumulative_returns

    results = pd.DataFrame({
        'portfolio_value': portfolio_values,
        'returns': daily_returns,
        'cash': initial_capital * 0.3,  # 30% cash
        'position': 100
    }, index=dates)

    # Calculate metrics
    perf = PerformanceMetrics(results, initial_capital=initial_capital)
    metrics = perf.calculate_all_metrics()

    # Print summary
    perf.print_summary(metrics)

    # Show individual metrics
    print("\nKey Metrics:")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {metrics['max_drawdown_pct']*100:.2f}%")
    print(f"Win Rate: {metrics['win_rate']*100:.2f}%")

