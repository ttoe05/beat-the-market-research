"""
Label Generator Module

Creates target labels for trading ML models:
- Binary classification (buy/no-buy)
- Multi-class classification (buy/sell/hold)
- Regression targets (forward returns)
"""

import pandas as pd
import numpy as np
from typing import Optional, Literal
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class LabelGenerator:
    """
    Generate labels for trading classification/regression tasks.
    By default, forward period is 5 days and threshold is 2% return for buy signals.

    Examples:
        >>> import pandas as pd
        >>> lg = LabelGenerator()
        >>> dates = pd.date_range('2020-01-01', periods=10)
        >>> data = pd.DataFrame({'close': [100, 102, 101, 103, 105, 104, 106, 108, 107, 109]}, index=dates)
        >>> labels = lg.generate_binary_labels(data, threshold=0.01)
        >>> labels.name
        'label'
    """

    def __init__(self, forward_period: int = 5, threshold: float = 0.02):
        """
        Initialize LabelGenerator.

        Args:
            forward_period: Number of periods to look ahead
            threshold: Threshold for classification (e.g., 0.02 = 2% return)
        """
        self.forward_period = forward_period
        self.threshold = threshold
        logger.info(f"Initialized LabelGenerator (forward_period={forward_period}, threshold={threshold})")

    def calculate_forward_returns(self, data: pd.DataFrame, column: str = 'close') -> pd.Series:
        """
        Calculate forward returns for labeling.

        Args:
            data: DataFrame with price data
            column: Column to use for calculation

        Returns:
            Series with forward returns
        """
        forward_returns = data[column].pct_change(self.forward_period).shift(-self.forward_period)
        logger.debug(f"Calculated forward returns for {self.forward_period} periods")
        return forward_returns

    def generate_binary_labels(self, data: pd.DataFrame,
                               threshold: Optional[float] = None,
                               column: str = 'close') -> pd.Series:
        """
        Generate binary labels (1 = buy, 0 = no buy).

        Args:
            data: DataFrame with price data
            threshold: Return threshold for buy signal (uses self.threshold if None)
            column: Column to use for calculation

        Returns:
            Series with binary labels
        """
        if threshold is None:
            threshold = self.threshold

        forward_returns = self.calculate_forward_returns(data, column)

        # 1 if forward return > threshold, else 0
        labels = (forward_returns > threshold).astype(int)
        labels.name = 'label'

        buy_count = labels.sum()
        total_count = labels.count()
        logger.info(f"Generated binary labels: {buy_count}/{total_count} buy signals ({buy_count/total_count*100:.2f}%)")

        return labels

    def generate_multiclass_labels(self, data: pd.DataFrame,
                                   buy_threshold: Optional[float] = None,
                                   sell_threshold: Optional[float] = None,
                                   column: str = 'close') -> pd.Series:
        """
        Generate multi-class labels (2 = buy, 1 = hold, 0 = sell).

        Args:
            data: DataFrame with price data
            buy_threshold: Return threshold for buy signal
            sell_threshold: Return threshold for sell signal (negative)
            column: Column to use for calculation

        Returns:
            Series with multi-class labels
        """
        if buy_threshold is None:
            buy_threshold = self.threshold
        if sell_threshold is None:
            sell_threshold = -self.threshold

        forward_returns = self.calculate_forward_returns(data, column)

        # Create multi-class labels
        labels = pd.Series(1, index=data.index, name='label')  # Default: hold
        labels[forward_returns > buy_threshold] = 2  # Buy
        labels[forward_returns < sell_threshold] = 0  # Sell

        buy_count = (labels == 2).sum()
        sell_count = (labels == 0).sum()
        hold_count = (labels == 1).sum()
        total_count = labels.count()

        logger.info(f"Generated multi-class labels: Buy={buy_count} ({buy_count/total_count*100:.2f}%), "
                   f"Hold={hold_count} ({hold_count/total_count*100:.2f}%), "
                   f"Sell={sell_count} ({sell_count/total_count*100:.2f}%)")

        return labels

    def generate_regression_labels(self, data: pd.DataFrame,
                                   column: str = 'close') -> pd.Series:
        """
        Generate regression labels (continuous forward returns).

        Args:
            data: DataFrame with price data
            column: Column to use for calculation

        Returns:
            Series with forward returns as labels
        """
        labels = self.calculate_forward_returns(data, column)
        labels.name = 'label'

        logger.info(f"Generated regression labels: mean={labels.mean():.4f}, std={labels.std():.4f}")

        return labels

    def generate_trend_labels(self, data: pd.DataFrame,
                             window: int = 20,
                             column: str = 'close') -> pd.Series:
        """
        Generate labels based on trend direction.

        Args:
            data: DataFrame with price data
            window: Window for trend calculation
            column: Column to use for calculation

        Returns:
            Series with trend labels (1 = uptrend, 0 = downtrend)
        """
        # Calculate trend using linear regression slope
        def calculate_slope(series):
            if len(series) < 2:
                return 0
            x = np.arange(len(series))
            slope = np.polyfit(x, series, 1)[0]
            return slope

        slopes = data[column].rolling(window=window).apply(calculate_slope, raw=False)
        labels = (slopes > 0).astype(int)
        labels.name = 'label'

        uptrend_count = labels.sum()
        total_count = labels.count()
        logger.info(f"Generated trend labels: {uptrend_count}/{total_count} uptrend ({uptrend_count/total_count*100:.2f}%)")

        return labels

    def generate_volatility_adjusted_labels(self, data: pd.DataFrame,
                                            volatility_window: int = 20,
                                            column: str = 'close') -> pd.Series:
        """
        Generate labels adjusted for volatility (Sharpe-like approach).

        Args:
            data: DataFrame with price data
            volatility_window: Window for volatility calculation
            column: Column to use for calculation

        Returns:
            Series with volatility-adjusted binary labels
        """
        forward_returns = self.calculate_forward_returns(data, column)

        # Calculate rolling volatility
        volatility = data[column].pct_change().rolling(window=volatility_window).std()

        # Adjust threshold by volatility
        adjusted_threshold = self.threshold * (volatility / volatility.mean())

        # Generate labels
        labels = (forward_returns > adjusted_threshold).astype(int)
        labels.name = 'label'

        buy_count = labels.sum()
        total_count = labels.count()
        logger.info(f"Generated volatility-adjusted labels: {buy_count}/{total_count} buy signals "
                   f"({buy_count/total_count*100:.2f}%)")

        return labels

    def generate_percentile_labels(self, data: pd.DataFrame,
                                   top_percentile: float = 0.3,
                                   bottom_percentile: float = 0.3,
                                   column: str = 'close') -> pd.Series:
        """
        Generate labels based on percentile ranking of forward returns.

        Args:
            data: DataFrame with price data
            top_percentile: Top percentile for buy signals (e.g., 0.3 = top 30%)
            bottom_percentile: Bottom percentile for sell signals
            column: Column to use for calculation

        Returns:
            Series with percentile-based labels (2 = buy, 1 = hold, 0 = sell)
        """
        forward_returns = self.calculate_forward_returns(data, column)

        # Calculate percentile thresholds
        buy_threshold = forward_returns.quantile(1 - top_percentile)
        sell_threshold = forward_returns.quantile(bottom_percentile)

        # Create labels
        labels = pd.Series(1, index=data.index, name='label')  # Default: hold
        labels[forward_returns >= buy_threshold] = 2  # Buy
        labels[forward_returns <= sell_threshold] = 0  # Sell

        buy_count = (labels == 2).sum()
        sell_count = (labels == 0).sum()
        hold_count = (labels == 1).sum()
        total_count = labels.count()

        logger.info(f"Generated percentile labels: Buy={buy_count} ({buy_count/total_count*100:.2f}%), "
                   f"Hold={hold_count} ({hold_count/total_count*100:.2f}%), "
                   f"Sell={sell_count} ({sell_count/total_count*100:.2f}%)")

        return labels

    def generate_labels(self, data: pd.DataFrame,
                       label_type: Literal['binary', 'multiclass', 'regression', 'trend',
                                          'volatility_adjusted', 'percentile'] = 'binary',
                       **kwargs) -> pd.Series:
        """
        Generate labels based on specified type.

        Args:
            data: DataFrame with price data
            label_type: Type of labels to generate
            **kwargs: Additional arguments for specific label types

        Returns:
            Series with generated labels
        """
        logger.info(f"Generating {label_type} labels")

        if label_type == 'binary':
            return self.generate_binary_labels(data, **kwargs)
        elif label_type == 'multiclass':
            return self.generate_multiclass_labels(data, **kwargs)
        elif label_type == 'regression':
            return self.generate_regression_labels(data, **kwargs)
        elif label_type == 'trend':
            return self.generate_trend_labels(data, **kwargs)
        elif label_type == 'volatility_adjusted':
            return self.generate_volatility_adjusted_labels(data, **kwargs)
        elif label_type == 'percentile':
            return self.generate_percentile_labels(data, **kwargs)
        else:
            raise ValueError(f"Unknown label type: {label_type}")


if __name__ == "__main__":
    # Example usage
    import logging
    from ..utils.logger import setup_logger

    setup_logger(__name__, level=logging.INFO)

    # Create sample data
    dates = pd.date_range('2020-01-01', periods=100)
    np.random.seed(42)

    # Simulate price data with trend
    price = 100
    prices = []
    for _ in range(100):
        price += np.random.randn() * 2 + 0.1  # Slight upward drift
        prices.append(price)

    data = pd.DataFrame({'close': prices}, index=dates)

    # Initialize label generator
    lg = LabelGenerator(forward_period=5, threshold=0.02)

    # Generate different types of labels
    print("=" * 60)
    print("LABEL GENERATION EXAMPLES")
    print("=" * 60)

    # Binary labels
    binary_labels = lg.generate_binary_labels(data)
    print(f"\nBinary labels:\n{binary_labels.value_counts()}")

    # Multi-class labels
    multiclass_labels = lg.generate_multiclass_labels(data)
    print(f"\nMulti-class labels:\n{multiclass_labels.value_counts()}")

    # Regression labels
    regression_labels = lg.generate_regression_labels(data)
    print(f"\nRegression labels (sample):\n{regression_labels.head(10)}")

    # Trend labels
    trend_labels = lg.generate_trend_labels(data)
    print(f"\nTrend labels:\n{trend_labels.value_counts()}")

    # Percentile labels
    percentile_labels = lg.generate_percentile_labels(data, top_percentile=0.3, bottom_percentile=0.3)
    print(f"\nPercentile labels:\n{percentile_labels.value_counts()}")

