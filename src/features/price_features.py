"""
Price Features Module

Generates price-based features for trading ML models including:
- Returns (simple, log, forward)
- Price changes and momentum
- Price ranges and spreads
- Moving averages and crossovers
"""

import pandas as pd
import numpy as np
from typing import Optional, List
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class PriceFeatures:
    """
    Generate price-based features for trading analysis.

    Examples:
        >>> import pandas as pd
        >>> pf = PriceFeatures()
        >>> dates = pd.date_range('2020-01-01', periods=5)
        >>> data = pd.DataFrame({'open': [100, 101, 102, 103, 104],
        ...                      'high': [105, 106, 107, 108, 109],
        ...                      'low': [95, 96, 97, 98, 99],
        ...                      'close': [102, 103, 104, 105, 106],
        ...                      'volume': [1000, 1100, 1200, 1300, 1400]}, index=dates)
        >>> features = pf.generate_all_features(data)
        >>> 'returns' in features.columns
        True
    """

    def __init__(self):
        """Initialize PriceFeatures generator."""
        logger.info("Initialized PriceFeatures")

    def calculate_returns(self, data: pd.DataFrame, periods: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        """
        Calculate simple and log returns for multiple periods.

        Args:
            data: DataFrame with 'close' column
            periods: List of periods for return calculation

        Returns:
            DataFrame with return features
        """
        features = pd.DataFrame(index=data.index)

        for period in periods:
            # Simple returns
            features[f'returns_{period}'] = data['close'].pct_change(period)

            # Log returns (prevent log of 0 or negative)
            price_ratio = data['close'] / data['close'].shift(period)
            # Replace 0 or negative ratios with very small positive number
            price_ratio = price_ratio.clip(lower=1e-10)
            features[f'log_returns_{period}'] = np.log(price_ratio)

        # Default single-period return
        features['returns'] = data['close'].pct_change()

        # Replace any inf values
        features = features.replace([np.inf, -np.inf], np.nan)

        logger.debug(f"Calculated returns for periods: {periods}")
        return features

    def calculate_forward_returns(self, data: pd.DataFrame, periods: List[int] = [1, 5, 10]) -> pd.DataFrame:
        """
        Calculate forward-looking returns (for label generation).

        Args:
            data: DataFrame with 'close' column
            periods: List of forward periods

        Returns:
            DataFrame with forward return features
        """
        features = pd.DataFrame(index=data.index)

        for period in periods:
            features[f'forward_returns_{period}'] = data['close'].pct_change(period).shift(-period)

        logger.debug(f"Calculated forward returns for periods: {periods}")
        return features

    def calculate_price_changes(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate various price changes and differentials.

        Args:
            data: DataFrame with OHLC columns

        Returns:
            DataFrame with price change features
        """
        features = pd.DataFrame(index=data.index)

        # Absolute price changes
        features['price_change'] = data['close'] - data['open']
        features['price_change_pct'] = (data['close'] - data['open']) / data['open'].replace(0, 1e-10)

        # High-Low range
        features['high_low_range'] = data['high'] - data['low']
        features['high_low_range_pct'] = (data['high'] - data['low']) / data['low'].replace(0, 1e-10)

        # Close position within range (prevent division by zero)
        denominator = (data['high'] - data['low']).replace(0, 1e-10)
        features['close_position'] = (data['close'] - data['low']) / denominator
        features['close_position'] = features['close_position'].clip(0, 1)  # Ensure 0-1 range

        # Gap (difference between today's open and yesterday's close)
        features['gap'] = data['open'] - data['close'].shift(1)
        features['gap_pct'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1).replace(0, 1e-10)

        # Replace any inf values
        features = features.replace([np.inf, -np.inf], np.nan)

        logger.debug("Calculated price change features")
        return features

    def calculate_moving_averages(self, data: pd.DataFrame,
                                  windows: List[int] = [5, 10, 20, 50, 200]) -> pd.DataFrame:
        """
        Calculate moving averages and related features.

        Args:
            data: DataFrame with 'close' column
            windows: List of window sizes for moving averages

        Returns:
            DataFrame with moving average features
        """
        features = pd.DataFrame(index=data.index)

        for window in windows:
            # Simple Moving Average
            features[f'sma_{window}'] = data['close'].rolling(window=window).mean()

            # Exponential Moving Average
            features[f'ema_{window}'] = data['close'].ewm(span=window, adjust=False).mean()

            # Price relative to MA (prevent division by zero)
            features[f'price_to_sma_{window}'] = data['close'] / features[f'sma_{window}'].replace(0, 1e-10) - 1
            features[f'price_to_ema_{window}'] = data['close'] / features[f'ema_{window}'].replace(0, 1e-10) - 1

        logger.debug(f"Calculated moving averages for windows: {windows}")
        return features

    def calculate_ma_crossovers(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate moving average crossover signals.

        Args:
            data: DataFrame with 'close' column

        Returns:
            DataFrame with crossover features
        """
        features = pd.DataFrame(index=data.index)

        # Common crossover pairs
        crossover_pairs = [(5, 20), (10, 50), (20, 50), (50, 200)]

        for fast, slow in crossover_pairs:
            fast_ma = data['close'].rolling(window=fast).mean()
            slow_ma = data['close'].rolling(window=slow).mean()

            # Difference between MAs
            features[f'ma_diff_{fast}_{slow}'] = fast_ma - slow_ma
            features[f'ma_diff_pct_{fast}_{slow}'] = (fast_ma - slow_ma) / slow_ma.replace(0, 1e-10)

            # Crossover signal (1 = bullish, -1 = bearish, 0 = no cross)
            prev_diff = (fast_ma - slow_ma).shift(1)
            curr_diff = fast_ma - slow_ma
            features[f'ma_cross_{fast}_{slow}'] = np.where(
                (prev_diff < 0) & (curr_diff > 0), 1,  # Bullish cross
                np.where((prev_diff > 0) & (curr_diff < 0), -1, 0)  # Bearish cross
            )

        # Replace any inf values
        features = features.replace([np.inf, -np.inf], np.nan)

        logger.debug(f"Calculated MA crossovers for pairs: {crossover_pairs}")
        return features

    def calculate_momentum(self, data: pd.DataFrame, periods: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Calculate price momentum features.

        Args:
            data: DataFrame with 'close' column
            periods: List of periods for momentum calculation

        Returns:
            DataFrame with momentum features
        """
        features = pd.DataFrame(index=data.index)

        for period in periods:
            # Rate of Change (ROC)
            features[f'roc_{period}'] = ((data['close'] - data['close'].shift(period)) /
                                         data['close'].shift(period).replace(0, 1e-10) * 100)

            # Momentum
            features[f'momentum_{period}'] = data['close'] - data['close'].shift(period)

        # Replace any inf values
        features = features.replace([np.inf, -np.inf], np.nan)

        logger.debug(f"Calculated momentum for periods: {periods}")
        return features

    def calculate_price_levels(self, data: pd.DataFrame, windows: List[int] = [20, 50]) -> pd.DataFrame:
        """
        Calculate price levels and extremes.

        Args:
            data: DataFrame with OHLC columns
            windows: List of window sizes

        Returns:
            DataFrame with price level features
        """
        features = pd.DataFrame(index=data.index)

        for window in windows:
            # Highest high and lowest low
            features[f'highest_high_{window}'] = data['high'].rolling(window=window).max()
            features[f'lowest_low_{window}'] = data['low'].rolling(window=window).min()

            # Price position relative to range (prevent division by zero)
            range_size = features[f'highest_high_{window}'] - features[f'lowest_low_{window}']
            range_size = range_size.replace(0, 1e-10)
            features[f'price_position_{window}'] = ((data['close'] - features[f'lowest_low_{window}']) /
                                                     range_size)
            features[f'price_position_{window}'] = features[f'price_position_{window}'].clip(0, 1)

        # Replace any inf values
        features = features.replace([np.inf, -np.inf], np.nan)

        logger.debug(f"Calculated price levels for windows: {windows}")
        return features

    def generate_price_lags(self, data: pd.DataFrame, lags: int,
                            smooth: bool = False,
                            smooth_window: int = 3) -> pd.DataFrame:
        """
        Generate lagged close price features with optional EWM smoothing.

        Args:
            data: DataFrame with 'close' column
            lags: Number of lags to generate
            smooth: Whether to apply EWM smoothing to each lag
            smooth_window: Span for the EWM smoothing (used when smooth=True)

        Returns:
            DataFrame with close price lag features. If smooth=False, columns are
            named close_lag_1 ... close_lag_q. If smooth=True, columns are named
            close_lag_ewm_1 ... close_lag_ewm_q.
        """
        features = pd.DataFrame(index=data.index)

        for lag in range(1, lags + 1):
            lagged = data['close'].shift(lag)
            if smooth:
                features[f'close_lag_ewm_{lag}'] = lagged.ewm(span=smooth_window, adjust=False).mean()
            else:
                features[f'close_lag_{lag}'] = lagged

        logger.debug(f"Generated {lags} close price lags (smooth={smooth}, smooth_window={smooth_window})")
        return features

    def generate_all_features(self, data: pd.DataFrame,
                            return_periods: List[int] = [1, 5, 10, 20],
                            ma_windows: List[int] = [5, 10, 20, 50, 200],
                            momentum_periods: List[int] = [5, 10, 20],
                            set_lags: bool = False,
                            lags: int = 1,
                            smooth: bool = False,
                            smooth_window: int = 3) -> pd.DataFrame:
        """
        Generate all price-based features.

        Args:
            data: DataFrame with OHLC columns
            return_periods: Periods for return calculation
            ma_windows: Windows for moving averages
            momentum_periods: Periods for momentum calculation
            set_lags: Whether to include lagged close price features
            lags: Number of close price lags to generate (used when set_lags=True)
            smooth: Whether to apply EWM smoothing to the price lags (used when set_lags=True)
            smooth_window: Span for the EWM smoothing (used when set_lags=True and smooth=True)

        Returns:
            DataFrame with all price features
        """
        logger.info("Generating all price features")

        feature_sets = [
            self.calculate_returns(data, return_periods),
            self.calculate_price_changes(data),
            self.calculate_moving_averages(data, ma_windows),
            self.calculate_ma_crossovers(data),
            self.calculate_momentum(data, momentum_periods),
            self.calculate_price_levels(data)
        ]

        if set_lags:
            feature_sets.append(self.generate_price_lags(data, lags, smooth, smooth_window))

        # Combine all features
        all_features = pd.concat(feature_sets, axis=1)

        logger.info(f"Generated {len(all_features.columns)} price features")
        return all_features


if __name__ == "__main__":
    # Example usage
    from ..utils.logger import setup_logger
    import logging

    setup_logger(__name__, level=logging.INFO)

    # Create sample data
    dates = pd.date_range('2020-01-01', periods=100)
    np.random.seed(42)

    data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'high': 102 + np.cumsum(np.random.randn(100) * 0.5),
        'low': 98 + np.cumsum(np.random.randn(100) * 0.5),
        'close': 101 + np.cumsum(np.random.randn(100) * 0.5),
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)

    # Ensure OHLC relationships are correct
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)

    # Generate features
    pf = PriceFeatures()
    features = pf.generate_all_features(data)

    print(f"Generated {len(features.columns)} features")
    print(f"\nFeature columns:\n{features.columns.tolist()}")
    print(f"\nSample features:\n{features.tail()}")

