"""
Volatility Features Module

Generates volatility-based features for trading ML models including:
- Historical volatility
- Average True Range (ATR)
- Bollinger Bands
- Keltner Channels
- Volatility ratios and comparisons
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class VolatilityFeatures:
    """
    Generate volatility-based features for trading analysis.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> vf = VolatilityFeatures()
        >>> dates = pd.date_range('2020-01-01', periods=20)
        >>> data = pd.DataFrame({'open': range(100, 120),
        ...                      'high': range(105, 125),
        ...                      'low': range(95, 115),
        ...                      'close': range(102, 122),
        ...                      'volume': range(1000, 1020)}, index=dates)
        >>> features = vf.generate_all_features(data)
        >>> 'historical_volatility_20' in features.columns
        True
    """

    def __init__(self):
        """Initialize VolatilityFeatures generator."""
        logger.info("Initialized VolatilityFeatures")

    def calculate_historical_volatility(self, data: pd.DataFrame,
                                       windows: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calculate historical volatility using standard deviation of returns.

        Args:
            data: DataFrame with 'close' column
            windows: List of window sizes for volatility calculation

        Returns:
            DataFrame with historical volatility features
        """
        if windows is None:
            windows = [5, 10, 20, 50]

        features = pd.DataFrame(index=data.index)

        # Calculate returns
        returns = data['close'].pct_change()

        for window in windows:
            # Historical volatility (annualized)
            features[f'historical_volatility_{window}'] = returns.rolling(window=window).std() * np.sqrt(252)

            # Non-annualized volatility
            features[f'volatility_{window}'] = returns.rolling(window=window).std()

            # Volatility change
            features[f'volatility_change_{window}'] = features[f'volatility_{window}'].pct_change()

        logger.debug(f"Calculated historical volatility for windows: {windows}")
        return features

    def calculate_atr(self, data: pd.DataFrame,
                     periods: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calculate Average True Range (ATR).

        ATR measures market volatility by decomposing the entire range of an asset.

        Args:
            data: DataFrame with OHLC columns
            periods: List of periods for ATR calculation

        Returns:
            DataFrame with ATR features
        """
        if periods is None:
            periods = [14, 20]

        features = pd.DataFrame(index=data.index)

        # True Range calculation
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        features['true_range'] = true_range

        # ATR calculation (exponential moving average of true range)
        for period in periods:
            features[f'atr_{period}'] = true_range.ewm(span=period, adjust=False).mean()

            # ATR percentage (ATR relative to price) - prevent division by zero
            features[f'atr_pct_{period}'] = features[f'atr_{period}'] / data['close'].replace(0, 1e-10) * 100

            # ATR change
            features[f'atr_change_{period}'] = features[f'atr_{period}'].pct_change()

        # Replace any inf values
        features = features.replace([np.inf, -np.inf], np.nan)

        logger.debug(f"Calculated ATR for periods: {periods}")
        return features

    def calculate_bollinger_bands(self, data: pd.DataFrame,
                                  window: int = 20,
                                  num_std: float = 2.0) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.

        Bollinger Bands consist of a middle band (SMA) and upper/lower bands
        at specified standard deviations.

        Args:
            data: DataFrame with 'close' column
            window: Window size for moving average
            num_std: Number of standard deviations for bands

        Returns:
            DataFrame with Bollinger Band features
        """
        features = pd.DataFrame(index=data.index)

        # Middle band (SMA)
        sma = data['close'].rolling(window=window).mean()
        std = data['close'].rolling(window=window).std()

        features[f'bb_middle_{window}'] = sma
        features[f'bb_upper_{window}'] = sma + (std * num_std)
        features[f'bb_lower_{window}'] = sma - (std * num_std)

        # Bandwidth (distance between upper and lower bands)
        features[f'bb_width_{window}'] = features[f'bb_upper_{window}'] - features[f'bb_lower_{window}']
        features[f'bb_width_pct_{window}'] = features[f'bb_width_{window}'] / sma.replace(0, 1e-10) * 100

        # %B (price position within bands) - prevent division by zero
        bb_width = features[f'bb_width_{window}'].replace(0, 1e-10)
        features[f'bb_pct_{window}'] = ((data['close'] - features[f'bb_lower_{window}']) / bb_width)

        # Distance from bands - prevent division by zero
        features[f'price_to_bb_upper_{window}'] = (data['close'] - features[f'bb_upper_{window}']) / data['close'].replace(0, 1e-10)
        features[f'price_to_bb_lower_{window}'] = (data['close'] - features[f'bb_lower_{window}']) / data['close'].replace(0, 1e-10)

        # Squeeze (narrow bands indicate low volatility)
        features[f'bb_squeeze_{window}'] = (features[f'bb_width_{window}'] <
                                            features[f'bb_width_{window}'].rolling(window=100).quantile(0.2)).astype(int)

        # Replace any inf values
        features = features.replace([np.inf, -np.inf], np.nan)

        logger.debug(f"Calculated Bollinger Bands with window={window}, std={num_std}")
        return features

    def calculate_keltner_channels(self, data: pd.DataFrame,
                                   window: int = 20,
                                   atr_period: int = 10,
                                   multiplier: float = 2.0) -> pd.DataFrame:
        """
        Calculate Keltner Channels.

        Keltner Channels use ATR instead of standard deviation for bands.

        Args:
            data: DataFrame with OHLC columns
            window: Window size for EMA
            atr_period: Period for ATR calculation
            multiplier: Multiplier for ATR

        Returns:
            DataFrame with Keltner Channel features
        """
        features = pd.DataFrame(index=data.index)

        # Middle line (EMA)
        ema = data['close'].ewm(span=window, adjust=False).mean()

        # Calculate ATR
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.ewm(span=atr_period, adjust=False).mean()

        features[f'kc_middle_{window}'] = ema
        features[f'kc_upper_{window}'] = ema + (atr * multiplier)
        features[f'kc_lower_{window}'] = ema - (atr * multiplier)

        # Width and position - prevent division by zero
        features[f'kc_width_{window}'] = features[f'kc_upper_{window}'] - features[f'kc_lower_{window}']
        kc_width = features[f'kc_width_{window}'].replace(0, 1e-10)
        features[f'kc_pct_{window}'] = ((data['close'] - features[f'kc_lower_{window}']) / kc_width)

        # Replace any inf values
        features = features.replace([np.inf, -np.inf], np.nan)

        logger.debug(f"Calculated Keltner Channels with window={window}")
        return features

    def calculate_parkinson_volatility(self, data: pd.DataFrame,
                                       windows: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calculate Parkinson's volatility estimator.

        Uses high-low range to estimate volatility, more efficient than close-to-close.

        Args:
            data: DataFrame with 'high' and 'low' columns
            windows: List of window sizes

        Returns:
            DataFrame with Parkinson volatility features
        """
        if windows is None:
            windows = [10, 20]

        features = pd.DataFrame(index=data.index)

        # Log of high/low ratio
        log_hl = np.log(data['high'] / data['low'])

        for window in windows:
            # Parkinson volatility
            features[f'parkinson_vol_{window}'] = np.sqrt(
                (1 / (4 * window * np.log(2))) * (log_hl ** 2).rolling(window=window).sum()
            ) * np.sqrt(252)  # Annualized

        logger.debug(f"Calculated Parkinson volatility for windows: {windows}")
        return features

    def calculate_garman_klass_volatility(self, data: pd.DataFrame,
                                         windows: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calculate Garman-Klass volatility estimator.

        Uses OHLC data for more accurate volatility estimation.

        Args:
            data: DataFrame with OHLC columns
            windows: List of window sizes

        Returns:
            DataFrame with Garman-Klass volatility features
        """
        if windows is None:
            windows = [10, 20]

        features = pd.DataFrame(index=data.index)

        log_hl = np.log(data['high'] / data['low'])
        log_co = np.log(data['close'] / data['open'])

        for window in windows:
            # Garman-Klass volatility
            features[f'garman_klass_vol_{window}'] = np.sqrt(
                (0.5 * (log_hl ** 2).rolling(window=window).mean()) -
                (2 * np.log(2) - 1) * (log_co ** 2).rolling(window=window).mean()
            ) * np.sqrt(252)  # Annualized

        logger.debug(f"Calculated Garman-Klass volatility for windows: {windows}")
        return features

    def calculate_volatility_ratios(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volatility ratios comparing different periods.

        Args:
            data: DataFrame with 'close' column

        Returns:
            DataFrame with volatility ratio features
        """
        features = pd.DataFrame(index=data.index)

        returns = data['close'].pct_change()

        # Short-term vs long-term volatility ratios
        vol_5 = returns.rolling(window=5).std()
        vol_20 = returns.rolling(window=20).std()
        vol_50 = returns.rolling(window=50).std()

        features['vol_ratio_5_20'] = vol_5 / vol_20
        features['vol_ratio_5_50'] = vol_5 / vol_50
        features['vol_ratio_20_50'] = vol_20 / vol_50

        logger.debug("Calculated volatility ratios")
        return features

    def calculate_realized_volatility(self, data: pd.DataFrame,
                                      windows: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calculate realized volatility (square root of sum of squared returns).

        Args:
            data: DataFrame with 'close' column
            windows: List of window sizes

        Returns:
            DataFrame with realized volatility features
        """
        if windows is None:
            windows = [5, 10, 20]

        features = pd.DataFrame(index=data.index)

        returns = data['close'].pct_change()

        for window in windows:
            # Realized volatility
            features[f'realized_vol_{window}'] = np.sqrt(
                (returns ** 2).rolling(window=window).sum()
            ) * np.sqrt(252 / window)  # Annualized

        logger.debug(f"Calculated realized volatility for windows: {windows}")
        return features

    def calculate_volatility_breakout(self, data: pd.DataFrame,
                                      window: int = 20,
                                      num_std: float = 2.0) -> pd.DataFrame:
        """
        Calculate volatility breakout signals.

        Args:
            data: DataFrame with 'close' column
            window: Window for moving average
            num_std: Number of standard deviations for breakout

        Returns:
            DataFrame with breakout features
        """
        features = pd.DataFrame(index=data.index)

        sma = data['close'].rolling(window=window).mean()
        std = data['close'].rolling(window=window).std()

        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)

        # Breakout signals
        features[f'vol_breakout_upper_{window}'] = (data['close'] > upper_band).astype(int)
        features[f'vol_breakout_lower_{window}'] = (data['close'] < lower_band).astype(int)
        features[f'vol_breakout_{window}'] = (features[f'vol_breakout_upper_{window}'] -
                                              features[f'vol_breakout_lower_{window}'])

        logger.debug(f"Calculated volatility breakout with window={window}")
        return features

    def calculate_ulcer_index(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Ulcer Index (downside volatility measure).

        Args:
            data: DataFrame with 'close' column
            period: Period for calculation

        Returns:
            DataFrame with Ulcer Index features
        """
        features = pd.DataFrame(index=data.index)

        # Calculate percentage drawdown from highest high
        rolling_max = data['close'].rolling(window=period).max()
        drawdown = ((data['close'] - rolling_max) / rolling_max) * 100

        # Ulcer Index
        features[f'ulcer_index_{period}'] = np.sqrt((drawdown ** 2).rolling(window=period).mean())

        logger.debug(f"Calculated Ulcer Index with period={period}")
        return features

    def calculate_volatility_clustering(self, data: pd.DataFrame,
                                       window: int = 20) -> pd.DataFrame:
        """
        Calculate features to detect volatility clustering (GARCH-like).

        Args:
            data: DataFrame with 'close' column
            window: Window for calculation

        Returns:
            DataFrame with volatility clustering features
        """
        features = pd.DataFrame(index=data.index)

        returns = data['close'].pct_change()
        squared_returns = returns ** 2

        # Moving average of squared returns (proxy for conditional variance)
        features[f'vol_cluster_{window}'] = squared_returns.rolling(window=window).mean()

        # Volatility persistence (autocorrelation of squared returns)
        features[f'vol_persistence_{window}'] = squared_returns.rolling(window=window).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
        )

        logger.debug(f"Calculated volatility clustering with window={window}")
        return features

    def generate_all_features(self, data: pd.DataFrame,
                            hv_windows: Optional[List[int]] = None,
                            atr_periods: Optional[List[int]] = None,
                            bb_windows: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Generate all volatility-based features.

        Args:
            data: DataFrame with OHLCV columns
            hv_windows: Windows for historical volatility
            atr_periods: Periods for ATR
            bb_windows: Windows for Bollinger Bands

        Returns:
            DataFrame with all volatility features
        """
        logger.info("Generating all volatility features")

        if hv_windows is None:
            hv_windows = [5, 10, 20, 50]
        if atr_periods is None:
            atr_periods = [14, 20]
        if bb_windows is None:
            bb_windows = [20]

        feature_sets = [
            self.calculate_historical_volatility(data, hv_windows),
            self.calculate_atr(data, atr_periods),
            self.calculate_parkinson_volatility(data),
            self.calculate_garman_klass_volatility(data),
            self.calculate_volatility_ratios(data),
            self.calculate_realized_volatility(data),
            self.calculate_ulcer_index(data)
        ]

        # Add Bollinger Bands for each window
        for window in bb_windows:
            feature_sets.append(self.calculate_bollinger_bands(data, window=window))

        # Add Keltner Channels
        feature_sets.append(self.calculate_keltner_channels(data))

        # Add volatility breakout
        feature_sets.append(self.calculate_volatility_breakout(data))

        # Add volatility clustering
        feature_sets.append(self.calculate_volatility_clustering(data))

        # Combine all features
        all_features = pd.concat(feature_sets, axis=1)

        logger.info(f"Generated {len(all_features.columns)} volatility features")
        return all_features


if __name__ == "__main__":
    # Example usage
    import logging
    from ..utils.logger import setup_logger

    setup_logger(__name__, level=logging.INFO)

    # Create sample data
    dates = pd.date_range('2020-01-01', periods=100)
    np.random.seed(42)

    # Simulate price data with varying volatility
    base_price = 100
    prices = [base_price]

    for i in range(1, 100):
        # Add volatility clustering (higher volatility after high volatility)
        if i > 20 and abs(prices[i-1] - prices[i-2]) > 2:
            volatility = 3  # High volatility
        else:
            volatility = 1  # Normal volatility

        price_change = np.random.randn() * volatility
        new_price = prices[-1] + price_change
        prices.append(max(new_price, 50))  # Ensure price stays positive

    data = pd.DataFrame({
        'open': [p - np.random.rand() for p in prices],
        'high': [p + np.random.rand() * 3 for p in prices],
        'low': [p - np.random.rand() * 3 for p in prices],
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)

    # Ensure OHLC relationships
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)

    # Generate features
    vf = VolatilityFeatures()
    features = vf.generate_all_features(data)

    print("=" * 60)
    print("VOLATILITY FEATURES GENERATION")
    print("=" * 60)
    print(f"\nGenerated {len(features.columns)} features")
    print(f"\nFeature categories:")
    print(f"  - Historical Volatility: historical_volatility_20, etc.")
    print(f"  - ATR: atr_14, atr_pct_14, etc.")
    print(f"  - Bollinger Bands: bb_upper_20, bb_width_20, bb_pct_20, etc.")
    print(f"  - Keltner Channels: kc_upper_20, kc_width_20, etc.")
    print(f"  - Advanced Estimators: parkinson_vol_20, garman_klass_vol_20")
    print(f"  - Volatility Ratios: vol_ratio_5_20, etc.")
    print(f"  - Realized Volatility: realized_vol_20, etc.")
    print(f"  - Ulcer Index: ulcer_index_14")

    print(f"\nSample features (last 5 rows):")
    print(features[['historical_volatility_20', 'atr_14', 'bb_width_20', 'vol_ratio_5_20', 'ulcer_index_14']].tail())

    print(f"\nFeature statistics:")
    print(features.describe().T[['mean', 'std', 'min', 'max']].head(10))

