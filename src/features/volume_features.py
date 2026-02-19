"""
Volume Features Module

Generates volume-based features for trading ML models including:
- Volume changes and ratios
- Volume moving averages
- On-Balance Volume (OBV)
- Volume price trends
- Accumulation/Distribution indicators
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class VolumeFeatures:
    """
    Generate volume-based features for trading analysis.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> vf = VolumeFeatures()
        >>> dates = pd.date_range('2020-01-01', periods=10)
        >>> data = pd.DataFrame({'open': range(100, 110),
        ...                      'high': range(105, 115),
        ...                      'low': range(95, 105),
        ...                      'close': range(102, 112),
        ...                      'volume': range(1000, 2000, 100)}, index=dates)
        >>> features = vf.generate_all_features(data)
        >>> 'volume_change' in features.columns
        True
    """

    def __init__(self):
        """Initialize VolumeFeatures generator."""
        logger.info("Initialized VolumeFeatures")

    def calculate_volume_changes(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume changes and ratios.

        Args:
            data: DataFrame with 'volume' column

        Returns:
            DataFrame with volume change features
        """
        features = pd.DataFrame(index=data.index)

        # Volume change (absolute and percentage)
        features['volume_change'] = data['volume'].diff()
        features['volume_change_pct'] = data['volume'].pct_change()

        # Volume ratio (current vs previous) - prevent division by zero
        features['volume_ratio'] = data['volume'] / data['volume'].shift(1).replace(0, 1e-10)

        # Log volume change - prevent log of 0 or negative
        volume_ratio = data['volume'] / data['volume'].shift(1)
        volume_ratio = volume_ratio.clip(lower=1e-10)
        features['log_volume_change'] = np.log(volume_ratio)

        # Replace any inf values
        features = features.replace([np.inf, -np.inf], np.nan)

        logger.debug("Calculated volume change features")
        return features

    def calculate_volume_ma(self, data: pd.DataFrame,
                           windows: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calculate volume moving averages.

        Args:
            data: DataFrame with 'volume' column
            windows: List of window sizes for moving averages

        Returns:
            DataFrame with volume MA features
        """
        if windows is None:
            windows = [5, 10, 20, 50]

        features = pd.DataFrame(index=data.index)

        for window in windows:
            # Volume moving average
            features[f'volume_ma_{window}'] = data['volume'].rolling(window=window).mean()

            # Volume relative to MA - prevent division by zero
            features[f'volume_to_ma_{window}'] = data['volume'] / features[f'volume_ma_{window}'].replace(0, 1e-10) - 1

            # Volume above/below MA (binary)
            features[f'volume_above_ma_{window}'] = (data['volume'] > features[f'volume_ma_{window}']).astype(int)

        logger.debug(f"Calculated volume MA for windows: {windows}")
        return features

    def calculate_volume_std(self, data: pd.DataFrame,
                            windows: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calculate volume standard deviation and z-scores.

        Args:
            data: DataFrame with 'volume' column
            windows: List of window sizes

        Returns:
            DataFrame with volume volatility features
        """
        if windows is None:
            windows = [10, 20, 50]

        features = pd.DataFrame(index=data.index)

        for window in windows:
            # Volume standard deviation
            vol_std = data['volume'].rolling(window=window).std()
            vol_mean = data['volume'].rolling(window=window).mean()

            features[f'volume_std_{window}'] = vol_std

            # Volume z-score (standardized volume) - prevent division by zero
            features[f'volume_zscore_{window}'] = (data['volume'] - vol_mean) / vol_std.replace(0, 1e-10)

            # Coefficient of variation - prevent division by zero
            features[f'volume_cv_{window}'] = vol_std / vol_mean.replace(0, 1e-10)

        # Replace any inf values
        features = features.replace([np.inf, -np.inf], np.nan)

        logger.debug(f"Calculated volume std for windows: {windows}")
        return features

    def calculate_obv(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate On-Balance Volume (OBV).

        OBV is a cumulative indicator that adds volume on up days and
        subtracts volume on down days.

        Args:
            data: DataFrame with 'close' and 'volume' columns

        Returns:
            DataFrame with OBV features
        """
        features = pd.DataFrame(index=data.index)

        # Calculate price direction
        price_change = data['close'].diff()

        # OBV calculation
        obv = pd.Series(0, index=data.index, dtype=float)
        obv[price_change > 0] = data['volume'][price_change > 0]
        obv[price_change < 0] = -data['volume'][price_change < 0]
        obv[price_change == 0] = 0

        features['obv'] = obv.cumsum()

        # OBV change
        features['obv_change'] = features['obv'].diff()
        features['obv_change_pct'] = features['obv'].pct_change()

        # OBV moving averages
        for window in [10, 20]:
            features[f'obv_ma_{window}'] = features['obv'].rolling(window=window).mean()
            features[f'obv_to_ma_{window}'] = features['obv'] / features[f'obv_ma_{window}'] - 1

        logger.debug("Calculated OBV features")
        return features

    def calculate_volume_price_trend(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Volume Price Trend (VPT) indicator.

        VPT is similar to OBV but considers the magnitude of price change.

        Args:
            data: DataFrame with 'close' and 'volume' columns

        Returns:
            DataFrame with VPT features
        """
        features = pd.DataFrame(index=data.index)

        # VPT calculation
        price_change_pct = data['close'].pct_change()
        vpt = (price_change_pct * data['volume']).cumsum()

        features['vpt'] = vpt
        features['vpt_change'] = vpt.diff()

        # VPT moving averages
        for window in [10, 20]:
            features[f'vpt_ma_{window}'] = vpt.rolling(window=window).mean()

        logger.debug("Calculated VPT features")
        return features

    def calculate_accumulation_distribution(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Accumulation/Distribution Line (A/D Line).

        A/D Line is a cumulative indicator that uses volume and price
        to assess whether a stock is being accumulated or distributed.

        Args:
            data: DataFrame with OHLC and volume columns

        Returns:
            DataFrame with A/D features
        """
        features = pd.DataFrame(index=data.index)

        # Money Flow Multiplier
        clv = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
        clv = clv.fillna(0)  # Handle division by zero

        # Money Flow Volume
        money_flow_volume = clv * data['volume']

        # A/D Line (cumulative)
        features['ad_line'] = money_flow_volume.cumsum()
        features['ad_change'] = features['ad_line'].diff()

        # A/D moving averages
        for window in [10, 20]:
            features[f'ad_ma_{window}'] = features['ad_line'].rolling(window=window).mean()

        logger.debug("Calculated A/D Line features")
        return features

    def calculate_volume_price_correlation(self, data: pd.DataFrame,
                                          windows: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calculate correlation between volume and price changes.

        Args:
            data: DataFrame with 'close' and 'volume' columns
            windows: List of window sizes for correlation

        Returns:
            DataFrame with volume-price correlation features
        """
        if windows is None:
            windows = [10, 20, 50]

        features = pd.DataFrame(index=data.index)

        price_change = data['close'].pct_change()
        volume_change = data['volume'].pct_change()

        for window in windows:
            # Rolling correlation
            features[f'volume_price_corr_{window}'] = price_change.rolling(window=window).corr(volume_change)

        logger.debug(f"Calculated volume-price correlation for windows: {windows}")
        return features

    def calculate_volume_momentum(self, data: pd.DataFrame,
                                  periods: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calculate volume momentum indicators.

        Args:
            data: DataFrame with 'volume' column
            periods: List of periods for momentum calculation

        Returns:
            DataFrame with volume momentum features
        """
        if periods is None:
            periods = [5, 10, 20]

        features = pd.DataFrame(index=data.index)

        for period in periods:
            # Volume ROC (Rate of Change)
            features[f'volume_roc_{period}'] = ((data['volume'] - data['volume'].shift(period)) /
                                                data['volume'].shift(period) * 100)

            # Volume momentum
            features[f'volume_momentum_{period}'] = data['volume'] - data['volume'].shift(period)

        logger.debug(f"Calculated volume momentum for periods: {periods}")
        return features

    def calculate_volume_weighted_price(self, data: pd.DataFrame,
                                       windows: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calculate Volume Weighted Average Price (VWAP).

        Args:
            data: DataFrame with OHLC and volume columns
            windows: List of window sizes for VWAP

        Returns:
            DataFrame with VWAP features
        """
        if windows is None:
            windows = [10, 20]

        features = pd.DataFrame(index=data.index)

        # Typical price
        typical_price = (data['high'] + data['low'] + data['close']) / 3

        for window in windows:
            # VWAP calculation
            vwap = (typical_price * data['volume']).rolling(window=window).sum() / data['volume'].rolling(window=window).sum()
            features[f'vwap_{window}'] = vwap

            # Price distance from VWAP
            features[f'price_to_vwap_{window}'] = (data['close'] - vwap) / vwap

        logger.debug(f"Calculated VWAP for windows: {windows}")
        return features

    def calculate_force_index(self, data: pd.DataFrame,
                             periods: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calculate Force Index.

        Force Index combines price change and volume to measure buying/selling pressure.

        Args:
            data: DataFrame with 'close' and 'volume' columns
            periods: List of EMA periods

        Returns:
            DataFrame with Force Index features
        """
        if periods is None:
            periods = [13]

        features = pd.DataFrame(index=data.index)

        # Raw Force Index
        force_index = data['close'].diff() * data['volume']
        features['force_index_raw'] = force_index

        # Smoothed Force Index (EMA)
        for period in periods:
            features[f'force_index_{period}'] = force_index.ewm(span=period, adjust=False).mean()

        logger.debug(f"Calculated Force Index for periods: {periods}")
        return features

    def calculate_ease_of_movement(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Ease of Movement (EMV).

        EMV relates price change to volume, showing how easily price moves.

        Args:
            data: DataFrame with OHLC and volume columns
            period: Period for smoothing

        Returns:
            DataFrame with EMV features
        """
        features = pd.DataFrame(index=data.index)

        # Distance moved
        high_low_mid = (data['high'] + data['low']) / 2
        distance_moved = high_low_mid.diff()

        # Box ratio
        box_ratio = (data['volume'] / 1000000) / (data['high'] - data['low'])
        box_ratio = box_ratio.replace([np.inf, -np.inf], np.nan).fillna(0)

        # EMV
        emv = distance_moved / box_ratio
        emv = emv.replace([np.inf, -np.inf], np.nan).fillna(0)

        features['emv'] = emv
        features[f'emv_{period}'] = emv.rolling(window=period).mean()

        logger.debug(f"Calculated EMV with period {period}")
        return features

    def generate_all_features(self, data: pd.DataFrame,
                            ma_windows: Optional[List[int]] = None,
                            std_windows: Optional[List[int]] = None,
                            corr_windows: Optional[List[int]] = None,
                            momentum_periods: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Generate all volume-based features.

        Args:
            data: DataFrame with OHLCV columns
            ma_windows: Windows for volume moving averages
            std_windows: Windows for volume standard deviation
            corr_windows: Windows for volume-price correlation
            momentum_periods: Periods for volume momentum

        Returns:
            DataFrame with all volume features
        """
        logger.info("Generating all volume features")

        if ma_windows is None:
            ma_windows = [5, 10, 20, 50]
        if std_windows is None:
            std_windows = [10, 20, 50]
        if corr_windows is None:
            corr_windows = [10, 20, 50]
        if momentum_periods is None:
            momentum_periods = [5, 10, 20]

        feature_sets = [
            self.calculate_volume_changes(data),
            self.calculate_volume_ma(data, ma_windows),
            self.calculate_volume_std(data, std_windows),
            self.calculate_obv(data),
            self.calculate_volume_price_trend(data),
            self.calculate_accumulation_distribution(data),
            self.calculate_volume_price_correlation(data, corr_windows),
            self.calculate_volume_momentum(data, momentum_periods),
            self.calculate_volume_weighted_price(data),
            self.calculate_force_index(data),
            self.calculate_ease_of_movement(data)
        ]

        # Combine all features
        all_features = pd.concat(feature_sets, axis=1)

        logger.info(f"Generated {len(all_features.columns)} volume features")
        return all_features


if __name__ == "__main__":
    # Example usage
    import logging
    from ..utils.logger import setup_logger

    setup_logger(__name__, level=logging.INFO)

    # Create sample data
    dates = pd.date_range('2020-01-01', periods=100)
    np.random.seed(42)

    # Simulate realistic price and volume data
    base_price = 100
    base_volume = 1000000

    prices = []
    volumes = []

    for i in range(100):
        # Price with trend and noise
        price = base_price + i * 0.1 + np.random.randn() * 2
        prices.append(price)

        # Volume correlated with price changes
        if i > 0:
            price_change = abs(price - prices[i-1])
            volume = base_volume * (1 + price_change/10) * (1 + np.random.randn() * 0.2)
        else:
            volume = base_volume
        volumes.append(max(volume, 100000))  # Ensure positive volume

    data = pd.DataFrame({
        'open': [p - np.random.rand() for p in prices],
        'high': [p + np.random.rand() * 2 for p in prices],
        'low': [p - np.random.rand() * 2 for p in prices],
        'close': prices,
        'volume': volumes
    }, index=dates)

    # Ensure OHLC relationships
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)

    # Generate features
    vf = VolumeFeatures()
    features = vf.generate_all_features(data)

    print("=" * 60)
    print("VOLUME FEATURES GENERATION")
    print("=" * 60)
    print(f"\nGenerated {len(features.columns)} features")
    print(f"\nFeature categories:")
    print(f"  - Volume changes: volume_change, volume_ratio, etc.")
    print(f"  - Volume MAs: volume_ma_5, volume_ma_10, etc.")
    print(f"  - OBV: obv, obv_change, etc.")
    print(f"  - VPT: vpt, vpt_change, etc.")
    print(f"  - A/D Line: ad_line, ad_change, etc.")
    print(f"  - VWAP: vwap_10, price_to_vwap_20, etc.")
    print(f"  - Force Index: force_index_13, etc.")
    print(f"  - EMV: emv, emv_14")

    print(f"\nSample features (last 5 rows):")
    print(features[['volume_change_pct', 'obv', 'vpt', 'ad_line', 'force_index_13']].tail())

    print(f"\nFeature statistics:")
    print(features.describe().T[['mean', 'std', 'min', 'max']].head(10))

