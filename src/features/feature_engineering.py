"""
Feature Engineering Pipeline

Orchestrates the creation of all features from different modules:
- Price features
- Technical indicators
- Volume features
- Volatility features
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from src.features.price_features import PriceFeatures
from src.features.technical_indicators import TechnicalIndicators
from src.features.volume_features import VolumeFeatures
from src.features.volatility_features import VolatilityFeatures
from src.features.label_generator import LabelGenerator
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class FeatureEngineering:
    """
    Main feature engineering pipeline that combines all feature generators.

    Examples:
        >>> import pandas as pd
        >>> fe = FeatureEngineering()
        >>> dates = pd.date_range('2020-01-01', periods=100)
        >>> data = pd.DataFrame({'open': range(100, 200),
        ...                      'high': range(105, 205),
        ...                      'low': range(95, 195),
        ...                      'close': range(102, 202),
        ...                      'volume': range(1000, 1100)}, index=dates)
        >>> features = fe.engineer_features(data)
        >>> len(features.columns) > 10
        True
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize FeatureEngineer with all feature generators.

        Args:
            config: Configuration dictionary for feature generation
        """
        self.config = config or {}

        # Initialize feature generators
        self.price_features = PriceFeatures()
        self.technical_indicators = TechnicalIndicators()
        self.volume_features = VolumeFeatures()
        self.volatility_features = VolatilityFeatures()

        logger.info("Initialized FeatureEngineer with all feature generators")

    def engineer_features(self, data: pd.DataFrame,
                         include_price: bool = True,
                         include_technical: bool = True,
                         include_volume: bool = True,
                         include_volatility: bool = True) -> pd.DataFrame:
        """
        Generate all features from input data.

        Args:
            data: DataFrame with OHLCV data
            include_price: Include price features
            include_technical: Include technical indicators
            include_volume: Include volume features
            include_volatility: Include volatility features

        Returns:
            DataFrame with all engineered features
        """
        logger.info("Starting feature engineering pipeline")

        feature_sets = []

        # Add original data
        feature_sets.append(data.copy())

        # Generate price features
        if include_price:
            logger.info("Generating price features")
            price_feats = self.price_features.generate_all_features(data)
            feature_sets.append(price_feats)

        # Generate technical indicators
        if include_technical:
            logger.info("Generating technical indicators")
            tech_feats = self.technical_indicators.add_all_indicators(data)
            feature_sets.append(tech_feats)

        # Generate volume features
        if include_volume:
            logger.info("Generating volume features")
            volume_feats = self.volume_features.generate_all_features(data)
            feature_sets.append(volume_feats)

        # Generate volatility features
        if include_volatility:
            logger.info("Generating volatility features")
            volatility_feats = self.volatility_features.generate_all_features(data)
            feature_sets.append(volatility_feats)

        # Combine all features
        all_features = pd.concat(feature_sets, axis=1)

        # Remove duplicate columns
        all_features = all_features.loc[:, ~all_features.columns.duplicated()]

        # Clean features (handle inf and extreme values)
        all_features = self._clean_features(all_features)

        logger.info(f"Feature engineering complete: {len(all_features.columns)} total features")

        return all_features

    def create_all_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Backward compatibility alias for engineer_features().

        Args:
            data: DataFrame with OHLCV data
            **kwargs: Additional arguments passed to engineer_features()

        Returns:
            DataFrame with all engineered features
        """
        return self.engineer_features(data, **kwargs)

    def create_feature_label_dataset(self, data: pd.DataFrame,
                                    label_type: str = 'binary',
                                    forward_period: int = 5,
                                    threshold: float = 0.02,
                                    **feature_kwargs) -> tuple[pd.DataFrame, pd.Series]:
        """
        Create complete dataset with features and labels.

        Args:
            data: DataFrame with OHLCV data
            label_type: Type of labels to generate
            forward_period: Forward period for label generation
            threshold: Threshold for classification labels
            **feature_kwargs: Additional arguments for feature engineering

        Returns:
            Tuple of (features DataFrame, labels Series)
        """
        logger.info(f"Creating feature-label dataset with {label_type} labels")

        # Engineer features
        features = self.engineer_features(data, **feature_kwargs)

        # Generate labels
        label_generator = LabelGenerator(forward_period=forward_period, threshold=threshold)
        labels = label_generator.generate_labels(data, label_type=label_type)

        # Align features and labels (remove rows with NaN labels)
        valid_indices = labels.notna()
        features = features[valid_indices]
        labels = labels[valid_indices]

        logger.info(f"Dataset created: {len(features)} samples, {len(features.columns)} features")

        return features, labels

    def select_features(self, features: pd.DataFrame,
                       method: str = 'variance',
                       threshold: float = 0.01,
                       k: Optional[int] = None) -> List[str]:
        """
        Select most important features based on various criteria.

        Args:
            features: DataFrame with features
            method: Selection method ('variance', 'correlation', 'mutual_info')
            threshold: Threshold for selection
            k: Number of top features to select (if None, use threshold)

        Returns:
            List of selected feature names
        """
        logger.info(f"Selecting features using {method} method")

        if method == 'variance':
            # Remove low variance features
            variances = features.var()
            selected = variances[variances > threshold].index.tolist()

        elif method == 'correlation':
            # Remove highly correlated features
            corr_matrix = features.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
            selected = [col for col in features.columns if col not in to_drop]

        elif method == 'mutual_info':
            # Placeholder for mutual information (requires labels)
            logger.warning("Mutual info selection requires labels, returning all features")
            selected = features.columns.tolist()

        else:
            raise ValueError(f"Unknown selection method: {method}")

        # If k is specified, select top k features by variance
        if k is not None and len(selected) > k:
            variances = features[selected].var()
            selected = variances.nlargest(k).index.tolist()

        logger.info(f"Selected {len(selected)} features out of {len(features.columns)}")

        return selected

    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Clean features by handling inf, NaN, and extreme values.

        Args:
            features: DataFrame with features

        Returns:
            DataFrame with cleaned features
        """
        logger.info("Cleaning features: handling inf, NaN, and extreme values")

        # Count issues before cleaning
        inf_count = np.isinf(features).sum().sum()
        nan_count = features.isna().sum().sum()

        if inf_count > 0:
            logger.warning(f"Found {inf_count} inf values")

        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values")

        # Replace inf with NaN
        features = features.replace([np.inf, -np.inf], np.nan)

        # Cap extreme values using percentiles (99th and 1st percentile)
        for col in features.columns:
            if features[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
                # Calculate percentiles
                q99 = features[col].quantile(0.99)
                q01 = features[col].quantile(0.01)

                # Cap extreme values
                features[col] = features[col].clip(lower=q01, upper=q99)

        # Fill remaining NaN values
        # First try forward fill, then backward fill, then fill with 0
        features = features.fillna(method='ffill')
        features = features.fillna(method='bfill')
        features = features.fillna(0)

        # Final check
        remaining_inf = np.isinf(features).sum().sum()
        remaining_nan = features.isna().sum().sum()

        if remaining_inf > 0 or remaining_nan > 0:
            logger.error(f"Still have {remaining_inf} inf and {remaining_nan} NaN values after cleaning")
        else:
            logger.info("Feature cleaning complete: all inf and NaN values handled")

        return features

    def handle_missing_values(self, features: pd.DataFrame,
                             method: str = 'forward_fill',
                             fill_value: float = 0.0) -> pd.DataFrame:
        """
        Handle missing values in features.

        Args:
            features: DataFrame with features
            method: Method for handling missing values
            fill_value: Value to use for 'constant' method

        Returns:
            DataFrame with missing values handled
        """
        logger.info(f"Handling missing values using {method} method")

        if method == 'forward_fill':
            features = features.fillna(method='ffill')
        elif method == 'backward_fill':
            features = features.fillna(method='bfill')
        elif method == 'mean':
            features = features.fillna(features.mean())
        elif method == 'median':
            features = features.fillna(features.median())
        elif method == 'constant':
            features = features.fillna(fill_value)
        elif method == 'drop':
            features = features.dropna()
        else:
            raise ValueError(f"Unknown method: {method}")

        # Fill any remaining NaNs with 0
        features = features.fillna(0)

        logger.info(f"Missing values handled: {features.shape[0]} rows, {features.shape[1]} columns")

        return features

    def get_feature_statistics(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Get statistics about the engineered features.

        Args:
            features: DataFrame with features

        Returns:
            DataFrame with feature statistics
        """
        stats = pd.DataFrame({
            'mean': features.mean(),
            'std': features.std(),
            'min': features.min(),
            'max': features.max(),
            'missing': features.isna().sum(),
            'missing_pct': features.isna().sum() / len(features) * 100
        })

        return stats


if __name__ == "__main__":
    # Example usage
    import logging
    from ..utils.logger import setup_logger

    setup_logger(__name__, level=logging.INFO)

    # Create sample data
    dates = pd.date_range('2020-01-01', periods=200)
    np.random.seed(42)

    data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(200) * 0.5),
        'high': 102 + np.cumsum(np.random.randn(200) * 0.5),
        'low': 98 + np.cumsum(np.random.randn(200) * 0.5),
        'close': 101 + np.cumsum(np.random.randn(200) * 0.5),
        'volume': np.random.randint(1000, 10000, 200)
    }, index=dates)

    # Ensure OHLC relationships
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)

    # Initialize feature engineer
    fe = FeatureEngineering()

    # Create features and labels
    features, labels = fe.create_feature_label_dataset(
        data,
        label_type='binary',
        forward_period=5,
        threshold=0.02
    )

    print("=" * 60)
    print("FEATURE ENGINEERING RESULTS")
    print("=" * 60)
    print(f"\nDataset shape: {features.shape}")
    print(f"Number of features: {len(features.columns)}")
    print(f"Number of samples: {len(features)}")
    print(f"\nLabel distribution:\n{labels.value_counts()}")

    # Get feature statistics
    stats = fe.get_feature_statistics(features)
    print(f"\nFeatures with >10% missing values:")
    print(stats[stats['missing_pct'] > 10][['missing', 'missing_pct']])

    # Feature selection
    selected = fe.select_features(features, method='variance', threshold=0.01)
    print(f"\nSelected {len(selected)} features using variance threshold")

    # Handle missing values
    features_clean = fe.handle_missing_values(features, method='forward_fill')
    print(f"\nCleaned features shape: {features_clean.shape}")
    print(f"Remaining missing values: {features_clean.isna().sum().sum()}")


