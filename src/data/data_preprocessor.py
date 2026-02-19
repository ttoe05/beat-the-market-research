# Data preprocessing module - cleans and prepares raw data
"""
Data preprocessing module for cleaning and preparing market data.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocesses raw market data for feature engineering.

    Tasks:
    - Handle missing values
    - Remove outliers
    - Normalize/scale data
    - Handle stock splits and dividends
    - Add derived columns

    Examples:
        preprocessor = DataPreprocessor()
        clean_data = preprocessor.preprocess(data)
    """

    def __init__(
            self,
            handle_missing: str = 'ffill',
            remove_outliers: bool = True,
            outlier_std: float = 5.0
    ):
        """
        Initialize preprocessor.

        Args:
            handle_missing: Method for handling missing values ('ffill', 'drop', 'interpolate')
            remove_outliers: Whether to remove outliers
            outlier_std: Number of standard deviations for outlier detection
        """
        self.handle_missing = handle_missing
        self.remove_outliers = remove_outliers
        self.outlier_std = outlier_std
        logger.info("Initialized DataPreprocessor")

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Main preprocessing pipeline.

        Args:
            data: Raw market data DataFrame

        Returns:
            Preprocessed DataFrame
        """
        df = data.copy()

        logger.info(f"Starting preprocessing: {len(df)} rows")

        # 1. Handle missing values
        df = self._handle_missing_values(df)

        # 2. Remove duplicates
        df = self._remove_duplicates(df)

        # 3. Handle outliers
        if self.remove_outliers:
            df = self._handle_outliers(df)

        # 4. Add derived columns
        df = self._add_derived_columns(df)

        # 5. Handle inf values (important for derived columns)
        df = self._handle_inf_values(df)

        # 6. Sort by date
        df = df.sort_index()

        logger.info(f"Preprocessing complete: {len(df)} rows")

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the data."""
        missing_count = df.isnull().sum().sum()

        if missing_count > 0:
            logger.warning(f"Found {missing_count} missing values")

            if self.handle_missing == 'ffill':
                df = df.fillna(method='ffill')
            elif self.handle_missing == 'drop':
                df = df.dropna()
            elif self.handle_missing == 'interpolate':
                df = df.interpolate(method='linear')

            # Fill any remaining NaNs at the start
            df = df.fillna(method='bfill')

        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        initial_len = len(df)
        df = df[~df.index.duplicated(keep='first')]

        removed = initial_len - len(df)
        if removed > 0:
            logger.warning(f"Removed {removed} duplicate rows")

        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and handle outliers using z-score method.

        Only applies to price columns to avoid removing legitimate
        extreme values in technical indicators.
        """
        price_columns = ['open', 'high', 'low', 'close']

        for col in price_columns:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()

                # Calculate z-scores
                z_scores = np.abs((df[col] - mean) / std)

                # Identify outliers
                outliers = z_scores > self.outlier_std
                outlier_count = outliers.sum()

                if outlier_count > 0:
                    logger.warning(
                        f"Found {outlier_count} outliers in {col} "
                        f"(>{self.outlier_std} std)"
                    )

                    # Cap outliers instead of removing
                    upper_bound = mean + (self.outlier_std * std)
                    lower_bound = mean - (self.outlier_std * std)
                    df.loc[df[col] > upper_bound, col] = upper_bound
                    df.loc[df[col] < lower_bound, col] = lower_bound

        return df

    def _handle_inf_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle infinite values in the data.

        Replaces inf/-inf with NaN, then fills with appropriate values.
        """
        # Check for inf values
        inf_count = np.isinf(df).sum().sum()

        if inf_count > 0:
            logger.warning(f"Found {inf_count} inf values, replacing with NaN")

            # Replace inf with NaN
            df = df.replace([np.inf, -np.inf], np.nan)

            # Fill NaN values using forward fill, then backward fill, then 0
            df = df.fillna(method='ffill')
            df = df.fillna(method='bfill')
            df = df.fillna(0)

        return df

    def _add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add useful derived columns."""
        # Daily returns
        if 'close' in df.columns:
            df['returns'] = df['close'].pct_change()

        # Price range
        if 'high' in df.columns and 'low' in df.columns:
            df['range'] = df['high'] - df['low']
            df['range_pct'] = (df['range'] / df['close']) * 100

        # Volume changes
        if 'volume' in df.columns:
            df['volume_change'] = df['volume'].pct_change()

        # Gap (difference between today's open and yesterday's close)
        if 'open' in df.columns and 'close' in df.columns:
            df['gap'] = df['open'] - df['close'].shift(1)
            df['gap_pct'] = (df['gap'] / df['close'].shift(1)) * 100

        return df

    def check_data_quality(self, df: pd.DataFrame) -> dict[str, any]:
        """
        Check data quality and return statistics.

        Returns:
            Dictionary with quality metrics
        """
        quality_report = {
            'total_rows': len(df),
            'date_range': (df.index.min(), df.index.max()),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_dates': df.index.duplicated().sum(),
            'columns': list(df.columns),
        }

        # Check for data gaps (missing trading days)
        date_diff = df.index.to_series().diff()
        gaps = date_diff[date_diff > pd.Timedelta(days=7)]  # More than a week
        quality_report['large_gaps'] = len(gaps)

        return quality_report


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create sample data
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    data = pd.DataFrame({
        'open': np.random.randn(len(dates)) * 10 + 100,
        'high': np.random.randn(len(dates)) * 10 + 105,
        'low': np.random.randn(len(dates)) * 10 + 95,
        'close': np.random.randn(len(dates)) * 10 + 100,
        'volume': np.random.randint(1000000, 10000000, len(dates)),
    }, index=dates)

    # Preprocess
    preprocessor = DataPreprocessor()
    clean_data = preprocessor.preprocess(data)

    # Check quality
    quality = preprocessor.check_data_quality(clean_data)
    print("Data Quality Report:")
    for key, value in quality.items():
        print(f"  {key}: {value}")
