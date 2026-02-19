# Data validation module - validates data quality and integrity
"""
Data validation module to ensure data quality and integrity.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validates market data for quality and consistency.

    Checks:
    - Required columns present
    - Data types correct
    - No invalid values (negative prices, etc.)
    - Reasonable value ranges
    - Temporal consistency

    Examples:
        >>> import pandas as pd
        >>> validator = DataValidator(strict=False)
        >>> dates = pd.date_range('2020-01-01', periods=1)
        >>> df = pd.DataFrame({'open': [100], 'high': [105], 'low': [95], 'close': [102], 'volume': [1000]}, index=dates)
        >>> is_valid, errors = validator.validate(df)
        >>> is_valid
        True
    """

    REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']

    def __init__(self, strict: bool = True):
        """
        Initialize validator.

        Args:
            strict: If True, raise exceptions on validation failures
        """
        self.strict = strict
        logger.info(f"Initialized DataValidator (strict={strict})")

    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate DataFrame.

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check required columns
        errors.extend(self._check_required_columns(df))

        # Check data types
        errors.extend(self._check_data_types(df))

        # Check for invalid values
        errors.extend(self._check_invalid_values(df))

        # Check OHLC relationships
        errors.extend(self._check_ohlc_relationships(df))

        # Check for data gaps
        errors.extend(self._check_data_gaps(df))

        is_valid = len(errors) == 0

        if not is_valid:
            logger.warning(f"Validation failed with {len(errors)} errors")
            for error in errors:
                logger.warning(f"  - {error}")

            if self.strict:
                raise ValueError(f"Data validation failed: {errors}")
        else:
            logger.info("Data validation passed")

        return is_valid, errors

    def _check_required_columns(self, df: pd.DataFrame) -> List[str]:
        """Check if all required columns are present."""
        errors = []
        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)

        if missing:
            errors.append(f"Missing required columns: {missing}")

        return errors

    def _check_data_types(self, df: pd.DataFrame) -> List[str]:
        """Check if data types are correct."""
        errors = []

        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    errors.append(f"Column '{col}' is not numeric")

        if not isinstance(df.index, pd.DatetimeIndex):
            errors.append("Index is not DatetimeIndex")

        return errors

    def _check_invalid_values(self, df: pd.DataFrame) -> List[str]:
        """Check for invalid values like negative prices."""
        errors = []

        # Check for negative prices
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df.columns:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    errors.append(
                        f"Found {negative_count} negative values in '{col}'"
                    )

        # Check for zero prices
        for col in price_columns:
            if col in df.columns:
                zero_count = (df[col] == 0).sum()
                if zero_count > 0:
                    errors.append(
                        f"Found {zero_count} zero values in '{col}'"
                    )

        # Check for zero volume
        if 'volume' in df.columns:
            zero_volume = (df['volume'] == 0).sum()
            if zero_volume > 0:
                logger.warning(
                    f"Found {zero_volume} zero volume values (may be valid)"
                )

        # Check for NaN/inf values
        if df.isnull().any().any():
            nan_counts = df.isnull().sum()
            nan_cols = nan_counts[nan_counts > 0]
            errors.append(f"Found NaN values: {nan_cols.to_dict()}")

        if np.isinf(df.select_dtypes(include=[np.number])).any().any():
            errors.append("Found infinite values in data")

        return errors

    def _check_ohlc_relationships(self, df: pd.DataFrame) -> List[str]:
        """
        Check OHLC price relationships.

        Rules:
        - High >= Open, Close, Low
        - Low <= Open, Close, High
        """
        errors = []

        required = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required):
            return errors  # Skip if columns missing

        # High should be >= Low
        invalid = df['high'] < df['low']
        if invalid.any():
            errors.append(
                f"Found {invalid.sum()} rows where high < low"
            )

        # High should be >= Open
        invalid = df['high'] < df['open']
        if invalid.any():
            errors.append(
                f"Found {invalid.sum()} rows where high < open"
            )

        # High should be >= Close
        invalid = df['high'] < df['close']
        if invalid.any():
            errors.append(
                f"Found {invalid.sum()} rows where high < close"
            )

        # Low should be <= Open
        invalid = df['low'] > df['open']
        if invalid.any():
            errors.append(
                f"Found {invalid.sum()} rows where low > open"
            )

        # Low should be <= Close
        invalid = df['low'] > df['close']
        if invalid.any():
            errors.append(
                f"Found {invalid.sum()} rows where low > close"
            )

        return errors

    def _check_data_gaps(self, df: pd.DataFrame) -> List[str]:
        """Check for unusual gaps in time series."""
        errors = []

        if len(df) < 2:
            return errors

        # Calculate time differences
        time_diffs = df.index.to_series().diff()

        # Identify large gaps (more than 30 days for daily data)
        large_gaps = time_diffs[time_diffs > pd.Timedelta(days=30)]

        if len(large_gaps) > 0:
            logger.warning(
                f"Found {len(large_gaps)} large time gaps (>30 days)"
            )
            # This is a warning, not an error

        return errors


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create valid data
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    valid_data = pd.DataFrame({
        'open': 100.0,
        'high': 105.0,
        'low': 95.0,
        'close': 102.0,
        'volume': 1000000,
    }, index=dates)

    validator = DataValidator(strict=False)
    is_valid, errors = validator.validate(valid_data)
    print(f"Valid data: {is_valid}")

    # Create invalid data (high < low)
    invalid_data = valid_data.copy()
    invalid_data.loc[invalid_data.index[0], 'high'] = 90.0

    is_valid_2, errors_2 = validator.validate(invalid_data)
    print(f"\nInvalid data: {is_valid_2}")
    print(f"Errors: {errors_2}")
