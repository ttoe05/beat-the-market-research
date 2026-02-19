# Data splitting module - splits data into train/validation/test sets
"""
Data splitting module for time series data.

IMPORTANT: Never shuffle time series data!
Always use chronological splits.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DataSplitter:
    """
    Splits time series data chronologically.

    Examples:
        >>> import pandas as pd
        >>> dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
        >>> data = pd.DataFrame({'value': range(len(dates))}, index=dates)
        >>> splitter = DataSplitter(train_ratio=0.7, val_ratio=0.15)
        >>> train, val, test = splitter.split(data)
    """

    def __init__(
            self,
            train_ratio: float = 0.7,
            val_ratio: float = 0.15,
            test_ratio: Optional[float] = None
    ):
        """
        Initialize splitter.

        Args:
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing (auto-calculated if None)
        """
        if test_ratio is None:
            test_ratio = 1.0 - train_ratio - val_ratio

        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("Ratios must sum to 1.0")

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        logger.info(
            f"Initialized DataSplitter: "
            f"train={train_ratio:.1%}, val={val_ratio:.1%}, test={test_ratio:.1%}"
        )

    def split(
            self,
            data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data chronologically into train/val/test.

        Args:
            data: DataFrame to split (must have DatetimeIndex)

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")

        # Sort by date to ensure chronological order
        data = data.sort_index()

        n = len(data)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        train = data.iloc[:train_end]
        val = data.iloc[train_end:val_end]
        test = data.iloc[val_end:]

        logger.info(
            f"Split data: train={len(train)} ({len(train) / n:.1%}), "
            f"val={len(val)} ({len(val) / n:.1%}), "
            f"test={len(test)} ({len(test) / n:.1%})"
        )
        logger.info(f"Train period: {train.index[0]} to {train.index[-1]}")
        logger.info(f"Val period: {val.index[0]} to {val.index[-1]}")
        logger.info(f"Test period: {test.index[0]} to {test.index[-1]}")

        return train, val, test

    def split_by_date(
            self,
            data: pd.DataFrame,
            train_end: str,
            val_end: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data by specific dates.

        Args:
            data: DataFrame to split
            train_end: End date for training set (YYYY-MM-DD)
            val_end: End date for validation set (YYYY-MM-DD)

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        data = data.sort_index()

        train = data[:train_end]
        val = data[train_end:val_end]
        test = data[val_end:]

        logger.info(f"Split by date:")
        logger.info(f"  Train: {len(train)} rows until {train_end}")
        logger.info(f"  Val: {len(val)} rows from {train_end} to {val_end}")
        logger.info(f"  Test: {len(test)} rows from {val_end} onwards")

        return train, val, test

    def walk_forward_split(
            self,
            data: pd.DataFrame,
            train_size: int,
            test_size: int,
            step: int = 1
    ) -> list[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create walk-forward splits for cross-validation.

        Useful for time series cross-validation where we want to
        maintain temporal order and test on future data.

        Args:
            data: DataFrame to split
            train_size: Number of samples in training set
            test_size: Number of samples in test set
            step: Step size between splits

        Returns:
            List of (train, test) tuples
        """
        data = data.sort_index()
        splits = []

        for i in range(0, len(data) - train_size - test_size + 1, step):
            train = data.iloc[i:i + train_size]
            test = data.iloc[i + train_size:i + train_size + test_size]
            splits.append((train, test))

        logger.info(
            f"Created {len(splits)} walk-forward splits "
            f"(train_size={train_size}, test_size={test_size}, step={step})"
        )

        return splits


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create sample data
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
    data = pd.DataFrame({
        'close': np.random.randn(len(dates)) * 10 + 100,
        'volume': np.random.randint(1000000, 10000000, len(dates)),
    }, index=dates)

    # Split by ratio
    splitter = DataSplitter(train_ratio=0.7, val_ratio=0.15)
    train, val, test = splitter.split(data)

    print(f"\nTrain shape: {train.shape}")
    print(f"Val shape: {val.shape}")
    print(f"Test shape: {test.shape}")

    # Split by date
    train2, val2, test2 = splitter.split_by_date(
        data,
        train_end='2023-01-01',
        val_end='2024-01-01'
    )
