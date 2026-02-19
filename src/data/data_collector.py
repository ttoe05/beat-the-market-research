"""
Data collection module for fetching market data from multiple sources.
Supports Yahoo Finance, Alpha Vantage, and other data providers.
"""

import yfinance as yf
import pandas as pd
from typing import Optional, List, Dict
from src.utils.logger import setup_logger
from pathlib import Path

logger = setup_logger(__name__)


class DataCollector:
    """
    Collects historical market data from various sources.

    Examples:
        collector = DataCollector(source='yahoo')
        data = collector.fetch_data('AAPL', start='2020-01-01', end='2024-12-31')
        collector.save_data(data, 'AAPL')
    """

    def __init__(self, source: str = 'yahoo', api_key: Optional[str] = None):
        """
        Initialize data collector.

        Args:
            source: Data source ('yahoo', 'alphavantage', 'iex')
            api_key: API key for paid data sources
        """
        self.source = source
        self.api_key = api_key

        logger.info(f"Initialized DataCollector with source: {source}")

    def fetch_data(
            self,
            symbol: str,
            start: str,
            end: str,
            interval: str = '1d',
            auto_adjust: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical market data.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL')
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            interval: Data interval ('1d', '1h', '5m')

        Returns:
            DataFrame with OHLCV data
        """
        if self.source == 'yahoo':
            return self._fetch_yahoo(symbol, start, end, interval, auto_adjust)
        elif self.source == 'alphavantage':
            return self._fetch_alphavantage(symbol, start, end)
        else:
            raise ValueError(f"Unsupported source: {self.source}")

    def _fetch_yahoo(
            self,
            symbol: str,
            start: str,
            end: str,
            interval: str,
            auto_adjust: bool = True
    ) -> pd.DataFrame:
        """Fetch data from Yahoo Finance."""
        try:
            logger.info(f"Fetching {symbol} from Yahoo Finance: {start} to {end}")
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end, interval=interval, auto_adjust=auto_adjust)

            if df.empty:
                raise ValueError(f"No data returned for {symbol}")

            # Standardize column names
            df.columns = df.columns.str.lower()
            df.index.name = 'date'

            logger.info(f"Successfully fetched {len(df)} rows for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise

    def _fetch_alphavantage(
            self,
            symbol: str,
            start: str,
            end: str
    ) -> pd.DataFrame:
        """Fetch data from Alpha Vantage."""
        # Implementation for Alpha Vantage API
        if not self.api_key:
            raise ValueError("API key required for Alpha Vantage")

        # Add Alpha Vantage API implementation here
        logger.info(f"Fetching {symbol} from Alpha Vantage")
        raise NotImplementedError("Alpha Vantage implementation not yet complete")

    def fetch_multiple_symbols(
            self,
            symbols: List[str],
            start: str,
            end: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols.

        Args:
            symbols: List of ticker symbols
            start: Start date
            end: End date

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        data = {}
        for symbol in symbols:
            try:
                data[symbol] = self.fetch_data(symbol, start, end)
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {str(e)}")

        return data

    def save_data(
            self,
            data: pd.DataFrame,
            symbol: str,
            output_dir: str = 'data/raw'
    ) -> None:
        """
        Save data to CSV file.

        Args:
            data: DataFrame to save
            symbol: Stock symbol
            output_dir: Output directory
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        filepath = Path(output_dir) / f"{symbol}_raw.csv"
        data.to_csv(filepath)
        logger.info(f"Saved data to {filepath}")

    def load_data(
            self,
            symbol: str,
            data_dir: str = 'data/raw'
    ) -> pd.DataFrame:
        """
        Load data from CSV file.

        Args:
            symbol: Stock symbol
            data_dir: Data directory

        Returns:
            DataFrame with loaded data
        """
        filepath = Path(data_dir) / f"{symbol}_raw.csv"
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        df = pd.read_csv(filepath, index_col='date', parse_dates=True)
        logger.info(f"Loaded {len(df)} rows from {filepath}")
        return df


if __name__ == "__main__":
    # Example usage
    collector = DataCollector(source='yahoo')
    data = collector.fetch_data('AAPL', start='2020-01-01', end='2024-12-31')
    print(data.head())
    print(data.columns)