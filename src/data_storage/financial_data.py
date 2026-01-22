"""
Financial Data Class for accessing S3-stored financial data and Yahoo Finance API.

This module provides a clean interface for retrieving financial data without exposing
S3IO operations or file path structures to end users.

Author: Claude
Status: Development
"""

import logging
from typing import Union, Optional
import polars as pl
import pandas as pd
import yfinance as yf
from .s3io import S3IO


class FinancialData:
    """
    A user-friendly interface for accessing financial data from S3 and Yahoo Finance.

    This class abstracts away S3IO operations and file path management, providing
    simple methods to retrieve ticker information, financial statements, and
    historical stock prices.

    Attributes
    ----------
    _s3io : S3IO
        S3 client for reading data from S3
    _tickers_df : pl.DataFrame or None
        Cached dataframe of available tickers (lazy loaded)
    _ticker_symbols : list or None
        Cached list of ticker symbols for validation (lazy loaded)
    _bucket : str
        AWS S3 bucket name
    _profile : str
        AWS credentials profile name
    """

    def __init__(self, bucket: str, profile: str = 'default'):
        """
        Initialize the FinancialData class.

        Parameters
        ----------
        bucket : str
            AWS S3 bucket name where financial data is stored
        profile : str, default='default'
            AWS credentials profile name
        """
        self._bucket = bucket
        self._profile = profile
        self._s3io = S3IO(bucket=bucket, profile=profile)
        self._tickers_df = None
        self._ticker_symbols = None

        logging.info(f"FinancialData initialized with bucket: {bucket}, profile: {profile}")

    def list_tickers(self, as_pandas: bool = False) -> Union[pl.DataFrame, pd.DataFrame]:
        """
        List all available ticker symbols.

        This method caches the tickers dataframe after the first load to minimize
        S3 calls. Subsequent calls return the cached data.

        Parameters
        ----------
        as_pandas : bool, default=False
            If True, return pandas DataFrame. If False, return polars DataFrame.

        Returns
        -------
        pl.DataFrame or pd.DataFrame
            DataFrame containing ticker information with at least a 'Symbol' column

        Raises
        ------
        KeyError
            If the 'Symbol' column is missing from the tickers dataframe
        Exception
            If there are errors reading from S3
        """
        # Load tickers if not already cached
        if self._tickers_df is None:
            try:
                logging.info("Loading tickers from S3: stock_tracker/tickers.parq")
                self._tickers_df = self._s3io.s3_read_parquet('stock_tracker/tickers.parq')

                # Validate that Symbol column exists
                if 'Symbol' not in self._tickers_df.columns:
                    raise KeyError("Tickers dataframe is missing the required 'Symbol' column")

                # Extract and cache ticker symbols for validation
                self._ticker_symbols = self._tickers_df['Symbol'].to_list()
                logging.info(f"Loaded {len(self._ticker_symbols)} tickers")

            except Exception as e:
                logging.error(f"Failed to load tickers from S3: {e}")
                raise

        # Return in requested format
        if as_pandas:
            return self._tickers_df.to_pandas()
        else:
            return self._tickers_df

    def _validate_ticker(self, ticker: str) -> str:
        """
        Validate that a ticker symbol exists in the available tickers list.

        This is an internal helper method that ensures ticker symbols are valid
        before making expensive S3 or API calls.

        Parameters
        ----------
        ticker : str
            Ticker symbol to validate

        Returns
        -------
        str
            Uppercased ticker symbol

        Raises
        ------
        ValueError
            If the ticker symbol is not found in the available tickers list
        """
        # Ensure tickers are loaded
        if self._ticker_symbols is None:
            self.list_tickers()

        # Convert to uppercase for case-insensitive comparison
        ticker_upper = ticker.upper()

        # Validate ticker exists
        if ticker_upper not in self._ticker_symbols:
            raise ValueError(
                f"Ticker '{ticker}' not found in available tickers. "
                f"Use list_tickers() to see all available tickers."
            )

        return ticker_upper

    def get_balance_sheet(
        self,
        ticker: str,
        as_pandas: bool = False
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """
        Retrieve balance sheet data for a specific ticker from S3.

        This method always reads fresh data from S3 (no caching).

        Parameters
        ----------
        ticker : str
            Ticker symbol (case-insensitive, will be converted to uppercase)
        as_pandas : bool, default=False
            If True, return pandas DataFrame. If False, return polars DataFrame.

        Returns
        -------
        pl.DataFrame or pd.DataFrame
            Balance sheet data for the specified ticker

        Raises
        ------
        ValueError
            If the ticker symbol is not found in available tickers
        Exception
            If there are errors reading from S3 (e.g., file not found)
        """
        # Validate ticker
        ticker_upper = self._validate_ticker(ticker)

        # Construct S3 path
        path = f"balance/{ticker_upper}/balance.parq"

        try:
            logging.info(f"Reading balance sheet for {ticker_upper} from S3")
            df = self._s3io.s3_read_parquet(path)

            # Return in requested format
            if as_pandas:
                return df.to_pandas()
            else:
                return df

        except Exception as e:
            logging.error(f"Failed to read balance sheet for {ticker_upper}: {e}")
            raise

    def get_income_statement(
        self,
        ticker: str,
        as_pandas: bool = False
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """
        Retrieve income statement data for a specific ticker from S3.

        This method always reads fresh data from S3 (no caching).

        Parameters
        ----------
        ticker : str
            Ticker symbol (case-insensitive, will be converted to uppercase)
        as_pandas : bool, default=False
            If True, return pandas DataFrame. If False, return polars DataFrame.

        Returns
        -------
        pl.DataFrame or pd.DataFrame
            Income statement data for the specified ticker

        Raises
        ------
        ValueError
            If the ticker symbol is not found in available tickers
        Exception
            If there are errors reading from S3 (e.g., file not found)
        """
        # Validate ticker
        ticker_upper = self._validate_ticker(ticker)

        # Construct S3 path
        path = f"income/{ticker_upper}/income.parq"

        try:
            logging.info(f"Reading income statement for {ticker_upper} from S3")
            df = self._s3io.s3_read_parquet(path)

            # Return in requested format
            if as_pandas:
                return df.to_pandas()
            else:
                return df

        except Exception as e:
            logging.error(f"Failed to read income statement for {ticker_upper}: {e}")
            raise

    def get_cash_flow(
        self,
        ticker: str,
        as_pandas: bool = False
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """
        Retrieve cash flow statement data for a specific ticker from S3.

        This method always reads fresh data from S3 (no caching).

        Parameters
        ----------
        ticker : str
            Ticker symbol (case-insensitive, will be converted to uppercase)
        as_pandas : bool, default=False
            If True, return pandas DataFrame. If False, return polars DataFrame.

        Returns
        -------
        pl.DataFrame or pd.DataFrame
            Cash flow statement data for the specified ticker

        Raises
        ------
        ValueError
            If the ticker symbol is not found in available tickers
        Exception
            If there are errors reading from S3 (e.g., file not found)
        """
        # Validate ticker
        ticker_upper = self._validate_ticker(ticker)

        # Construct S3 path
        path = f"cash/{ticker_upper}/cash.parq"

        try:
            logging.info(f"Reading cash flow for {ticker_upper} from S3")
            df = self._s3io.s3_read_parquet(path)

            # Return in requested format
            if as_pandas:
                return df.to_pandas()
            else:
                return df

        except Exception as e:
            logging.error(f"Failed to read cash flow for {ticker_upper}: {e}")
            raise

    def get_historical_stock_prices(
        self,
        ticker: str,
        as_pandas: bool = False,
        period: str = 'max',
        interval: str = '1d'
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """
        Retrieve historical stock price data from Yahoo Finance.

        This method always fetches fresh data from Yahoo Finance (no caching).

        Parameters
        ----------
        ticker : str
            Ticker symbol (case-insensitive, will be converted to uppercase)
        as_pandas : bool, default=False
            If True, return pandas DataFrame. If False, return polars DataFrame.
        period : str, default='max'
            Data period to download. Valid periods include:
            - '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'
        interval : str, default='1d'
            Data interval. Valid intervals include:
            - '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'

        Returns
        -------
        pl.DataFrame or pd.DataFrame
            Historical stock price data with columns like Open, High, Low, Close, Volume, etc.

        Raises
        ------
        ValueError
            If the ticker symbol is not found in available tickers, or if no data is returned
        Exception
            If there are errors fetching data from Yahoo Finance
        """
        # Validate ticker
        ticker_upper = self._validate_ticker(ticker)

        try:
            logging.info(
                f"Fetching historical prices for {ticker_upper} from Yahoo Finance "
                f"(period={period}, interval={interval})"
            )

            # Fetch data from Yahoo Finance
            ticker_obj = yf.Ticker(ticker_upper)
            hist = ticker_obj.history(period=period, interval=interval)

            # Check if data was returned
            if hist.empty:
                raise ValueError(
                    f"No historical data returned from Yahoo Finance for ticker '{ticker_upper}'. "
                    f"The ticker may not exist or may not have data for the specified period."
                )

            # Return in requested format
            if as_pandas:
                return hist
            else:
                # Convert pandas DataFrame to polars
                return pl.from_pandas(hist)

        except ValueError:
            # Re-raise ValueError as-is (already has good error message)
            raise
        except Exception as e:
            logging.error(f"Failed to fetch historical prices for {ticker_upper}: {e}")
            raise
