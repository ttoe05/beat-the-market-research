# Technical indicators - calculates technical trading indicators

"""
Technical indicators module.

Implements common technical analysis indicators:
- Trend: SMA, EMA, MACD
- Momentum: RSI, Stochastic
- Volatility: Bollinger Bands, ATR
- Volume: OBV, Volume MA
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Calculate technical indicators for trading analysis.

    Examples:
        >>> import pandas as pd
        >>> dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
        >>> data = pd.DataFrame({
        ...     'open': np.random.randn(len(dates)) * 10 + 100,
        ...     'high': np.random.randn(len(dates)) * 10 + 105,
        ...     'low': np.random.randn(len(dates)) * 10 + 95,
        ...     'close': np.random.randn(len(dates)) * 10 + 100,
        ...     'volume': np.random.randint(1000000, 10000000, len(dates)),
        ... }, index=dates)
        >>> ti = TechnicalIndicators()
        >>> data_with_indicators = ti.add_all_indicators(data)
    """

    def __init__(self):
        logger.info("Initialized TechnicalIndicators")

    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """
        Simple Moving Average.

        Args:
            data: Price series
            period: Number of periods

        Returns:
            SMA series
        """
        return data.rolling(window=period).mean()

    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """
        Exponential Moving Average.

        Args:
            data: Price series
            period: Number of periods

        Returns:
            EMA series
        """
        return data.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index.

        Args:
            data: Price series
            period: RSI period (default 14)

        Returns:
            RSI series (0-100)
        """
        # Calculate price changes
        delta = data.diff()

        # Separate gains and losses
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        # Calculate RS and RSI, handling division by zero
        # When loss is 0, RS should be very high, making RSI approach 100
        rs = gain / loss.replace(0, 1e-10)  # Replace 0 with very small number
        rsi = 100 - (100 / (1 + rs))

        # Replace any inf values with 100 (max RSI)
        rsi = rsi.replace([np.inf, -np.inf], 100)

        return rsi

    @staticmethod
    def macd(
            data: pd.Series,
            fast: int = 12,
            slow: int = 26,
            signal: int = 9
    ) -> pd.DataFrame:
        """
        Moving Average Convergence Divergence.

        Args:
            data: Price series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period

        Returns:
            DataFrame with macd, signal, and histogram
        """
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return pd.DataFrame({
            'macd': macd_line,
            'macd_signal': signal_line,
            'macd_histogram': histogram
        })

    @staticmethod
    def bollinger_bands(
            data: pd.Series,
            period: int = 20,
            std_dev: float = 2.0
    ) -> pd.DataFrame:
        """
        Bollinger Bands.

        Args:
            data: Price series
            period: Moving average period
            std_dev: Number of standard deviations

        Returns:
            DataFrame with upper, middle, and lower bands
        """
        middle = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return pd.DataFrame({
            'bb_upper': upper,
            'bb_middle': middle,
            'bb_lower': lower
        })

    @staticmethod
    def atr(
            high: pd.Series,
            low: pd.Series,
            close: pd.Series,
            period: int = 14
    ) -> pd.Series:
        """
        Average True Range.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period

        Returns:
            ATR series
        """
        # True Range components
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        # True Range is the maximum of the three
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR is the moving average of TR
        atr = tr.rolling(window=period).mean()

        return atr

    @staticmethod
    def stochastic(
            high: pd.Series,
            low: pd.Series,
            close: pd.Series,
            period: int = 14,
            smooth_k: int = 3,
            smooth_d: int = 3
    ) -> pd.DataFrame:
        """
        Stochastic Oscillator.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: Lookback period
            smooth_k: %K smoothing period
            smooth_d: %D smoothing period

        Returns:
            DataFrame with %K and %D
        """
        # Lowest low and highest high over period
        low_min = low.rolling(window=period).min()
        high_max = high.rolling(window=period).max()

        # Fast %K (prevent division by zero)
        denominator = high_max - low_min
        denominator = denominator.replace(0, 1e-10)  # Replace 0 with very small number
        k_fast = 100 * (close - low_min) / denominator

        # Clip to 0-100 range and handle any inf values
        k_fast = k_fast.clip(0, 100).replace([np.inf, -np.inf], 50)

        # Slow %K (smoothed)
        k_slow = k_fast.rolling(window=smooth_k).mean()

        # %D (smoothed %K)
        d = k_slow.rolling(window=smooth_d).mean()

        return pd.DataFrame({
            'stoch_k': k_slow,
            'stoch_d': d
        })

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        On-Balance Volume.

        Args:
            close: Close prices
            volume: Volume

        Returns:
            OBV series
        """
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv

    @staticmethod
    def adx(
            high: pd.Series,
            low: pd.Series,
            close: pd.Series,
            period: int = 14
    ) -> pd.Series:
        """
        Average Directional Index.

        Measures trend strength (0-100).
        Values > 25 indicate strong trend.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ADX period

        Returns:
            ADX series
        """
        # Calculate directional movement
        high_diff = high.diff()
        low_diff = -low.diff()

        pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

        # Calculate ATR
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        # Calculate directional indicators
        pos_di = 100 * (pos_dm.rolling(window=period).mean() / atr)
        neg_di = 100 * (neg_dm.rolling(window=period).mean() / atr)

        # Calculate ADX
        dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
        adx = dx.rolling(window=period).mean()

        return adx

    @staticmethod
    def ppo(
            self,
            data: pd.Series,
            fast: int = 12,
            slow: int = 26
    ) -> pd.Series:
        """
        Percentage Price Oscillator.

        Args:
            data: Price series
            fast: Fast EMA period
            slow: Slow EMA period

        Returns:
            PPO series (% difference between EMAs)
        """
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()

        ppo = ((ema_fast - ema_slow) / ema_slow) * 100
        return ppo

    @staticmethod
    def roc(
            self,
            data: pd.Series,
            period: int = 12
    ) -> pd.Series:
        """
        Rate of Change.

        Args:
            data: Price series
            period: Lookback period

        Returns:
            ROC series (% change)
        """
        roc = ((data - data.shift(period)) / data.shift(period)) * 100
        return roc

    @staticmethod
    def z_score(
            self,
            data: pd.Series,
            period: int = 20
    ) -> pd.Series:
        """
        Z-Score.

        Args:
            data: Price series
            period: Lookback period
        Returns:
            Z-Score series (number of std devs from mean)
        """
        rolling_mean = data.rolling(window=period).mean()
        rolling_std = data.rolling(window=period).std()

        # Handle division by zero
        z_score = (data - rolling_mean) / rolling_std.replace(0, 1e-10)
        return z_score

    def add_all_indicators(
            self,
            df: pd.DataFrame,
            include_all: bool = True
    ) -> pd.DataFrame:
        """
        Add all technical indicators to DataFrame.

        Args:
            df: DataFrame with OHLCV data
            include_all: If True, add all indicators

        Returns:
            DataFrame with added indicator columns
        """
        result = df.copy()

        logger.info("Adding technical indicators...")

        # Moving averages
        for period in [5, 10, 20, 50, 200]:
            result[f'sma_{period}'] = self.sma(df['close'], period)
            result[f'ema_{period}'] = self.ema(df['close'], period)

        # RSI
        result['rsi'] = self.rsi(df['close'])
        result['rsi_14'] = self.rsi(df['close'], 14)
        result['rsi_7'] = self.rsi(df['close'], 7)

        # MACD
        macd_df = self.macd(df['close'])
        result = pd.concat([result, macd_df], axis=1)

        # Bollinger Bands
        bb_df = self.bollinger_bands(df['close'])
        result = pd.concat([result, bb_df], axis=1)

        # ATR
        result['atr'] = self.atr(df['high'], df['low'], df['close'])
        result['atr_14'] = self.atr(df['high'], df['low'], df['close'], 14)

        # Stochastic
        stoch_df = self.stochastic(df['high'], df['low'], df['close'])
        result = pd.concat([result, stoch_df], axis=1)

        # OBV
        result['obv'] = self.obv(df['close'], df['volume'])

        # ADX
        result['adx'] = self.adx(df['high'], df['low'], df['close'])

        # Volume moving average
        result['volume_sma_20'] = self.sma(df['volume'], 20)

        #PPO
        result['ppo'] = self.ppo(df['close'])

        # ROC
        result['roc'] = self.roc(df['close'])

        # Z-Score
        result['z_score'] = self.z_score(df['close'])

        logger.info(f"Added {len(result.columns) - len(df.columns)} indicators")

        return result




# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create sample data
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
    data = pd.DataFrame({
        'open': np.random.randn(len(dates)) * 10 + 100,
        'high': np.random.randn(len(dates)) * 10 + 105,
        'low': np.random.randn(len(dates)) * 10 + 95,
        'close': np.random.randn(len(dates)) * 10 + 100,
        'volume': np.random.randint(1000000, 10000000, len(dates)),
    }, index=dates)

    # Add indicators
    ti = TechnicalIndicators()
    data_with_indicators = ti.add_all_indicators(data)

    print(f"\nOriginal columns: {list(data.columns)}")
    print(f"With indicators: {len(data_with_indicators.columns)} columns")
    print(f"\nNew indicators added:")
    new_cols = set(data_with_indicators.columns) - set(data.columns)
    for col in sorted(new_cols):
        print(f"  - {col}")