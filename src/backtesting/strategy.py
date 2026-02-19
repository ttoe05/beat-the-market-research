"""
Trading Strategy Implementation

Base classes and concrete implementations for trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
from abc import ABC, abstractmethod
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class Strategy(ABC):
    """
    Abstract base class for trading strategies.

    Subclasses must implement the generate_signals method.

    Examples:
        >>> import pandas as pd
        >>> dates = pd.date_range('2020-01-01', periods=10)
        >>> data = pd.DataFrame({'close': range(100, 110)}, index=dates)
        >>> strategy = SimpleMovingAverageStrategy(short_window=2, long_window=5)
        >>> signals = strategy.generate_signals(data)
        >>> len(signals)
        10
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize Strategy.

        Args:
            name: Strategy name
        """
        self.name = name or self.__class__.__name__
        self.parameters = {}
        logger.info(f"Initialized strategy: {self.name}")

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals from market data.

        Args:
            data: DataFrame with market data (OHLCV)

        Returns:
            Series with trading signals:
                1 = Buy signal
                0 = Hold/No signal
               -1 = Sell signal
        """
        pass

    def set_parameters(self, **kwargs) -> None:
        """
        Set strategy parameters.

        Args:
            **kwargs: Parameter name-value pairs
        """
        self.parameters.update(kwargs)
        logger.info(f"Updated {self.name} parameters: {kwargs}")

    def get_parameters(self) -> Dict[str, Any]:
        """Get current strategy parameters."""
        return self.parameters.copy()

    def __repr__(self) -> str:
        """String representation of strategy."""
        return f"{self.name}(parameters={self.parameters})"


class SimpleMovingAverageStrategy(Strategy):
    """
    Simple Moving Average (SMA) crossover strategy.

    Generates buy signals when short-term SMA crosses above long-term SMA,
    and sell signals when it crosses below.

    Examples:
        >>> import pandas as pd
        >>> dates = pd.date_range('2020-01-01', periods=20)
        >>> prices = [100, 101, 102, 103, 104, 105, 104, 103, 102, 101,
        ...          100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        >>> data = pd.DataFrame({'close': prices}, index=dates)
        >>> strategy = SimpleMovingAverageStrategy(short_window=3, long_window=7)
        >>> signals = strategy.generate_signals(data)
        >>> len(signals)
        20
    """

    def __init__(self, short_window: int = 20, long_window: int = 50):
        """
        Initialize SMA Strategy.

        Args:
            short_window: Short-term moving average window
            long_window: Long-term moving average window
        """
        super().__init__()
        self.short_window = short_window
        self.long_window = long_window
        self.parameters = {
            'short_window': short_window,
            'long_window': long_window
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate signals based on SMA crossover.

        Args:
            data: DataFrame with 'close' column

        Returns:
            Series with trading signals
        """
        # Calculate moving averages
        short_ma = data['close'].rolling(window=self.short_window).mean()
        long_ma = data['close'].rolling(window=self.long_window).mean()

        # Initialize signals
        signals = pd.Series(0, index=data.index)

        # Generate signals
        # Buy when short MA crosses above long MA
        signals[short_ma > long_ma] = 1
        # Sell when short MA crosses below long MA
        signals[short_ma < long_ma] = -1

        # Only signal on actual crossovers (not continuous)
        # Calculate differences to detect crossovers
        signal_diff = signals.diff()
        signals = pd.Series(0, index=data.index)
        signals[signal_diff == 2] = 1   # Crossover up
        signals[signal_diff == -2] = -1  # Crossover down

        logger.info(f"Generated {(signals != 0).sum()} signals using SMA strategy "
                   f"(short={self.short_window}, long={self.long_window})")

        return signals


class MomentumStrategy(Strategy):
    """
    Momentum-based trading strategy.

    Generates buy signals when returns exceed a threshold,
    and sell signals when they fall below.

    Examples:
        >>> import pandas as pd
        >>> dates = pd.date_range('2020-01-01', periods=10)
        >>> data = pd.DataFrame({'close': [100, 102, 101, 103, 105, 104, 106, 108, 107, 109]}, index=dates)
        >>> strategy = MomentumStrategy(lookback=3, threshold=0.02)
        >>> signals = strategy.generate_signals(data)
        >>> len(signals)
        10
    """

    def __init__(self, lookback: int = 10, threshold: float = 0.02):
        """
        Initialize Momentum Strategy.

        Args:
            lookback: Lookback period for calculating momentum
            threshold: Momentum threshold for generating signals
        """
        super().__init__()
        self.lookback = lookback
        self.threshold = threshold
        self.parameters = {
            'lookback': lookback,
            'threshold': threshold
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate signals based on momentum.

        Args:
            data: DataFrame with 'close' column

        Returns:
            Series with trading signals
        """
        # Calculate momentum (percentage change over lookback period)
        momentum = data['close'].pct_change(periods=self.lookback)

        # Initialize signals
        signals = pd.Series(0, index=data.index)

        # Generate signals
        signals[momentum > self.threshold] = 1   # Buy on strong positive momentum
        signals[momentum < -self.threshold] = -1  # Sell on strong negative momentum

        logger.info(f"Generated {(signals != 0).sum()} signals using Momentum strategy "
                   f"(lookback={self.lookback}, threshold={self.threshold})")

        return signals


class MLPredictionStrategy(Strategy):
    """
    Machine Learning prediction-based strategy.

    Uses model predictions directly as trading signals.

    Examples:
        >>> import pandas as pd
        >>> dates = pd.date_range('2020-01-01', periods=10)
        >>> data = pd.DataFrame({'prediction': [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]}, index=dates)
        >>> strategy = MLPredictionStrategy(prediction_column='prediction')
        >>> signals = strategy.generate_signals(data)
        >>> signals.sum()
        5
    """

    def __init__(self, prediction_column: str = 'prediction',
                 threshold: float = 0.5,
                 hold_on_neutral: bool = True):
        """
        Initialize ML Prediction Strategy.

        Args:
            prediction_column: Column name containing model predictions
            threshold: Threshold for converting probabilities to signals
            hold_on_neutral: Whether to hold (0) on neutral predictions
        """
        super().__init__()
        self.prediction_column = prediction_column
        self.threshold = threshold
        self.hold_on_neutral = hold_on_neutral
        self.parameters = {
            'prediction_column': prediction_column,
            'threshold': threshold,
            'hold_on_neutral': hold_on_neutral
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate signals from model predictions.

        Args:
            data: DataFrame with prediction column

        Returns:
            Series with trading signals
        """
        if self.prediction_column not in data.columns:
            raise ValueError(f"Column '{self.prediction_column}' not found in data")

        predictions = data[self.prediction_column]

        # Initialize signals
        signals = pd.Series(0, index=data.index)

        # Convert predictions to signals
        # Assuming predictions are binary (0/1) or probabilities
        if predictions.max() <= 1.0 and predictions.min() >= 0.0:
            # Probability or binary predictions
            signals[predictions >= self.threshold] = 1
            if not self.hold_on_neutral:
                signals[predictions < self.threshold] = -1
        else:
            # Assume predictions are already signals (-1, 0, 1)
            signals = predictions

        logger.info(f"Generated {(signals != 0).sum()} signals from ML predictions")

        return signals


class RSIStrategy(Strategy):
    """
    Relative Strength Index (RSI) strategy.

    Generates buy signals when RSI indicates oversold conditions,
    and sell signals when it indicates overbought conditions.

    Examples:
        >>> import pandas as pd
        >>> dates = pd.date_range('2020-01-01', periods=20)
        >>> data = pd.DataFrame({'close': range(100, 120)}, index=dates)
        >>> strategy = RSIStrategy(period=14, oversold=30, overbought=70)
        >>> signals = strategy.generate_signals(data)
        >>> len(signals)
        20
    """

    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        """
        Initialize RSI Strategy.

        Args:
            period: RSI calculation period
            oversold: RSI level indicating oversold (buy signal)
            overbought: RSI level indicating overbought (sell signal)
        """
        super().__init__()
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.parameters = {
            'period': period,
            'oversold': oversold,
            'overbought': overbought
        }

    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """
        Calculate RSI indicator.

        Args:
            prices: Series of prices

        Returns:
            Series with RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate signals based on RSI.

        Args:
            data: DataFrame with 'close' column

        Returns:
            Series with trading signals
        """
        # Calculate RSI
        rsi = self.calculate_rsi(data['close'])

        # Initialize signals
        signals = pd.Series(0, index=data.index)

        # Generate signals
        signals[rsi < self.oversold] = 1   # Buy when oversold
        signals[rsi > self.overbought] = -1  # Sell when overbought

        logger.info(f"Generated {(signals != 0).sum()} signals using RSI strategy "
                   f"(period={self.period}, oversold={self.oversold}, overbought={self.overbought})")

        return signals


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("TRADING STRATEGY EXAMPLES")
    print("=" * 60)

    # Create sample data
    dates = pd.date_range('2020-01-01', periods=100)
    np.random.seed(42)
    prices = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.02))
    data = pd.DataFrame({'close': prices}, index=dates)

    print(f"\nSample data: {len(data)} days")
    print(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")

    # Test SMA Strategy
    print("\n" + "=" * 60)
    print("SIMPLE MOVING AVERAGE STRATEGY")
    print("=" * 60)
    sma_strategy = SimpleMovingAverageStrategy(short_window=10, long_window=30)
    sma_signals = sma_strategy.generate_signals(data)
    print(f"Buy signals: {(sma_signals == 1).sum()}")
    print(f"Sell signals: {(sma_signals == -1).sum()}")
    print(f"Hold signals: {(sma_signals == 0).sum()}")

    # Test Momentum Strategy
    print("\n" + "=" * 60)
    print("MOMENTUM STRATEGY")
    print("=" * 60)
    momentum_strategy = MomentumStrategy(lookback=5, threshold=0.03)
    momentum_signals = momentum_strategy.generate_signals(data)
    print(f"Buy signals: {(momentum_signals == 1).sum()}")
    print(f"Sell signals: {(momentum_signals == -1).sum()}")
    print(f"Hold signals: {(momentum_signals == 0).sum()}")

    # Test RSI Strategy
    print("\n" + "=" * 60)
    print("RSI STRATEGY")
    print("=" * 60)
    rsi_strategy = RSIStrategy(period=14, oversold=30, overbought=70)
    rsi_signals = rsi_strategy.generate_signals(data)
    print(f"Buy signals: {(rsi_signals == 1).sum()}")
    print(f"Sell signals: {(rsi_signals == -1).sum()}")
    print(f"Hold signals: {(rsi_signals == 0).sum()}")

    # Test ML Prediction Strategy
    print("\n" + "=" * 60)
    print("ML PREDICTION STRATEGY")
    print("=" * 60)
    # Add random predictions to data
    data['prediction'] = np.random.choice([0, 1], size=len(data), p=[0.6, 0.4])
    ml_strategy = MLPredictionStrategy(prediction_column='prediction')
    ml_signals = ml_strategy.generate_signals(data)
    print(f"Buy signals: {(ml_signals == 1).sum()}")
    print(f"Sell signals: {(ml_signals == -1).sum()}")
    print(f"Hold signals: {(ml_signals == 0).sum()}")
