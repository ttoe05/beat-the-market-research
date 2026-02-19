"""
Risk Management Module

Manages position sizing, risk controls, and portfolio risk.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class RiskManager:
    """
    Manages risk controls and position sizing for trading.

    Examples:
        >>> config = {'max_position_size': 0.2, 'max_portfolio_risk': 0.02}
        >>> risk_manager = RiskManager(config=config)
        >>> position_size = risk_manager.calculate_position_size(100.0, 100000.0, 50000.0)
        >>> position_size <= 20000.0  # 20% of portfolio
        True
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize RiskManager.

        Args:
            config: Configuration dictionary with risk parameters
        """
        # Default configuration
        default_config = {
            'max_position_size': 0.25,      # Max 25% of portfolio in single position
            'max_portfolio_risk': 0.02,     # Max 2% portfolio risk per trade
            'stop_loss_pct': 0.05,          # 5% stop loss
            'max_drawdown': 0.20,           # Max 20% drawdown before stopping
            'max_leverage': 1.0,            # No leverage by default
            'position_sizing_method': 'fixed_fraction',  # or 'kelly', 'volatility'
            'volatility_window': 20,        # Window for volatility calculation
            'risk_free_rate': 0.02          # Annual risk-free rate
        }

        # Merge with provided config
        self.config = default_config.copy()
        if config:
            self.config.update(config)

        # Risk tracking
        self.peak_value = 0
        self.current_drawdown = 0
        self.risk_violations = []

        logger.info(f"Initialized RiskManager with config: {self.config}")

    def calculate_position_size(self, price: float, portfolio_value: float,
                                available_cash: float,
                                volatility: Optional[float] = None) -> float:
        """
        Calculate position size based on risk parameters.

        Args:
            price: Current price per share
            portfolio_value: Total portfolio value
            available_cash: Available cash for trading
            volatility: Optional volatility estimate for position sizing

        Returns:
            Dollar amount to invest in position
        """
        method = self.config['position_sizing_method']

        if method == 'fixed_fraction':
            position_size = self._fixed_fraction_sizing(portfolio_value)

        elif method == 'volatility':
            if volatility is None:
                logger.warning("Volatility not provided, using fixed fraction")
                position_size = self._fixed_fraction_sizing(portfolio_value)
            else:
                position_size = self._volatility_based_sizing(portfolio_value, volatility)

        elif method == 'kelly':
            # Simplified Kelly criterion (requires win rate and avg win/loss)
            # For now, fall back to fixed fraction
            logger.warning("Kelly criterion not fully implemented, using fixed fraction")
            position_size = self._fixed_fraction_sizing(portfolio_value)

        else:
            logger.warning(f"Unknown sizing method '{method}', using fixed fraction")
            position_size = self._fixed_fraction_sizing(portfolio_value)

        # Apply constraints
        position_size = min(position_size, available_cash)
        position_size = min(position_size,
                           portfolio_value * self.config['max_position_size'])

        logger.debug(f"Position size calculated: ${position_size:,.2f} "
                    f"({position_size/portfolio_value*100:.1f}% of portfolio)")

        return position_size

    def _fixed_fraction_sizing(self, portfolio_value: float) -> float:
        """
        Fixed fraction position sizing.

        Args:
            portfolio_value: Total portfolio value

        Returns:
            Position size in dollars
        """
        return portfolio_value * self.config['max_position_size']

    def _volatility_based_sizing(self, portfolio_value: float,
                                 volatility: float) -> float:
        """
        Volatility-based position sizing.

        Higher volatility = smaller position size.

        Args:
            portfolio_value: Total portfolio value
            volatility: Annualized volatility

        Returns:
            Position size in dollars
        """
        # Target risk per trade
        target_risk = portfolio_value * self.config['max_portfolio_risk']

        # Adjust position size based on volatility
        # Higher volatility -> smaller position
        volatility_adjusted_fraction = self.config['max_position_size'] / (1 + volatility)

        position_size = portfolio_value * volatility_adjusted_fraction

        # Ensure we don't exceed risk limits
        max_position = target_risk / (volatility * np.sqrt(252))
        position_size = min(position_size, max_position)

        return position_size

    def check_stop_loss(self, entry_price: float, current_price: float,
                       position_type: str = 'long') -> bool:
        """
        Check if stop loss is triggered.

        Args:
            entry_price: Entry price of position
            current_price: Current market price
            position_type: 'long' or 'short'

        Returns:
            True if stop loss is triggered
        """
        stop_loss_pct = self.config['stop_loss_pct']

        if position_type == 'long':
            loss_pct = (entry_price - current_price) / entry_price
        else:  # short
            loss_pct = (current_price - entry_price) / entry_price

        if loss_pct >= stop_loss_pct:
            logger.warning(f"Stop loss triggered: {loss_pct*100:.2f}% loss")
            return True

        return False

    def check_max_drawdown(self, current_value: float) -> bool:
        """
        Check if maximum drawdown is exceeded.

        Args:
            current_value: Current portfolio value

        Returns:
            True if max drawdown is exceeded
        """
        # Update peak value
        if current_value > self.peak_value:
            self.peak_value = current_value

        # Calculate current drawdown
        if self.peak_value > 0:
            self.current_drawdown = (self.peak_value - current_value) / self.peak_value

            if self.current_drawdown > self.config['max_drawdown']:
                logger.error(f"Maximum drawdown exceeded: {self.current_drawdown*100:.2f}%")
                self.risk_violations.append({
                    'type': 'max_drawdown',
                    'value': self.current_drawdown,
                    'threshold': self.config['max_drawdown']
                })
                return True

        return False

    def validate_trade(self, position_value: float, portfolio_value: float,
                      current_positions: int = 1) -> Dict[str, Any]:
        """
        Validate if a trade meets risk requirements.

        Args:
            position_value: Value of proposed position
            portfolio_value: Total portfolio value
            current_positions: Number of current open positions

        Returns:
            Dictionary with validation results
        """
        violations = []

        # Check position size
        position_pct = position_value / portfolio_value
        if position_pct > self.config['max_position_size']:
            violations.append(f"Position size {position_pct*100:.1f}% exceeds max "
                            f"{self.config['max_position_size']*100:.1f}%")

        # Check leverage
        total_exposure = position_value * current_positions
        leverage = total_exposure / portfolio_value
        if leverage > self.config['max_leverage']:
            violations.append(f"Leverage {leverage:.2f} exceeds max "
                            f"{self.config['max_leverage']:.2f}")

        # Check drawdown
        if self.check_max_drawdown(portfolio_value):
            violations.append(f"Drawdown {self.current_drawdown*100:.1f}% exceeds max "
                            f"{self.config['max_drawdown']*100:.1f}%")

        is_valid = len(violations) == 0

        return {
            'valid': is_valid,
            'violations': violations,
            'position_pct': position_pct,
            'leverage': leverage,
            'drawdown': self.current_drawdown
        }

    def calculate_portfolio_risk(self, positions: pd.DataFrame,
                                 correlation_matrix: Optional[pd.DataFrame] = None) -> float:
        """
        Calculate total portfolio risk.

        Args:
            positions: DataFrame with position details (symbol, value, volatility)
            correlation_matrix: Optional correlation matrix between positions

        Returns:
            Portfolio risk (standard deviation)
        """
        if positions.empty:
            return 0.0

        # Simple volatility-weighted risk if no correlation matrix
        if correlation_matrix is None or 'volatility' not in positions.columns:
            if 'volatility' in positions.columns:
                weights = positions['value'] / positions['value'].sum()
                portfolio_risk = np.sqrt(np.sum((weights * positions['volatility']) ** 2))
            else:
                # Assume average volatility
                portfolio_risk = 0.15  # 15% default
        else:
            # Portfolio variance using correlation matrix
            weights = positions['value'] / positions['value'].sum()
            volatilities = positions['volatility'].values

            # Covariance matrix
            cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix.values

            # Portfolio variance
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_risk = np.sqrt(portfolio_variance)

        return portfolio_risk

    def get_risk_metrics(self) -> Dict[str, Any]:
        """
        Get current risk metrics.

        Returns:
            Dictionary with risk metrics
        """
        return {
            'current_drawdown': self.current_drawdown,
            'peak_value': self.peak_value,
            'max_drawdown_limit': self.config['max_drawdown'],
            'max_position_size': self.config['max_position_size'],
            'max_leverage': self.config['max_leverage'],
            'risk_violations': len(self.risk_violations),
            'violations_detail': self.risk_violations
        }

    def reset(self) -> None:
        """Reset risk tracking metrics."""
        self.peak_value = 0
        self.current_drawdown = 0
        self.risk_violations = []
        logger.info("Risk manager reset")


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("RISK MANAGER EXAMPLES")
    print("=" * 60)

    # Initialize risk manager
    config = {
        'max_position_size': 0.20,
        'max_portfolio_risk': 0.02,
        'stop_loss_pct': 0.05,
        'max_drawdown': 0.15,
        'position_sizing_method': 'fixed_fraction'
    }

    risk_manager = RiskManager(config=config)
    print(f"\nRisk Manager initialized with config:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Test position sizing
    print("\n" + "=" * 60)
    print("POSITION SIZING")
    print("=" * 60)

    portfolio_value = 100000
    available_cash = 50000
    price = 150.0

    position_size = risk_manager.calculate_position_size(
        price=price,
        portfolio_value=portfolio_value,
        available_cash=available_cash
    )

    shares = int(position_size / price)
    print(f"Portfolio Value: ${portfolio_value:,.2f}")
    print(f"Available Cash: ${available_cash:,.2f}")
    print(f"Stock Price: ${price:.2f}")
    print(f"Recommended Position Size: ${position_size:,.2f}")
    print(f"Number of Shares: {shares}")
    print(f"Position as % of Portfolio: {position_size/portfolio_value*100:.1f}%")

    # Test stop loss
    print("\n" + "=" * 60)
    print("STOP LOSS CHECK")
    print("=" * 60)

    entry_price = 150.0
    test_prices = [145.0, 142.5, 140.0]

    for current_price in test_prices:
        loss_pct = (entry_price - current_price) / entry_price * 100
        is_stopped = risk_manager.check_stop_loss(entry_price, current_price, 'long')
        print(f"Current Price: ${current_price:.2f} (Loss: {loss_pct:.1f}%) - "
              f"Stop Loss: {'TRIGGERED' if is_stopped else 'OK'}")

    # Test drawdown check
    print("\n" + "=" * 60)
    print("DRAWDOWN CHECK")
    print("=" * 60)

    portfolio_values = [100000, 95000, 90000, 85000, 82000]

    for value in portfolio_values:
        exceeded = risk_manager.check_max_drawdown(value)
        print(f"Portfolio Value: ${value:,.2f} - "
              f"Drawdown: {risk_manager.current_drawdown*100:.1f}% - "
              f"Status: {'EXCEEDED' if exceeded else 'OK'}")

    # Test trade validation
    print("\n" + "=" * 60)
    print("TRADE VALIDATION")
    print("=" * 60)

    test_positions = [15000, 25000, 35000]

    for position_value in test_positions:
        validation = risk_manager.validate_trade(
            position_value=position_value,
            portfolio_value=portfolio_value,
            current_positions=2
        )

        print(f"\nPosition Value: ${position_value:,.2f}")
        print(f"Valid: {validation['valid']}")
        print(f"Position %: {validation['position_pct']*100:.1f}%")
        print(f"Leverage: {validation['leverage']:.2f}")
        if validation['violations']:
            print("Violations:")
            for v in validation['violations']:
                print(f"  - {v}")

    # Risk metrics summary
    print("\n" + "=" * 60)
    print("RISK METRICS SUMMARY")
    print("=" * 60)

    metrics = risk_manager.get_risk_metrics()
    for key, value in metrics.items():
        if key != 'violations_detail':
            print(f"{key}: {value}")
