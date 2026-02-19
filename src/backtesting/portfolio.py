"""
Portfolio Management

Manages trading positions, cash, and portfolio value tracking.
"""

import pandas as pd
from typing import Dict, Optional
from datetime import datetime
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class Portfolio:
    """
    Manages portfolio positions, cash, and value tracking.

    Examples:
        >>> portfolio = Portfolio(initial_capital=100000)
        >>> portfolio.cash
        100000
        >>> portfolio.update_position('AAPL', 100, 150.0, 15.0)
        >>> portfolio.get_position('AAPL')
        100
        >>> portfolio.get_total_value({'AAPL': 155.0})
        115485.0
    """

    def __init__(self, initial_capital: float = 100000):
        """
        Initialize Portfolio.

        Args:
            initial_capital: Starting capital amount
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, float] = {}  # symbol -> shares
        self.cost_basis: Dict[str, float] = {}  # symbol -> average cost per share
        self.position_values: Dict[str, float] = {}  # symbol -> current value

        # Track history
        self.trade_history = []

        logger.info(f"Initialized Portfolio with ${initial_capital:,.2f}")

    def get_position(self, symbol: str) -> float:
        """
        Get current position for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Number of shares held (0 if no position)
        """
        return self.positions.get(symbol, 0)

    def get_cost_basis(self, symbol: str) -> float:
        """
        Get average cost basis for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Average cost per share (0 if no position)
        """
        return self.cost_basis.get(symbol, 0)

    def update_position(self, symbol: str, shares: float, price: float,
                       commission: float = 0) -> None:
        """
        Update position for a symbol.

        Args:
            symbol: Stock symbol
            shares: Number of shares to add (positive) or remove (negative)
            price: Execution price per share
            commission: Commission paid on this trade
        """
        current_position = self.positions.get(symbol, 0)

        if shares > 0:  # Buy
            # Calculate new average cost basis
            if current_position > 0:
                current_value = current_position * self.cost_basis.get(symbol, 0)
                new_value = shares * price + commission
                total_shares = current_position + shares
                self.cost_basis[symbol] = (current_value + new_value) / total_shares
            else:
                self.cost_basis[symbol] = (shares * price + commission) / shares

            # Update position and cash
            self.positions[symbol] = current_position + shares
            self.cash -= (shares * price + commission)

            logger.debug(f"BUY: {shares} shares of {symbol} @ ${price:.2f} "
                        f"(Commission: ${commission:.2f}, Cash: ${self.cash:,.2f})")

        elif shares < 0:  # Sell
            shares_to_sell = abs(shares)

            if current_position < shares_to_sell:
                logger.warning(f"Attempting to sell {shares_to_sell} shares of {symbol}, "
                             f"but only {current_position} shares available")
                shares_to_sell = current_position

            # Update position and cash
            self.positions[symbol] = current_position - shares_to_sell
            self.cash += (shares_to_sell * price - commission)

            # Remove cost basis if position is closed
            if self.positions[symbol] == 0:
                self.cost_basis.pop(symbol, None)

            logger.debug(f"SELL: {shares_to_sell} shares of {symbol} @ ${price:.2f} "
                        f"(Commission: ${commission:.2f}, Cash: ${self.cash:,.2f})")

        # Record trade in history
        self.trade_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'shares': shares,
            'price': price,
            'commission': commission,
            'cash_after': self.cash,
            'position_after': self.positions.get(symbol, 0)
        })

    def get_position_value(self, symbol: str, current_price: float) -> float:
        """
        Get current value of a position.

        Args:
            symbol: Stock symbol
            current_price: Current market price

        Returns:
            Current market value of position
        """
        shares = self.positions.get(symbol, 0)
        return shares * current_price

    def get_total_value(self, current_prices: Optional[Dict[str, float]] = None,
                       default_price: Optional[float] = None) -> float:
        """
        Get total portfolio value (cash + positions).

        Args:
            current_prices: Dictionary of symbol -> current price
            default_price: Default price to use if current_prices not provided
                          (for single-symbol portfolios)

        Returns:
            Total portfolio value
        """
        total_value = self.cash

        for symbol, shares in self.positions.items():
            if shares > 0:
                if current_prices and symbol in current_prices:
                    price = current_prices[symbol]
                elif default_price is not None:
                    price = default_price
                else:
                    logger.warning(f"No price available for {symbol}, using cost basis")
                    price = self.cost_basis.get(symbol, 0)

                total_value += shares * price

        return total_value

    def get_unrealized_pnl(self, symbol: str, current_price: float) -> float:
        """
        Get unrealized profit/loss for a position.

        Args:
            symbol: Stock symbol
            current_price: Current market price

        Returns:
            Unrealized P&L
        """
        shares = self.positions.get(symbol, 0)
        if shares == 0:
            return 0

        cost_basis = self.cost_basis.get(symbol, 0)
        current_value = shares * current_price
        cost_value = shares * cost_basis

        return current_value - cost_value

    def get_positions_summary(self, current_prices: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Get summary of all positions.

        Args:
            current_prices: Dictionary of symbol -> current price

        Returns:
            DataFrame with position details
        """
        if not self.positions:
            return pd.DataFrame()

        summary_data = []

        for symbol, shares in self.positions.items():
            if shares > 0:
                cost_basis = self.cost_basis.get(symbol, 0)
                cost_value = shares * cost_basis

                if current_prices and symbol in current_prices:
                    current_price = current_prices[symbol]
                    current_value = shares * current_price
                    unrealized_pnl = current_value - cost_value
                    pnl_pct = (unrealized_pnl / cost_value) * 100 if cost_value > 0 else 0
                else:
                    current_price = cost_basis
                    current_value = cost_value
                    unrealized_pnl = 0
                    pnl_pct = 0

                summary_data.append({
                    'symbol': symbol,
                    'shares': shares,
                    'cost_basis': cost_basis,
                    'current_price': current_price,
                    'cost_value': cost_value,
                    'current_value': current_value,
                    'unrealized_pnl': unrealized_pnl,
                    'pnl_pct': pnl_pct
                })

        return pd.DataFrame(summary_data)

    def get_trade_history(self) -> pd.DataFrame:
        """
        Get trade history as DataFrame.

        Returns:
            DataFrame with trade history
        """
        if not self.trade_history:
            return pd.DataFrame()

        return pd.DataFrame(self.trade_history)

    def reset(self) -> None:
        """Reset portfolio to initial state."""
        self.cash = self.initial_capital
        self.positions = {}
        self.cost_basis = {}
        self.position_values = {}
        self.trade_history = []

        logger.info(f"Portfolio reset to ${self.initial_capital:,.2f}")

    def __repr__(self) -> str:
        """String representation of portfolio."""
        return (f"Portfolio(cash=${self.cash:,.2f}, "
                f"positions={len(self.positions)}, "
                f"total_value=${self.get_total_value():,.2f})")


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("PORTFOLIO MANAGEMENT EXAMPLE")
    print("=" * 60)

    # Initialize portfolio
    portfolio = Portfolio(initial_capital=100000)
    print(f"\nInitial Portfolio: {portfolio}")

    # Buy AAPL
    print("\n--- Buy 100 shares of AAPL @ $150 ---")
    portfolio.update_position('AAPL', 100, 150.0, commission=15.0)
    print(f"Portfolio: {portfolio}")
    print(f"AAPL Position: {portfolio.get_position('AAPL')} shares")
    print(f"Cost Basis: ${portfolio.get_cost_basis('AAPL'):.2f}")

    # Buy more AAPL
    print("\n--- Buy 50 more shares of AAPL @ $155 ---")
    portfolio.update_position('AAPL', 50, 155.0, commission=7.5)
    print(f"Portfolio: {portfolio}")
    print(f"AAPL Position: {portfolio.get_position('AAPL')} shares")
    print(f"Average Cost Basis: ${portfolio.get_cost_basis('AAPL'):.2f}")

    # Check unrealized P&L
    current_price = 160.0
    print(f"\n--- Current AAPL Price: ${current_price} ---")
    unrealized_pnl = portfolio.get_unrealized_pnl('AAPL', current_price)
    print(f"Unrealized P&L: ${unrealized_pnl:,.2f}")
    print(f"Total Portfolio Value: ${portfolio.get_total_value({'AAPL': current_price}):,.2f}")

    # Sell some shares
    print("\n--- Sell 75 shares of AAPL @ $162 ---")
    portfolio.update_position('AAPL', -75, 162.0, commission=12.0)
    print(f"Portfolio: {portfolio}")
    print(f"AAPL Position: {portfolio.get_position('AAPL')} shares")

    # Position summary
    print("\n--- Position Summary ---")
    summary = portfolio.get_positions_summary({'AAPL': current_price})
    if not summary.empty:
        print(summary.to_string(index=False))

    # Trade history
    print("\n--- Trade History ---")
    history = portfolio.get_trade_history()
    if not history.empty:
        print(history[['symbol', 'shares', 'price', 'commission', 'cash_after']].to_string(index=False))
