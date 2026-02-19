"""
Run Backtest Script

Runs backtests using trained models and historical data.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import json
import yaml
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.backtesting.backtest_engine import BacktestEngine
from src.models.model_persistence import ModelPersistence
from src.features.feature_engineering import FeatureEngineer
from src.utils.logger import setup_logger
from src.utils.file_utils import ensure_dir

logger = setup_logger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model_and_metadata(model_path: str, models_dir: str = 'data/models'):
    """Load trained model and its metadata."""
    persistence = ModelPersistence(base_dir=models_dir)

    if not Path(model_path).exists():
        model_path = Path(models_dir) / model_path

    logger.info(f"Loading model from {model_path}")
    model = persistence.load_model(str(model_path))

    metadata_path = str(model_path).replace('.pkl', '_metadata.json')
    if Path(metadata_path).exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}

    return model, metadata


def load_data(symbol: str, data_dir: str = 'data/processed') -> pd.DataFrame:
    """Load historical data for backtesting."""
    file_path = Path(data_dir) / f"{symbol}.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    logger.info(f"Loading data from {file_path}")
    data = pd.read_csv(file_path, index_col=0, parse_dates=True)

    logger.info(f"Loaded {len(data)} rows of data for {symbol}")
    return data


def prepare_data_with_signals(data: pd.DataFrame, model, metadata: dict) -> pd.DataFrame:
    """
    Prepare data with features and generate trading signals.

    Args:
        data: Raw OHLCV data
        model: Trained model
        metadata: Model metadata

    Returns:
        DataFrame with signals
    """
    logger.info("Preparing features and generating signals")

    # Generate features
    fe = FeatureEngineer()
    features = fe.engineer_features(data)
    features = fe.handle_missing_values(features, method='forward_fill')

    # Select features used during training
    if 'feature_names' in metadata:
        feature_names = metadata['feature_names']
        available_features = [f for f in feature_names if f in features.columns]
        missing_features = [f for f in feature_names if f not in features.columns]

        if missing_features:
            for feature in missing_features:
                features[feature] = 0

        features = features[feature_names]

    # Generate signals
    predictions = model.predict(features)

    # Create result DataFrame
    result = data.copy()
    result['signal'] = predictions

    # Convert binary predictions to trading signals (1=buy, 0=hold)
    # For multi-class: 2=buy, 1=hold, 0=sell -> convert to 1, 0, -1
    if set(predictions).issubset({0, 1, 2}):
        result['signal'] = predictions.map({2: 1, 1: 0, 0: -1})

    logger.info(f"Generated {len(predictions)} signals")
    return result


def run_backtest(model_path: str, symbol: str, config: dict,
                data_dir: str = 'data/processed',
                models_dir: str = 'data/models',
                output_dir: str = 'reports'):
    """
    Run backtest using trained model.

    Args:
        model_path: Path to trained model
        symbol: Stock symbol
        config: Configuration dictionary
        data_dir: Directory containing data
        models_dir: Directory containing models
        output_dir: Directory to save results
    """
    logger.info(f"Running backtest for {symbol}")

    # Load model and metadata
    model, metadata = load_model_and_metadata(model_path, models_dir)

    # Load data
    data = load_data(symbol, data_dir)

    # Prepare data with signals
    data_with_signals = prepare_data_with_signals(data, model, metadata)

    # Initialize backtest engine
    backtest_config = config.get('backtest', {})
    engine = BacktestEngine(
        initial_capital=backtest_config.get('initial_capital', 100000),
        commission=backtest_config.get('commission', 0.001),
        slippage=backtest_config.get('slippage', 0.0005),
        use_risk_manager=backtest_config.get('use_risk_manager', True),
        risk_config=backtest_config.get('risk_config')
    )

    # Run backtest
    results = engine.run(data_with_signals, symbol, signal_column='signal')

    # Get metrics and trades
    metrics = engine.get_performance_metrics()
    trades = engine.get_trades()
    trade_stats = engine.get_trade_statistics()

    # Print results
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)
    print(f"\nSymbol: {symbol}")
    print(f"Model: {Path(model_path).stem}")
    print(f"Period: {results.index[0]} to {results.index[-1]}")
    print(f"Trading days: {len(results)}")

    print("\nPerformance Metrics:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")

    print("\nTrade Statistics:")
    for stat, value in trade_stats.items():
        if isinstance(value, float):
            print(f"  {stat}: {value:.2f}")
        else:
            print(f"  {stat}: {value}")

    # Save results
    ensure_dir(output_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save backtest results
    results_path = Path(output_dir) / f"{symbol}_backtest_results_{timestamp}.csv"
    results.to_csv(results_path)
    logger.info(f"Saved backtest results to {results_path}")

    # Save trades
    if len(trades) > 0:
        trades_path = Path(output_dir) / f"{symbol}_trades_{timestamp}.csv"
        trades.to_csv(trades_path, index=False)
        logger.info(f"Saved trades to {trades_path}")

    # Save metrics
    report = {
        'symbol': symbol,
        'model_path': str(model_path),
        'backtest_period': {
            'start': str(results.index[0]),
            'end': str(results.index[-1]),
            'days': len(results)
        },
        'performance_metrics': metrics,
        'trade_statistics': trade_stats,
        'config': backtest_config,
        'run_at': datetime.now().isoformat()
    }

    report_path = Path(output_dir) / f"{symbol}_backtest_report_{timestamp}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved backtest report to {report_path}")

    # Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Portfolio value
    axes[0].plot(results.index, results['portfolio_value'], label='Portfolio Value')
    axes[0].axhline(y=engine.initial_capital, color='r', linestyle='--', label='Initial Capital')
    axes[0].set_title('Portfolio Value Over Time')
    axes[0].set_ylabel('Value ($)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Price and signals
    axes[1].plot(results.index, results['close'], label='Price', alpha=0.7)
    buy_signals = results[results['signal'] == 1]
    sell_signals = results[results['signal'] == -1]
    axes[1].scatter(buy_signals.index, buy_signals['close'], color='green',
                   marker='^', s=100, label='Buy Signal', zorder=5)
    axes[1].scatter(sell_signals.index, sell_signals['close'], color='red',
                   marker='v', s=100, label='Sell Signal', zorder=5)
    axes[1].set_title('Price and Trading Signals')
    axes[1].set_ylabel('Price ($)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Returns
    axes[2].plot(results.index, results['returns'] * 100, label='Cumulative Return')
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[2].set_title('Cumulative Returns')
    axes[2].set_ylabel('Return (%)')
    axes[2].set_xlabel('Date')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = Path(output_dir) / 'figures' / f"{symbol}_backtest_{timestamp}.png"
    ensure_dir(plot_path.parent)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved backtest plot to {plot_path}")

    return results, metrics, trades


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run backtest with trained model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model file')
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                       help='Directory containing data')
    parser.add_argument('--models-dir', type=str, default='data/models',
                       help='Directory containing models')
    parser.add_argument('--output-dir', type=str, default='reports',
                       help='Output directory for results')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')

    args = parser.parse_args()

    # Setup logging
    setup_logger(__name__, level=getattr(logger, args.log_level))

    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.warning(f"Config file not found: {args.config}, using defaults")
        config = {'backtest': {}}

    # Run backtest
    try:
        results, metrics, trades = run_backtest(
            args.model,
            args.symbol,
            config,
            args.data_dir,
            args.models_dir,
            args.output_dir
        )

        print(f"\nBacktest complete!")
        print(f"Output directory: {args.output_dir}")

    except Exception as e:
        logger.error(f"Error running backtest: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
