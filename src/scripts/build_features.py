"""
Build Features Script

Loads raw data, engineers features, and saves them for model training.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features.feature_engineering import FeatureEngineer
from src.utils.logger import setup_logger
from src.utils.file_utils import ensure_dir

logger = setup_logger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_raw_data(symbol: str, data_dir: str = 'data/processed') -> pd.DataFrame:
    """
    Load raw/processed data for a symbol.

    Args:
        symbol: Stock symbol
        data_dir: Directory containing data files

    Returns:
        DataFrame with OHLCV data
    """
    file_path = Path(data_dir) / f"{symbol}.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    logger.info(f"Loading data from {file_path}")
    data = pd.read_csv(file_path, index_col=0, parse_dates=True)

    logger.info(f"Loaded {len(data)} rows of data for {symbol}")
    return data


def build_features(symbol: str, config: dict, output_dir: str = 'data/features'):
    """
    Build features for a given symbol.

    Args:
        symbol: Stock symbol
        config: Configuration dictionary
        output_dir: Directory to save features
    """
    logger.info(f"Building features for {symbol}")

    # Load data
    data = load_raw_data(symbol, config.get('data_dir', 'data/processed'))

    # Initialize feature engineer
    fe = FeatureEngineer(config=config.get('feature_config', {}))

    # Engineer features
    feature_config = config.get('feature_config', {})
    features, labels = fe.create_feature_label_dataset(
        data,
        label_type=feature_config.get('label_type', 'binary'),
        forward_period=feature_config.get('forward_period', 5),
        threshold=feature_config.get('threshold', 0.02),
        include_price=feature_config.get('include_price', True),
        include_technical=feature_config.get('include_technical', True),
        include_volume=feature_config.get('include_volume', True),
        include_volatility=feature_config.get('include_volatility', True)
    )

    # Handle missing values
    features = fe.handle_missing_values(
        features,
        method=feature_config.get('missing_value_method', 'forward_fill')
    )

    # Feature selection (optional)
    if feature_config.get('feature_selection', False):
        selected_features = fe.select_features(
            features,
            method=feature_config.get('selection_method', 'variance'),
            threshold=feature_config.get('selection_threshold', 0.01)
        )
        features = features[selected_features]

    # Save features and labels
    ensure_dir(output_dir)

    features_path = Path(output_dir) / f"{symbol}_features.csv"
    labels_path = Path(output_dir) / f"{symbol}_labels.csv"

    features.to_csv(features_path)
    labels.to_csv(labels_path)

    logger.info(f"Saved features to {features_path}")
    logger.info(f"Saved labels to {labels_path}")

    # Save feature statistics
    stats = fe.get_feature_statistics(features)
    stats_path = Path(output_dir) / f"{symbol}_feature_stats.csv"
    stats.to_csv(stats_path)
    logger.info(f"Saved feature statistics to {stats_path}")

    return features, labels


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Build features from raw data')
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol')
    parser.add_argument('--config', type=str, default='configs/feature_config.yaml',
                       help='Path to feature configuration file')
    parser.add_argument('--output-dir', type=str, default='data/features',
                       help='Output directory for features')
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
        config = {}

    # Build features
    try:
        features, labels = build_features(args.symbol, config, args.output_dir)

        print("\n" + "=" * 60)
        print("FEATURE BUILDING COMPLETE")
        print("=" * 60)
        print(f"Symbol: {args.symbol}")
        print(f"Features shape: {features.shape}")
        print(f"Number of features: {len(features.columns)}")
        print(f"Number of samples: {len(features)}")
        print(f"\nLabel distribution:")
        print(labels.value_counts())
        print(f"\nOutput directory: {args.output_dir}")

    except Exception as e:
        logger.error(f"Error building features: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
