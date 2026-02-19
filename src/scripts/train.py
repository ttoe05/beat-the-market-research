"""
Training Script

Trains machine learning models for trading signal prediction.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import yaml
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_splitter import DataSplitter
from src.models.random_forest_classifier import RandomForestClassifier
from src.models.xgboost_classifier import XGBoostClassifier
from src.models.lightgbm_classifier import LightGBMClassifier
from src.models.model_trainer import ModelTrainer
from src.models.model_evaluator import ModelEvaluator
from src.models.model_persistence import ModelPersistence
from src.utils.logger import setup_logger
from src.utils.file_utils import ensure_dir

logger = setup_logger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_features_and_labels(symbol: str, features_dir: str = 'data/features'):
    """
    Load features and labels for a symbol.

    Args:
        symbol: Stock symbol
        features_dir: Directory containing feature files

    Returns:
        Tuple of (features DataFrame, labels Series)
    """
    features_path = Path(features_dir) / f"{symbol}_features.csv"
    labels_path = Path(features_dir) / f"{symbol}_labels.csv"

    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    logger.info(f"Loading features from {features_path}")
    features = pd.read_csv(features_path, index_col=0, parse_dates=True)

    logger.info(f"Loading labels from {labels_path}")
    labels = pd.read_csv(labels_path, index_col=0, parse_dates=True).squeeze()

    logger.info(f"Loaded {len(features)} samples with {len(features.columns)} features")

    return features, labels


def get_model(model_type: str, config: dict):
    """
    Get model instance based on type.

    Args:
        model_type: Type of model
        config: Model configuration

    Returns:
        Model instance
    """
    models = {
        'random_forest': RandomForestClassifier,
        'xgboost': XGBoostClassifier,
        'lightgbm': LightGBMClassifier
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")

    return models[model_type](config=config)


def train_model(symbol: str, model_type: str, config: dict,
                features_dir: str = 'data/features',
                output_dir: str = 'data/models'):
    """
    Train a model for a given symbol.

    Args:
        symbol: Stock symbol
        model_type: Type of model to train
        config: Configuration dictionary
        features_dir: Directory containing features
        output_dir: Directory to save model
    """
    logger.info(f"Training {model_type} model for {symbol}")

    # Load features and labels
    features, labels = load_features_and_labels(symbol, features_dir)

    # Split data
    splitter_config = config.get('data_split', {})
    splitter = DataSplitter(
        test_size=splitter_config.get('test_size', 0.2),
        validation_size=splitter_config.get('validation_size', 0.1),
        method=splitter_config.get('method', 'time_series')
    )

    splits = splitter.split(features, labels)
    X_train, X_val, X_test = splits['X_train'], splits['X_val'], splits['X_test']
    y_train, y_val, y_test = splits['y_train'], splits['y_val'], splits['y_test']

    logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Initialize model
    model_config = config.get('model_config', {}).get(model_type, {})
    model = get_model(model_type, model_config)

    # Train model
    trainer = ModelTrainer(model)
    trainer.train(X_train, y_train, X_val, y_val)

    # Evaluate model
    evaluator = ModelEvaluator()

    # Evaluate on validation set
    logger.info("Evaluating on validation set")
    val_metrics = evaluator.evaluate(model, X_val, y_val)

    # Evaluate on test set
    logger.info("Evaluating on test set")
    test_metrics = evaluator.evaluate(model, X_test, y_test)

    # Print results
    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    print(f"\nValidation Metrics:")
    for metric, value in val_metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")

    print(f"\nTest Metrics:")
    for metric, value in test_metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")

    # Save model
    ensure_dir(output_dir)
    model_name = f"{symbol}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    persistence = ModelPersistence(base_dir=output_dir)
    model_path = persistence.save_model(model, model_name)

    # Save metadata
    metadata = {
        'symbol': symbol,
        'model_type': model_type,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'num_features': len(features.columns),
        'feature_names': features.columns.tolist(),
        'validation_metrics': {k: float(v) if isinstance(v, (int, float)) else str(v)
                              for k, v in val_metrics.items()},
        'test_metrics': {k: float(v) if isinstance(v, (int, float)) else str(v)
                        for k, v in test_metrics.items()},
        'config': model_config,
        'trained_at': datetime.now().isoformat()
    }

    metadata_path = Path(output_dir) / f"{model_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved model to {model_path}")
    logger.info(f"Saved metadata to {metadata_path}")

    return model, test_metrics


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train trading ML model')
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol')
    parser.add_argument('--model', type=str, default='random_forest',
                       choices=['random_forest', 'xgboost', 'lightgbm'],
                       help='Model type to train')
    parser.add_argument('--config', type=str, default='configs/model_config.yaml',
                       help='Path to model configuration file')
    parser.add_argument('--features-dir', type=str, default='data/features',
                       help='Directory containing features')
    parser.add_argument('--output-dir', type=str, default='data/models',
                       help='Output directory for models')
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

    # Train model
    try:
        model, metrics = train_model(
            args.symbol,
            args.model,
            config,
            args.features_dir,
            args.output_dir
        )

        print(f"\nModel training complete!")
        print(f"Output directory: {args.output_dir}")

    except Exception as e:
        logger.error(f"Error training model: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

