"""
Prediction Script

Makes predictions using trained models.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.model_persistence import ModelPersistence
from src.features.feature_engineering import FeatureEngineer
from src.utils.logger import setup_logger
from src.utils.file_utils import ensure_dir

logger = setup_logger(__name__)


def load_model_and_metadata(model_path: str, models_dir: str = 'data/models'):
    """
    Load trained model and its metadata.

    Args:
        model_path: Path to model file or model name
        models_dir: Directory containing models

    Returns:
        Tuple of (model, metadata)
    """
    persistence = ModelPersistence(base_dir=models_dir)

    # If model_path is just a name, construct full path
    if not Path(model_path).exists():
        model_path = Path(models_dir) / model_path

    # Load model
    logger.info(f"Loading model from {model_path}")
    model = persistence.load_model(str(model_path))

    # Load metadata
    metadata_path = str(model_path).replace('.pkl', '_metadata.json')
    if Path(metadata_path).exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logger.info("Loaded model metadata")
    else:
        logger.warning(f"Metadata file not found: {metadata_path}")
        metadata = {}

    return model, metadata


def load_data(symbol: str, data_dir: str = 'data/processed') -> pd.DataFrame:
    """
    Load data for prediction.

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


def prepare_features(data: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """
    Prepare features for prediction using same pipeline as training.

    Args:
        data: Raw OHLCV data
        metadata: Model metadata containing feature information

    Returns:
        DataFrame with features
    """
    logger.info("Preparing features for prediction")

    # Initialize feature engineer
    fe = FeatureEngineer()

    # Generate features (without labels)
    features = fe.engineer_features(data)

    # Handle missing values
    features = fe.handle_missing_values(features, method='forward_fill')

    # Select only features used during training
    if 'feature_names' in metadata:
        feature_names = metadata['feature_names']
        # Filter to only include features that exist
        available_features = [f for f in feature_names if f in features.columns]
        missing_features = [f for f in feature_names if f not in features.columns]

        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features from training: {missing_features[:5]}...")
            # Add missing features as zeros
            for feature in missing_features:
                features[feature] = 0

        features = features[feature_names]
        logger.info(f"Selected {len(feature_names)} features matching training")

    return features


def make_predictions(model, features: pd.DataFrame,
                     include_proba: bool = True) -> pd.DataFrame:
    """
    Make predictions using trained model.

    Args:
        model: Trained model
        features: Features DataFrame
        include_proba: Include prediction probabilities

    Returns:
        DataFrame with predictions
    """
    logger.info(f"Making predictions for {len(features)} samples")

    # Make predictions
    predictions = model.predict(features)

    # Create results DataFrame
    results = pd.DataFrame(index=features.index)
    results['prediction'] = predictions

    # Add probabilities if available
    if include_proba and hasattr(model, 'predict_proba'):
        try:
            probabilities = model.predict_proba(features)
            if probabilities.shape[1] == 2:  # Binary classification
                results['probability'] = probabilities[:, 1]
            else:  # Multi-class
                for i in range(probabilities.shape[1]):
                    results[f'probability_class_{i}'] = probabilities[:, i]
        except Exception as e:
            logger.warning(f"Could not get probabilities: {e}")

    logger.info("Predictions complete")
    return results


def predict(model_path: str, symbol: str,
            data_dir: str = 'data/processed',
            models_dir: str = 'data/models',
            output_dir: str = 'data/predictions',
            save_results: bool = True) -> pd.DataFrame:
    """
    Make predictions for a symbol using a trained model.

    Args:
        model_path: Path to trained model
        symbol: Stock symbol
        data_dir: Directory containing data
        models_dir: Directory containing models
        output_dir: Directory to save predictions
        save_results: Whether to save results to file

    Returns:
        DataFrame with predictions
    """
    # Load model and metadata
    model, metadata = load_model_and_metadata(model_path, models_dir)

    # Load data
    data = load_data(symbol, data_dir)

    # Prepare features
    features = prepare_features(data, metadata)

    # Make predictions
    results = make_predictions(model, features, include_proba=True)

    # Add original data columns
    results = pd.concat([data, results], axis=1)

    # Save results
    if save_results:
        ensure_dir(output_dir)
        output_path = Path(output_dir) / f"{symbol}_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results.to_csv(output_path)
        logger.info(f"Saved predictions to {output_path}")

    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Make predictions with trained model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model file or model name')
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                       help='Directory containing data')
    parser.add_argument('--models-dir', type=str, default='data/models',
                       help='Directory containing models')
    parser.add_argument('--output-dir', type=str, default='data/predictions',
                       help='Output directory for predictions')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save predictions to file')
    parser.add_argument('--tail', type=int, default=10,
                       help='Number of recent predictions to display')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')

    args = parser.parse_args()

    # Setup logging
    setup_logger(__name__, level=getattr(logger, args.log_level))

    # Make predictions
    try:
        results = predict(
            args.model,
            args.symbol,
            args.data_dir,
            args.models_dir,
            args.output_dir,
            save_results=not args.no_save
        )

        print("\n" + "=" * 60)
        print("PREDICTION RESULTS")
        print("=" * 60)
        print(f"\nSymbol: {args.symbol}")
        print(f"Total predictions: {len(results)}")

        # Show prediction distribution
        print(f"\nPrediction distribution:")
        print(results['prediction'].value_counts().sort_index())

        # Show recent predictions
        print(f"\nLast {args.tail} predictions:")
        display_cols = ['close', 'prediction']
        if 'probability' in results.columns:
            display_cols.append('probability')
        print(results[display_cols].tail(args.tail))

        if not args.no_save:
            print(f"\nResults saved to: {args.output_dir}")

    except Exception as e:
        logger.error(f"Error making predictions: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

