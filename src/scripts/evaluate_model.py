"""
Model Evaluation Script

Comprehensive evaluation of trained models on test data.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.model_persistence import ModelPersistence
from src.models.model_evaluator import ModelEvaluator
from src.visualization.model_visualizer import ModelVisualizer
from src.utils.logger import setup_logger
from src.utils.file_utils import ensure_dir

logger = setup_logger(__name__)


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


def load_test_data(symbol: str, features_dir: str = 'data/features'):
    """Load test features and labels."""
    features_path = Path(features_dir) / f"{symbol}_features.csv"
    labels_path = Path(features_dir) / f"{symbol}_labels.csv"

    if not features_path.exists() or not labels_path.exists():
        raise FileNotFoundError(f"Test data not found in {features_dir}")

    features = pd.read_csv(features_path, index_col=0, parse_dates=True)
    labels = pd.read_csv(labels_path, index_col=0, parse_dates=True).squeeze()

    return features, labels


def evaluate_model(model_path: str, symbol: str,
                  features_dir: str = 'data/features',
                  models_dir: str = 'data/models',
                  output_dir: str = 'reports',
                  create_plots: bool = True):
    """
    Comprehensive evaluation of a trained model.

    Args:
        model_path: Path to trained model
        symbol: Stock symbol
        features_dir: Directory containing features
        models_dir: Directory containing models
        output_dir: Directory to save evaluation results
        create_plots: Whether to create visualization plots
    """
    logger.info(f"Evaluating model for {symbol}")

    # Load model and metadata
    model, metadata = load_model_and_metadata(model_path, models_dir)

    # Load test data
    X_test, y_test = load_test_data(symbol, features_dir)

    # Filter features if metadata contains feature names
    if 'feature_names' in metadata:
        available_features = [f for f in metadata['feature_names'] if f in X_test.columns]
        X_test = X_test[available_features]

    logger.info(f"Evaluating on {len(X_test)} test samples")

    # Evaluate model
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(model, X_test, y_test)

    # Print metrics
    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nSymbol: {symbol}")
    print(f"Model: {Path(model_path).stem}")
    print(f"Test samples: {len(X_test)}")

    print("\nMetrics:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        elif isinstance(value, dict):
            print(f"  {metric}:")
            for k, v in value.items():
                print(f"    {k}: {v:.4f}")
        else:
            print(f"  {metric}: {value}")

    # Create visualizations if requested
    if create_plots:
        logger.info("Creating evaluation visualizations")
        ensure_dir(output_dir / 'figures')

        visualizer = ModelVisualizer()

        # Get predictions
        y_pred = model.predict(X_test)
        y_pred_proba = None
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)

        # Create plots
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Confusion Matrix
        fig = visualizer.plot_confusion_matrix(y_test, y_pred)
        confusion_path = output_dir / 'figures' / f"{symbol}_confusion_matrix_{timestamp}.png"
        fig.savefig(confusion_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved confusion matrix to {confusion_path}")

        # ROC Curve (for binary classification)
        if y_pred_proba is not None and len(set(y_test)) == 2:
            fig = visualizer.plot_roc_curve(y_test, y_pred_proba[:, 1])
            roc_path = output_dir / 'figures' / f"{symbol}_roc_curve_{timestamp}.png"
            fig.savefig(roc_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Saved ROC curve to {roc_path}")

        # Feature Importance
        if hasattr(model, 'get_feature_importance'):
            importance = model.get_feature_importance()
            fig = visualizer.plot_feature_importance(importance, top_n=20)
            importance_path = output_dir / 'figures' / f"{symbol}_feature_importance_{timestamp}.png"
            fig.savefig(importance_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Saved feature importance to {importance_path}")

        # Prediction Distribution
        fig = visualizer.plot_prediction_distribution(y_test, y_pred)
        dist_path = output_dir / 'figures' / f"{symbol}_prediction_distribution_{timestamp}.png"
        fig.savefig(dist_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved prediction distribution to {dist_path}")

    # Save evaluation report
    ensure_dir(output_dir / 'metrics')
    report = {
        'symbol': symbol,
        'model_path': str(model_path),
        'test_samples': len(X_test),
        'num_features': len(X_test.columns),
        'metrics': {k: float(v) if isinstance(v, (int, float)) else v
                   for k, v in metrics.items()},
        'evaluated_at': datetime.now().isoformat()
    }

    if metadata:
        report['model_metadata'] = metadata

    report_path = output_dir / 'metrics' / f"{symbol}_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"Saved evaluation report to {report_path}")

    return metrics


def compare_models(model_paths: list, symbol: str,
                  features_dir: str = 'data/features',
                  models_dir: str = 'data/models'):
    """
    Compare multiple models on the same test set.

    Args:
        model_paths: List of paths to trained models
        symbol: Stock symbol
        features_dir: Directory containing features
        models_dir: Directory containing models
    """
    logger.info(f"Comparing {len(model_paths)} models for {symbol}")

    # Load test data once
    X_test, y_test = load_test_data(symbol, features_dir)

    evaluator = ModelEvaluator()
    comparison_results = []

    for model_path in model_paths:
        model, metadata = load_model_and_metadata(model_path, models_dir)

        # Filter features if needed
        X_test_filtered = X_test.copy()
        if 'feature_names' in metadata:
            available_features = [f for f in metadata['feature_names'] if f in X_test.columns]
            X_test_filtered = X_test[available_features]

        metrics = evaluator.evaluate(model, X_test_filtered, y_test)

        comparison_results.append({
            'model': Path(model_path).stem,
            'model_type': metadata.get('model_type', 'unknown'),
            **{k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        })

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_results)

    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(comparison_df.to_string(index=False))

    return comparison_df


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model file (use comma-separated for comparison)')
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol')
    parser.add_argument('--features-dir', type=str, default='data/features',
                       help='Directory containing features')
    parser.add_argument('--models-dir', type=str, default='data/models',
                       help='Directory containing models')
    parser.add_argument('--output-dir', type=str, default='reports',
                       help='Output directory for evaluation results')
    parser.add_argument('--no-plots', action='store_true',
                       help='Do not create visualization plots')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple models (use comma-separated model paths)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')

    args = parser.parse_args()

    # Setup logging
    setup_logger(__name__, level=getattr(logger, args.log_level))

    output_dir = Path(args.output_dir)

    try:
        if args.compare or ',' in args.model:
            # Compare multiple models
            model_paths = [m.strip() for m in args.model.split(',')]
            comparison_df = compare_models(
                model_paths,
                args.symbol,
                args.features_dir,
                args.models_dir
            )

            # Save comparison results
            ensure_dir(output_dir / 'metrics')
            comparison_path = output_dir / 'metrics' / f"{args.symbol}_model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            comparison_df.to_csv(comparison_path, index=False)
            logger.info(f"Saved comparison results to {comparison_path}")

        else:
            # Evaluate single model
            metrics = evaluate_model(
                args.model,
                args.symbol,
                args.features_dir,
                args.models_dir,
                output_dir,
                create_plots=not args.no_plots
            )

        print(f"\nEvaluation complete!")
        print(f"Output directory: {output_dir}")

    except Exception as e:
        logger.error(f"Error evaluating model: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

