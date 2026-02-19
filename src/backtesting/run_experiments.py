"""
Walk-Forward Classifier Experiment Runner

Fetches market data, builds features and labels, then runs the
WalkForwardClassifier for a given ticker symbol.

Usage:
    python -m src.backtesting.run_experiments --symbol AAPL

    # Multiclass labels, recall scoring, growing window
    python -m src.backtesting.run_experiments \\
        --symbol AAPL \\
        --start 2015-01-01 \\
        --end 2024-12-31 \\
        --label-type multiclass \\
        --scoring recall \\
        --window-type growing \\
        --output-dir results/walk_forward

    # Quick smoke-test on last 30 windows
    python -m src.backtesting.run_experiments --symbol AAPL --testing
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

# Allow running from the project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_collector import DataCollector
from src.features.feature_engineering import FeatureEngineering
from src.features.label_generator import LabelGenerator
from src.backtesting.walk_forward_classifier import WalkForwardClassifier
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run walk-forward classification experiment for a ticker symbol.'
    )

    # Data
    parser.add_argument('--symbol', type=str, required=True,
                        help='Stock ticker symbol (e.g. AAPL)')
    parser.add_argument('--start', type=str, default='2015-01-01',
                        help='Data start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=datetime.today().strftime('%Y-%m-%d'),
                        help='Data end date (YYYY-MM-DD)')

    # Labels
    parser.add_argument('--label-type', type=str, default='binary',
                        choices=['binary', 'multiclass'],
                        help='Classification label type')
    parser.add_argument('--forward-period', type=int, default=5,
                        help='Forward-looking periods used for label construction')
    parser.add_argument('--threshold', type=float, default=0.02,
                        help='Return threshold for buy/sell signals')

    # Walk-forward window
    parser.add_argument('--window-size', type=int, default=252,
                        help='Training window size in trading days')
    parser.add_argument('--min-window-size', type=int, default=126,
                        help='Minimum rows required before the first training step')
    parser.add_argument('--window-type', type=str, default='sliding',
                        choices=['sliding', 'growing'],
                        help='Window expansion strategy')
    parser.add_argument('--step-size', type=int, default=1,
                        help='Steps to advance per iteration')

    # Model / training
    parser.add_argument('--model-retrain-interval', type=int, default=20,
                        help='Retrain the model every N prediction steps')
    parser.add_argument('--scoring', type=str, default='precision',
                        choices=['precision', 'recall'],
                        help='GridSearchCV scoring metric')
    parser.add_argument('--n-cv-splits', type=int, default=5,
                        help='Number of TimeSeriesSplit folds for GridSearchCV')
    parser.add_argument('--no-scaling', action='store_true',
                        help='Disable StandardScaler on features')

    # Output
    parser.add_argument('--output-dir', type=str, default='results/walk_forward',
                        help='Directory to write result parquet files')

    # Misc
    parser.add_argument('--testing', action='store_true',
                        help='Run on the last 30 windows only (smoke test)')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging verbosity')

    return parser.parse_args()


def fetch_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance via DataCollector."""
    collector = DataCollector(source='yahoo')
    raw_data = collector.fetch_data(symbol=symbol, start=start, end=end)
    logger.info(f"Fetched {len(raw_data)} rows for {symbol} ({start} → {end})")
    return raw_data


def build_dataset(
    raw_data: pd.DataFrame,
    label_type: str,
    forward_period: int,
    threshold: float,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Apply feature engineering and label generation, then join into one DataFrame.

    Args:
        raw_data: OHLCV DataFrame.
        label_type: 'binary' or 'multiclass'.
        forward_period: Look-ahead periods for label construction.
        threshold: Return threshold for buy/sell classification.

    Returns:
        dataset: Combined features + label DataFrame with NaNs dropped.
        feature_cols: List of feature column names.
    """
    # 1. Engineer features
    fe = FeatureEngineering()
    features = fe.engineer_features(raw_data)
    logger.info(f"Feature engineering complete: {features.shape[1]} features")

    # 2. Generate labels
    lg = LabelGenerator(forward_period=forward_period, threshold=threshold)
    labels = lg.generate_labels(raw_data, label_type=label_type)
    logger.info(f"Labels generated — type={label_type}, distribution:\n"
                f"{labels.value_counts().to_string()}")

    # 3. Join and clean
    dataset = features.join(labels).dropna()
    feature_cols = [c for c in dataset.columns if c != 'label']

    logger.info(
        f"Dataset ready: {len(dataset)} rows, {len(feature_cols)} features, "
        f"label column='label'"
    )
    return dataset, feature_cols


def run_experiment(args: argparse.Namespace) -> None:
    """End-to-end experiment: fetch → build → validate → export."""

    logger.info(
        f"Starting experiment — symbol={args.symbol}, "
        f"label_type={args.label_type}, scoring={args.scoring}, "
        f"window_type={args.window_type}"
    )

    # Step 1: Fetch raw market data
    raw_data = fetch_data(args.symbol, args.start, args.end)

    # Step 2: Build feature + label dataset
    dataset, feature_cols = build_dataset(
        raw_data=raw_data,
        label_type=args.label_type,
        forward_period=args.forward_period,
        threshold=args.threshold,
    )

    # Step 3: Initialise walk-forward classifier
    wfc = WalkForwardClassifier(
        data=dataset,
        dependent_var='label',
        feature_columns=feature_cols,
        window_size=args.window_size,
        min_window_size=args.min_window_size,
        window_type=args.window_type,
        step_size=args.step_size,
        model_retrain_interval=args.model_retrain_interval,
        forward_period=args.forward_period,
        use_scaling=not args.no_scaling,
        scoring=args.scoring,
        n_cv_splits=args.n_cv_splits,
        testing=args.testing,
    )

    # Step 4: Run validation loop
    wfc.run_walk_forward_validation()

    # Step 5: Export results
    output_dir = Path(args.output_dir) / args.symbol
    wfc.export_results(str(output_dir))

    # Step 6: Print a brief summary
    results_df = pd.DataFrame(wfc.results_list)
    if not results_df.empty:
        correct = (results_df['actual'] == results_df['predicted_class']).sum()
        total = len(results_df)
        accuracy = correct / total * 100
        print("\n" + "=" * 60)
        print("WALK-FORWARD EXPERIMENT SUMMARY")
        print("=" * 60)
        print(f"  Symbol          : {args.symbol}")
        print(f"  Date range      : {args.start} → {args.end}")
        print(f"  Label type      : {args.label_type}")
        print(f"  Scoring         : {args.scoring}")
        print(f"  Window type     : {args.window_type} ({args.window_size} days)")
        print(f"  Total steps     : {total}")
        print(f"  Overall accuracy: {accuracy:.2f}%")
        print(f"  Results saved to: {output_dir}")
        print("=" * 60)


def main() -> None:
    args = parse_args()

    import logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    try:
        run_experiment(args)
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
# run_experiments.py [-h] --symbol SYMBOL [--start START] [--end END]
#                           [--label-type {binary,multiclass}]
#                           [--forward-period FORWARD_PERIOD]
#                           [--threshold THRESHOLD] [--window-size WINDOW_SIZE]
#                           [--min-window-size MIN_WINDOW_SIZE]
#                           [--window-type {sliding,growing}]
#                           [--step-size STEP_SIZE]
#                           [--model-retrain-interval MODEL_RETRAIN_INTERVAL]
#                           [--scoring {precision,recall}]
#                           [--n-cv-splits N_CV_SPLITS] [--no-scaling]
#                           [--output-dir OUTPUT_DIR] [--testing]
#                           [--log-level {DEBUG,INFO,WARNING,ERROR}]
# run_experiments.py: error: the following arguments are required: --symbol