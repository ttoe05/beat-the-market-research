"""
Walk-Forward Classifier

Implements a rolling/growing walk-forward validation loop for classification
models. Starts with XGBoost as the base classifier.

Preprocessing (features + labels) is the caller's responsibility:
    1. Apply FeatureEngineering.engineer_features()
    2. Generate labels via LabelGenerator.generate_labels()
    3. Join features + label into a single DataFrame
    4. Pass the joined DataFrame along with dependent_var and feature_columns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, accuracy_score
from tqdm import tqdm

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class WalkForwardClassifier:
    """
    Walk-forward validation framework for classification models.

    Runs a time-series-safe rolling or growing window validation loop,
    retraining an XGBoost classifier at a configurable interval using
    TimeSeriesSplit + GridSearchCV.

    Args:
        data: Full dataset with feature columns and label column joined.
        dependent_var: Column name of the classification label.
        feature_columns: List of feature column names.
        window_size: Training window size in rows (trading days).
        min_window_size: Minimum rows required before the first training step.
        window_type: 'sliding' (fixed-length window) or 'growing' (expanding).
        step_size: Number of steps to advance per iteration.
        model_retrain_interval: Retrain the model every N prediction steps.
        forward_period: Number of forward-looking periods used to construct
            the label. The last `forward_period` rows of each training window
            are excluded to prevent data leakage.
        use_scaling: Whether to apply StandardScaler to features.
        scoring: GridSearchCV scoring metric — 'precision' (conservative,
            default) or 'recall' (aggressive).
        xgb_param_grid: Optional override for the XGBoost hyperparameter grid.
        n_cv_splits: Number of splits for TimeSeriesSplit inside GridSearchCV.
        testing: If True, run only on the last 30 windows for quick testing.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 dependent_var: str,
                 feature_columns: List[str],
                 window_size: int = 252,
                 min_window_size: int = 126,
                 window_type: str = 'sliding',
                 step_size: int = 1,
                 model_retrain_interval: int = 20,
                 forward_period: int = 5,
                 use_scaling: bool = True,
                 scoring: str = 'precision',
                 xgb_param_grid: Optional[Dict[str, Any]] = None,
                 n_cv_splits: int = 5,
                 testing: bool = False):

        self.data = data
        self.dependent_var = dependent_var
        self.feature_columns = feature_columns
        self.window_size = window_size
        self.min_window_size = min_window_size
        self.window_type = window_type
        self.step_size = step_size
        self.model_retrain_interval = model_retrain_interval
        self.forward_period = forward_period
        self.use_scaling = use_scaling
        self.scoring = scoring
        self.n_cv_splits = n_cv_splits
        self.testing = testing

        self.param_grid = xgb_param_grid if xgb_param_grid is not None \
            else self._build_default_param_grid()

        # Model state
        self.model: Optional[XGBClassifier] = None
        self.initial_run: bool = True
        self.model_retrain_counter: int = 0

        # Scaler
        if self.use_scaling:
            self.scaler_x = StandardScaler()
            self.scalers_fitted: bool = False
        else:
            self.scaler_x = None
            self.scalers_fitted: bool = False

        # Results storage
        self.results_list: List[Dict] = []
        self.model_summaries: List[Dict] = []

        logger.info(
            f"Initialized WalkForwardClassifier: window_size={window_size}, "
            f"min_window_size={min_window_size}, window_type={window_type}, "
            f"forward_period={forward_period}, scoring={scoring}"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_default_param_grid(self) -> Dict[str, Any]:
        """Return the default XGBoost hyperparameter grid for GridSearchCV."""
        return {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.9, 1.0],
            'min_child_weight': [1, 3, 5],
        }

    def _get_time_windows(self) -> List[Tuple[int, int]]:
        """
        Generate (train_start_idx, train_end_idx) index pairs for the loop.

        Returns:
            List of (start, end) integer index tuples.
        """
        n_samples = len(self.data)
        windows: List[Tuple[int, int]] = []

        if self.window_type == 'sliding':
            start_i = max(self.window_size, self.min_window_size)
            for i in range(start_i, n_samples, self.step_size):
                start_idx = max(0, i - self.window_size)
                windows.append((start_idx, i))

        elif self.window_type == 'growing':
            for end_idx in range(self.min_window_size, n_samples, self.step_size):
                windows.append((0, end_idx))

        else:
            raise ValueError(
                f"Invalid window_type '{self.window_type}'. "
                "Must be 'sliding' or 'growing'."
            )

        if self.testing:
            logger.info(f"Testing mode: using last 30 of {len(windows)} windows")
            windows = windows[-30:]

        logger.info(f"Generated {len(windows)} {self.window_type} time windows")
        return windows

    def _train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Tuple[XGBClassifier, Dict[str, Any]]:
        """
        Train an XGBClassifier using TimeSeriesSplit + GridSearchCV.

        Optimizes for self.scoring ('precision' or 'recall') and computes
        training-set metrics (precision, recall, accuracy, class distribution).

        Args:
            X_train: Scaled feature matrix.
            y_train: Classification labels.

        Returns:
            best_model: Fitted XGBClassifier with best hyperparameters.
            model_info: Dict with best_params, cv score, and training metrics.
        """
        estimator = XGBClassifier(
            eval_metric='logloss',
            random_state=42,
            verbosity=0,
        )

        tscv = TimeSeriesSplit(n_splits=self.n_cv_splits)

        search = GridSearchCV(
            estimator=estimator,
            param_grid=self.param_grid,
            cv=tscv,
            scoring=self.scoring,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)

        best_model: XGBClassifier = search.best_estimator_

        # Training-set metrics using the best model
        y_pred_train = best_model.predict(X_train)
        train_precision = precision_score(y_train, y_pred_train, average='weighted', zero_division=0)
        train_recall = recall_score(y_train, y_pred_train, average='weighted', zero_division=0)
        train_accuracy = accuracy_score(y_train, y_pred_train)

        # Class distribution as percentage
        class_counts = y_train.value_counts()
        class_distribution = {
            int(cls): round(count / len(y_train) * 100, 2)
            for cls, count in class_counts.items()
        }

        model_info: Dict[str, Any] = {
            'best_params': search.best_params_,
            'cv_best_score': round(search.best_score_, 6),
            'train_precision': round(train_precision, 6),
            'train_recall': round(train_recall, 6),
            'train_accuracy': round(train_accuracy, 6),
            'class_distribution': class_distribution,
        }

        logger.debug(
            f"Model trained — best_score={search.best_score_:.4f}, "
            f"precision={train_precision:.4f}, recall={train_recall:.4f}, "
            f"accuracy={train_accuracy:.4f}"
        )

        return best_model, model_info

    def _predict(
        self, X_predict: pd.DataFrame
    ) -> Tuple[int, Dict[int, float]]:
        """
        Generate a class prediction and class probabilities.

        Args:
            X_predict: Single-row scaled feature DataFrame.

        Returns:
            predicted_class: Predicted class label (argmax of probabilities).
            probabilities: Dict mapping each class label to its probability.
        """
        proba = self.model.predict_proba(X_predict)[0]  # shape: (n_classes,)
        predicted_class = int(np.argmax(proba))
        classes = self.model.classes_
        probabilities = {int(cls): round(float(p), 6) for cls, p in zip(classes, proba)}

        return predicted_class, probabilities

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run_single_step(
        self,
        train_start_idx: int,
        train_end_idx: int,
        predict_idx: int,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Execute one walk-forward step: train (if due) and predict.

        Args:
            train_start_idx: Start index of the training window.
            train_end_idx: End index of the training window (exclusive).
            predict_idx: Row index of the point to predict.

        Returns:
            result: Prediction result dict for this step.
            model_summary: Model info dict (populated only on retrain steps).
        """
        # 1. Extract training window
        window_data = self.data.iloc[train_start_idx:train_end_idx]

        # 2. Prevent data leakage — drop the last `forward_period` rows whose
        #    labels overlap with future prices at/beyond the prediction point.
        if self.forward_period > 0 and len(window_data) > self.forward_period:
            window_data = window_data.iloc[:-self.forward_period]

        # 3. Split into features and label
        X_train = window_data[self.feature_columns]
        y_train = window_data[self.dependent_var]

        needs_retrain = (
            self.model_retrain_counter == self.model_retrain_interval
            or self.initial_run
        )

        # 4. Scaling — fit on retrain, transform-only otherwise
        if self.use_scaling:
            if needs_retrain:
                X_train_scaled = pd.DataFrame(
                    self.scaler_x.fit_transform(X_train),
                    columns=X_train.columns,
                    index=X_train.index,
                )
                self.scalers_fitted = True
            else:
                X_train_scaled = pd.DataFrame(
                    self.scaler_x.transform(X_train),
                    columns=X_train.columns,
                    index=X_train.index,
                )
        else:
            X_train_scaled = X_train

        # 5. Retrain if due
        model_summary: Dict[str, Any] = {}
        if needs_retrain:
            logger.debug(
                f"Retraining model on window [{train_start_idx}:{train_end_idx}]"
            )
            self.model, model_info = self._train_model(X_train_scaled, y_train)
            model_summary = model_info
            model_summary['train_start_idx'] = train_start_idx
            model_summary['train_end_idx'] = train_end_idx
            self.model_retrain_counter = 0
            if self.initial_run:
                self.initial_run = False
        # else: reuse self.model

        # 6. Scale prediction point
        X_predict = self.data.iloc[[predict_idx]][self.feature_columns]
        if self.use_scaling and self.scalers_fitted:
            X_predict_scaled = pd.DataFrame(
                self.scaler_x.transform(X_predict),
                columns=X_predict.columns,
                index=X_predict.index,
            )
        else:
            X_predict_scaled = X_predict

        # 7. Predict
        predicted_class, probabilities = self._predict(X_predict_scaled)

        # 8. Actual label and date
        actual = int(self.data.iloc[predict_idx][self.dependent_var])
        predict_date = self.data.index[predict_idx]

        # 9. Build result
        result: Dict[str, Any] = {
            'date': predict_date,
            'actual': actual,
            'predicted_class': predicted_class,
            'probabilities': probabilities,
            'retrain': bool(model_summary),
        }

        # 10. Increment retrain counter
        self.model_retrain_counter += 1

        logger.debug(
            f"Step {predict_date}: actual={actual}, predicted={predicted_class}"
        )

        return result, model_summary

    def run_walk_forward_validation(self) -> None:
        """
        Run the complete walk-forward validation loop.

        Iterates over all time windows, calling run_single_step() at each
        position. Results are stored in self.results_list and
        self.model_summaries.
        """
        windows = self._get_time_windows()
        n_windows = len(windows)

        logger.info(
            f"Starting walk-forward classification validation — "
            f"{n_windows} steps, window_type={self.window_type}, "
            f"scoring={self.scoring}"
        )

        for train_start_idx, train_end_idx in tqdm(
            windows,
            desc='Walk-Forward Classifier',
            total=n_windows,
        ):
            predict_idx = train_end_idx

            if predict_idx >= len(self.data):
                logger.warning(
                    f"Prediction index {predict_idx} out of bounds — stopping."
                )
                break

            result, model_summary = self.run_single_step(
                train_start_idx=train_start_idx,
                train_end_idx=train_end_idx,
                predict_idx=predict_idx,
            )

            self.results_list.append(result)
            if model_summary:
                model_summary['date'] = result['date']
                self.model_summaries.append(model_summary)

        logger.info(
            f"Walk-forward validation complete — "
            f"{len(self.results_list)} predictions recorded"
        )

    def export_results(self, filepath: str) -> None:
        """
        Export prediction results and model summaries to parquet files.

        Args:
            filepath: Directory path where output files will be written.
        """
        output_dir = Path(filepath)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export prediction results
        try:
            results_df = pd.DataFrame(self.results_list)
            results_path = output_dir / 'walk_forward_classifier_results.parquet'
            results_df.to_parquet(results_path, index=False)
            logger.info(f"Results exported to {results_path}")
        except Exception as e:
            logger.error(f"Error exporting results: {e}")

        # Export model summaries
        try:
            summary_df = pd.DataFrame(self.model_summaries)
            summary_path = output_dir / 'walk_forward_classifier_model_summary.parquet'
            summary_df.to_parquet(summary_path, index=False)
            logger.info(f"Model summaries exported to {summary_path}")
        except Exception as e:
            logger.error(f"Error exporting model summaries: {e}")
