"""
Model Evaluator Module

Evaluates classification models with standard ML metrics and trading-specific metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    log_loss,
    matthews_corrcoef,
    balanced_accuracy_score
)
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class ModelEvaluator:
    """
    Evaluate classification models with comprehensive metrics.

    Provides both standard ML metrics and trading-specific interpretations.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> evaluator = ModelEvaluator()
        >>>
        >>> # Mock predictions
        >>> y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
        >>> y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
        >>>
        >>> metrics = evaluator.evaluate_predictions(y_true, y_pred)
        >>> print(f"Accuracy: {metrics['accuracy']:.3f}")
        Accuracy: 0.800
    """

    def __init__(self):
        """Initialize ModelEvaluator."""
        logger.info("Initialized ModelEvaluator")

    def evaluate(
        self,
        model: Any,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Evaluate a fitted model on test data.

        Args:
            model: Fitted model with predict and predict_proba methods
            X: Test features
            y: True labels

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating model on {len(X)} samples")

        # Get predictions
        y_pred = model.predict(X)

        # Get probabilities if available
        y_proba = None
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X)
            except Exception as e:
                logger.warning(f"Could not get prediction probabilities: {e}")

        # Evaluate
        metrics = self.evaluate_predictions(y, y_pred, y_proba)

        logger.info(f"Evaluation complete: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")

        return metrics

    def evaluate_predictions(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Evaluate predictions with comprehensive metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)

        Returns:
            Dictionary of evaluation metrics
        """
        # Convert to numpy arrays
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values

        # Determine if binary or multiclass
        n_classes = len(np.unique(y_true))

        if n_classes == 2:
            return self._evaluate_binary(y_true, y_pred, y_proba)
        else:
            return self._evaluate_multiclass(y_true, y_pred, y_proba)

    def _evaluate_binary(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Evaluate binary classification predictions.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm

        # Extract confusion matrix components
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_positives'] = int(tp)
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)

            # Trading-specific metrics
            metrics['buy_precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
            metrics['buy_recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics['hold_precision'] = tn / (tn + fn) if (tn + fn) > 0 else 0
            metrics['hold_recall'] = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Probability-based metrics
        if y_proba is not None:
            try:
                # ROC AUC
                if len(y_proba.shape) == 2:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba)

                # Log loss
                metrics['log_loss'] = log_loss(y_true, y_proba)
            except Exception as e:
                logger.warning(f"Could not calculate probability-based metrics: {e}")

        # Support (number of samples per class)
        unique, counts = np.unique(y_true, return_counts=True)
        metrics['support'] = dict(zip(unique.tolist(), counts.tolist()))

        return metrics

    def _evaluate_multiclass(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Evaluate multiclass classification predictions.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Basic metrics (macro average)
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)

        # Weighted metrics
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

        metrics['precision_per_class'] = precision_per_class.tolist()
        metrics['recall_per_class'] = recall_per_class.tolist()
        metrics['f1_per_class'] = f1_per_class.tolist()

        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)

        # Probability-based metrics
        if y_proba is not None:
            try:
                # ROC AUC (one-vs-rest)
                metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
                metrics['roc_auc_ovo'] = roc_auc_score(y_true, y_proba, multi_class='ovo')
                metrics['log_loss'] = log_loss(y_true, y_proba)
            except Exception as e:
                logger.warning(f"Could not calculate probability-based metrics: {e}")

        # Support
        unique, counts = np.unique(y_true, return_counts=True)
        metrics['support'] = dict(zip(unique.tolist(), counts.tolist()))

        return metrics

    def print_report(self, metrics: Dict[str, Any]) -> None:
        """
        Print formatted evaluation report.

        Args:
            metrics: Dictionary of metrics from evaluate()
        """
        print("\n" + "=" * 70)
        print("MODEL EVALUATION REPORT")
        print("=" * 70)

        # Basic metrics
        print("\nClassification Metrics:")
        print(f"  Accuracy:          {metrics.get('accuracy', 0):.4f}")

        if 'balanced_accuracy' in metrics:
            print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")

        if 'precision' in metrics:  # Binary classification
            print(f"  Precision:         {metrics['precision']:.4f}")
            print(f"  Recall:            {metrics['recall']:.4f}")
            print(f"  F1 Score:          {metrics['f1']:.4f}")

            if 'mcc' in metrics:
                print(f"  Matthews Corr:     {metrics['mcc']:.4f}")

        elif 'precision_macro' in metrics:  # Multiclass
            print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
            print(f"  Recall (macro):    {metrics['recall_macro']:.4f}")
            print(f"  F1 Score (macro):  {metrics['f1_macro']:.4f}")

        # Probability-based metrics
        if 'roc_auc' in metrics:
            print(f"  ROC AUC:           {metrics['roc_auc']:.4f}")
        if 'log_loss' in metrics:
            print(f"  Log Loss:          {metrics['log_loss']:.4f}")

        # Confusion matrix
        if 'confusion_matrix' in metrics:
            print("\nConfusion Matrix:")
            cm = metrics['confusion_matrix']
            print(cm)

        # Binary classification details
        if 'true_positives' in metrics:
            print("\nDetailed Breakdown:")
            print(f"  True Positives:    {metrics['true_positives']}")
            print(f"  True Negatives:    {metrics['true_negatives']}")
            print(f"  False Positives:   {metrics['false_positives']}")
            print(f"  False Negatives:   {metrics['false_negatives']}")

            print("\nTrading Signal Interpretation:")
            print(f"  Correct BUY:       {metrics['true_positives']}")
            print(f"  Correct HOLD:      {metrics['true_negatives']}")
            print(f"  Wrong BUY:         {metrics['false_positives']} (bought but should hold)")
            print(f"  Missed BUY:        {metrics['false_negatives']} (held but should buy)")

            if 'buy_precision' in metrics:
                print("\nBuy Signal Quality:")
                print(f"  Buy Precision:     {metrics['buy_precision']:.4f} (% of buy signals that were correct)")
                print(f"  Buy Recall:        {metrics['buy_recall']:.4f} (% of opportunities captured)")

        # Support
        if 'support' in metrics:
            print("\nClass Distribution:")
            for cls, count in metrics['support'].items():
                print(f"  Class {cls}: {count} samples")

        print("=" * 70 + "\n")

    def compare_models(
        self,
        models_metrics: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Compare metrics across multiple models.

        Args:
            models_metrics: Dictionary mapping model names to their metrics

        Returns:
            DataFrame with comparison
        """
        comparison_data = []

        for model_name, metrics in models_metrics.items():
            row = {'model': model_name}

            # Extract key metrics
            for key in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'balanced_accuracy']:
                if key in metrics:
                    row[key] = metrics[key]
                elif f'{key}_macro' in metrics:
                    row[key] = metrics[f'{key}_macro']

            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        # Sort by F1 score if available
        if 'f1' in df.columns:
            df = df.sort_values('f1', ascending=False)
        elif 'accuracy' in df.columns:
            df = df.sort_values('accuracy', ascending=False)

        return df

    def get_classification_report(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        target_names: Optional[list] = None
    ) -> str:
        """
        Get sklearn classification report.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            target_names: Names for classes

        Returns:
            Classification report string
        """
        return classification_report(y_true, y_pred, target_names=target_names)


if __name__ == "__main__":
    # Example usage
    import logging
    from ..utils.logger import setup_logger

    setup_logger(__name__, level=logging.INFO)

    # Create sample predictions
    np.random.seed(42)

    # Binary classification example
    print("=" * 70)
    print("BINARY CLASSIFICATION EXAMPLE")
    print("=" * 70)

    n_samples = 100
    y_true = np.random.randint(0, 2, n_samples)
    y_pred = y_true.copy()
    # Add some errors
    error_idx = np.random.choice(n_samples, size=20, replace=False)
    y_pred[error_idx] = 1 - y_pred[error_idx]

    # Create probabilities
    y_proba = np.column_stack([1 - y_pred, y_pred]) + np.random.randn(n_samples, 2) * 0.1
    y_proba = np.clip(y_proba, 0, 1)
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_predictions(y_true, y_pred, y_proba)
    evaluator.print_report(metrics)

    # Multiclass classification example
    print("\n" + "=" * 70)
    print("MULTICLASS CLASSIFICATION EXAMPLE")
    print("=" * 70)

    y_true_multi = np.random.randint(0, 3, n_samples)  # 3 classes: sell, hold, buy
    y_pred_multi = y_true_multi.copy()
    error_idx = np.random.choice(n_samples, size=25, replace=False)
    y_pred_multi[error_idx] = np.random.randint(0, 3, 25)

    y_proba_multi = np.random.randn(n_samples, 3)
    y_proba_multi = np.exp(y_proba_multi) / np.exp(y_proba_multi).sum(axis=1, keepdims=True)

    metrics_multi = evaluator.evaluate_predictions(y_true_multi, y_pred_multi, y_proba_multi)
    evaluator.print_report(metrics_multi)

    # Model comparison example
    print("\n" + "=" * 70)
    print("MODEL COMPARISON EXAMPLE")
    print("=" * 70)

    models_metrics = {
        'Model A': metrics,
        'Model B': {'accuracy': 0.75, 'precision': 0.72, 'recall': 0.78, 'f1': 0.75, 'roc_auc': 0.80}
    }

    comparison = evaluator.compare_models(models_metrics)
    print("\nModel Comparison:")
    print(comparison.to_string(index=False))

