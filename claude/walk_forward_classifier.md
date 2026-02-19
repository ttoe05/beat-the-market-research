# Walk-Forward Classifier Plan

## Overview

Implement `WalkForwardClassifier` in `src/backtesting/walk_forward_classifier.py`.
The class runs a rolling/growing walk-forward validation loop for classification models,
starting with XGBoost. The design mirrors the structure of the existing
`WalkForwardValidator` in `src/backtesting/walk_forward.py` but is rebuilt
specifically for the classification flow shown in `walkforward_classifier_flow.png`.

---

## Preprocessing (Caller's Responsibility)

Before instantiating `WalkForwardClassifier`, the caller must:

1. **Apply feature engineering** via `FeatureEngineering.engineer_features()`
   (`src/features/feature_engineering.py`)
2. **Generate classification labels** via `LabelGenerator.generate_labels()`
   (`src/features/label_generator.py`) — supports `binary` or `multiclass`
3. **Join features + label** into a single DataFrame. The label column name is
   passed as `dependent_var`. All other selected columns are treated as features.
4. **Pass the joined DataFrame** to `WalkForwardClassifier` along with
   `dependent_var` and `feature_columns`.

This keeps the validator stateless with respect to feature/label construction and
consistent with the existing `WalkForwardValidator` pattern.

---

## Class: `WalkForwardClassifier`

**File:** `src/backtesting/walk_forward_classifier.py`

### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | `pd.DataFrame` | required | Full dataset with features + label column |
| `dependent_var` | `str` | required | Column name of the classification label |
| `feature_columns` | `List[str]` | required | Feature column names |
| `window_size` | `int` | `252` | Training window size (trading days) |
| `min_window_size` | `int` | `126` | Minimum rows required to start training |
| `window_type` | `str` | `'sliding'` | `'sliding'` or `'growing'` |
| `step_size` | `int` | `1` | How many steps to advance per iteration |
| `model_retrain_interval` | `int` | `20` | Retrain model every N steps |
| `forward_period` | `int` | `5` | Periods used by label — last `m` rows excluded from training window to prevent data leakage |
| `use_scaling` | `bool` | `True` | Apply `StandardScaler` to features |
| `scoring` | `str` | `'precision'` | GridSearchCV metric: `'precision'` (default/conservative) or `'recall'` (aggressive) |
| `xgb_param_grid` | `Optional[Dict]` | `None` | Override default XGBoost hyperparameter grid |
| `n_cv_splits` | `int` | `5` | Number of splits for `TimeSeriesSplit` inside GridSearchCV |
| `testing` | `bool` | `False` | If True, run only on the last 30 windows |

### Instance Attributes (set in `__init__`)

```
self.data               # Full DataFrame
self.dependent_var      # Label column name
self.feature_columns    # Feature column names
self.model              # Current best XGBClassifier (updated on retrain)
self.scaler_x           # StandardScaler for features (or None)
self.scalers_fitted     # bool
self.results_list       # List[Dict] — one entry per prediction step
self.model_summaries    # List[Dict] — one entry per retrain event
self.initial_run        # bool — True until first train completes
self.model_retrain_counter  # int
```

---

## Methods

### 1. `_get_time_windows() -> List[Tuple[int, int]]`

Mirrors `DataLoader.get_time_windows()` but operates directly on `self.data`.

- **Sliding**: `start = max(0, i - window_size)`, `end = i` for
  `i in range(max(window_size, min_window_size), n_samples)`
- **Growing**: `start = 0`, `end` increments from `min_window_size` to `n_samples`
- If `testing=True`, return only the last 30 windows.

---

### 2. `_build_default_param_grid() -> Dict`

Returns the default XGBoost hyperparameter grid used by GridSearchCV:

```python
{
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.9, 1.0],
    'min_child_weight': [1, 3, 5],
}
```

If `xgb_param_grid` is passed at construction, that grid is used instead.

---

### 3. `_train_model(X_train, y_train) -> Tuple[XGBClassifier, Dict]`

Implements the **Model Training block** from the flow diagram.

**Steps:**

1. Instantiate `XGBClassifier(use_label_encoder=False, eval_metric='logloss')`
2. Create `TimeSeriesSplit(n_splits=self.n_cv_splits)`
3. Run `GridSearchCV(estimator, param_grid, cv=tscv, scoring=self.scoring, n_jobs=-1)`
4. Extract `best_estimator_` as the model for this window
5. Compute training metrics on the full `X_train / y_train` using the best estimator:
   - `precision_score` (weighted)
   - `recall_score` (weighted)
   - `accuracy_score`
   - Class distribution as a percentage dict (e.g. `{0: 45.2%, 1: 54.8%}`)
6. Return `(best_model, model_info_dict)`

`model_info_dict` structure:
```python
{
    'best_params': {...},
    'cv_best_score': float,
    'train_precision': float,
    'train_recall': float,
    'train_accuracy': float,
    'class_distribution': {class_label: pct, ...},
}
```

---

### 4. `_predict(X_predict) -> Tuple[int, np.ndarray]`

Implements the **Predict** step from the flow diagram.

1. Call `self.model.predict_proba(X_predict)` → class probabilities for each class
2. Select predicted class via `np.argmax(probabilities)`
3. Return `(predicted_class, probabilities)`

---

### 5. `run_single_step(train_start_idx, train_end_idx, predict_idx) -> Tuple[Dict, Dict]`

Runs one step of the walk-forward loop.

**Steps:**

1. **Extract training window** from `self.data.iloc[train_start_idx:train_end_idx]`

2. **Prevent data leakage**: Drop the last `forward_period` rows from the
   training window. Because labels are forward-looking (e.g. 5-day forward return),
   the last `m` rows of any window have labels that overlap with the prediction
   target, introducing leakage. This mirrors the diagram note:
   *"Eliminate t_i − m records to prevent data leakage."*

3. **Split** cleaned window into `X_train` (feature columns) and
   `y_train` (dependent_var).

4. **Apply StandardScaler** to `X_train` (fit on retrain, transform otherwise).
   Labels are not scaled — they are discrete classes.

5. **Retrain check** (`model_retrain_counter == model_retrain_interval or initial_run`):
   - If retrain: call `_train_model(X_train_scaled, y_train)`, store best model
     and model summary, reset counter.
   - If not retrain: skip training, reuse existing `self.model`.

6. **Scale prediction point** `X_predict` using fitted scaler (transform only).

7. **Predict**: call `_predict(X_predict_scaled)` → `(predicted_class, probabilities)`

8. **Inverse transform** features back if needed for result storage (note: the
   scaler is on features only, so the predicted class and probabilities need
   no inverse transform).

9. **Get actual label** from `self.data.iloc[predict_idx][self.dependent_var]`.

10. **Build result dict**:
    ```python
    {
        'date': predict_date,
        'actual': int,
        'predicted_class': int,
        'probabilities': {class_label: float, ...},
        'retrain': bool,
    }
    ```

11. **Increment** `model_retrain_counter`.

12. Return `(result, model_summary)` — model_summary is empty dict on non-retrain steps.

---

### 6. `run_walk_forward_validation() -> None`

Main orchestration loop.

```
windows = _get_time_windows()
for (train_start_idx, train_end_idx) in windows:
    predict_idx = train_end_idx
    if predict_idx >= len(self.data): break
    result, model_summary = run_single_step(train_start_idx, train_end_idx, predict_idx)
    self.results_list.append(result)
    if model_summary:
        self.model_summaries.append(model_summary)
```

Wraps the loop with `tqdm` progress bar identical to the existing validator.
Logs start/end and total number of steps.

---

### 7. `export_results(filepath: str) -> None`

Exports results after validation is complete.

- `results_df = pd.DataFrame(self.results_list)` → saved as
  `{filepath}/walk_forward_classifier_results.parquet`
- `model_summary_df = pd.DataFrame(self.model_summaries)` → saved as
  `{filepath}/walk_forward_classifier_model_summary.parquet`
- Creates parent directories if they do not exist.
- Logs success or error on each export.

---

## Data Leakage Design Note

The label at row `t` is based on the forward return over `forward_period` days
starting from `t`. This means rows `[train_end_idx - forward_period : train_end_idx]`
in the training window have labels that are computed using prices that extend into
or beyond the prediction date. Excluding these rows (via
`window_data = window_data.iloc[:-forward_period]`) ensures the model never sees
information from the future during training.

---

## File Structure

```
src/
└── backtesting/
    ├── walk_forward.py          # Existing (regression / forecasting)
    └── walk_forward_classifier.py   # New (classification)
```

---

## Dependencies

```
xgboost
scikit-learn  (GridSearchCV, TimeSeriesSplit, StandardScaler, precision_score,
               recall_score, accuracy_score)
pandas
numpy
tqdm
```

All already present in `requirements.txt`.

---

## Example Usage

```python
from src.features.feature_engineering import FeatureEngineering
from src.features.label_generator import LabelGenerator
from src.backtesting.walk_forward_classifier import WalkForwardClassifier

# 1. Build features
fe = FeatureEngineering()
features = fe.engineer_features(raw_data)

# 2. Generate labels
lg = LabelGenerator(forward_period=5, threshold=0.02)
labels = lg.generate_binary_labels(raw_data)

# 3. Join and drop NaNs
dataset = features.join(labels).dropna()
feature_cols = [c for c in dataset.columns if c != 'label']

# 4. Run walk-forward validation
wfc = WalkForwardClassifier(
    data=dataset,
    dependent_var='label',
    feature_columns=feature_cols,
    window_size=252,
    min_window_size=126,
    window_type='sliding',
    forward_period=5,
    scoring='precision',   # or 'recall' for aggressive mode
    model_retrain_interval=20,
)
wfc.run_walk_forward_validation()
wfc.export_results('results/walk_forward')
```
