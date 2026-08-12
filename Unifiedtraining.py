"""
Unified benchmarking pipeline across all PeptiVerse endpoints.

Auto-detects, per embedding-variant parquet file:
  - which column(s) hold embeddings (list/array-typed columns) -> concatenated
    if more than one (handles binding_affinity's paired target+binder case)
  - which column holds the label/target
  - whether the task is classification or regression (cross-checked against
    the hardcoded ENDPOINTS config below — mismatches are flagged, not trained)

Run from the directory that directly contains binding_affinity/, half_life/,
hemolysis/, nf/, permeability_caco2/, permeability_pampa/,
permeability_penetrance/, solubility/ :

    python train_all_endpoints.py                 # inspect + train everything
    python train_all_endpoints.py --inspect-only   # just print detected schema, don't train
    python train_all_endpoints.py --endpoint hemolysis   # just one endpoint
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    RandomForestRegressor, ExtraTreesRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix, r2_score, mean_squared_error, mean_absolute_error,
)
import xgboost as xgb

warnings.filterwarnings("ignore")

ROOT = Path(".")
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Endpoint -> expected task type, per the audit doc. Used as a cross-check,
# not a blind assumption — detected label cardinality is compared against this.
ENDPOINTS = {
    "hemolysis": "classification",
    "nf": "classification",
    "permeability_penetrance": "classification",
    "solubility": "classification",
    "half_life": "regression",
    "permeability_caco2": "regression",
    "permeability_pampa": "regression",
    "binding_affinity": "regression",
}

CLASSIFICATION_MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=2000, n_jobs=-1),
    "RandomForest": RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=500, n_jobs=-1, random_state=42),
    "SVM": SVC(probability=True, kernel="rbf", random_state=42),
    "XGBoost": xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                                   eval_metric="logloss", n_jobs=-1, random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=500, random_state=42),
}

REGRESSION_MODELS = {
    "LinearRegression": LinearRegression(n_jobs=-1),
    "RandomForest": RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=500, n_jobs=-1, random_state=42),
    "SVR": SVR(kernel="rbf"),
    "XGBoost": xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                                  n_jobs=-1, random_state=42),
    "MLP": MLPRegressor(hidden_layer_sizes=(256, 64), max_iter=500, random_state=42),
}


# --------------------------------------------------------------------------
# Schema auto-detection
# --------------------------------------------------------------------------
def _is_embedding_column(series: pd.Series) -> bool:
    sample = series.dropna().iloc[0] if series.notna().any() else None
    return isinstance(sample, (list, np.ndarray))


def detect_schema(df: pd.DataFrame):
    """Returns (embedding_cols: list[str], label_col: str, task_hint: str)"""
    embedding_cols = [c for c in df.columns if _is_embedding_column(df[c])]

    non_embedding = [c for c in df.columns if c not in embedding_cols]
    # Prefer common label-ish names; fall back to first numeric non-id column.
    label_priority = ["label", "Label", "value", "Value", "target", "Target",
                       "affinity", "half_life", "y"]
    label_col = None
    for cand in label_priority:
        if cand in non_embedding:
            label_col = cand
            break
    if label_col is None:
        for c in non_embedding:
            if pd.api.types.is_numeric_dtype(df[c]) and c.lower() not in ("id", "index", "cluster_id"):
                label_col = c
                break

    task_hint = None
    if label_col is not None:
        n_unique = df[label_col].nunique()
        task_hint = "classification" if n_unique <= 2 else "regression"

    return embedding_cols, label_col, task_hint


def load_features_labels(parquet_path: Path):
    df = pd.read_parquet(parquet_path)
    embedding_cols, label_col, task_hint = detect_schema(df)

    if not embedding_cols or label_col is None:
        return None, None, None, {
            "columns": list(df.columns), "embedding_cols": embedding_cols,
            "label_col": label_col, "task_hint": task_hint, "shape": df.shape,
        }

    feature_blocks = [np.stack(df[c].values) for c in embedding_cols]
    X = np.concatenate(feature_blocks, axis=1) if len(feature_blocks) > 1 else feature_blocks[0]
    y = df[label_col].values

    info = {
        "columns": list(df.columns), "embedding_cols": embedding_cols,
        "label_col": label_col, "task_hint": task_hint, "shape": df.shape,
        "feature_dim": X.shape[1],
    }
    return X, y, task_hint, info


def discover_variants(endpoint_dir: Path):
    """Find embedding-variant subfolders containing train/val parquet files."""
    variants = []
    for sub in sorted(endpoint_dir.iterdir()):
        if not sub.is_dir():
            continue
        train_files = list(sub.glob("train-*.parquet"))
        val_files = list(sub.glob("val-*.parquet"))
        if train_files and val_files:
            variants.append((sub.name, train_files[0], val_files[0]))
    return variants


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def classification_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Sensitivity": recall_score(y_true, y_pred),
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "AUROC": roc_auc_score(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob),
    }


def regression_metrics(y_true, y_pred):
    pearson_r, _ = pearsonr(y_true, y_pred)
    spearman_r, _ = spearmanr(y_true, y_pred)
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "Pearson_r": pearson_r,
        "Spearman_r": spearman_r,
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def run_endpoint(endpoint_name, task_type, inspect_only=False):
    endpoint_dir = ROOT / endpoint_name
    if not endpoint_dir.exists():
        print(f"  [skip] {endpoint_name}: directory not found")
        return None

    variants = discover_variants(endpoint_dir)
    if not variants:
        print(f"  [skip] {endpoint_name}: no embedding-variant subfolders found")
        return None

    model_suite = CLASSIFICATION_MODELS if task_type == "classification" else REGRESSION_MODELS
    metric_fn = classification_metrics if task_type == "classification" else regression_metrics
    all_results = []

    for variant_name, train_path, val_path in variants:
        print(f"\n  Variant: {variant_name}")
        X_train, y_train, task_hint, train_info = load_features_labels(train_path)
        X_val, y_val, _, val_info = load_features_labels(val_path)

        print(f"    Detected — embedding cols: {train_info['embedding_cols']}, "
              f"label col: '{train_info['label_col']}', "
              f"feature dim: {train_info.get('feature_dim')}, "
              f"detected task: {task_hint} (expected: {task_type})")

        if X_train is None:
            print(f"    ⚠️  Could not auto-detect schema — columns were: {train_info['columns']}")
            continue

        if task_hint != task_type:
            print(f"    ⚠️  MISMATCH: expected '{task_type}' but label looks like '{task_hint}' "
                  f"({len(set(y_train))} unique values). Skipping this variant — verify manually.")
            continue

        if inspect_only:
            continue

        for model_name, model in model_suite.items():
            model_instance = type(model)(**model.get_params())
            print(f"    Training {model_name}...")
            model_instance.fit(X_train, y_train)

            if task_type == "classification":
                y_prob = model_instance.predict_proba(X_val)[:, 1]
                y_pred = (y_prob >= 0.5).astype(int)
                metrics = metric_fn(y_val, y_pred, y_prob)
            else:
                y_pred = model_instance.predict(X_val)
                metrics = metric_fn(y_val, y_pred)

            metrics["endpoint"] = endpoint_name
            metrics["variant"] = variant_name
            metrics["model"] = model_name
            all_results.append(metrics)

            primary = metrics.get("MCC", metrics.get("R2"))
            print(f"      primary_metric={primary:.4f}")

    if inspect_only or not all_results:
        return None

    df = pd.DataFrame(all_results)
    sort_col = "MCC" if task_type == "classification" else "R2"
    df = df.sort_values(sort_col, ascending=False)
    out_path = RESULTS_DIR / f"leaderboard_{endpoint_name}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved leaderboard to {out_path}")
    print(df.head(5).to_string(index=False))
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect-only", action="store_true",
                         help="Only print detected schema per variant, don't train.")
    parser.add_argument("--endpoint", type=str, default=None,
                         help="Run only this endpoint (default: all).")
    args = parser.parse_args()

    targets = {args.endpoint: ENDPOINTS[args.endpoint]} if args.endpoint else ENDPOINTS

    for endpoint_name, task_type in targets.items():
        print(f"\n{'='*90}\nENDPOINT: {endpoint_name}  (expected task: {task_type})\n{'='*90}")
        run_endpoint(endpoint_name, task_type, inspect_only=args.inspect_only)


if __name__ == "__main__":
    main()
