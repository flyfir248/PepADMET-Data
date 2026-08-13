"""
Solubility — binary classification.

Two embedding variants confirmed from the project audit:
  solubility__sol_chemberta_with_embeddings  (384-dim ChemBERTa)
  solubility__sol_wt_with_embeddings          (1280-dim WT)

Schema confirmed identical to permeability_penetrance: sequence, embedding, label.

Run from the directory containing solubility/:
    python train_solubility.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix,
)
import xgboost as xgb

warnings.filterwarnings("ignore")

ROOT = Path(".")
ENDPOINT_DIR = ROOT / "solubility"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

VARIANTS = [
    "sol_chemberta_with_embeddings",
    "sol_wt_with_embeddings",
]

LABEL_CANDIDATES = ["label", "Label"]

MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=2000, n_jobs=-1),
    "RandomForest": RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=500, n_jobs=-1, random_state=42),
    "SVM": SVC(probability=True, kernel="rbf", random_state=42),
    "XGBoost": xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                                   eval_metric="logloss", n_jobs=-1, random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=500, random_state=42),
}


def find_label_col(df, path):
    for cand in LABEL_CANDIDATES:
        if cand in df.columns:
            return cand
    raise ValueError(
        f"None of {LABEL_CANDIDATES} found in {path}. Actual columns: {list(df.columns)}"
    )


def load_split(variant_dir, split):
    path = ENDPOINT_DIR / f"solubility__{variant_dir}" / f"{split}-00000-of-00001.parquet"
    df = pd.read_parquet(path)

    if "embedding" not in df.columns:
        raise ValueError(f"No 'embedding' column in {path}. Actual columns: {list(df.columns)}")

    label_col = find_label_col(df, path)
    X = np.stack(df["embedding"].values)
    y = df[label_col].values
    return X, y, label_col


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


def main():
    all_results = []

    for variant_dir in VARIANTS:
        print(f"\n{'='*90}\nVariant: {variant_dir}\n{'='*90}")

        try:
            X_train, y_train, label_col = load_split(variant_dir, "train")
            X_val, y_val, _ = load_split(variant_dir, "val")
        except FileNotFoundError as e:
            print(f"  [skip] {e}")
            continue

        print(f"  Detected label column: '{label_col}'")
        print(f"  Train: {X_train.shape}, Val: {X_val.shape}")
        print(f"  Class balance — train: {pd.Series(y_train).value_counts().to_dict()}  "
              f"val: {pd.Series(y_val).value_counts().to_dict()}")

        embedding_name = "ChemBERTa" if "chemberta" in variant_dir else "WT"

        for model_name, model in MODELS.items():
            model_instance = type(model)(**model.get_params())
            print(f"  Training {model_name}...")
            model_instance.fit(X_train, y_train)
            y_prob = model_instance.predict_proba(X_val)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            metrics = classification_metrics(y_val, y_pred, y_prob)
            metrics["variant"] = variant_dir
            metrics["embedding"] = embedding_name
            metrics["model"] = model_name
            all_results.append(metrics)

            print(f"    MCC={metrics['MCC']:.4f}  AUROC={metrics['AUROC']:.4f}  "
                  f"AUPRC={metrics['AUPRC']:.4f}")

    results_df = pd.DataFrame(all_results)
    cols = ["embedding", "variant", "model", "MCC", "AUROC", "AUPRC",
            "Sensitivity", "Specificity", "Balanced_Accuracy", "F1", "Accuracy", "Precision"]
    results_df = results_df[cols].sort_values("MCC", ascending=False)

    out_path = RESULTS_DIR / "leaderboard_solubility.csv"
    results_df.to_csv(out_path, index=False)

    print(f"\n{'='*90}\nFULL LEADERBOARD (sorted by MCC)\n{'='*90}")
    print(results_df.to_string(index=False))
    print(f"\nSaved to {out_path}")

    print(f"\n{'='*90}\nChemBERTa vs WT comparison (best model per embedding)\n{'='*90}")
    best_per_embedding = results_df.loc[results_df.groupby("embedding")["MCC"].idxmax()]
    print(best_per_embedding[["embedding", "model", "MCC", "AUROC"]].to_string(index=False))


if __name__ == "__main__":
    main()
