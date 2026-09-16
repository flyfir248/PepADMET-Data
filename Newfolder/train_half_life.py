"""
Trains half-life (regression) across the ChemBERTa and WT embedding
variants, same two-variant pattern as hemolysis/solubility but for a
continuous target, and exports the winner to:
  models/half_life_best_model.joblib
  models/half_life_best_model_metadata.json

These are the exact filenames app.py already looks for, so no changes to
app.py are needed -- just run this script once, then start app.py.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb

from export_utils import export_best_model

warnings.filterwarnings("ignore")

ROOT = Path(".")
ENDPOINT_DIR = ROOT / "half_life"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Adjust these two folder names if yours differ slightly on disk.
VARIANTS = [
    "half_life__halflife_chemberta_with_embeddings",
    "half_life__halflife_wt_with_embeddings",
]
LABEL_CANDIDATES = ["label", "Label", "half_life", "HalfLife"]

MODELS = {
    "LinearRegression": LinearRegression(n_jobs=-1),
    "RandomForest": RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=500, n_jobs=-1, random_state=42),
    "SVR": SVR(kernel="rbf"),
    "XGBoost": xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                                 n_jobs=-1, random_state=42),
    "MLP": MLPRegressor(hidden_layer_sizes=(256, 64), max_iter=500, random_state=42),
}


def find_label_col(df, path):
    for cand in LABEL_CANDIDATES:
        if cand in df.columns:
            return cand
    raise ValueError(
        f"None of {LABEL_CANDIDATES} found in {path}. Actual columns: {list(df.columns)}"
    )


def load_split(variant_dir, split):
    path = ENDPOINT_DIR / variant_dir / f"{split}-00000-of-00001.parquet"
    df = pd.read_parquet(path)
    if "embedding" not in df.columns:
        raise ValueError(f"No 'embedding' column in {path}. Actual columns: {list(df.columns)}")
    label_col = find_label_col(df, path)
    X = np.stack(df["embedding"].values)
    y = df[label_col].values.astype(float)
    return X, y, label_col


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


def main():
    all_results = []
    for variant_dir in VARIANTS:
        print(f"\n{'=' * 90}\nVariant: {variant_dir}\n{'=' * 90}")
        try:
            X_train, y_train, label_col = load_split(variant_dir, "train")
            X_val, y_val, _ = load_split(variant_dir, "val")
        except FileNotFoundError as e:
            print(f"  [skip] {e}")
            continue
        print(f"  Detected label column: '{label_col}'")
        print(f"  Train: {X_train.shape}, Val: {X_val.shape}")
        print(f"  Label range - train: [{y_train.min():.3f}, {y_train.max():.3f}] "
              f"val: [{y_val.min():.3f}, {y_val.max():.3f}]")
        embedding_name = "ChemBERTa" if "chemberta" in variant_dir else "WT"
        for model_name, model in MODELS.items():
            model_instance = type(model)(**model.get_params())
            print(f"  Training {model_name}...")
            model_instance.fit(X_train, y_train)
            y_pred = model_instance.predict(X_val)
            metrics = regression_metrics(y_val, y_pred)
            metrics["variant"] = variant_dir
            metrics["embedding"] = embedding_name
            metrics["model"] = model_name
            all_results.append(metrics)
            print(f"    R2={metrics['R2']:.4f}  RMSE={metrics['RMSE']:.4f}  "
                  f"Pearson_r={metrics['Pearson_r']:.4f}  Spearman_r={metrics['Spearman_r']:.4f}")

    if not all_results:
        print("\nNo datasets were successfully processed. Check VARIANTS folder names above.")
        return

    results_df = pd.DataFrame(all_results)
    cols = ["embedding", "variant", "model", "R2", "Pearson_r", "Spearman_r", "RMSE", "MAE"]
    results_df = results_df[cols].sort_values("R2", ascending=False)
    out_path = RESULTS_DIR / "leaderboard_half_life.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n{'=' * 90}\nFULL LEADERBOARD (sorted by R2)\n{'=' * 90}")
    print(results_df.to_string(index=False))
    print(f"\nSaved to {out_path}")
    print(f"\n{'=' * 90}\nChemBERTa vs WT comparison (best model per embedding)\n{'=' * 90}")
    best_per_embedding = results_df.loc[results_df.groupby("embedding")["R2"].idxmax()]
    print(best_per_embedding.to_string(index=False))

    def load_variant_adapter(variant_name, config, split):
        return load_split(variant_name, split)

    export_best_model(
        endpoint_key="half_life",
        results_df=results_df,
        task="regression",
        primary_metric="R2",
        model_factory_by_name=MODELS,
        variant_configs={v: {} for v in VARIANTS},
        load_variant_fn=load_variant_adapter,
    )


if __name__ == "__main__":
    main()
