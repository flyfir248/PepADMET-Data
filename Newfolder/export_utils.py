"""
Shared helper used by the train_*.py scripts to export the best-performing
model + a metadata.json describing exactly how to reproduce its input
features at inference time. This is what lets app.py load a model without
guessing which variant/embedding it needs.

Import this from each train_*.py and call export_best_model(...) at the end
of main(), after you've built `results_df`.
"""
import json
from pathlib import Path

import joblib
import numpy as np

MODELS_DIR = Path(".") / "models"
MODELS_DIR.mkdir(exist_ok=True)


def export_best_model(
    endpoint_key: str,
    results_df,
    task: str,                      # "classification" | "regression"
    primary_metric: str,            # "R2" (regression) or "MCC" (classification)
    model_factory_by_name: dict,    # e.g. MODELS = {"XGBoost": xgb.XGBRegressor(...), ...}
    variant_configs: dict,          # variant_name -> config dict (VARIANTS)
    load_variant_fn,                # (variant_name, config, split) -> (X, y, ...)
    extra_metadata: dict = None,
):
    """
    Picks the best row in results_df by primary_metric (highest wins),
    retrains that exact (variant, model) combination on TRAIN+VAL combined
    (more data for the final artifact than either split alone), and saves:

      models/<endpoint_key>_best_model.joblib
      models/<endpoint_key>_best_model_metadata.json

    The filenames always use endpoint_key, never the winning model's name,
    so app.py can find them without guessing.
    """
    best_row = results_df.sort_values(primary_metric, ascending=False).iloc[0]
    variant_name = best_row["variant"]
    model_name = best_row["model"]
    print(f"\n>>> Exporting best model for '{endpoint_key}': "
          f"variant={variant_name}, model={model_name}, "
          f"{primary_metric}={best_row[primary_metric]:.4f}")

    config = variant_configs[variant_name]

    train_result = load_variant_fn(variant_name, config, "train")
    val_result = load_variant_fn(variant_name, config, "val")
    X_train, y_train = train_result[0], train_result[1]
    X_val, y_val = val_result[0], val_result[1]
    X_full = np.concatenate([X_train, X_val], axis=0)
    y_full = np.concatenate([y_train, y_val], axis=0)

    model_template = model_factory_by_name[model_name]
    final_model = type(model_template)(**model_template.get_params())
    final_model.fit(X_full, y_full)

    model_path = MODELS_DIR / f"{endpoint_key}_best_model.joblib"
    joblib.dump(final_model, model_path)

    metadata = {
        "endpoint": endpoint_key,
        "task": task,
        "variant": variant_name,
        "model_name": model_name,
        "primary_metric": primary_metric,
        "metrics": {
            k: float(v) for k, v in best_row.items()
            if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool)
        },
        "trained_on_rows": int(X_full.shape[0]),
        "feature_dim": int(X_full.shape[1]),
    }
    if task == "classification":
        metadata["classification_threshold"] = 0.5

    # Record how to rebuild the feature vector at inference time.
    if "target_col" in config:
        metadata["embed_type"] = "pair"
        metadata["target_col"] = config["target_col"]
        metadata["target_dim"] = config["target_dim"]
        metadata["binder_col"] = config["binder_col"]
        metadata["binder_dim"] = config["binder_dim"]
    else:
        metadata["embed_type"] = "chemberta" if "chemberta" in variant_name else "wt"

    if extra_metadata:
        metadata.update(extra_metadata)

    metadata_path = MODELS_DIR / f"{endpoint_key}_best_model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"    Saved model    -> {model_path}")
    print(f"    Saved metadata -> {metadata_path}")
    return model_path, metadata_path
