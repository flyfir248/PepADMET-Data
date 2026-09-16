"""
Unified Peptide ADMET / Toxicity app -- ONE SCRIPT.
====================================================

Run:  python app.py

What happens on startup, in order:
  1. For every endpoint that has training logic built into this file
     (binding_affinity, permeability_penetrance, permeability_caco2,
     permeability_pampa, solubility) it checks whether a trained model
     already exists at the expected path.
       - If YES: skips training, just loads it later when a prediction is
         requested.
       - If NO: runs that endpoint's training sweep (same models/metrics as
         your original train_*.py scripts), picks the best (variant, model)
         by its primary metric, retrains it on train+val combined, and
         saves models/<endpoint>_best_model.joblib + _metadata.json.
       - If the underlying parquet data isn't found on disk, it prints a
         clear [skip] message and moves on -- it never crashes the app.
  2. toxicity, hemolysis, half_life, nf have no training logic in this file
     (toxicity already has a model from your existing pipeline; hemolysis /
     half_life / nf training scripts were never shared with me). For these,
     it just checks whether a model file exists and reports status.
  3. Starts the Flask app with one tab per endpoint.

Directory layout expected (same as your project):

  ROOT/
    binding_affinity/binding_affinity_<variant>/{train,val}-00000-of-00001.parquet
    permeability_penetrance/<variant>/{train,val}-00000-of-00001.parquet
    permeability_caco2/<variant>/{train,val}-00000-of-00001.parquet
    permeability_pampa/<variant>/{train,val}-00000-of-00001.parquet
    solubility/solubility_<variant>/{train,val}-00000-of-00001.parquet
    models/
      toxicity_chemberta_xgboost.joblib (+ _metadata.json)          [expected to already exist]
      hemolysis_best_model.joblib (+ _metadata.json)                [only if you've trained it]
      half_life_best_model.joblib (+ _metadata.json)                [only if you've trained it]
      nf_best_model.joblib (+ _metadata.json)                       [only if you've trained it]
      binding_affinity_best_model.joblib (+ _metadata.json)         [auto-trained here if missing]
      permeability_penetrance_best_model.joblib (+ _metadata.json)  [auto-trained here if missing]
      permeability_caco2_best_model.joblib (+ _metadata.json)       [auto-trained here if missing]
      permeability_pampa_best_model.joblib (+ _metadata.json)       [auto-trained here if missing]
      solubility_best_model.joblib (+ _metadata.json)               [auto-trained here if missing]
    results/
      leaderboard_*.csv   (written automatically during training)
    src/03_generate_embeddings.py   (needed for the Predict tab -- must
      expose embed_smiles(list[str]) and embed_wt(list[str]))
"""

import io
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request, jsonify, send_file
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    RandomForestClassifier, ExtraTreesClassifier,
)
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix,
)
import xgboost as xgb

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# PATH CONFIG -- adjust if your layout differs
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# The shared embedding-generation module (embed_smiles / embed_wt) lives at
# ToxicityModel/src/03_generate_embeddings.py -- a sibling of this script's
# own folder (EmbeddingPeptiverse), not nested inside it or inside
# Hemolytik2. Listed in priority order; the loader tries each until one
# actually contains 03_generate_embeddings.py.
EMBEDDINGS_MODULE_CANDIDATES = [
    ROOT.parent / "src",              # confirmed location: ToxicityModel/src
    ROOT / "src",                      # in case it's ever moved in-repo
    ROOT.parent / "Hemolytik2" / "src",  # legacy toxicity-only location
]

# Toxicity's model lives in a different project folder (Hemolytik2/models),
# a sibling of this script's own folder (EmbeddingPeptiverse) -- not inside
# the local models/ dir like every other endpoint. Adjust if your layout
# differs.
TOXICITY_MODELS_DIR = ROOT.parent / "Hemolytik2" / "models"

MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
for _cand in EMBEDDINGS_MODULE_CANDIDATES:
    sys.path.insert(0, str(_cand))
sys.path.insert(0, str(ROOT))

app = Flask(__name__)

# ==========================================================================
# SECTION 1 -- MODEL ZOOS used during training
# ==========================================================================
REGRESSION_MODELS = {
    "LinearRegression": LinearRegression(n_jobs=-1),
    "RandomForest": RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=500, n_jobs=-1, random_state=42),
    "SVR": SVR(kernel="rbf"),
    "XGBoost": xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                                 n_jobs=-1, random_state=42),
    "MLP": MLPRegressor(hidden_layer_sizes=(256, 64), max_iter=500, random_state=42),
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


def save_export(endpoint_key, model, task, primary_metric, metrics_row, extra_metadata):
    """Write models/<endpoint_key>_best_model.joblib + _metadata.json."""
    model_path = MODELS_DIR / f"{endpoint_key}_best_model.joblib"
    joblib.dump(model, model_path)
    metadata = {
        "endpoint": endpoint_key,
        "task": task,
        "primary_metric": primary_metric,
        "metrics": {k: float(v) for k, v in metrics_row.items()
                    if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool)},
    }
    if task == "classification":
        metadata["classification_threshold"] = 0.5
    metadata.update(extra_metadata)
    metadata_path = MODELS_DIR / f"{endpoint_key}_best_model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"    Saved model    -> {model_path}")
    print(f"    Saved metadata -> {metadata_path}")


# ==========================================================================
# SECTION 2 -- TRAINING / EXPORT PER ENDPOINT
# Each function: trains every (variant, model) combo, writes a leaderboard
# CSV, then retrains the winner on train+val combined and exports it.
# Each is wrapped in try/except by the caller so missing data never kills
# the whole app.
# ==========================================================================

def train_binding_affinity():
    endpoint_key = "binding_affinity"
    endpoint_dir = ROOT / "binding_affinity"
    label_col = "label"
    affinity_check_col = "affinity"
    variants = {
        "pair_wt_chemberta_pooled": {
            "target_col": "target_embedding", "target_dim": 1280,
            "binder_col": "binder_embedding", "binder_dim": 384,
        },
        "pair_wt_wt_pooled": {
            "target_col": "target_embedding", "target_dim": 1280,
            "binder_col": "embedding", "binder_dim": 1280,
        },
    }

    def load_variant(variant_name, config, split):
        path = endpoint_dir / f"binding_affinity_{variant_name}" / f"{split}-00000-of-00001.parquet"
        df = pd.read_parquet(path)
        required = {config["target_col"], config["binder_col"], label_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns {missing}. Actual: {list(df.columns)}")
        target_emb = np.stack(df[config["target_col"]].values)
        binder_emb = np.stack(df[config["binder_col"]].values)
        if target_emb.shape[1] != config["target_dim"] or binder_emb.shape[1] != config["binder_dim"]:
            raise ValueError(f"{path}: unexpected embedding dims.")
        X = np.concatenate([target_emb, binder_emb], axis=1)
        y = df[label_col].values.astype(float)
        if affinity_check_col in df.columns:
            mismatch = ~np.isclose(df[label_col].astype(float), df[affinity_check_col].astype(float))
            if mismatch.any():
                print(f"    \u26a0 {mismatch.sum()} rows where label != affinity in {path}")
        return X, y

    print(f"\n{'=' * 90}\nTraining: {endpoint_key}\n{'=' * 90}")
    all_results = []
    for variant_name, config in variants.items():
        X_train, y_train = load_variant(variant_name, config, "train")
        X_val, y_val = load_variant(variant_name, config, "val")
        print(f"  [{variant_name}] Train: {X_train.shape}, Val: {X_val.shape}")
        for model_name, model in REGRESSION_MODELS.items():
            m = type(model)(**model.get_params())
            m.fit(X_train, y_train)
            y_pred = m.predict(X_val)
            metrics = regression_metrics(y_val, y_pred)
            metrics.update({"variant": variant_name, "model": model_name, "feature_dim": X_train.shape[1]})
            all_results.append(metrics)
            print(f"    {model_name}: R2={metrics['R2']:.4f}")

    results_df = pd.DataFrame(all_results).sort_values("R2", ascending=False)
    results_df.to_csv(RESULTS_DIR / "leaderboard_binding_affinity.csv", index=False)

    best = results_df.iloc[0]
    variant_name, model_name = best["variant"], best["model"]
    config = variants[variant_name]
    X_train, y_train = load_variant(variant_name, config, "train")
    X_val, y_val = load_variant(variant_name, config, "val")
    X_full = np.concatenate([X_train, X_val], axis=0)
    y_full = np.concatenate([y_train, y_val], axis=0)
    final_model = type(REGRESSION_MODELS[model_name])(**REGRESSION_MODELS[model_name].get_params())
    final_model.fit(X_full, y_full)
    print(f"  >>> Best: variant={variant_name}, model={model_name}, R2={best['R2']:.4f}")
    save_export(endpoint_key, final_model, "regression", "R2", best, {
        "variant": variant_name, "embed_type": "pair",
        "target_col": config["target_col"], "target_dim": config["target_dim"],
        "binder_col": config["binder_col"], "binder_dim": config["binder_dim"],
        "trained_on_rows": int(X_full.shape[0]), "feature_dim": int(X_full.shape[1]),
    })


def _train_single_embedding_classifier(endpoint_key, endpoint_dir, path_prefix, variants,
                                        leaderboard_name, label_candidates=("label", "Label")):
    def find_label_col(df, path):
        for cand in label_candidates:
            if cand in df.columns:
                return cand
        raise ValueError(f"None of {label_candidates} found in {path}. Actual: {list(df.columns)}")

    def load_split(variant_dir, split):
        path = endpoint_dir / f"{path_prefix}{variant_dir}" / f"{split}-00000-of-00001.parquet"
        df = pd.read_parquet(path)
        if "embedding" not in df.columns:
            raise ValueError(f"No 'embedding' column in {path}. Actual: {list(df.columns)}")
        lc = find_label_col(df, path)
        X = np.stack(df["embedding"].values)
        y = df[lc].values
        return X, y

    print(f"\n{'=' * 90}\nTraining: {endpoint_key}\n{'=' * 90}")
    all_results = []
    for variant_dir in variants:
        try:
            X_train, y_train = load_split(variant_dir, "train")
            X_val, y_val = load_split(variant_dir, "val")
        except FileNotFoundError as e:
            print(f"  [skip variant] {e}")
            continue
        embedding_name = "ChemBERTa" if "chemberta" in variant_dir else "WT"
        print(f"  [{variant_dir}] Train: {X_train.shape}, Val: {X_val.shape}")
        for model_name, model in CLASSIFICATION_MODELS.items():
            m = type(model)(**model.get_params())
            m.fit(X_train, y_train)
            y_prob = m.predict_proba(X_val)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            metrics = classification_metrics(y_val, y_pred, y_prob)
            metrics.update({"variant": variant_dir, "embedding": embedding_name, "model": model_name})
            all_results.append(metrics)
            print(f"    {model_name}: MCC={metrics['MCC']:.4f}")

    if not all_results:
        raise FileNotFoundError(f"No variants for '{endpoint_key}' had data on disk.")

    results_df = pd.DataFrame(all_results).sort_values("MCC", ascending=False)
    results_df.to_csv(RESULTS_DIR / leaderboard_name, index=False)

    best = results_df.iloc[0]
    variant_name, model_name = best["variant"], best["model"]
    X_train, y_train = load_split(variant_name, "train")
    X_val, y_val = load_split(variant_name, "val")
    X_full = np.concatenate([X_train, X_val], axis=0)
    y_full = np.concatenate([y_train, y_val], axis=0)
    final_model = type(CLASSIFICATION_MODELS[model_name])(**CLASSIFICATION_MODELS[model_name].get_params())
    final_model.fit(X_full, y_full)
    print(f"  >>> Best: variant={variant_name}, model={model_name}, MCC={best['MCC']:.4f}")
    save_export(endpoint_key, final_model, "classification", "MCC", best, {
        "variant": variant_name,
        "embed_type": "chemberta" if "chemberta" in variant_name else "wt",
        "trained_on_rows": int(X_full.shape[0]), "feature_dim": int(X_full.shape[1]),
    })


def train_permeability_penetrance():
    _train_single_embedding_classifier(
        endpoint_key="permeability_penetrance",
        endpoint_dir=ROOT / "permeability_penetrance",
        path_prefix="",
        variants=[
            "permeability_penetrance__perm_chemberta_with_embeddings",
            "permeability_penetrance__perm_wt_with_embeddings_pooled",
        ],
        leaderboard_name="leaderboard_permeability_penetrance.csv",
    )


def train_solubility():
    _train_single_embedding_classifier(
        endpoint_key="solubility",
        endpoint_dir=ROOT / "solubility",
        path_prefix="solubility_",
        variants=["sol_chemberta_with_embeddings", "sol_wt_with_embeddings"],
        leaderboard_name="leaderboard_solubility.csv",
    )


def train_permeability_regression():
    """Trains caco2 and PAMPA (both regression, single ChemBERTa embedding) and
    exports each as its own model."""
    endpoints = {
        "permeability_caco2": {
            "embedding_dir": "permeability_caco2__caco2_chemberta_with_embeddings",
            "label_candidates": ["label", "Caco2", "caco2"],
        },
        "permeability_pampa": {
            "embedding_dir": "permeability_pampa__pampa_chemberta_with_embeddings",
            "label_candidates": ["label", "PAMPA", "pampa"],
        },
    }

    def find_label_col(df, candidates, path):
        for cand in candidates:
            if cand in df.columns:
                return cand
        raise ValueError(f"None of {candidates} found in {path}. Actual: {list(df.columns)}")

    def load_split(endpoint_name, config, split):
        path = ROOT / endpoint_name / config["embedding_dir"] / f"{split}-00000-of-00001.parquet"
        df = pd.read_parquet(path)
        if "embedding" not in df.columns:
            raise ValueError(f"No 'embedding' column in {path}. Actual: {list(df.columns)}")
        lc = find_label_col(df, config["label_candidates"], path)
        X = np.stack(df["embedding"].values)
        y = df[lc].values.astype(float)
        return X, y

    for endpoint_key, config in endpoints.items():
        print(f"\n{'=' * 90}\nTraining: {endpoint_key}\n{'=' * 90}")
        try:
            X_train, y_train = load_split(endpoint_key, config, "train")
            X_val, y_val = load_split(endpoint_key, config, "val")
        except FileNotFoundError as e:
            print(f"  [skip] {e}")
            continue
        print(f"  Train: {X_train.shape}, Val: {X_val.shape}")
        rows = []
        for model_name, model in REGRESSION_MODELS.items():
            m = type(model)(**model.get_params())
            m.fit(X_train, y_train)
            y_pred = m.predict(X_val)
            metrics = regression_metrics(y_val, y_pred)
            metrics.update({"model": model_name})
            rows.append(metrics)
            print(f"    {model_name}: R2={metrics['R2']:.4f}")

        endpoint_df = pd.DataFrame(rows).sort_values("R2", ascending=False)
        leaderboard_path = RESULTS_DIR / "leaderboard_permeability_regression.csv"
        if leaderboard_path.exists():
            existing = pd.read_csv(leaderboard_path)
            existing = existing[existing.get("endpoint") != endpoint_key] if "endpoint" in existing.columns else existing
            endpoint_df_labeled = endpoint_df.copy()
            endpoint_df_labeled["endpoint"] = endpoint_key
            combined = pd.concat([existing, endpoint_df_labeled], ignore_index=True)
        else:
            combined = endpoint_df.copy()
            combined["endpoint"] = endpoint_key
        combined.to_csv(leaderboard_path, index=False)

        best = endpoint_df.iloc[0]
        model_name = best["model"]
        X_full = np.concatenate([X_train, X_val], axis=0)
        y_full = np.concatenate([y_train, y_val], axis=0)
        final_model = type(REGRESSION_MODELS[model_name])(**REGRESSION_MODELS[model_name].get_params())
        final_model.fit(X_full, y_full)
        print(f"  >>> Best: model={model_name}, R2={best['R2']:.4f}")
        save_export(endpoint_key, final_model, "regression", "R2", best, {
            "embed_type": "chemberta",
            "trained_on_rows": int(X_full.shape[0]), "feature_dim": int(X_full.shape[1]),
        })


# Endpoints this script knows how to auto-train if their model is missing.
AUTO_TRAINERS = {
    "binding_affinity": train_binding_affinity,
    "permeability_penetrance": train_permeability_penetrance,
    "solubility": train_solubility,
    "permeability_caco2": train_permeability_regression,   # trains both caco2 + pampa
    "permeability_pampa": train_permeability_regression,   # same call, guarded against double-run below
}


def ensure_models_exported():
    """For every endpoint below, train+export only if the model file is missing."""
    already_ran = set()
    for endpoint_key, cfg in ENDPOINTS.items():
        if cfg["model_path"].exists():
            print(f"[ready]   {endpoint_key}: model already exists at {cfg['model_path']}, skipping training.")
            continue
        trainer = AUTO_TRAINERS.get(endpoint_key)
        if trainer is None:
            print(f"[missing] {endpoint_key}: no model found and no training logic available in this "
                  f"script for it yet. Predict tab will show 'model not ready'.")
            continue
        if trainer in already_ran:
            continue  # permeability_regression trains both caco2 and pampa in one call
        try:
            trainer()
            already_ran.add(trainer)
        except FileNotFoundError as e:
            print(f"[skip]    {endpoint_key}: training data not found ({e}).")
        except Exception as e:
            print(f"[error]   {endpoint_key}: training failed: {e}")


# ==========================================================================
# SECTION 3 -- INFERENCE-TIME EMBEDDING (for the Predict tab)
# ==========================================================================
_gen_embeddings = None


def _load_embedding_module():
    global _gen_embeddings
    if _gen_embeddings is None:
        from importlib import import_module
        errors = []
        for cand_dir in EMBEDDINGS_MODULE_CANDIDATES:
            module_file = cand_dir / "03_generate_embeddings.py"
            if not module_file.exists():
                errors.append(f"  - {module_file} (not found)")
                continue
            try:
                _gen_embeddings = import_module("03_generate_embeddings")
                print(f"[info] Loaded embedding module from {module_file}")
                break
            except Exception as e:
                errors.append(f"  - {module_file} (found, but import failed: {e})")
        if _gen_embeddings is None:
            raise RuntimeError(
                "Could not load 03_generate_embeddings.py from any known location:\n"
                + "\n".join(errors)
                + "\nEdit EMBEDDINGS_MODULE_CANDIDATES near the top of this file if it "
                  "lives somewhere else."
            )
    return _gen_embeddings


def embed_single(text: str, metadata: dict) -> np.ndarray:
    embed_type = (metadata or {}).get("embed_type", "chemberta")
    mod = _load_embedding_module()
    if embed_type == "chemberta":
        if not hasattr(mod, "embed_smiles"):
            raise RuntimeError("03_generate_embeddings.py has no embed_smiles() function.")
        return np.asarray(mod.embed_smiles([text]))[0]
    if embed_type == "wt":
        if not hasattr(mod, "embed_wt"):
            raise RuntimeError("03_generate_embeddings.py has no embed_wt() function.")
        return np.asarray(mod.embed_wt([text]))[0]
    raise RuntimeError(f"Unsupported embed_type '{embed_type}'.")


def embed_pair(binder_text: str, target_text: str, metadata: dict) -> np.ndarray:
    mod = _load_embedding_module()
    variant = (metadata or {}).get("variant", "")
    if not hasattr(mod, "embed_wt"):
        raise RuntimeError("03_generate_embeddings.py has no embed_wt() function.")
    target_emb = np.asarray(mod.embed_wt([target_text]))[0]
    if "chemberta" in variant:
        if not hasattr(mod, "embed_smiles"):
            raise RuntimeError("03_generate_embeddings.py has no embed_smiles() function.")
        binder_emb = np.asarray(mod.embed_smiles([binder_text]))[0]
    else:
        binder_emb = np.asarray(mod.embed_wt([binder_text]))[0]
    return np.concatenate([target_emb, binder_emb])


# ==========================================================================
# SECTION 4 -- ENDPOINT REGISTRY (UI + file locations)
# ==========================================================================
ENDPOINTS = {
    "toxicity": {
        "label": "Toxicity", "task": "classification", "input_kind": "single",
        "model_path": TOXICITY_MODELS_DIR / "toxicity_chemberta_xgboost.joblib",
        "metadata_path": TOXICITY_MODELS_DIR / "toxicity_chemberta_xgboost_metadata.json",
        "leaderboard_csv": RESULTS_DIR / "leaderboard_toxicity.csv",
        "positive_label": "Toxic", "negative_label": "Non-toxic",
        "examples": [
            {"label": "Melittin (known toxic)", "input": "GIGAVLKVLTTGLPALISWIKRKRQQ"},
            {"label": "Magainin 2 (mild)", "input": "GIGKFLHSAKKFGKAFVGEIMNS"},
            {"label": "Short benign peptide", "input": "AAAAAAAAAA"},
        ],
    },
    "hemolysis": {
        "label": "Hemolysis", "task": "classification", "input_kind": "single",
        "model_path": MODELS_DIR / "hemolysis_best_model.joblib",
        "metadata_path": MODELS_DIR / "hemolysis_best_model_metadata.json",
        "leaderboard_csv": RESULTS_DIR / "leaderboard_hemolysis.csv",
        "positive_label": "Hemolytic", "negative_label": "Non-hemolytic",
        "examples": [
            {"label": "Melittin (known hemolytic)", "input": "GIGAVLKVLTTGLPALISWIKRKRQQ"},
            {"label": "KLA peptide", "input": "KLAKLAKKLAKLAK"},
            {"label": "Short benign peptide", "input": "AAAAAAAAAA"},
        ],
    },
    "half_life": {
        "label": "Half-life", "task": "regression", "input_kind": "single",
        "model_path": MODELS_DIR / "half_life_best_model.joblib",
        "metadata_path": MODELS_DIR / "half_life_best_model_metadata.json",
        "leaderboard_csv": RESULTS_DIR / "leaderboard_half_life.csv",
        "unit": "hours",
        "examples": [
            {"label": "Insulin B-chain", "input": "FVNQHLCGSHLVEALYLVCGERGFFYTPKA"},
            {"label": "D-amino acid stabilized (example)", "input": "kLAkLAkKLAkLAk"},
            {"label": "Short linear peptide", "input": "GLFDIVKKVVGALGSL"},
        ],
    },
    "nf": {
        "label": "NF", "task": "regression", "input_kind": "single",
        "model_path": MODELS_DIR / "nf_best_model.joblib",
        "metadata_path": MODELS_DIR / "nf_best_model_metadata.json",
        "leaderboard_csv": RESULTS_DIR / "leaderboard_nf.csv",
        "unit": "",
        "examples": [
            {"label": "Example peptide 1", "input": "KLALKLALKAWKAALKLA"},
            {"label": "Example peptide 2", "input": "GLFDIVKKVVGALGSL"},
            {"label": "Example peptide 3", "input": "RRWWRRWRR"},
        ],
    },
    "binding_affinity": {
        "label": "Binding Affinity", "task": "regression", "input_kind": "pair",
        "model_path": MODELS_DIR / "binding_affinity_best_model.joblib",
        "metadata_path": MODELS_DIR / "binding_affinity_best_model_metadata.json",
        "leaderboard_csv": RESULTS_DIR / "leaderboard_binding_affinity.csv",
        "unit": "affinity units (see metadata)",
        "examples": [
            {"label": "Example binder/target pair 1",
             "input": "KLALKLALKAWKAALKLA", "input2": "GIGAVLKVLTTGLPALISWIKRKRQQ"},
            {"label": "Example binder/target pair 2",
             "input": "RRWWRRWRR", "input2": "GLFDIVKKVVGALGSL"},
        ],
    },
    "permeability_caco2": {
        "label": "Permeability (Caco-2)", "task": "regression", "input_kind": "single",
        "model_path": MODELS_DIR / "permeability_caco2_best_model.joblib",
        "metadata_path": MODELS_DIR / "permeability_caco2_best_model_metadata.json",
        "leaderboard_csv": RESULTS_DIR / "leaderboard_permeability_regression.csv",
        "unit": "log Papp",
        "examples": [
            {"label": "Cyclosporine A (SMILES)",
             "input": "CC1C(C(CC(N(C(C(C(CC(N(C(C(N(C(C(N(C(C(N(C(C(N1C(=O)C(C(C)CC=CC)NC(=O)C(C(C)C)N(C)C(=O)C(C(C)C)N(C)C(=O)CN(C)C(=O)C(C(C)C)NC(=O)C(C)NC(=O)C(CC(C)C)N(C)C(=O)C(CC(C)C)N(C)C(=O)C(C)NC(=O)C(C(C)C)N(C)C(=O)C(CC(C)C)N(C)C(=O)C(C)NC1=O)C)C(=O)N(C)C)C)C)C(=O)N(C)C)C)C(=O)N(C)C)C)C(C)C)C)O)C"},
            {"label": "Short linear peptide", "input": "KLALKLALKAWKAALKLA"},
        ],
    },
    "permeability_pampa": {
        "label": "Permeability (PAMPA)", "task": "regression", "input_kind": "single",
        "model_path": MODELS_DIR / "permeability_pampa_best_model.joblib",
        "metadata_path": MODELS_DIR / "permeability_pampa_best_model_metadata.json",
        "leaderboard_csv": RESULTS_DIR / "leaderboard_permeability_regression.csv",
        "unit": "log Pe",
        "examples": [
            {"label": "Short linear peptide", "input": "KLALKLALKAWKAALKLA"},
            {"label": "Cyclic-like short peptide", "input": "GLFDIVKKVVGALGSL"},
        ],
    },
    "permeability_penetrance": {
        "label": "Permeability (Penetrance)", "task": "classification", "input_kind": "single",
        "model_path": MODELS_DIR / "permeability_penetrance_best_model.joblib",
        "metadata_path": MODELS_DIR / "permeability_penetrance_best_model_metadata.json",
        "leaderboard_csv": RESULTS_DIR / "leaderboard_permeability_penetrance.csv",
        "positive_label": "Penetrant", "negative_label": "Non-penetrant",
        "examples": [
            {"label": "Penetratin (known CPP)", "input": "RQIKIWFQNRRMKWKK"},
            {"label": "TAT peptide (known CPP)", "input": "GRKKRRQRRRPPQ"},
            {"label": "Short benign peptide", "input": "AAAAAAAAAA"},
        ],
    },
    "solubility": {
        "label": "Solubility", "task": "classification", "input_kind": "single",
        "model_path": MODELS_DIR / "solubility_best_model.joblib",
        "metadata_path": MODELS_DIR / "solubility_best_model_metadata.json",
        "leaderboard_csv": RESULTS_DIR / "leaderboard_solubility.csv",
        "positive_label": "Soluble", "negative_label": "Insoluble",
        "examples": [
            {"label": "Polar/charged peptide", "input": "DEDEDEDEDEDEDEDE"},
            {"label": "Hydrophobic peptide", "input": "LVLVLVLVLVLVLVLV"},
            {"label": "Mixed peptide", "input": "KLALKLALKAWKAALKLA"},
        ],
    },
}

# --------------------------------------------------------------------------
# MODEL CACHE (inference)
# --------------------------------------------------------------------------
_model_cache = {}
_metadata_cache = {}


def get_model_and_metadata(endpoint_key: str):
    if endpoint_key in _model_cache:
        return _model_cache[endpoint_key], _metadata_cache.get(endpoint_key)
    cfg = ENDPOINTS[endpoint_key]
    model_path, metadata_path = cfg["model_path"], cfg["metadata_path"]
    if not model_path.exists():
        raise FileNotFoundError(f"No trained model found at {model_path}.")
    model = joblib.load(model_path)
    metadata = None
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    _model_cache[endpoint_key] = model
    _metadata_cache[endpoint_key] = metadata
    return model, metadata


def endpoint_status(endpoint_key: str) -> dict:
    cfg = ENDPOINTS[endpoint_key]
    return {
        "key": endpoint_key, "label": cfg["label"], "task": cfg["task"],
        "input_kind": cfg["input_kind"], "model_ready": cfg["model_path"].exists(),
        "model_path": str(cfg["model_path"]), "leaderboard_ready": cfg["leaderboard_csv"].exists(),
    }


# --------------------------------------------------------------------------
# PREDICTION CORE
# --------------------------------------------------------------------------
def predict_single(endpoint_key: str, text: str, text2: str = None) -> dict:
    cfg = ENDPOINTS[endpoint_key]
    if not text or not str(text).strip():
        return {"error": "No input provided."}
    try:
        model, metadata = get_model_and_metadata(endpoint_key)
    except FileNotFoundError as e:
        return {"error": str(e)}

    try:
        if cfg["input_kind"] == "pair":
            if not text2 or not str(text2).strip():
                return {"error": "This endpoint needs both a binder and a target sequence."}
            X = embed_pair(text.strip(), text2.strip(), metadata).reshape(1, -1)
        else:
            X = embed_single(text.strip(), metadata).reshape(1, -1)
    except Exception as e:
        return {"error": f"Feature extraction failed: {e}"}

    try:
        if cfg["task"] == "classification":
            prob = float(model.predict_proba(X)[:, 1][0])
            threshold = (metadata or {}).get("classification_threshold", 0.5)
            pred_class = int(prob >= threshold)
            label = cfg["positive_label"] if pred_class == 1 else cfg["negative_label"]
            dist = abs(prob - threshold)
            confidence = "High" if dist > 0.35 else "Moderate" if dist > 0.15 else "Low"
            return {
                "input": text if cfg["input_kind"] != "pair" else {"binder": text, "target": text2},
                "prediction": label, "probability": round(prob, 4), "class": pred_class,
                "confidence": confidence, "error": None,
            }
        else:
            value = float(model.predict(X)[0])
            return {
                "input": text if cfg["input_kind"] != "pair" else {"binder": text, "target": text2},
                "prediction": round(value, 4), "unit": cfg.get("unit", ""), "error": None,
            }
    except Exception as e:
        return {"error": f"Prediction failed: {e}"}


def predict_batch(endpoint_key: str, items: list) -> list:
    cfg = ENDPOINTS[endpoint_key]
    results = []
    for item in items:
        if cfg["input_kind"] == "pair":
            binder, target = (item + [None, None])[:2] if isinstance(item, list) else (item, None)
            results.append(predict_single(endpoint_key, binder, target))
        else:
            results.append(predict_single(endpoint_key, item))
    return results


# --------------------------------------------------------------------------
# ROUTES
# --------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template_string(INDEX_HTML, endpoints_json=json.dumps({
        k: {"label": c["label"], "task": c["task"], "input_kind": c["input_kind"],
            "examples": c.get("examples", [])}
        for k, c in ENDPOINTS.items()
    }))


@app.route("/api/status")
def api_status():
    return jsonify({k: endpoint_status(k) for k in ENDPOINTS})


@app.route("/api/predict/<endpoint_key>", methods=["POST"])
def api_predict(endpoint_key):
    if endpoint_key not in ENDPOINTS:
        return jsonify({"error": f"Unknown endpoint '{endpoint_key}'."}), 404
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("input") or "").strip()
    text2 = (data.get("input2") or "").strip() or None
    result = predict_single(endpoint_key, text, text2)
    return jsonify(result), (400 if result.get("error") else 200)


@app.route("/api/predict_batch/<endpoint_key>", methods=["POST"])
def api_predict_batch(endpoint_key):
    if endpoint_key not in ENDPOINTS:
        return jsonify({"error": f"Unknown endpoint '{endpoint_key}'."}), 404
    data = request.get_json(force=True, silent=True) or {}
    raw = data.get("input_block", "")
    cfg = ENDPOINTS[endpoint_key]
    if cfg["input_kind"] == "pair":
        items = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                items.append([parts[0], parts[1]])
    else:
        items = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not items:
        return jsonify({"error": "No valid input rows found."}), 400
    if len(items) > 500:
        return jsonify({"error": "Max 500 rows via paste. Use CSV upload for more."}), 400
    return jsonify({"results": predict_batch(endpoint_key, items)})


@app.route("/api/predict_csv/<endpoint_key>", methods=["POST"])
def api_predict_csv(endpoint_key):
    if endpoint_key not in ENDPOINTS:
        return jsonify({"error": f"Unknown endpoint '{endpoint_key}'."}), 404
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    cfg = ENDPOINTS[endpoint_key]
    file = request.files["file"]
    col = request.form.get("input_col", "SMILES")
    col2 = request.form.get("input_col2", "target")
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Could not read CSV: {e}"}), 400
    if col not in df.columns:
        return jsonify({"error": f"Column '{col}' not found. Available: {list(df.columns)}"}), 400
    if len(df) > 5000:
        return jsonify({"error": "Max 5000 rows per CSV upload."}), 400
    if cfg["input_kind"] == "pair":
        if col2 not in df.columns:
            return jsonify({"error": f"Column '{col2}' not found. Available: {list(df.columns)}"}), 400
        items = [list(x) for x in zip(df[col].astype(str), df[col2].astype(str))]
    else:
        items = df[col].astype(str).tolist()
    predictions = predict_batch(endpoint_key, items)
    pred_df = pd.DataFrame(predictions)
    out = pd.concat([df.reset_index(drop=True), pred_df], axis=1)
    buf = io.StringIO()
    out.to_csv(buf, index=False)
    buf.seek(0)
    mem = io.BytesIO(buf.getvalue().encode("utf-8"))
    return send_file(mem, mimetype="text/csv", as_attachment=True,
                      download_name=f"{endpoint_key}_predictions.csv")


@app.route("/api/leaderboard/<endpoint_key>")
def api_leaderboard(endpoint_key):
    if endpoint_key not in ENDPOINTS:
        return jsonify({"error": f"Unknown endpoint '{endpoint_key}'."}), 404
    path = ENDPOINTS[endpoint_key]["leaderboard_csv"]
    if not path.exists():
        return jsonify({"error": f"No leaderboard found at {path}."}), 404
    try:
        df = pd.read_csv(path)
        if "endpoint" in df.columns:
            df = df[df["endpoint"] == endpoint_key]
    except Exception as e:
        return jsonify({"error": f"Could not read leaderboard: {e}"}), 400
    return jsonify({"columns": list(df.columns), "rows": df.to_dict(orient="records")})


# ==========================================================================
# SECTION 5 -- FRONTEND (inlined, single file)
# ==========================================================================
INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Peptide ADMET / Toxicity Predictor</title>
<style>
  :root{
    --bg:#f5f6fa; --panel:#ffffff; --panel2:#f0f2f7; --border:#dfe3ec;
    --text:#1c1f2b; --muted:#6b7280; --accent:#3b6fe0; --accent-soft:#eaf0ff;
    --good:#1fa971; --bad:#e0483f;
    --shadow: 0 1px 3px rgba(20,24,40,.06), 0 1px 2px rgba(20,24,40,.04);
  }
  *{box-sizing:border-box;}
  body{margin:0;font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--text);}
  header{padding:20px 28px;border-bottom:1px solid var(--border);background:var(--panel);}
  header h1{margin:0;font-size:20px;}
  header p{margin:4px 0 0;color:var(--muted);font-size:13px;}
  .tabs{display:flex;flex-wrap:wrap;gap:4px;padding:12px 24px 0;border-bottom:1px solid var(--border);background:var(--panel);}
  .tab{padding:9px 14px;border-radius:8px 8px 0 0;cursor:pointer;color:var(--muted);font-size:13px;user-select:none;}
  .tab.active{background:var(--accent-soft);color:var(--accent);font-weight:600;}
  .tab .dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:6px;}
  .dot.ready{background:var(--good);} .dot.missing{background:var(--bad);}
  .content{padding:24px;max-width:1000px;margin:0 auto;}
  .panel{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:20px;margin-bottom:16px;box-shadow:var(--shadow);}
  .panel h2{margin-top:0;font-size:16px;}
  .subtabs{display:flex;gap:8px;margin-bottom:14px;}
  .subtab{padding:6px 12px;border-radius:6px;background:var(--panel2);cursor:pointer;font-size:13px;color:var(--muted);}
  .subtab.active{background:var(--accent);color:#ffffff;font-weight:600;}
  textarea,input[type=text]{width:100%;background:var(--panel2);border:1px solid var(--border);color:var(--text);
    border-radius:8px;padding:10px;font-family:monospace;font-size:13px;}
  textarea{min-height:90px;resize:vertical;}
  button{background:var(--accent);color:#ffffff;border:none;padding:9px 16px;border-radius:8px;
    font-weight:600;cursor:pointer;font-size:13px;margin-top:10px;}
  button:hover{opacity:.9;}
  .examples{display:flex;flex-wrap:wrap;gap:8px;margin:6px 0 4px;}
  .example-btn{background:var(--accent-soft);color:var(--accent);border:1px solid var(--border);
    padding:6px 12px;border-radius:20px;font-weight:600;font-size:12px;cursor:pointer;margin-top:0;}
  .example-btn:hover{background:var(--accent);color:#fff;opacity:1;}
  .row{display:flex;gap:12px;flex-wrap:wrap;}
  .row>div{flex:1;min-width:220px;}
  label{font-size:12px;color:var(--muted);display:block;margin-bottom:4px;}
  .result{margin-top:14px;padding:14px;border-radius:8px;background:var(--panel2);font-size:13px;white-space:pre-wrap;border:1px solid var(--border);}
  .result.err{border:1px solid var(--bad);color:var(--bad);background:#fdeceb;}
  .badge{display:inline-block;padding:2px 8px;border-radius:12px;font-size:12px;font-weight:600;}
  .badge.pos{background:#fdeceb;color:var(--bad);}
  .badge.neg{background:#e8f8f1;color:var(--good);}
  table{width:100%;border-collapse:collapse;font-size:12px;margin-top:10px;}
  th,td{padding:6px 8px;border-bottom:1px solid var(--border);text-align:left;white-space:nowrap;}
  th{color:var(--muted);font-weight:600;}
  .missing-note{color:var(--bad);font-size:13px;}
</style>
</head>
<body>
<header>
  <h1>Peptide ADMET / Toxicity Predictor</h1>
  <p>One model per endpoint &middot; auto-trains missing models on startup &middot; single script</p>
</header>
<div class="tabs" id="tabs"></div>
<div class="content" id="content"></div>
<script>
const ENDPOINTS = {{ endpoints_json | safe }};
let status = {};
let activeTab = Object.keys(ENDPOINTS)[0];
let activeSub = {};

async function loadStatus(){
  const res = await fetch('/api/status');
  status = await res.json();
  renderTabs();
  renderContent();
}
function renderTabs(){
  const tabs = document.getElementById('tabs');
  tabs.innerHTML = '';
  Object.entries(ENDPOINTS).forEach(([key, cfg]) => {
    const s = status[key] || {};
    const div = document.createElement('div');
    div.className = 'tab' + (key === activeTab ? ' active' : '');
    div.innerHTML = `<span class="dot ${s.model_ready ? 'ready' : 'missing'}"></span>${cfg.label}`;
    div.onclick = () => { activeTab = key; renderTabs(); renderContent(); };
    tabs.appendChild(div);
  });
}
function renderContent(){
  const cfg = ENDPOINTS[activeTab];
  const s = status[activeTab] || {};
  const sub = activeSub[activeTab] || 'predict';
  const c = document.getElementById('content');
  let html = `<div class="panel"><div class="subtabs">
    <div class="subtab ${sub==='predict'?'active':''}" onclick="setSub('predict')">Predict</div>
    <div class="subtab ${sub==='metrics'?'active':''}" onclick="setSub('metrics')">Metrics / Leaderboard</div>
  </div>`;
  if (!s.model_ready) {
    html += `<p class="missing-note">No trained model found at <code>${s.model_path}</code>.
      Run this script again after the training data is available, or check the
      terminal output from startup for training errors.</p></div>`;
    c.innerHTML = html; return;
  }
  if (sub === 'predict') html += renderPredictPanel(cfg);
  else html += `<div id="metrics-area">Loading metrics&hellip;</div>`;
  html += `</div>`;
  c.innerHTML = html;
  if (sub === 'metrics') loadMetrics(activeTab);
}
function setSub(name){ activeSub[activeTab] = name; renderContent(); }
function renderExampleButtons(cfg){
  if (!cfg.examples || !cfg.examples.length) return '';
  const btns = cfg.examples.map((ex, i) =>
    `<span class="example-btn" onclick="fillExample(${i})">${ex.label}</span>`
  ).join('');
  return `<div class="examples">${btns}</div>`;
}
function fillExample(i){
  const cfg = ENDPOINTS[activeTab];
  const ex = cfg.examples[i];
  if (cfg.input_kind === 'pair') {
    document.getElementById('single-input2a').value = ex.input || '';
    document.getElementById('single-input2b').value = ex.input2 || '';
  } else {
    document.getElementById('single-input').value = ex.input || '';
  }
  runSingle();
}
function renderPredictPanel(cfg){
  if (cfg.input_kind === 'pair') {
    return `
      <h2>Single prediction</h2>
      ${renderExampleButtons(cfg)}
      <div class="row">
        <div><label>Binder sequence</label><input type="text" id="single-input2a" placeholder="e.g. KLALKLALKAWKAALKLA"></div>
        <div><label>Target sequence</label><input type="text" id="single-input2b" placeholder="e.g. target protein sequence"></div>
      </div>
      <button onclick="runSingle()">Predict</button>
      <div id="single-result"></div>
      <h2 style="margin-top:24px;">Batch (paste, one pair per line: binder,target)</h2>
      <textarea id="batch-input" placeholder="binder1,target1&#10;binder2,target2"></textarea>
      <button onclick="runBatch()">Predict batch</button>
      <div id="batch-result"></div>
      <h2 style="margin-top:24px;">CSV upload</h2>
      <div class="row">
        <div><label>CSV file</label><input type="file" id="csv-file"></div>
        <div><label>Binder column name</label><input type="text" id="csv-col" value="SMILES"></div>
        <div><label>Target column name</label><input type="text" id="csv-col2" value="target"></div>
      </div>
      <button onclick="runCsv()">Upload &amp; download predictions</button>
      <div id="csv-result"></div>`;
  }
  return `
    <h2>Single prediction</h2>
    ${renderExampleButtons(cfg)}
    <label>SMILES / sequence</label>
    <input type="text" id="single-input" placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O or KLALKLALKAWKAALKLA">
    <button onclick="runSingle()">Predict</button>
    <div id="single-result"></div>
    <h2 style="margin-top:24px;">Batch (paste, one per line)</h2>
    <textarea id="batch-input" placeholder="one SMILES/sequence per line"></textarea>
    <button onclick="runBatch()">Predict batch</button>
    <div id="batch-result"></div>
    <h2 style="margin-top:24px;">CSV upload</h2>
    <div class="row">
      <div><label>CSV file</label><input type="file" id="csv-file"></div>
      <div><label>Column name</label><input type="text" id="csv-col" value="SMILES"></div>
    </div>
    <button onclick="runCsv()">Upload &amp; download predictions</button>
    <div id="csv-result"></div>`;
}
function formatResult(r){
  if (r.error) return `<div class="result err">${r.error}</div>`;
  if ('probability' in r) {
    const badgeClass = r.class === 1 ? 'pos' : 'neg';
    return `<div class="result"><span class="badge ${badgeClass}">${r.prediction}</span>
      &nbsp; probability: ${r.probability} &nbsp; confidence: ${r.confidence}</div>`;
  }
  return `<div class="result">Prediction: <b>${r.prediction}</b> ${r.unit || ''}</div>`;
}
async function runSingle(){
  const cfg = ENDPOINTS[activeTab];
  let body;
  if (cfg.input_kind === 'pair') {
    body = { input: document.getElementById('single-input2a').value,
             input2: document.getElementById('single-input2b').value };
  } else {
    body = { input: document.getElementById('single-input').value };
  }
  const res = await fetch(`/api/predict/${activeTab}`, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  const data = await res.json();
  document.getElementById('single-result').innerHTML = formatResult(data);
}
async function runBatch(){
  const raw = document.getElementById('batch-input').value;
  const res = await fetch(`/api/predict_batch/${activeTab}`, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({input_block: raw})});
  const data = await res.json();
  const el = document.getElementById('batch-result');
  if (data.error) { el.innerHTML = `<div class="result err">${data.error}</div>`; return; }
  let rows = data.results.map(r => `<tr><td>${JSON.stringify(r.input)}</td><td>${r.error ? r.error : (r.prediction)}</td><td>${r.probability ?? ''}</td></tr>`).join('');
  el.innerHTML = `<table><tr><th>Input</th><th>Prediction</th><th>Probability</th></tr>${rows}</table>`;
}
async function runCsv(){
  const fileInput = document.getElementById('csv-file');
  if (!fileInput.files.length) { alert('Choose a CSV file first.'); return; }
  const cfg = ENDPOINTS[activeTab];
  const fd = new FormData();
  fd.append('file', fileInput.files[0]);
  fd.append('input_col', document.getElementById('csv-col').value);
  if (cfg.input_kind === 'pair') fd.append('input_col2', document.getElementById('csv-col2').value);
  const res = await fetch(`/api/predict_csv/${activeTab}`, {method:'POST', body: fd});
  if (!res.ok) {
    const data = await res.json();
    document.getElementById('csv-result').innerHTML = `<div class="result err">${data.error || 'Failed.'}</div>`;
    return;
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = `${activeTab}_predictions.csv`; a.click();
  document.getElementById('csv-result').innerHTML = `<div class="result">Downloaded.</div>`;
}
async function loadMetrics(endpointKey){
  const res = await fetch(`/api/leaderboard/${endpointKey}`);
  const data = await res.json();
  const el = document.getElementById('metrics-area');
  if (!el) return;
  if (data.error) { el.innerHTML = `<p class="missing-note">${data.error}</p>`; return; }
  let head = data.columns.map(c => `<th>${c}</th>`).join('');
  let rows = data.rows.map(r => `<tr>${data.columns.map(c => `<td>${typeof r[c] === 'number' ? r[c].toFixed(4) : r[c]}</td>`).join('')}</tr>`).join('');
  el.innerHTML = `<table><tr>${head}</tr>${rows}</table>`;
}
loadStatus();
</script>
</body>
</html>
"""

# ==========================================================================
# SECTION 6 -- STARTUP
# ==========================================================================
if __name__ == "__main__":
    print(f"{'=' * 90}\nChecking / exporting models for each endpoint\n{'=' * 90}")
    ensure_models_exported()

    print(f"\n{'=' * 90}\nFinal endpoint status\n{'=' * 90}")
    for key in ENDPOINTS:
        s = endpoint_status(key)
        flag = "OK" if s["model_ready"] else "MISSING MODEL"
        print(f"  [{flag:14s}] {key:28s} -> {s['model_path']}")

    print("\nStarting Flask app on http://0.0.0.0:5000 ...")
    app.run(debug=True, host="0.0.0.0", port=5000)
