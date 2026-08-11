"""
baselines.py
------------
Step 5 baselines: Random Forest Regressor and Support Vector Regression on
1024-bit Morgan fingerprints (no graph, no 3D), each hyperparameter-tuned by
grid search rather than left at library defaults -- so the gap between these
and the MAT model isolates what the graph + 3D distance information buys you.

PATCH NOTE: main() now also pickles each best-fit estimator (rfr_model.pkl /
svr_model.pkl) alongside baseline_results.json, so export_models.py has
something to bundle for the Flask app. Nothing about featurization or
grid search changed.

Usage:
  python baselines.py --train splits/PAMPA_train.csv --val splits/PAMPA_val.csv \
      --test splits/PAMPA_test.csv --out_dir runs/pampa_baselines
"""
import argparse
import json
import os
import pickle
import time
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")


def smiles_to_morgan(smiles: str, n_bits: int = 1024, radius: int = 2):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def featurize_df(df: pd.DataFrame, n_bits: int = 1024, radius: int = 2, desc: str = "fingerprints"):
    X, y, kept = [], [], []
    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"[baselines] {desc}"):
        fp = smiles_to_morgan(row["smiles"], n_bits=n_bits, radius=radius)
        if fp is None:
            continue
        X.append(fp)
        y.append(row["logp_exp"])
        kept.append(i)
    if len(X) < len(df):
        print(f"[baselines] dropped {len(df) - len(X)}/{len(df)} unparseable SMILES")
    return np.stack(X), np.array(y, dtype=np.float32)


RFR_GRID_FAST = {"n_estimators": [200], "max_depth": [None, 20]}
RFR_GRID_FULL = {"n_estimators": [200, 500], "max_depth": [None, 10, 20, 40]}
# Exact grid from Jiang, Chen & Du (2025), Methods 2.4: n_estimators fixed at
# 100, depth swept over [5,10,15,20,25,30,35,40] (paper found 20 optimal).
RFR_GRID_PAPER = {"n_estimators": [100], "max_depth": [5, 10, 15, 20, 25, 30, 35, 40]}

SVR_GRID_FAST = {"C": [1, 10], "epsilon": [0.1], "kernel": ["rbf"]}
SVR_GRID_FULL = {"C": [1, 10, 100], "epsilon": [0.01, 0.1, 0.5], "kernel": ["rbf"]}
# Exact grid from the paper: C in [0.1,1,10,100], epsilon in [0.01,0.1,0.5,1], RBF kernel.
SVR_GRID_PAPER = {"C": [0.1, 1, 10, 100], "epsilon": [0.01, 0.1, 0.5, 1], "kernel": ["rbf"]}


def _combined_split(X_train, y_train, X_val, y_val):
    """Build a PredefinedSplit so GridSearchCV tunes against the val set
    (matching the paper's train/val/test protocol) instead of doing its
    own internal k-fold CV on the training data alone."""
    X = np.concatenate([X_train, X_val], axis=0)
    y = np.concatenate([y_train, y_val], axis=0)
    test_fold = np.concatenate([-np.ones(len(X_train)), np.zeros(len(X_val))])
    return X, y, PredefinedSplit(test_fold)


def fit_and_eval(name, estimator, grid, X_train, y_train, X_val, y_val, X_test, y_test):
    n_candidates = 1
    for v in grid.values():
        n_candidates *= len(v)
    print(f"[{name}] fitting {n_candidates} hyperparameter candidate(s) on "
          f"{len(X_train) + (len(X_val) if X_val is not None else 0)} peptides "
          f"(this can take a few minutes for RFR with large n_estimators)...")
    t0 = time.time()

    if X_val is not None:
        X, y, split = _combined_split(X_train, y_train, X_val, y_val)
        gs = GridSearchCV(estimator, grid, cv=split, scoring="r2", n_jobs=-1, verbose=2)
        gs.fit(X, y)
    else:
        gs = GridSearchCV(estimator, grid, cv=5, scoring="r2", n_jobs=-1, verbose=2)
        gs.fit(X_train, y_train)

    best = gs.best_estimator_
    preds = best.predict(X_test)
    metrics = {
        "mse": mean_squared_error(y_test, preds),
        "mae": mean_absolute_error(y_test, preds),
        "r2": r2_score(y_test, preds),
        "best_params": gs.best_params_,
    }
    print(f"[{name}] done in {time.time() - t0:.1f}s | best_params={gs.best_params_} "
          f"test_r2={metrics['r2']:.4f} test_mae={metrics['mae']:.4f} test_mse={metrics['mse']:.4f}")
    return best, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", default=None)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_bits", type=int, default=1024)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--grid", choices=["fast", "full", "paper"], default="fast",
                     help="'paper' reproduces the exact RFR/SVR grids from Jiang et al. 2025 Methods 2.4")
    args = ap.parse_args()

    rfr_grid = {"fast": RFR_GRID_FAST, "full": RFR_GRID_FULL, "paper": RFR_GRID_PAPER}[args.grid]
    svr_grid = {"fast": SVR_GRID_FAST, "full": SVR_GRID_FULL, "paper": SVR_GRID_PAPER}[args.grid]

    os.makedirs(args.out_dir, exist_ok=True)
    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)
    val_df = pd.read_csv(args.val) if args.val else None

    X_train, y_train = featurize_df(train_df, args.n_bits, args.radius, desc="train fingerprints")
    X_test, y_test = featurize_df(test_df, args.n_bits, args.radius, desc="test fingerprints")
    X_val, y_val = (featurize_df(val_df, args.n_bits, args.radius, desc="val fingerprints")
                     if val_df is not None else (None, None))

    results = {}
    rfr_best, results["RFR"] = fit_and_eval("RFR", RandomForestRegressor(random_state=0, n_jobs=-1), rfr_grid,
                                             X_train, y_train, X_val, y_val, X_test, y_test)
    svr_best, results["SVR"] = fit_and_eval("SVR", SVR(), svr_grid,
                                             X_train, y_train, X_val, y_val, X_test, y_test)

    with open(os.path.join(args.out_dir, "baseline_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[write] {os.path.join(args.out_dir, 'baseline_results.json')}")

    # Pickle the actual fitted estimators too -- baseline_results.json only had
    # metrics before, which is all y_randomization.py needed, but export_models.py
    # needs the models themselves to serve predictions from.
    for name, model in [("rfr", rfr_best), ("svr", svr_best)]:
        pkl_path = os.path.join(args.out_dir, f"{name}_model.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(model, f)
        print(f"[write] {pkl_path}")


if __name__ == "__main__":
    main()
