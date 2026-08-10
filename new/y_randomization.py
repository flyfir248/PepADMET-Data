"""
y_randomization.py
-------------------
Step 7: Y-randomization control. Randomly permutes the training labels so
each peptide is paired with a different peptide's permeability value, then
retrains the identical model on the scrambled data. Repeated 20x by default.

A large gap between real-label R2 and scrambled-label R2 is evidence the
model is learning genuine structure-permeability relationships rather than
exploiting a split leak or memorized artifact -- this is exactly the check
worth running against the duplicate-peptide / cross-source leakage risk in
CycPeptMPDB before trusting a scaffold split (see the methodology doc's
"What this means for your own pipeline" section).

By default this scrambles labels for the fast RFR-on-Morgan-fingerprints
baseline (20 runs of full MAT training is expensive); pass --model mat to
run the full MAT model instead if you have the compute budget.

PATCH NOTE: --model mat now requires a GPU via train.get_device() (raises
RuntimeError if none is visible; pass --allow_cpu to override). --model rfr
is sklearn/CPU-bound regardless of GPU availability and is unaffected.

Usage:
  python y_randomization.py --train splits/PAMPA_train.csv --test splits/PAMPA_test.csv \
      --out_dir runs/pampa_yrand --n_runs 20 --model rfr
"""
import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from baselines import featurize_df
from dataset_mat import PeptidePermeabilityDataset, collate_fn, atom_feature_dim
from model_mat import MATModel, LAMBDA_PRESETS
from train import run_epoch, get_device


def yrand_rfr(train_df, test_df, n_runs, seed0=0):
    X_train, y_train = featurize_df(train_df)
    X_test, y_test = featurize_df(test_df)

    real_model = RandomForestRegressor(n_estimators=500, random_state=0).fit(X_train, y_train)
    real_r2 = r2_score(y_test, real_model.predict(X_test))

    scrambled_r2s = []
    rng = np.random.RandomState(seed0)
    for run in range(n_runs):
        y_perm = rng.permutation(y_train)
        model = RandomForestRegressor(n_estimators=500, random_state=run).fit(X_train, y_perm)
        r2 = r2_score(y_test, model.predict(X_test))
        scrambled_r2s.append(r2)
        print(f"  [y-rand rfr] run {run+1}/{n_runs}: scrambled test r2 = {r2:.4f}")

    return real_r2, scrambled_r2s


def yrand_mat(train_csv, test_csv, n_runs, args, device):
    train_df = pd.read_csv(train_csv)

    def _train_and_eval(train_df_local, seed, epochs=30):
        tmp_csv = os.path.join(args.out_dir, f"_tmp_train_seed{seed}.csv")
        train_df_local.to_csv(tmp_csv, index=False)
        train_ds = PeptidePermeabilityDataset(tmp_csv, max_atoms=args.max_atoms)
        test_ds = PeptidePermeabilityDataset(test_csv, max_atoms=args.max_atoms)
        pin = device.type == "cuda"
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn, pin_memory=pin)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn, pin_memory=pin)

        # Note: for scrambled runs this is computed on the permuted labels, but a
        # permutation preserves mean/std, so this matches what train.py would
        # compute for the same (real or scrambled) split.
        y_mean, y_std = train_ds.target_stats()

        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        model = MATModel(atom_feat_dim=atom_feature_dim(), d_model=64, n_heads=4, n_layers=2,
                          lambdas=LAMBDA_PRESETS["balanced"], y_mean=y_mean, y_std=y_std).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        for _ in range(epochs):
            run_epoch(model, train_loader, device, optimizer)
        test_metrics, _, _ = run_epoch(model, test_loader, device, optimizer=None)
        os.remove(tmp_csv)
        return test_metrics["r2"]

    real_r2 = _train_and_eval(train_df, seed=999)

    scrambled_r2s = []
    rng = np.random.RandomState(0)
    for run in range(n_runs):
        perm_df = train_df.copy()
        perm_df["logp_exp"] = rng.permutation(perm_df["logp_exp"].values)
        r2 = _train_and_eval(perm_df, seed=run)
        scrambled_r2s.append(r2)
        print(f"  [y-rand mat] run {run+1}/{n_runs}: scrambled test r2 = {r2:.4f}")

    return real_r2, scrambled_r2s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_runs", type=int, default=20)
    ap.add_argument("--model", choices=["rfr", "mat"], default="rfr")
    ap.add_argument("--max_atoms", type=int, default=200)
    ap.add_argument("--allow_cpu", action="store_true",
                     help="allow --model mat to fall back to CPU if no GPU is found (default: hard error). "
                          "Has no effect on --model rfr, which is always CPU/sklearn.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.model == "rfr":
        train_df = pd.read_csv(args.train)
        test_df = pd.read_csv(args.test)
        real_r2, scrambled_r2s = yrand_rfr(train_df, test_df, args.n_runs)
    else:
        device = get_device(allow_cpu=args.allow_cpu)
        real_r2, scrambled_r2s = yrand_mat(args.train, args.test, args.n_runs, args, device)

    result = {
        "model": args.model,
        "real_label_r2": real_r2,
        "scrambled_r2_mean": float(np.mean(scrambled_r2s)),
        "scrambled_r2_std": float(np.std(scrambled_r2s)),
        "scrambled_r2s": scrambled_r2s,
    }
    print("\n=== Y-randomization result ===")
    print(json.dumps(result, indent=2))
    print(f"\nReal-label R2 ({real_r2:.3f}) vs. scrambled-label R2 "
          f"({result['scrambled_r2_mean']:.3f} +/- {result['scrambled_r2_std']:.3f}).")
    print("A large gap here is the reassuring outcome: it means the model is "
          "learning real structure-permeability relationships, not exploiting a leak.")

    with open(os.path.join(args.out_dir, "y_randomization.json"), "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
