"""
ablation.py
-----------
Step 8: removes each of the three structural inputs in turn (distance
matrix, dummy node, adjacency matrix) and retrains, to measure each one's
individual contribution to R2 -- reproduces the paper's ablation table.

Usage:
  python ablation.py --train splits/PAMPA_train.csv --val splits/PAMPA_val.csv \
      --test splits/PAMPA_test.csv --out_dir runs/pampa_ablation --epochs 60
"""
import argparse
import json
import os
import torch

from train import train_one_run


CONFIGS = {
    "baseline (nothing removed)": {},
    "distance matrix removed": {"no_distance": True},
    "dummy node removed": {"no_dummy_node": True},
    "adjacency matrix removed": {"no_adjacency": True},
}


class Args:
    """Small shim so train_one_run's `args.<flag>` attribute access works
    without needing argparse.Namespace boilerplate for every config."""
    def __init__(self, base_args, overrides):
        for k, v in vars(base_args).items():
            setattr(self, k, v)
        for k, v in overrides.items():
            setattr(self, k, v)
        # ablation configs default these False unless overridden above
        for flag in ("no_distance", "no_adjacency", "no_dummy_node"):
            if not hasattr(self, flag):
                setattr(self, flag, False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", default=None)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--max_atoms", type=int, default=200)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--lambdas", default="balanced")
    ap.add_argument("--force_field", default="MMFF")
    ap.add_argument("--no_nb", action="store_true")
    ap.add_argument("--init_checkpoint", default=None)
    base_args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(base_args.out_dir, exist_ok=True)

    results = {}
    for name, overrides in CONFIGS.items():
        print(f"\n=== ablation: {name} ===")
        run_args = Args(base_args, overrides)
        run_dir = os.path.join(base_args.out_dir, name.replace(" ", "_").replace("(", "").replace(")", ""))
        r2s = []
        for r in range(base_args.repeats):
            metrics, _ = train_one_run(base_args.train, base_args.val, base_args.test, run_dir,
                                        seed=r, args=run_args, device=device)
            r2s.append(metrics["r2"])
        results[name] = {"r2_runs": r2s, "r2_mean": sum(r2s) / len(r2s)}
        print(f"{name}: R2 = {results[name]['r2_mean']:.3f}")

    print("\n=== Ablation summary ===")
    for name, res in results.items():
        print(f"  {name:35s} R2 = {res['r2_mean']:.3f}")

    with open(os.path.join(base_args.out_dir, "ablation_summary.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
