"""
export_models.py
-----------------
Bundles trained RFR, SVR, and MAT checkpoints into per-model pickle files
plus a single models_manifest.json, ready to be loaded by model_registry.py
(used by app.py).

Assumes:
  - baselines.py has been (re-)run with the pickle-export patch, producing
    rfr_model.pkl / svr_model.pkl / baseline_results.json in --baselines_dir.
  - train.py has produced a MAT checkpoint (best_seed<N>.pt) and summary.json
    in --mat_dir. Pass the SAME architecture flags you trained with
    (--d_model, --lambdas, --no_distance, etc.) or the state_dict load below
    will fail with a shape-mismatch error -- that's intentional, it's a
    guard against silently serving a mis-configured model.

Usage:
  python export_models.py \
      --baselines_dir cpmp_runs/pampa_baselines \
      --mat_dir cpmp_runs/pampa_mat --mat_seed 0 \
      --out_dir exported_models/pampa
"""
import argparse
import json
import os
import pickle
import shutil

import torch

from model_mat import MATModel, LAMBDA_PRESETS
from dataset_mat import atom_feature_dim


def export_baselines(baselines_dir: str, out_dir: str) -> dict:
    manifest = {}
    results_path = os.path.join(baselines_dir, "baseline_results.json")
    with open(results_path) as f:
        results = json.load(f)

    for name in ["RFR", "SVR"]:
        src_pkl = os.path.join(baselines_dir, f"{name.lower()}_model.pkl")
        if not os.path.exists(src_pkl):
            raise FileNotFoundError(
                f"{src_pkl} not found. Re-run the patched baselines.py (it now pickles the "
                f"fitted {name} estimator) before exporting."
            )
        dst_pkl = os.path.join(out_dir, f"{name.lower()}_model.pkl")
        shutil.copy(src_pkl, dst_pkl)
        manifest[name.lower()] = {
            "type": name,
            "pickle_path": os.path.basename(dst_pkl),
            "featurization": {"kind": "morgan_fingerprint", "n_bits": 1024, "radius": 2},
            "test_metrics": {
                "r2": results[name]["r2"],
                "mae": results[name]["mae"],
                "mse": results[name]["mse"],
            },
            "best_params": results[name]["best_params"],
        }
    return manifest


def export_mat(mat_dir: str, mat_seed: int, out_dir: str, max_atoms: int, force_field: str,
                non_bonded: bool, lambdas_key: str, use_distance: bool, use_adjacency: bool,
                use_dummy_node: bool, d_model: int, n_heads: int, n_layers: int, d_ff: int,
                n_dense: int = 1, leaky_slope: float = None,
                distance_kernel_kind: str = "softmax_neg") -> dict:
    ckpt_path = os.path.join(mat_dir, f"best_seed{mat_seed}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"{ckpt_path} not found. Check --mat_dir / --mat_seed.")
    state_dict = torch.load(ckpt_path, map_location="cpu")

    config = {
        "atom_feat_dim": atom_feature_dim(add_dummy_node=use_dummy_node),
        "d_model": d_model, "n_heads": n_heads, "n_layers": n_layers, "d_ff": d_ff,
        "lambdas": LAMBDA_PRESETS[lambdas_key],
        "use_distance": use_distance, "use_adjacency": use_adjacency, "use_dummy_node": use_dummy_node,
        "max_atoms": max_atoms, "force_field": force_field, "non_bonded": non_bonded,
        "n_dense": n_dense, "leaky_slope": leaky_slope, "distance_kernel_kind": distance_kernel_kind,
    }

    # Fail loudly here, at export time, rather than inside the Flask app:
    # if the architecture flags don't match what the checkpoint was trained
    # with, load_state_dict raises a clear shape-mismatch error.
    model = MATModel(atom_feat_dim=config["atom_feat_dim"], d_model=d_model, n_heads=n_heads,
                      n_layers=n_layers, d_ff=d_ff, lambdas=config["lambdas"],
                      use_distance=use_distance, use_adjacency=use_adjacency,
                      use_dummy_node=use_dummy_node, n_dense=n_dense, leaky_slope=leaky_slope,
                      distance_kernel_kind=distance_kernel_kind)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        # Unexpected keys (vs. just missing y_mean/y_std) means a real
        # architecture mismatch -- fail loudly rather than export a broken model.
        raise RuntimeError(f"Checkpoint has unexpected keys {unexpected} for this architecture; "
                            f"double-check --d_model/--n_heads/--n_layers/--lambdas/etc. match training.")
    if missing:
        print(f"[export] WARNING: {missing} not in checkpoint (predates the y_mean/y_std patch); "
              f"this MAT model will predict on the RAW target scale, not normalized.")

    dst_pkl = os.path.join(out_dir, "mat_model.pkl")
    with open(dst_pkl, "wb") as f:
        pickle.dump({"state_dict": state_dict, "config": config}, f)

    test_metrics = None
    summary_path = os.path.join(mat_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        test_metrics = {k: summary["summary"][k]["mean"] for k in ["r2", "mae", "mse"]}
    else:
        print(f"[export] WARNING: {summary_path} not found, MAT test_metrics will be null "
              f"(the frontend will still show live predictions, just no comparison numbers for MAT)")

    return {
        "mat": {
            "type": "MAT",
            "pickle_path": os.path.basename(dst_pkl),
            "featurization": {"kind": "3d_conformer", "max_atoms": max_atoms,
                               "force_field": force_field, "non_bonded": non_bonded},
            "test_metrics": test_metrics,
        }
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines_dir", required=True)
    ap.add_argument("--mat_dir", required=True)
    ap.add_argument("--mat_seed", type=int, default=0)
    ap.add_argument("--out_dir", required=True)

    # These must match the flags train.py was run with for this checkpoint.
    ap.add_argument("--max_atoms", type=int, default=200)
    ap.add_argument("--force_field", default="MMFF", choices=["UFF", "MMFF"])
    ap.add_argument("--no_nb", action="store_true")
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--d_ff", type=int, default=256)
    ap.add_argument("--n_dense", type=int, default=1)
    ap.add_argument("--leaky_slope", type=float, default=None)
    ap.add_argument("--distance_kernel_kind", choices=["softmax_neg", "exp_elementwise", "exp_neg"],
                     default="softmax_neg")
    ap.add_argument("--lambdas", default="balanced", choices=list(LAMBDA_PRESETS.keys()))
    ap.add_argument("--paper_faithful", action="store_true",
                     help="architecture flags matching a checkpoint trained with train.py --paper_faithful")
    ap.add_argument("--no_distance", action="store_true")
    ap.add_argument("--no_adjacency", action="store_true")
    ap.add_argument("--no_dummy_node", action="store_true")
    args = ap.parse_args()

    if args.paper_faithful:
        for k, v in dict(d_model=64, n_heads=64, n_layers=6, n_dense=2, leaky_slope=0.16,
                          lambdas="cpmp_pampa", distance_kernel_kind="exp_elementwise",
                          force_field="UFF", no_nb=True).items():
            setattr(args, k, v)

    os.makedirs(args.out_dir, exist_ok=True)

    manifest = {}
    manifest.update(export_baselines(args.baselines_dir, args.out_dir))
    manifest.update(export_mat(
        args.mat_dir, args.mat_seed, args.out_dir,
        max_atoms=args.max_atoms, force_field=args.force_field, non_bonded=not args.no_nb,
        lambdas_key=args.lambdas, use_distance=not args.no_distance, use_adjacency=not args.no_adjacency,
        use_dummy_node=not args.no_dummy_node, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, d_ff=args.d_ff, n_dense=args.n_dense, leaky_slope=args.leaky_slope,
        distance_kernel_kind=args.distance_kernel_kind,
    ))

    manifest_path = os.path.join(args.out_dir, "models_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[export] wrote {manifest_path}")
    for name, info in manifest.items():
        print(f"  {name}: {info['pickle_path']}  test_metrics={info.get('test_metrics')}")


if __name__ == "__main__":
    main()
