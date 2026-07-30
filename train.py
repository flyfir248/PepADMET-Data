"""
train.py
--------
Step 4 training protocol + Step 6 evaluation metrics.

Two modes, matching the paper:
  - from_scratch: train on <assay>_train.csv, model-select on <assay>_val.csv
    (PAMPA / Caco-2). Plain MSE loss, Adam.
  - finetune: load a checkpoint pretrained on a larger related assay
    (e.g. Caco-2), then continue training on a small assay's train split
    (RRCK / MDCK) before evaluating on its test split. Use this for the
    RRCK/MDCK "pretrain on Caco-2, then fine-tune" strategy (Step 5.2).

Each run is repeated `--repeats` times (paper uses 3) with different seeds
and metrics are reported as mean +/- std, to match Step 6.

Usage:
  # PAMPA / Caco-2, from scratch
  python train.py --train splits/PAMPA_train.csv --val splits/PAMPA_val.csv \
      --test splits/PAMPA_test.csv --out_dir runs/pampa --repeats 3

  # RRCK, pretrain-then-finetune
  python train.py --train splits/RRCK_train.csv --test splits/RRCK_test.csv \
      --out_dir runs/rrck_finetuned --init_checkpoint runs/caco2/best_seed0.pt --repeats 3
"""
import argparse
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from dataset_mat import PeptidePermeabilityDataset, collate_fn, atom_feature_dim
from model_mat import MATModel, LAMBDA_PRESETS


def build_model(device, lambdas_key="balanced", use_distance=True, use_adjacency=True,
                 use_dummy_node=True, d_model=128, n_heads=8, n_layers=4, d_ff=256):
    model = MATModel(
        atom_feat_dim=atom_feature_dim(),
        d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff,
        lambdas=LAMBDA_PRESETS[lambdas_key],
        use_distance=use_distance, use_adjacency=use_adjacency, use_dummy_node=use_dummy_node,
    ).to(device)
    return model


def run_epoch(model, loader, device, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    total_loss, n = 0.0, 0
    preds, trues = [], []
    for batch in loader:
        atom = batch["atom"].to(device)
        adj = batch["adj"].to(device)
        dist = batch["dist"].to(device)
        mask = batch["mask"].to(device)
        y = batch["y"].to(device)

        with torch.set_grad_enabled(train_mode):
            pred = model(atom, adj, dist, mask)
            loss = torch.nn.functional.mse_loss(pred, y)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * y.size(0)
        n += y.size(0)
        preds.append(pred.detach().cpu().numpy())
        trues.append(y.detach().cpu().numpy())

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    metrics = {
        "loss": total_loss / max(n, 1),
        "mse": mean_squared_error(trues, preds),
        "mae": mean_absolute_error(trues, preds),
        "r2": r2_score(trues, preds) if len(set(trues.tolist())) > 1 else float("nan"),
    }
    return metrics, preds, trues


def train_one_run(train_csv, val_csv, test_csv, out_dir, seed, args, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = PeptidePermeabilityDataset(train_csv, max_atoms=args.max_atoms,
                                           force_field=args.force_field, non_bonded=not args.no_nb, seed=seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    val_loader = None
    if val_csv:
        val_ds = PeptidePermeabilityDataset(val_csv, max_atoms=args.max_atoms,
                                             force_field=args.force_field, non_bonded=not args.no_nb, seed=seed)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    test_ds = PeptidePermeabilityDataset(test_csv, max_atoms=args.max_atoms,
                                          force_field=args.force_field, non_bonded=not args.no_nb, seed=seed)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = build_model(device, lambdas_key=args.lambdas, use_distance=not args.no_distance,
                         use_adjacency=not args.no_adjacency, use_dummy_node=not args.no_dummy_node,
                         d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers)

    if args.init_checkpoint:
        print(f"[finetune] loading pretrained weights from {args.init_checkpoint}")
        model.load_state_dict(torch.load(args.init_checkpoint, map_location=device))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_r2 = -np.inf
    best_state = None
    patience_left = args.patience
    for epoch in range(1, args.epochs + 1):
        train_metrics, _, _ = run_epoch(model, train_loader, device, optimizer)
        monitor_loader = val_loader if val_loader is not None else train_loader
        eval_metrics, _, _ = run_epoch(model, monitor_loader, device, optimizer=None)

        print(f"  epoch {epoch:3d} | train_loss {train_metrics['loss']:.4f} | "
              f"{'val' if val_loader else 'train'}_r2 {eval_metrics['r2']:.4f}")

        if eval_metrics["r2"] > best_val_r2:
            best_val_r2 = eval_metrics["r2"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"best_seed{seed}.pt")
    torch.save(model.state_dict(), ckpt_path)

    test_metrics, preds, trues = run_epoch(model, test_loader, device, optimizer=None)
    print(f"  TEST seed={seed}: mse={test_metrics['mse']:.4f} mae={test_metrics['mae']:.4f} r2={test_metrics['r2']:.4f}")
    return test_metrics, ckpt_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", default=None, help="omit for small-assay 7:3 splits (no val set)")
    ap.add_argument("--test", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--init_checkpoint", default=None, help="pretrained weights for finetuning (Step 5.2)")

    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--max_atoms", type=int, default=200)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--lambdas", default="balanced", choices=list(LAMBDA_PRESETS.keys()))
    ap.add_argument("--force_field", default="MMFF", choices=["UFF", "MMFF"])
    ap.add_argument("--no_nb", action="store_true", help="use the '-NB' (no non-bonded) force-field variant")

    # ablation flags (Step 8)
    ap.add_argument("--no_distance", action="store_true")
    ap.add_argument("--no_adjacency", action="store_true")
    ap.add_argument("--no_dummy_node", action="store_true")

    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    results = []
    for r in range(args.repeats):
        print(f"\n=== run {r+1}/{args.repeats} (seed={r}) ===")
        metrics, ckpt = train_one_run(args.train, args.val, args.test, args.out_dir, seed=r,
                                       args=args, device=device)
        results.append(metrics)

    summary = {}
    for key in ["mse", "mae", "r2"]:
        vals = np.array([m[key] for m in results])
        summary[key] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}
    print("\n=== summary over repeats ===")
    print(json.dumps(summary, indent=2))

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"per_run": results, "summary": summary, "args": vars(args)}, f, indent=2)


if __name__ == "__main__":
    main()
