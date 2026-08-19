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

PATCH NOTE: this build requires a GPU. `get_device()` raises RuntimeError
instead of silently falling back to CPU if CUDA isn't available. Pass
--allow_cpu to opt back into the old fallback behavior if you ever need it
(e.g. for a quick sanity check on a machine with no GPU).

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

# PATCH NOTE (real lambda grid search): the paper determines lambda_a/d/g via
# grid search (Methods 2.2), not by picking among a few named presets. This
# is a coarse simplex grid at 0.2 resolution (lambda_a, lambda_d, lambda_g >= 0,
# summing to 1) -- fine enough to matter, coarse enough that a short-budget
# search over it is tractable. Pass --lambdas grid_search to use it.
def _lambda_simplex_grid(step: float = 0.2):
    vals = [round(i * step, 2) for i in range(int(round(1 / step)) + 1)]
    grid = []
    for a in vals:
        for d in vals:
            g = round(1.0 - a - d, 2)
            if -1e-9 <= g <= 1.0 + 1e-9:
                grid.append((a, d, round(g, 2)))
    return grid


LAMBDA_SEARCH_GRID = _lambda_simplex_grid(step=0.2)

# Paper grid-searches all 4 combinations and picks the best per dataset
# (Methods 3.4 / Supplementary Table S1): UFF-NB won PAMPA, MMFF-NB won Caco-2.
FORCE_FIELD_GRID = [("UFF", True), ("UFF", False), ("MMFF", True), ("MMFF", False)]


def get_device(allow_cpu: bool = False) -> torch.device:
    """
    GPU-only by default. Raises RuntimeError with actionable diagnostics if
    no CUDA device is visible, instead of silently training on CPU (which is
    what `torch.device("cuda" if torch.cuda.is_available() else "cpu")` does).
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        idx = torch.cuda.current_device()
        print(f"[device] using GPU {idx}: {torch.cuda.get_device_name(idx)}")
        return device

    if allow_cpu:
        print("[device] WARNING: no CUDA device found, falling back to CPU because --allow_cpu was set")
        return torch.device("cpu")

    raise RuntimeError(
        "No CUDA GPU detected (torch.cuda.is_available() == False), and this run requires GPU.\n"
        "Checks to run:\n"
        "  1. `nvidia-smi` -- confirms the driver sees a GPU at all.\n"
        "  2. `python -c \"import torch; print(torch.__version__, torch.version.cuda)\"` -- confirms "
        "this is a CUDA build of torch, not a CPU-only wheel (a common cause after `pip install torch` "
        "without a --index-url pointing at a CUDA build).\n"
        "  3. If you're in a container/devcontainer, confirm it was started with GPU passthrough "
        "(e.g. `docker run --gpus all ...`) and the NVIDIA container toolkit is installed on the host.\n"
        "Pass --allow_cpu to override this check and run on CPU anyway."
    )


def build_model(device, lambdas_key="balanced", use_distance=True, use_adjacency=True,
                 use_dummy_node=True, d_model=128, n_heads=8, n_layers=4, d_ff=256,
                 y_mean=0.0, y_std=1.0, n_dense=1, leaky_slope=None, distance_kernel_kind="softmax_neg"):
    lambdas = LAMBDA_PRESETS[lambdas_key] if isinstance(lambdas_key, str) else tuple(lambdas_key)
    model = MATModel(
        atom_feat_dim=atom_feature_dim(),
        d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff,
        lambdas=lambdas,
        use_distance=use_distance, use_adjacency=use_adjacency, use_dummy_node=use_dummy_node,
        y_mean=y_mean, y_std=y_std, n_dense=n_dense, leaky_slope=leaky_slope,
        distance_kernel_kind=distance_kernel_kind,
    ).to(device)
    return model


def search_best_lambdas(train_csv, val_csv, args, device, search_epochs: int = 20):
    """Short-budget grid search over LAMBDA_SEARCH_GRID, ranked by val R2.
    NOT a substitute for full training -- this only ranks candidates cheaply
    (search_epochs << args.epochs) so the final model can be trained at full
    budget with the winning lambdas. Mirrors the paper's 'optimal lambda
    determined via grid search' (Methods 2.2) without the compute cost of
    fully training every candidate."""
    print(f"\n[lambda search] scanning {len(LAMBDA_SEARCH_GRID)} (la, ld, lg) candidates "
          f"at {search_epochs} epochs each (short budget -- for RANKING only)")
    best_lambdas, best_r2 = None, -np.inf
    for lambdas in LAMBDA_SEARCH_GRID:
        torch.manual_seed(0)
        train_ds = PeptidePermeabilityDataset(train_csv, max_atoms=args.max_atoms,
                                               force_field=args.force_field, non_bonded=not args.no_nb, seed=0, num_conformers=args.num_conformers)
        val_ds = PeptidePermeabilityDataset(val_csv, max_atoms=args.max_atoms,
                                             force_field=args.force_field, non_bonded=not args.no_nb, seed=0, num_conformers=args.num_conformers)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        y_mean, y_std = train_ds.target_stats()

        model = build_model(device, lambdas_key=lambdas, use_distance=not args.no_distance,
                             use_adjacency=not args.no_adjacency, use_dummy_node=not args.no_dummy_node,
                             d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
                             y_mean=y_mean, y_std=y_std, n_dense=args.n_dense, leaky_slope=args.leaky_slope,
                             distance_kernel_kind=args.distance_kernel_kind)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for _ in range(search_epochs):
            run_epoch(model, train_loader, device, optimizer, loss_reduction=args.loss_reduction)
        val_metrics, _, _ = run_epoch(model, val_loader, device, optimizer=None, loss_reduction=args.loss_reduction)
        print(f"  [lambda search] lambdas={lambdas} val_r2={val_metrics['r2']:.4f}")
        if val_metrics["r2"] > best_r2:
            best_r2, best_lambdas = val_metrics["r2"], lambdas

    print(f"[lambda search] best: lambdas={best_lambdas} (short-budget val_r2={best_r2:.4f})\n")
    return best_lambdas


def search_best_force_field(train_csv, val_csv, args, device, search_epochs: int = 20):
    """Short-budget search over (force_field, non_bonded) combos, ranked by
    val R2 -- mirrors the paper's UFF/UFF-NB/MMFF/MMFF-NB comparison
    (Methods 3.4). Same short-budget-for-ranking caveat as search_best_lambdas."""
    print(f"\n[force field search] scanning {len(FORCE_FIELD_GRID)} configs at "
          f"{search_epochs} epochs each (short budget -- for RANKING only)")
    best_cfg, best_r2 = None, -np.inf
    for ff, nb in FORCE_FIELD_GRID:
        torch.manual_seed(0)
        train_ds = PeptidePermeabilityDataset(train_csv, max_atoms=args.max_atoms,
                                               force_field=ff, non_bonded=nb, seed=0, num_conformers=args.num_conformers)
        val_ds = PeptidePermeabilityDataset(val_csv, max_atoms=args.max_atoms,
                                             force_field=ff, non_bonded=nb, seed=0, num_conformers=args.num_conformers)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        y_mean, y_std = train_ds.target_stats()

        model = build_model(device, lambdas_key=args.lambdas if args.lambdas != "grid_search" else "balanced",
                             use_distance=not args.no_distance, use_adjacency=not args.no_adjacency,
                             use_dummy_node=not args.no_dummy_node, d_model=args.d_model,
                             n_heads=args.n_heads, n_layers=args.n_layers, y_mean=y_mean, y_std=y_std,
                             n_dense=args.n_dense, leaky_slope=args.leaky_slope,
                             distance_kernel_kind=args.distance_kernel_kind)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for _ in range(search_epochs):
            run_epoch(model, train_loader, device, optimizer, loss_reduction=args.loss_reduction)
        val_metrics, _, _ = run_epoch(model, val_loader, device, optimizer=None, loss_reduction=args.loss_reduction)
        label = f"{ff}{'+NB' if nb else '-NB'}"
        print(f"  [force field search] {label} val_r2={val_metrics['r2']:.4f}")
        if val_metrics["r2"] > best_r2:
            best_r2, best_cfg = val_metrics["r2"], (ff, nb)

    label = f"{best_cfg[0]}{'+NB' if best_cfg[1] else '-NB'}"
    print(f"[force field search] best: {label} (short-budget val_r2={best_r2:.4f})\n")
    return best_cfg


def run_epoch(model, loader, device, optimizer=None, loss_reduction: str = "mean"):
    """
    Loss is computed against the NORMALIZED target ((y - model.y_mean) /
    model.y_std), matching what model.forward() outputs -- this keeps
    gradients at an O(1) scale regardless of the raw LogP range, instead of
    the small fixed lr=1e-4 having to push through a loss computed directly
    on ~-10..0-scale targets. Reported preds/metrics are always converted
    back to the real LogP scale before being returned, so mse/mae/r2 stay
    directly comparable to the RFR/SVR baselines.

    loss_reduction: "mean" (default) backprops the per-sample-mean MSE for
    each batch. "sum" backprops the per-batch SUM of squared errors instead
    -- this is what the actual CPMP source (train_pampa.py) does
    (nn.MSELoss(reduction='sum')), which effectively scales the gradient by
    batch_size relative to "mean". Use "sum" together with --lr 1e-3 (their
    value) to replicate their training dynamics; mixing "sum" with a
    mean-tuned lr will very likely diverge or train far too aggressively.
    """
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    total_loss, n = 0.0, 0
    preds, trues = [], []
    for batch in loader:
        atom = batch["atom"].to(device, non_blocking=True)
        adj = batch["adj"].to(device, non_blocking=True)
        dist = batch["dist"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        y_norm = (y - model.y_mean) / model.y_std

        with torch.set_grad_enabled(train_mode):
            pred_norm = model(atom, adj, dist, mask)
            if loss_reduction == "sum":
                loss = torch.nn.functional.mse_loss(pred_norm, y_norm, reduction="sum")
            else:
                loss = torch.nn.functional.mse_loss(pred_norm, y_norm, reduction="mean")
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        pred_real = pred_norm.detach() * model.y_std + model.y_mean
        # Always accumulate as a per-sample-mean-equivalent for reporting,
        # regardless of which reduction was used for the backward pass.
        batch_loss_per_sample = loss.item() / y.size(0) if loss_reduction == "sum" else loss.item()
        total_loss += batch_loss_per_sample * y.size(0)
        n += y.size(0)
        preds.append(pred_real.cpu().numpy())
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
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    train_ds = PeptidePermeabilityDataset(train_csv, max_atoms=args.max_atoms,
                                           force_field=args.force_field, non_bonded=not args.no_nb, seed=seed, num_conformers=args.num_conformers)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                               pin_memory=pin)

    val_loader = None
    if val_csv:
        val_ds = PeptidePermeabilityDataset(val_csv, max_atoms=args.max_atoms,
                                             force_field=args.force_field, non_bonded=not args.no_nb, seed=seed, num_conformers=args.num_conformers)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                                 pin_memory=pin)

    test_ds = PeptidePermeabilityDataset(test_csv, max_atoms=args.max_atoms,
                                          force_field=args.force_field, non_bonded=not args.no_nb, seed=seed, num_conformers=args.num_conformers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                              pin_memory=pin)

    y_mean, y_std = train_ds.target_stats()
    print(f"[target] logp_exp train stats: mean={y_mean:.3f} std={y_std:.3f} "
          f"(MAT trains against the normalized target, see model_mat.py)")

    model = build_model(device, lambdas_key=args.lambdas, use_distance=not args.no_distance,
                         use_adjacency=not args.no_adjacency, use_dummy_node=not args.no_dummy_node,
                         d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
                         y_mean=y_mean, y_std=y_std, n_dense=args.n_dense, leaky_slope=args.leaky_slope,
                         distance_kernel_kind=args.distance_kernel_kind)

    if args.init_checkpoint:
        print(f"[finetune] loading pretrained weights from {args.init_checkpoint}")
        ckpt_state = torch.load(args.init_checkpoint, map_location=device)
        # strict=False: checkpoints saved before the y_mean/y_std buffers were
        # added won't have those keys. missing/unexpected are reported so you
        # notice, but this shouldn't hard-fail a finetune from an old checkpoint.
        missing, unexpected = model.load_state_dict(ckpt_state, strict=False)
        if missing or unexpected:
            print(f"[finetune] WARNING: missing={missing} unexpected={unexpected} "
                  f"(expected if {args.init_checkpoint} predates the y_mean/y_std patch)")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_r2 = -np.inf
    best_state = None
    patience_left = args.patience
    for epoch in range(1, args.epochs + 1):
        train_metrics, _, _ = run_epoch(model, train_loader, device, optimizer, loss_reduction=args.loss_reduction)
        monitor_loader = val_loader if val_loader is not None else train_loader
        eval_metrics, _, _ = run_epoch(model, monitor_loader, device, optimizer=None, loss_reduction=args.loss_reduction)

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

    test_metrics, preds, trues = run_epoch(model, test_loader, device, optimizer=None, loss_reduction=args.loss_reduction)
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
    ap.add_argument("--loss_reduction", choices=["mean", "sum"], default="mean",
                     help="'sum' replicates CPMP's nn.MSELoss(reduction='sum') per-batch backward -- "
                          "only combine with --lr 1e-3 (their value), not a mean-tuned lr")
    ap.add_argument("--max_atoms", type=int, default=200)
    ap.add_argument("--num_conformers", type=int, default=1,
                     help="candidate conformers to embed per molecule, keeping the lowest-energy one "
                          "(1=fast/previous behavior; 5-10 recommended for final results -- see "
                          "features_mat.py's embed_conformer docstring)")
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_dense", type=int, default=1, help="hidden layers in the readout head (CPMP uses 2)")
    ap.add_argument("--leaky_slope", type=float, default=None,
                     help="LeakyReLU negative_slope for encoder FF + readout activations; "
                          "None = plain ReLU (our default). CPMP uses 0.16")
    ap.add_argument("--distance_kernel_kind", choices=["softmax_neg", "exp_elementwise", "exp_neg"],
                     default="softmax_neg",
                     help="'exp_elementwise' (unnormalized g(d)=exp(-d)) is what CPMP's real source uses; "
                          "'softmax_neg' was our original guess")
    ap.add_argument("--lambdas", default="balanced",
                     help=f"a named preset ({list(LAMBDA_PRESETS.keys())}) or 'grid_search' to run "
                          f"search_best_lambdas() first and use the winner")
    ap.add_argument("--force_field", default="MMFF", choices=["UFF", "MMFF", "auto"],
                     help="'auto' runs search_best_force_field() first and uses the winner")
    ap.add_argument("--no_nb", action="store_true", help="use the '-NB' (no non-bonded) force-field variant")
    ap.add_argument("--allow_cpu", action="store_true",
                     help="allow falling back to CPU if no GPU is found (default: hard error)")
    ap.add_argument("--paper_faithful", action="store_true",
                     help="Override architecture/training hyperparameters to match "
                          "github.com/panda1103/CPMP/blob/main/train_pampa.py exactly: d_model=64, "
                          "n_heads=64, n_layers=6, n_dense=2, leaky_slope=0.16, "
                          "lambdas=cpmp_pampa (0.1/0.6/0.3), distance_kernel_kind=exp_elementwise, "
                          "lr=1e-3, loss_reduction=sum, weight_decay=0, force_field=UFF --no_nb, "
                          "epochs=600, patience=600 (i.e. effectively no early stopping). Checkpoint "
                          "selection still uses validation R2, NOT the original's test-set peeking -- "
                          "see train.py's module docstring / the chat writeup for why.")

    # ablation flags (Step 8)
    ap.add_argument("--no_distance", action="store_true")
    ap.add_argument("--no_adjacency", action="store_true")
    ap.add_argument("--no_dummy_node", action="store_true")

    args = ap.parse_args()

    if args.paper_faithful:
        overrides = dict(d_model=64, n_heads=64, n_layers=6, n_dense=2, leaky_slope=0.16,
                          lambdas="cpmp_pampa", distance_kernel_kind="exp_elementwise",
                          lr=1e-3, loss_reduction="sum", weight_decay=0.0,
                          force_field="UFF", no_nb=True, epochs=600, patience=600)
        print("[paper_faithful] overriding hyperparameters to match CPMP's train_pampa.py:")
        for k, v in overrides.items():
            print(f"    --{k} = {v}  (was {getattr(args, k)})")
            setattr(args, k, v)

    device = get_device(allow_cpu=args.allow_cpu)

    if args.lambdas == "grid_search" or args.force_field == "auto":
        if not args.val:
            raise ValueError("--lambdas grid_search / --force_field auto need --val "
                              "(the search ranks candidates by validation R2)")
    if args.force_field == "auto":
        best_ff, best_nb = search_best_force_field(args.train, args.val, args, device)
        args.force_field, args.no_nb = best_ff, not best_nb
        print(f"[main] locking force_field={args.force_field} non_bonded={not args.no_nb} for all repeats")
    if args.lambdas == "grid_search":
        args.lambdas = search_best_lambdas(args.train, args.val, args, device)
        print(f"[main] locking lambdas={args.lambdas} for all repeats")

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
