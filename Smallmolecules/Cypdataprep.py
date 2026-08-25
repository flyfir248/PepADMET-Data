"""
cyp_data_prep.py
-----------------
Prepares CYP450 inhibition challenge CSVs (CYP1A2, CYP2C9, CYP2D6, CYP3A4)
for the same MAT/baseline pipeline built for peptide permeability
(baselines.py, dataset_mat.py, train.py, ablation.py, y_randomization.py,
export_models.py, model_registry.py, app.py). Target is pIC50 (regression),
higher = stronger inhibitor.

IMPORTANT NAMING NOTE: the downstream pipeline's internal CSV schema uses a
generic "logp_exp" column name for whatever regression target is being
modeled -- a naming holdover from the pipeline's original permeability use
case. This script deliberately keeps that same column name in its output
(rather than renaming it, e.g. to "pic50") specifically so every existing
downstream script keeps working completely unchanged, with zero risk to
the already-validated peptide-permeability pipeline. For these CYP splits,
the "logp_exp" column holds pIC50 values, not LogP -- it's just the
pipeline's generic target-column name at this point.

Differences from the peptide pipeline's data_prep.py:
  - Does NOT apply the CycPeptMPDB "-10.0 detection floor" filter -- that's
    a permeability-assay-specific artifact and meaningless for pIC50.
  - Handles the CYP challenge files' inconsistent headers: CYP1A2's export
    uses "Compound ID" / "pIC50"; the other three use "Molecule_Name" /
    "<ASSAY>_pIC50_direct_inhibition".
  - Handles quoted SMILES fields containing embedded CXSMILES enhanced-
    stereochemistry annotations (e.g. "...N1O  |&1:15,19|") -- these parse
    fine with pandas' standard CSV quoting and with RDKit's MolFromSmiles
    directly (modern RDKit parses the trailing "|...|" block natively).
  - No detection-floor filter also means every valid-SMILES/valid-pIC50 row
    is kept; the only rows dropped are unparseable SMILES or missing pIC50.
  - These files are the CHALLENGE TRAINING data (filenames say "TRAIN"), so
    this script carves its own train/val/test split out of them (default
    8:1:1) purely so the existing pipeline has something to validate/test
    against locally. If the challenge organizers provide a separate held-out
    test set to submit predictions against, that's independent of this
    local split and should be scored with predict-style inference, not by
    retraining -- ping me when you have that file and I'll wire it in.

Usage:
  python cyp_data_prep.py --csv cyp-sm/cyp-challenge-TRAIN_CYP3A4_inhibition.csv --assay CYP3A4 --out_dir splits_cyp
  # repeat per target: CYP1A2, CYP2C9, CYP2D6, CYP3A4 (see run_cyp_pipeline.sh
  # for a script that loops over all four automatically)
"""
import argparse
import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

ID_COL_CANDIDATES = ["Molecule_Name", "Compound ID", "Compound_ID", "compound_id", "id", "ID"]
SMILES_COL_CANDIDATES = ["SMILES", "smiles"]


def _first_present(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def _target_col_for_assay(columns, assay: str) -> str:
    """CYP1A2's export just has a plain "pIC50" column; the other three
    have "<ASSAY>_pIC50_direct_inhibition". Try the assay-specific name
    first, then the plain name, then fall back to any column whose name
    contains "pic50" (case-insensitive) so a slightly different export
    format doesn't silently fail to find the target."""
    explicit = f"{assay}_pIC50_direct_inhibition"
    if explicit in columns:
        return explicit
    if "pIC50" in columns:
        return "pIC50"
    for c in columns:
        if "pic50" in c.lower():
            return c
    raise ValueError(f"Could not find a pIC50 column among {list(columns)} for assay {assay}")


def _canonical_smiles(smi: str):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def load_cyp_csv(path: str, assay: str) -> pd.DataFrame:
    df = pd.read_csv(path)  # standard comma-separated, quote-aware -- handles
    # the CXSMILES rows whose SMILES field is quoted because it contains a
    # literal comma inside the "|&1:7,11|" enhanced-stereo annotation.
    id_col = _first_present(df.columns, ID_COL_CANDIDATES)
    smi_col = _first_present(df.columns, SMILES_COL_CANDIDATES)
    tgt_col = _target_col_for_assay(df.columns, assay)

    if smi_col is None:
        raise ValueError(f"Could not find a SMILES column in {path}. Columns: {list(df.columns)}")

    out = pd.DataFrame({
        "id": df[id_col] if id_col else np.arange(len(df)),
        "smiles": df[smi_col],
        "logp_exp": pd.to_numeric(df[tgt_col], errors="coerce"),  # holds pIC50 -- see module docstring
    })
    out["assay"] = assay
    before = len(out)
    out = out.dropna(subset=["smiles", "logp_exp"]).reset_index(drop=True)
    if len(out) < before:
        print(f"[load] dropped {before - len(out)}/{before} rows with missing SMILES or pIC50")
    return out


def dedup_smiles(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate compounds by canonical SMILES, keeping the first
    occurrence, BEFORE splitting -- so the same compound can never land in
    both train and test."""
    before = len(df)
    canon = df["smiles"].apply(_canonical_smiles)
    n_unparseable = canon.isna().sum()
    if n_unparseable:
        print(f"[dedup] {n_unparseable} rows had unparseable SMILES (kept for now; "
              f"downstream featurization will drop them)")
    dedup_key = canon.fillna(df["smiles"] + "__unparseable_" + df.index.astype(str))
    kept = df.loc[~dedup_key.duplicated(keep="first")].reset_index(drop=True)
    print(f"[dedup] dropped {before - len(kept)} / {before} duplicate-SMILES rows")
    return kept


def split_dataset(df: pd.DataFrame, assay: str, seed: int = 42,
                   train_frac: float = 0.8, val_frac: float = 0.1):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(df))
    df = df.iloc[idx].reset_index(drop=True)
    n = len(df)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    splits = {
        "train": df.iloc[:n_train].reset_index(drop=True),
        "val": df.iloc[n_train:n_train + n_val].reset_index(drop=True),
        "test": df.iloc[n_train + n_val:].reset_index(drop=True),
    }
    for name, sub in splits.items():
        print(f"[split] {assay} {name}: {len(sub)} compounds")
    return splits


def prepare(csv_path: str, assay: str, out_dir: str, seed: int = 42,
            train_frac: float = 0.8, val_frac: float = 0.1, skip_dedup: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    df = load_cyp_csv(csv_path, assay)
    print(f"[load] {assay}: {len(df)} compounds with valid SMILES + pIC50")
    if not skip_dedup:
        df = dedup_smiles(df)
    splits = split_dataset(df, assay, seed=seed, train_frac=train_frac, val_frac=val_frac)
    for name, sub in splits.items():
        out_path = os.path.join(out_dir, f"{assay}_{name}.csv")
        sub.to_csv(out_path, index=False)
        print(f"[write] {out_path}")
    return splits


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Prepare a CYP450 challenge inhibition CSV for the MAT pipeline")
    ap.add_argument("--csv", required=True, help="Path to cyp-challenge-TRAIN_<ASSAY>_inhibition.csv")
    ap.add_argument("--assay", required=True, choices=["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"])
    ap.add_argument("--out_dir", default="./splits_cyp")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--skip_dedup", action="store_true", help="disable SMILES deduplication (not recommended)")
    args = ap.parse_args()
    prepare(args.csv, args.assay, args.out_dir, args.seed, args.train_frac, args.val_frac, args.skip_dedup)
