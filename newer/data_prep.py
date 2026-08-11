"""
data_prep.py
------------
Loads a CycPeptMPDB per-assay export (e.g. CycPeptMPDB_Peptide_Assay_PAMPA.csv,
CycPeptMPDB_Peptide_Assay_Caco2.csv), applies the CPMP paper's detection-floor
filter, and produces train/val/test (or train/test) splits.

Mirrors "Step 1 - Dataset Preparation" from CPMP (Jiang, Chen & Du, 2025):
  - Drop any peptide with LogPexp < -10.0 (assay detection floor).
  - PAMPA / Caco-2 (large sets): 8:1:1 train/val/test split.
  - RRCK / MDCK (small sets): 7:3 train/test split, no validation carve-out.

PATCH NOTE (dedup): the paper reports PAMPA n=6,701 and Caco-2 n=1,310 after
the detection-floor filter; a raw CycPeptMPDB export can come in noticeably
larger than that (e.g. 7,298 for PAMPA) if it contains duplicate SMILES rows
(the same peptide tested/logged more than once). This matters for two
reasons: (1) it's the most likely explanation for your row counts not
matching the paper's, and (2) if duplicates aren't removed BEFORE splitting,
the same peptide can land in both train and test, which is a genuine leakage
path that would make test R2 look better than true generalization -- not
your current problem (you're underperforming, not overperforming), but
worth closing regardless. Dedup is on canonical SMILES (RDKit-canonicalized,
so different-but-equivalent SMILES strings for the same molecule are also
caught), keeping the first occurrence.

Usage:
    python data_prep.py --csv /workspaces/PepADMET-Data/Permeability/CycPeptMPDB/CycPeptMPDB_Peptide_Assay_PAMPA.csv --assay PAMPA
"""
import argparse
import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

LOGP_FLOOR = -10.0

# Assay column CycPeptMPDB uses in its per-assay exports, in order of preference.
TARGET_COL_CANDIDATES = ["Permeability", "PAMPA", "Caco2", "MDCK", "RRCK", "logp_exp", "LogPexp"]
SMILES_COL_CANDIDATES = ["SMILES", "smiles"]
ID_COL_CANDIDATES = ["ID", "id"]

LARGE_ASSAYS = {"PAMPA", "Caco2", "CACO2", "CACO-2"}


def _first_present(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def _canonical_smiles(smi: str):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def load_assay_csv(path: str, assay: str, target_col: str = None, smiles_col: str = None) -> pd.DataFrame:
    """Load a single-assay CycPeptMPDB export and return the columns we need."""
    df = pd.read_csv(path, sep=None, engine="python")  # sniff tab vs comma

    smi_col = smiles_col or _first_present(df.columns, SMILES_COL_CANDIDATES)
    id_col = _first_present(df.columns, ID_COL_CANDIDATES)
    tgt_col = target_col or _first_present(df.columns, TARGET_COL_CANDIDATES)

    if smi_col is None:
        raise ValueError(f"Could not find a SMILES column in {path}. Columns: {list(df.columns)}")
    if tgt_col is None:
        raise ValueError(f"Could not find a permeability/target column in {path}. Columns: {list(df.columns)}")

    out = pd.DataFrame({
        "id": df[id_col] if id_col else np.arange(len(df)),
        "smiles": df[smi_col],
        "logp_exp": pd.to_numeric(df[tgt_col], errors="coerce"),
    })
    out["assay"] = assay
    out = out.dropna(subset=["smiles", "logp_exp"]).reset_index(drop=True)
    return out


def dedup_smiles(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate peptides by canonical SMILES, keeping the first
    occurrence. Run this BEFORE splitting so the same peptide can never
    land in both train and test."""
    before = len(df)
    canon = df["smiles"].apply(_canonical_smiles)
    n_unparseable = canon.isna().sum()
    if n_unparseable:
        print(f"[dedup] {n_unparseable} rows had unparseable SMILES (kept for now; "
              f"downstream featurization will drop them)")
    # Rows with unparseable SMILES keep a unique per-row key so they aren't
    # accidentally collapsed together by a shared NaN canonical value.
    dedup_key = canon.fillna(df["smiles"] + "__unparseable_" + df.index.astype(str))
    kept = df.loc[~dedup_key.duplicated(keep="first")].reset_index(drop=True)
    print(f"[dedup] dropped {before - len(kept)} / {before} duplicate-SMILES rows")
    return kept


def apply_detection_floor(df: pd.DataFrame, floor: float = LOGP_FLOOR) -> pd.DataFrame:
    """Drop rows at/under the assay detection floor (Step 1 exclusion filter)."""
    before = len(df)
    kept = df[df["logp_exp"] >= floor].reset_index(drop=True)
    print(f"[detection floor] dropped {before - len(kept)} / {before} rows with LogPexp < {floor}")
    return kept


def split_dataset(df: pd.DataFrame, assay: str, seed: int = 42):
    """
    8:1:1 train/val/test for large assays (PAMPA, Caco-2);
    7:3 train/test (no val) for small assays (RRCK, MDCK).
    Returns a dict of DataFrames.
    """
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(df))
    df = df.iloc[idx].reset_index(drop=True)
    n = len(df)

    if assay in LARGE_ASSAYS:
        n_train = int(round(0.8 * n))
        n_val = int(round(0.1 * n))
        splits = {
            "train": df.iloc[:n_train].reset_index(drop=True),
            "val": df.iloc[n_train:n_train + n_val].reset_index(drop=True),
            "test": df.iloc[n_train + n_val:].reset_index(drop=True),
        }
    else:
        n_train = int(round(0.7 * n))
        splits = {
            "train": df.iloc[:n_train].reset_index(drop=True),
            "test": df.iloc[n_train:].reset_index(drop=True),
        }

    for name, sub in splits.items():
        print(f"[split] {assay} {name}: {len(sub)} peptides")
    return splits


def prepare(csv_path: str, assay: str, out_dir: str, seed: int = 42, skip_dedup: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    df = load_assay_csv(csv_path, assay)
    if not skip_dedup:
        df = dedup_smiles(df)
    df = apply_detection_floor(df)
    print(f"[prepare] {assay}: {len(df)} peptides after dedup + detection floor "
          f"(paper reports 6,701 for PAMPA / 1,310 for Caco-2 -- if you're still "
          f"noticeably above that, check for near-duplicate variants the canonical "
          f"SMILES comparison didn't catch, e.g. differing stereochemistry annotations)")
    splits = split_dataset(df, assay, seed=seed)
    for name, sub in splits.items():
        out_path = os.path.join(out_dir, f"{assay}_{name}.csv")
        sub.to_csv(out_path, index=False)
        print(f"[write] {out_path}")
    return splits


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Prepare CycPeptMPDB assay CSV for CPMP-style modeling")
    ap.add_argument("--csv", required=True, help="Path to CycPeptMPDB_Peptide_Assay_<Assay>.csv")
    ap.add_argument("--assay", required=True, choices=["PAMPA", "Caco2", "RRCK", "MDCK"])
    ap.add_argument("--out_dir", default="./splits")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_dedup", action="store_true", help="disable SMILES deduplication (not recommended)")
    args = ap.parse_args()
    prepare(args.csv, args.assay, args.out_dir, args.seed, args.skip_dedup)
