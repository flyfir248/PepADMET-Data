"""
dataset_mat.py
--------------
Wraps features_mat.py in a torch Dataset + collate_fn so a split CSV
(id, smiles, logp_exp, assay) can be fed straight into MATModel.
Failed SMILES (unparseable, or too large if max_atoms is set) are dropped
with a warning rather than crashing the run.

PATCH NOTE: _precompute_all() now reports progress via tqdm. Conformer
embedding (ETKDG + force-field minimization) is done per-molecule with no
parallelism, so for large train splits (thousands of macrocyclic peptides)
this step alone can take a long time. Previously it printed nothing until
finished, which looked identical to the pipeline hanging. Now you'll see
a live bar + it/s so you can tell "working slowly" from "actually stuck".
"""
from typing import List, Optional
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from features_mat import smiles_to_matrices_cached as smiles_to_matrices, pad_matrices, MolMatrices


class PeptidePermeabilityDataset(Dataset):
    def __init__(self, csv_path: str, max_atoms: int = 200, force_field: str = "MMFF",
                 non_bonded: bool = True, use_dummy_node: bool = True, seed: int = 42,
                 num_conformers: int = 1, precompute: bool = True):
        self.df = pd.read_csv(csv_path)
        self.max_atoms = max_atoms
        self.force_field = force_field
        self.non_bonded = non_bonded
        self.use_dummy_node = use_dummy_node
        self.seed = seed
        self.num_conformers = num_conformers
        self.csv_path = csv_path

        self._mats: List[Optional[MolMatrices]] = [None] * len(self.df)
        self._valid_idx = []
        if precompute:
            self._precompute_all()

    def _precompute_all(self):
        dropped = 0
        conformer_failures = 0
        t0 = time.time()
        smiles_list = self.df["smiles"].tolist()
        desc = f"[dataset] {self.csv_path} conformers ({self.force_field}, {self.num_conformers} confs)"
        for i, smi in enumerate(tqdm(smiles_list, desc=desc, total=len(smiles_list))):
            m = smiles_to_matrices(smi, force_field=self.force_field, non_bonded=self.non_bonded,
                                    seed=self.seed, max_atoms=self.max_atoms,
                                    num_conformers=self.num_conformers)
            if m is None:
                dropped += 1
                continue
            if not m.conformer_ok:
                conformer_failures += 1
            self._mats[i] = m
            self._valid_idx.append(i)

        elapsed = time.time() - t0
        rate = len(smiles_list) / elapsed if elapsed > 0 else float("inf")
        print(f"[dataset] {self.csv_path}: {len(self._valid_idx)}/{len(self.df)} peptides embedded "
              f"in {elapsed:.1f}s ({rate:.2f} mol/s)")
        if dropped:
            print(f"[dataset] dropped {dropped}/{len(self.df)} peptides (unparseable SMILES or > max_atoms)")
        if conformer_failures:
            print(f"[dataset] {conformer_failures}/{len(self._valid_idx)} kept peptides fell back to "
                  f"topological distance (3D embedding failed)")

    def target_stats(self):
        """Mean/std of logp_exp over the rows that actually survived
        featurization (excludes dropped/unparseable SMILES). Used by
        train.py to normalize the MAT training target -- see model_mat.py's
        PATCH NOTE for why."""
        y = self.df.loc[self._valid_idx, "logp_exp"].to_numpy()
        return float(y.mean()), float(y.std())

    def __len__(self):
        return len(self._valid_idx)

    def __getitem__(self, idx):
        real_idx = self._valid_idx[idx]
        mats = self._mats[real_idx]
        atom_pad, adj_pad, dist_pad, mask = pad_matrices(mats, self.max_atoms, self.use_dummy_node)
        y = float(self.df.iloc[real_idx]["logp_exp"])
        return {
            "atom": torch.from_numpy(atom_pad),
            "adj": torch.from_numpy(adj_pad),
            "dist": torch.from_numpy(dist_pad),
            "mask": torch.from_numpy(mask),
            "y": torch.tensor(y, dtype=torch.float32),
        }


def collate_fn(batch):
    return {
        "atom": torch.stack([b["atom"] for b in batch]),
        "adj": torch.stack([b["adj"] for b in batch]),
        "dist": torch.stack([b["dist"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
    }


def atom_feature_dim(add_dummy_node: bool = True) -> int:
    """Convenience: feature width produced by features_mat.atom_features,
    plus the dummy-node indicator column pad_matrices() adds when
    add_dummy_node=True (used to size MATModel's input_proj layer)."""
    from features_mat import ATOM_FEATURE_DIM
    return ATOM_FEATURE_DIM + (1 if add_dummy_node else 0)
