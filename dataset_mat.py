"""
dataset_mat.py
--------------
Wraps features_mat.py in a torch Dataset + collate_fn so a split CSV
(id, smiles, logp_exp, assay) can be fed straight into MATModel.
Failed SMILES (unparseable, or too large if max_atoms is set) are dropped
with a warning rather than crashing the run.
"""
from typing import List, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from features_mat import smiles_to_matrices, pad_matrices, MolMatrices


class PeptidePermeabilityDataset(Dataset):
    def __init__(self, csv_path: str, max_atoms: int = 200, force_field: str = "MMFF",
                 non_bonded: bool = True, use_dummy_node: bool = True, seed: int = 42,
                 precompute: bool = True):
        self.df = pd.read_csv(csv_path)
        self.max_atoms = max_atoms
        self.force_field = force_field
        self.non_bonded = non_bonded
        self.use_dummy_node = use_dummy_node
        self.seed = seed

        self._mats: List[Optional[MolMatrices]] = [None] * len(self.df)
        self._valid_idx = []
        if precompute:
            self._precompute_all()

    def _precompute_all(self):
        dropped = 0
        for i, smi in enumerate(self.df["smiles"].tolist()):
            m = smiles_to_matrices(smi, force_field=self.force_field, non_bonded=self.non_bonded,
                                    seed=self.seed, max_atoms=self.max_atoms)
            if m is None:
                dropped += 1
                continue
            self._mats[i] = m
            self._valid_idx.append(i)
        if dropped:
            print(f"[dataset] dropped {dropped}/{len(self.df)} peptides (unparseable SMILES or > max_atoms)")

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


def atom_feature_dim() -> int:
    """Convenience: feature width produced by features_mat.atom_features (used to size MATModel)."""
    from features_mat import ATOM_LIST, HYBRIDIZATIONS
    return (len(ATOM_LIST) + 1) + (6 + 1) + (5 + 1) + (len(HYBRIDIZATIONS) + 1) + (5 + 1) + 1 + 1 + 1
