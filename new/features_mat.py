"""
features_mat.py
----------------
Turns a SMILES string into the three matrices MAT needs (Step 2 of the
methodology): atom feature matrix, adjacency matrix, and 3D distance matrix.
Conformer generation follows Step 9: RDKit ETKDG embedding + a fast force
field (UFF or MMFF, each with an optional "-NB" no-non-bonded variant).

All three matrices are computed over heavy atoms only (Hs added transiently
for a more realistic 3D embedding, then stripped), which is the convention
the original MAT implementation and CPMP both use.

PATCH NOTES (this version):
1. BUG FIX -- dummy-node distance. pad_matrices() previously set the dummy
   node's distance to every real atom to 0.0. Under the softmax(-D) distance
   kernel used in model_mat.py, distance 0 is the CLOSEST possible value, so
   the dummy node -- which has no real 3D position -- was getting maximal
   weight through the distance-attention channel instead of ~zero weight.
   The original MAT design uses a large sentinel distance for the dummy node
   so it's reached only through learned attention / adjacency, not the 3D
   channel. Fixed via DUMMY_DISTANCE below.
2. Added a disk-backed cache (smiles_to_matrices_cached) so the same SMILES
   isn't re-embedded from scratch on every training repeat / ablation config.
   Conformer generation is deterministic given (smiles, force_field,
   non_bonded, seed), so this is safe to reuse across runs.
"""
import os
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from joblib import Memory

RDLogger.DisableLog("rdApp.*")

# Sentinel "infinite" distance for the dummy node. Large enough that
# softmax(-DUMMY_DISTANCE) is ~0 relative to real inter-atom distances
# (which are on the order of 1-50 Angstroms for these peptides), so the
# dummy node gets negligible weight through the distance-kernel channel.
DUMMY_DISTANCE = 1e6

_CACHE_DIR = os.environ.get("MAT_CONFORMER_CACHE", ".conformer_cache")
_memory = Memory(location=_CACHE_DIR, verbose=0)

# ---- Atom featurization -----------------------------------------------
# Standard RDKit atom-featurization set used by every atom-level GNN
# (atom type one-hot, degree, formal charge, hybridization, aromaticity,
# implicit/explicit H count, ring membership).
ATOM_LIST = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "Se", "H", "*"]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]


def _one_hot(value, choices):
    vec = [0] * (len(choices) + 1)  # +1 slot for "other"
    if value in choices:
        vec[choices.index(value)] = 1
    else:
        vec[-1] = 1
    return vec


def atom_features(atom: Chem.Atom) -> np.ndarray:
    feats = []
    feats += _one_hot(atom.GetSymbol(), ATOM_LIST)
    feats += _one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
    feats += _one_hot(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])
    feats += _one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
    feats += _one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    feats.append(1.0 if atom.GetIsAromatic() else 0.0)
    feats.append(1.0 if atom.IsInRing() else 0.0)
    feats.append(float(atom.GetMass()) / 100.0)  # light scaling
    return np.array(feats, dtype=np.float32)


@dataclass
class MolMatrices:
    atom_matrix: np.ndarray       # (N, F)
    adjacency_matrix: np.ndarray  # (N, N) binary
    distance_matrix: np.ndarray   # (N, N) Euclidean, from the 3D conformer
    n_atoms: int
    conformer_ok: bool            # False if 3D embedding failed and we fell back to 2D-only


def embed_conformer(mol: Chem.Mol, force_field: str = "MMFF", non_bonded: bool = True,
                     seed: int = 42) -> Tuple[Optional[Chem.Mol], bool]:
    """
    Step 9: ETKDG embedding + a fast force field.
    force_field: 'UFF' or 'MMFF'. non_bonded: whether to include non-bonded terms
    (the "+NB" vs "-NB" variants the paper tests). Returns (mol_with_conformer, ok).
    """
    molH = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    embed_status = AllChem.EmbedMolecule(molH, params)
    if embed_status != 0:
        # retry with random coords as a fallback, then give up
        params.useRandomCoords = True
        embed_status = AllChem.EmbedMolecule(molH, params)
        if embed_status != 0:
            return None, False

    try:
        if force_field.upper() == "UFF":
            ff = AllChem.UFFGetMoleculeForceField(molH, ignoreInterfragInteractions=not non_bonded)
        else:
            mp = AllChem.MMFFGetMoleculeProperties(molH)
            ff = AllChem.MMFFGetMoleculeForceField(molH, mp, ignoreInterfragInteractions=not non_bonded)
        if ff is None:
            return None, False
        ff.Minimize(maxIts=500)
    except Exception:
        return None, False

    return molH, True


def smiles_to_matrices(smiles: str, force_field: str = "MMFF", non_bonded: bool = True,
                        seed: int = 42, max_atoms: Optional[int] = None) -> Optional[MolMatrices]:
    """
    Full Step-2 pipeline for one molecule. Returns None if the SMILES cannot
    be parsed at all; sets conformer_ok=False (distance matrix falls back to
    the topological/bond-graph distance) if 3D embedding fails, which does
    happen for a small fraction of large or unusual peptides.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    n_atoms = mol.GetNumAtoms()
    if max_atoms is not None and n_atoms > max_atoms:
        return None

    # adjacency + atom features computed on the heavy-atom graph
    adjacency = Chem.GetAdjacencyMatrix(mol).astype(np.float32)
    feats = np.stack([atom_features(a) for a in mol.GetAtoms()], axis=0)

    molH, ok = embed_conformer(mol, force_field=force_field, non_bonded=non_bonded, seed=seed)
    if ok:
        molH_noH = Chem.RemoveHs(molH)
        conf = molH_noH.GetConformer()
        coords = conf.GetPositions()[:n_atoms]
        dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1).astype(np.float32)
    else:
        # Fallback: topological (bond-count) distance instead of a failed 3D embed.
        dist = Chem.GetDistanceMatrix(mol).astype(np.float32)

    return MolMatrices(
        atom_matrix=feats,
        adjacency_matrix=adjacency,
        distance_matrix=dist,
        n_atoms=n_atoms,
        conformer_ok=ok,
    )


# Disk-backed memoized version. Conformer generation is deterministic given
# (smiles, force_field, non_bonded, seed, max_atoms), so caching is safe and
# means the same SMILES is only ever embedded once across however many
# repeats/ablation configs you run, instead of once per run.
#
# Cache location: $MAT_CONFORMER_CACHE, default ./.conformer_cache -- this
# directory can be gigabytes for large datasets; delete it to force a clean
# re-embed (e.g. if you change atom_features() or embed_conformer() logic,
# since joblib.Memory only invalidates on *argument* changes, not on the
# smiles_to_matrices source code changing in ways that don't touch its
# signature).
smiles_to_matrices_cached = _memory.cache(smiles_to_matrices)


def pad_matrices(mats: MolMatrices, max_atoms: int, add_dummy_node: bool = True,
                  dummy_distance: float = DUMMY_DISTANCE):
    """
    Pad a single molecule's matrices to a fixed max_atoms size and build a
    boolean mask. If add_dummy_node, one extra slot is reserved and bonded to
    every real atom with adjacency=1. The dummy node's distance to every real
    atom is set to `dummy_distance` (a large sentinel, NOT 0) -- it has no
    real 3D position, so under the softmax(-D) distance kernel this pushes
    its weight in the distance-attention channel to ~0, matching the
    original MAT design (the dummy is reached via learned attention and
    adjacency instead). Returns (atom_matrix, adjacency_matrix,
    distance_matrix, mask) all at size (max_atoms [+1], ...).
    """
    n = mats.n_atoms
    size = max_atoms + (1 if add_dummy_node else 0)
    f = mats.atom_matrix.shape[1]

    atom_pad = np.zeros((size, f), dtype=np.float32)
    adj_pad = np.zeros((size, size), dtype=np.float32)
    dist_pad = np.zeros((size, size), dtype=np.float32)
    mask = np.zeros((size,), dtype=np.float32)

    atom_pad[:n] = mats.atom_matrix
    adj_pad[:n, :n] = mats.adjacency_matrix
    dist_pad[:n, :n] = mats.distance_matrix
    mask[:n] = 1.0

    if add_dummy_node:
        dummy_idx = max_atoms  # last slot
        atom_pad[dummy_idx] = 0.0
        adj_pad[dummy_idx, :n] = 1.0
        adj_pad[:n, dummy_idx] = 1.0
        dist_pad[dummy_idx, :n] = dummy_distance
        dist_pad[:n, dummy_idx] = dummy_distance
        dist_pad[dummy_idx, dummy_idx] = 0.0  # self-distance stays 0
        mask[dummy_idx] = 1.0

    return atom_pad, adj_pad, dist_pad, mask


if __name__ == "__main__":
    # smoke test on a couple of SMILES pulled from the sample rows shared in chat
    test_smiles = {
        "Cyclosporine A": "C/C=C/C[C@@H](C)[C@@H](O)[C@H]1C(=O)N[C@@H](CC)C(=O)N(C)CC(=O)N(C)[C@@H](CC(C)C)C(=O)N[C@@H](C(C)C)C(=O)N(C)[C@@H](CC(C)C)C(=O)N[C@@H](C)C(=O)N[C@H](C)C(=O)N(C)[C@@H](CC(C)C)C(=O)N(C)[C@@H](CC(C)C)C(=O)N(C)[C@@H](C(C)C)C(=O)N1C",
        "small hexapeptide": "CC(C)C[C@@H]1NC(=O)[C@@H](CC(C)C)NC(=O)[C@@H](CC(C)C)NC(=O)[C@H](Cc2ccc(O)cc2)NC(=O)[C@@H]2CCCN2C(=O)[C@@H](CC(C)C)NC1=O",
    }
    for name, smi in test_smiles.items():
        m = smiles_to_matrices_cached(smi, force_field="MMFF", non_bonded=True)
        print(f"{name}: n_atoms={m.n_atoms}, conformer_ok={m.conformer_ok}, "
              f"atom_matrix={m.atom_matrix.shape}, adjacency={m.adjacency_matrix.shape}, "
              f"distance_range=({m.distance_matrix.min():.2f}, {m.distance_matrix.max():.2f})")
