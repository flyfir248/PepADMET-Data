"""
model_registry.py
------------------
Loads the pickled RFR / SVR / MAT models produced by export_models.py and
exposes a single predict_all(smiles), keeping all model-specific
featurization logic in one place so app.py stays thin. Loaded once at
Flask startup, not per-request.
"""
import json
import os
import pickle

import torch

from baselines import smiles_to_morgan
from features_mat import smiles_to_matrices_cached as smiles_to_matrices, pad_matrices
from model_mat import MATModel


class ModelRegistry:
    def __init__(self, models_dir: str, device: str = None):
        self.models_dir = models_dir
        # Inference-only, so unlike training this quietly falls back to CPU
        # if no GPU is present -- a single-molecule forward pass is cheap;
        # the only genuinely slow part (conformer generation) is CPU-bound
        # in RDKit regardless of what device we pick here.
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"[registry] inference device: {self.device}")

        with open(os.path.join(models_dir, "models_manifest.json")) as f:
            self.manifest = json.load(f)

        self.rfr = self._load_pickle("rfr")
        self.svr = self._load_pickle("svr")
        self.mat, self.mat_config = self._load_mat()
        print(f"[registry] loaded rfr, svr, mat from {models_dir}")

    def _load_pickle(self, key):
        path = os.path.join(self.models_dir, self.manifest[key]["pickle_path"])
        with open(path, "rb") as f:
            return pickle.load(f)

    def _load_mat(self):
        path = os.path.join(self.models_dir, self.manifest["mat"]["pickle_path"])
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        config = bundle["config"]
        model = MATModel(
            atom_feat_dim=config["atom_feat_dim"], d_model=config["d_model"],
            n_heads=config["n_heads"], n_layers=config["n_layers"], d_ff=config["d_ff"],
            lambdas=tuple(config["lambdas"]), use_distance=config["use_distance"],
            use_adjacency=config["use_adjacency"], use_dummy_node=config["use_dummy_node"],
            n_dense=config.get("n_dense", 1), leaky_slope=config.get("leaky_slope", None),
            distance_kernel_kind=config.get("distance_kernel_kind", "softmax_neg"),
        ).to(self.device)
        missing, unexpected = model.load_state_dict(bundle["state_dict"], strict=False)
        if missing or unexpected:
            print(f"[registry] WARNING loading MAT checkpoint: missing={missing} unexpected={unexpected} "
                  f"(expected if this checkpoint predates the y_mean/y_std normalization patch -- "
                  f"predictions will use the default y_mean=0/y_std=1, i.e. no normalization)")
        model.eval()
        return model, config

    def _predict_rfr_svr(self, model, manifest_key: str, smiles: str) -> dict:
        feat_cfg = self.manifest[manifest_key]["featurization"]
        fp = smiles_to_morgan(smiles, n_bits=feat_cfg["n_bits"], radius=feat_cfg["radius"])
        if fp is None:
            return {"error": "RDKit could not parse this SMILES string"}
        pred = float(model.predict(fp.reshape(1, -1))[0])
        return {"prediction": pred, "test_metrics": self.manifest[manifest_key]["test_metrics"]}

    def _predict_mat(self, smiles: str) -> dict:
        cfg = self.mat_config
        mats = smiles_to_matrices(smiles, force_field=cfg["force_field"], non_bonded=cfg["non_bonded"],
                                   max_atoms=cfg["max_atoms"])
        if mats is None:
            return {"error": "RDKit could not parse this SMILES string, or it exceeds max_atoms"}

        atom, adj, dist, mask = pad_matrices(mats, cfg["max_atoms"], add_dummy_node=cfg["use_dummy_node"])
        with torch.no_grad():
            def to_batch(arr):
                return torch.from_numpy(arr).unsqueeze(0).to(self.device)
            pred = self.mat.predict(to_batch(atom), to_batch(adj), to_batch(dist), to_batch(mask)).item()

        return {
            "prediction": float(pred),
            "test_metrics": self.manifest["mat"]["test_metrics"],
            "conformer_ok": mats.conformer_ok,
        }

    def predict_all(self, smiles: str) -> dict:
        return {
            "smiles": smiles,
            "rfr": self._predict_rfr_svr(self.rfr, "rfr", smiles),
            "svr": self._predict_rfr_svr(self.svr, "svr", smiles),
            "mat": self._predict_mat(smiles),
        }

    def predict_batch(self, smiles_list: list) -> list:
        """Run predict_all for each SMILES, isolating failures per-item so one
        malformed/oversized peptide doesn't abort predictions for the rest of
        the batch. MAT's conformer generation runs sequentially here, so a
        batch of N compounds takes roughly N times as long as a single one."""
        results = []
        for smiles in smiles_list:
            try:
                results.append(self.predict_all(smiles))
            except Exception as e:
                results.append({"smiles": smiles, "error": f"prediction failed: {e}"})
        return results
