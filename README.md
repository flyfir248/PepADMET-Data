# CPMP-style permeability model for your CycPeptMPDB PAMPA / Caco-2 exports

This reproduces the CPMP methodology (Jiang, Chen & Du, 2025 — the paper in
`CPMP_Methodology_Deep_Dive.docx`) against your own data at:

- `/workspaces/PepADMET-Data/Permeability/CycPeptMPDB/CycPeptMPDB_Peptide_Assay_PAMPA.csv`
- `/workspaces/PepADMET-Data/Permeability/CycPeptMPDB/CycPeptMPDB_Peptide_Assay_Caco2.csv`

None of this ran against your real files — I don't have access to
`/workspaces/...` from here. Everything below was built and smoke-tested
against a handful of rows from the sample table you pasted in chat, so the
plumbing (featurization → model → training → baselines → Y-randomization →
ablation) is verified to run without errors; you'll run it for real in your
own Codespace.

## Files

| File | Methodology step | What it does |
|---|---|---|
| `data_prep.py` | Step 1 | Loads a per-assay CSV, drops LogPexp < −10.0, splits 8:1:1 (PAMPA/Caco-2). |
| `features_mat.py` | Steps 2, 9 | SMILES → atom feature matrix, adjacency matrix, 3D distance matrix (ETKDG + UFF/MMFF). |
| `model_mat.py` | Step 3 | The Molecular Attention Transformer: `A = λₐ·attn + λ_d·g(D) + λ_g·A_adj`, dummy node, ablation flags. |
| `dataset_mat.py` | — | torch `Dataset`/collate glue between the two above. |
| `train.py` | Steps 4, 6 | Training loop (train-from-scratch or `--init_checkpoint` finetune), MSE/MAE/R² over repeated runs. |
| `baselines.py` | Step 5 | RFR / SVR on 1024-bit Morgan fingerprints, grid-searched. |
| `y_randomization.py` | Step 7 | Label-permutation control, 20 runs by default. |
| `ablation.py` | Step 8 | Retrains with distance / dummy node / adjacency each removed in turn. |
| `run_pipeline.sh` | — | Runs steps 1–5 end to end against your real CSV paths. |

## Quickstart in your Codespace

```bash
cd /workspaces/PermeabilityML   # or wherever you want this to live
pip install -r requirements.txt
chmod +x run_pipeline.sh
./run_pipeline.sh
```

That runs, in order: data prep → baselines → MAT training (3 repeats each
for PAMPA and Caco-2) → Y-randomization → ablation. Adjust `--epochs` /
`--repeats` down first if you just want to confirm everything runs before
committing to full training time.

## Column detection

`data_prep.py` looks for a SMILES column (`SMILES`) and a target column,
trying `Permeability`, then `PAMPA`, `Caco2`, `MDCK`, `RRCK` in that order —
this matches the column names visible in the sample rows you pasted. If your
actual CSVs use different headers, pass `--target_col` / edit
`TARGET_COL_CANDIDATES` at the top of `data_prep.py`.

## RRCK / MDCK (pretrain-then-finetune)

Once you export RRCK and MDCK from CycPeptMPDB the same way, Step 5.2 is:

```bash
python data_prep.py --csv <RRCK_csv> --assay RRCK --out_dir splits
python train.py --train splits/RRCK_train.csv --test splits/RRCK_test.csv \
    --out_dir cpmp_runs/rrck_finetuned \
    --init_checkpoint cpmp_runs/caco2_mat/best_seed0.pt --repeats 3 --epochs 60
```

`--val` is omitted for RRCK/MDCK since `data_prep.py` gives them a 7:3
train/test split only (no validation carve-out), matching the paper.

## Honest caveats vs. the paper

- **λ weights**: the paper grid-searches λₐ/λ_d/λ_g; I don't have their
  found values, so `model_mat.py` ships three presets
  (`balanced` / `distance_heavy` / `attention_heavy`) — grid-search these
  yourself with `--lambdas`, or expose them as a proper hyperparameter sweep
  if that becomes your bottleneck.
- **MGNN baseline** isn't implemented — only RFR/SVR (Step 5's fingerprint
  baselines). Worth adding a plain 2D-GNN baseline later if you want the
  "MGNN vs CPMP isolates the 3D contribution" comparison the paper makes;
  your existing PermeabilityML XGBoost/LightGBM models can substitute in the
  meantime.
- **3D embedding can fail** for a small fraction of large/unusual peptides;
  `features_mat.py` falls back to RDKit's topological (bond-count) distance
  matrix for those rather than dropping them, and flags this via
  `conformer_ok=False` — worth checking how often that fires on your real
  data before trusting the results uniformly.
- Not yet wired to your `compare_cycpep_databases.py` duplicate-linking —
  if a peptide appears in both your train and test splits across sources,
  that's a leak the Y-randomization check (Step 7) is specifically meant to
  catch, but a scaffold- or duplicate-aware split would prevent it upstream.
