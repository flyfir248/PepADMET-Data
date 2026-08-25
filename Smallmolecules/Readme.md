Based on your directory structure from the screenshots:

- **Your working pipeline** (`train.py`, `baselines.py`, `features_mat.py`, etc.) lives in `~/0_APSL/AJ/Anoop/MolecularAttentionTransformer/peptideADMET/`
- **Your CYP CSVs** live in a sibling folder: `~/0_APSL/AJ/Anoop/MolecularAttentionTransformer/cyp-sm/`

## Where to put the two new files

Drop both `cyp_data_prep.py` and `run_cyp_pipeline.sh` **into `peptideADMET/`**, alongside everything else — they import from `features_mat.py`, `dataset_mat.py`, `baselines.py`, `train.py`, so they need to sit in the same directory those live in.

```bash
cd ~/0_APSL/AJ/Anoop/MolecularAttentionTransformer/peptideADMET
# copy cyp_data_prep.py and run_cyp_pipeline.sh here
chmod +x run_cyp_pipeline.sh
```

## Where to run it, and pointing it at your CSVs

Since `cyp-sm` is a **sibling** folder, not inside `peptideADMET`, tell the script where to find it via the `CYP_DATA_DIR` environment variable (the script defaults to `./cyp-sm`, which would be wrong here):

```bash
cd ~/0_APSL/AJ/Anoop/MolecularAttentionTransformer/peptideADMET
CYP_DATA_DIR=../cyp-sm ./run_cyp_pipeline.sh
```

That's it — one command. It'll create `splits_cyp/` and `cyp_runs/` inside `peptideADMET/`, looping data prep → baselines → MAT training across all four targets (CYP1A2, CYP2C9, CYP2D6, CYP3A4) in sequence.

**If you'd rather test one target first** before committing to all four back-to-back (reasonable, given it's new code on a new dataset):

```bash
python cyp_data_prep.py --csv ../cyp-sm/cyp-challenge-TRAIN_CYP1A2_inhibition.csv --assay CYP1A2 --out_dir splits_cyp
python baselines.py --train splits_cyp/CYP1A2_train.csv --val splits_cyp/CYP1A2_val.csv --test splits_cyp/CYP1A2_test.csv --out_dir cyp_runs/CYP1A2_baselines --grid paper
python train.py --train splits_cyp/CYP1A2_train.csv --val splits_cyp/CYP1A2_val.csv --test splits_cyp/CYP1A2_test.csv --out_dir cyp_runs/CYP1A2_mat --lambdas balanced --force_field MMFF --num_conformers 5 --repeats 3 --epochs 100
```

If that completes cleanly and the numbers look sane, run the full `run_cyp_pipeline.sh` for all four.
