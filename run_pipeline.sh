#!/usr/bin/env bash
# End-to-end CPMP-style pipeline against your CycPeptMPDB exports.
# Edit the two CSV paths below if they differ, then:
#   chmod +x run_pipeline.sh && ./run_pipeline.sh
set -euo pipefail

PAMPA_CSV="/workspaces/PepADMET-Data/Permeability/CycPeptMPDB/CycPeptMPDB_Peptide_Assay_PAMPA.csv"
CACO2_CSV="/workspaces/PepADMET-Data/Permeability/CycPeptMPDB/CycPeptMPDB_Peptide_Assay_Caco2.csv"
OUT_ROOT="./cpmp_runs"

echo "### 1. Data prep (detection-floor filter + 8:1:1 split) ###"
python data_prep.py --csv "$PAMPA_CSV" --assay PAMPA --out_dir splits
python data_prep.py --csv "$CACO2_CSV" --assay Caco2 --out_dir splits

echo "### 2. Baselines (RFR / SVR on Morgan fingerprints) ###"
python baselines.py --train splits/PAMPA_train.csv --val splits/PAMPA_val.csv \
    --test splits/PAMPA_test.csv --out_dir "$OUT_ROOT/pampa_baselines"
python baselines.py --train splits/Caco2_train.csv --val splits/Caco2_val.csv \
    --test splits/Caco2_test.csv --out_dir "$OUT_ROOT/caco2_baselines"

echo "### 3. MAT model, train from scratch (PAMPA, Caco-2) ###"
python train.py --train splits/PAMPA_train.csv --val splits/PAMPA_val.csv \
    --test splits/PAMPA_test.csv --out_dir "$OUT_ROOT/pampa_mat" --repeats 3 --epochs 100
python train.py --train splits/Caco2_train.csv --val splits/Caco2_val.csv \
    --test splits/Caco2_test.csv --out_dir "$OUT_ROOT/caco2_mat" --repeats 3 --epochs 100

echo "### 4. Y-randomization control (PAMPA) ###"
python y_randomization.py --train splits/PAMPA_train.csv --test splits/PAMPA_test.csv \
    --out_dir "$OUT_ROOT/pampa_yrand" --n_runs 20 --model rfr

echo "### 5. Ablation study (PAMPA) - distance / dummy node / adjacency ###"
python ablation.py --train splits/PAMPA_train.csv --val splits/PAMPA_val.csv \
    --test splits/PAMPA_test.csv --out_dir "$OUT_ROOT/pampa_ablation" --epochs 60 --repeats 1

echo "### Done. See $OUT_ROOT for checkpoints and JSON result summaries. ###"

# --- Optional: RRCK / MDCK pretrain-then-finetune (Step 5.2) ---
# Once you have Caco2 CSVs for RRCK/MDCK too:
#   python data_prep.py --csv <RRCK_csv> --assay RRCK --out_dir splits
#   python train.py --train splits/RRCK_train.csv --test splits/RRCK_test.csv \
#       --out_dir "$OUT_ROOT/rrck_finetuned" \
#       --init_checkpoint "$OUT_ROOT/caco2_mat/best_seed0.pt" --repeats 3 --epochs 60
