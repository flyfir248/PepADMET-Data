#!/usr/bin/env bash
# End-to-end pipeline for the four CYP450 inhibition challenge targets
# (CYP1A2, CYP2C9, CYP2D6, CYP3A4). Target is pIC50 regression.
#
# Edit CYP_DATA_DIR below if your CSVs live elsewhere, then:
#   chmod +x run_cyp_pipeline.sh && ./run_cyp_pipeline.sh
set -euo pipefail

CYP_DATA_DIR="${CYP_DATA_DIR:-./cyp-sm}"
SPLITS_DIR="splits_cyp"
OUT_ROOT="./cyp_runs"
ASSAYS=("CYP1A2" "CYP2C9" "CYP2D6" "CYP3A4")

for ASSAY in "${ASSAYS[@]}"; do
  echo "=============================================="
  echo "### ${ASSAY}: 1. Data prep ###"
  echo "=============================================="
  python cyp_data_prep.py \
      --csv "${CYP_DATA_DIR}/cyp-challenge-TRAIN_${ASSAY}_inhibition.csv" \
      --assay "${ASSAY}" --out_dir "${SPLITS_DIR}"

  echo "### ${ASSAY}: 2. Baselines (RFR / SVR on Morgan fingerprints) ###"
  python baselines.py \
      --train "${SPLITS_DIR}/${ASSAY}_train.csv" \
      --val "${SPLITS_DIR}/${ASSAY}_val.csv" \
      --test "${SPLITS_DIR}/${ASSAY}_test.csv" \
      --out_dir "${OUT_ROOT}/${ASSAY}_baselines" --grid paper

  echo "### ${ASSAY}: 3. MAT model, train from scratch ###"
  python train.py \
      --train "${SPLITS_DIR}/${ASSAY}_train.csv" \
      --val "${SPLITS_DIR}/${ASSAY}_val.csv" \
      --test "${SPLITS_DIR}/${ASSAY}_test.csv" \
      --out_dir "${OUT_ROOT}/${ASSAY}_mat" \
      --lambdas balanced --force_field MMFF --num_conformers 5 \
      --repeats 3 --epochs 100

  echo "### ${ASSAY}: done -- results in ${OUT_ROOT}/${ASSAY}_mat/summary.json ###"
done

echo "=============================================="
echo "### ALL FOUR CYP TARGETS COMPLETE ###"
echo "See ${OUT_ROOT}/<ASSAY>_baselines and ${OUT_ROOT}/<ASSAY>_mat for results."
echo "=============================================="
