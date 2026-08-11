# 1. Re-prep data with dedup — check the printed counts against paper's 6,701 / 1,310
python data_prep.py --csv .../CycPeptMPDB_Peptide_Assay_PAMPA.csv --assay PAMPA --out_dir splits
python data_prep.py --csv .../CycPeptMPDB_Peptide_Assay_Caco2.csv --assay Caco2 --out_dir splits

# 2. Baselines with the paper's exact grid
python baselines.py --train splits/PAMPA_train.csv --val splits/PAMPA_val.csv \
    --test splits/PAMPA_test.csv --out_dir cpmp_runs/pampa_baselines --grid paper
python baselines.py --train splits/Caco2_train.csv --val splits/Caco2_val.csv \
    --test splits/Caco2_test.csv --out_dir cpmp_runs/caco2_baselines --grid paper

# 3. MAT with real lambda + force-field grid search, full training budget
python train.py --train splits/PAMPA_train.csv --val splits/PAMPA_val.csv \
    --test splits/PAMPA_test.csv --out_dir cpmp_runs/pampa_mat \
    --lambdas grid_search --force_field auto --repeats 3 --epochs 100

python train.py --train splits/Caco2_train.csv --val splits/Caco2_val.csv \
    --test splits/Caco2_test.csv --out_dir cpmp_runs/caco2_mat \
    --lambdas grid_search --force_field auto --repeats 3 --epochs 100

# 4. Ablation — now uses the SAME budget as step 3, so the comparison is fair
python ablation.py --train splits/PAMPA_train.csv --val splits/PAMPA_val.csv \
    --test splits/PAMPA_test.csv --out_dir cpmp_runs/pampa_ablation
