python cyp_data_prep.py --csv ../cyp-sm/cyp-challenge-TRAIN_CYP1A2_inhibition.csv --assay CYP1A2 --out_dir splits_cyp
python baselines.py --train splits_cyp/CYP1A2_train.csv --val splits_cyp/CYP1A2_val.csv --test splits_cyp/CYP1A2_test.csv --out_dir cyp_runs/CYP1A2_baselines --grid paper
python train.py --train splits_cyp/CYP1A2_train.csv --val splits_cyp/CYP1A2_val.csv --test splits_cyp/CYP1A2_test.csv --out_dir cyp_runs/CYP1A2_mat --lambdas balanced --force_field MMFF --num_conformers 5 --repeats 3 --epochs 100
