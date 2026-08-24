python train.py --train splits/PAMPA_train.csv --val splits/PAMPA_val.csv \
    --test splits/PAMPA_test.csv --out_dir cpmp_runs/pampa_mat_diag \
    --paper_faithful_arch_only --repeats 1


python train.py --train splits/PAMPA_train.csv --val splits/PAMPA_val.csv --test splits/PAMPA_test.csv --out_dir cpmp_runs/pampa_mat_diag --paper_faithful_arch_only --repeats 1

python train.py --train splits/PAMPA_train.csv --val splits/PAMPA_val.csv \
    --test splits/PAMPA_test.csv --out_dir cpmp_runs/pampa_mat_diag \
    --paper_faithful_arch_only --repeats 1




# _______


python train.py --train splits/PAMPA_train.csv --val splits/PAMPA_val.csv \
    --test splits/PAMPA_test.csv --out_dir cpmp_runs/pampa_mat_tuned \
    --lambdas grid_search --force_field auto --num_conformers 5 --repeats 3


# _______________________







python train.py --train splits/PAMPA_train.csv --val splits/PAMPA_val.csv \
    --test splits/PAMPA_test.csv --out_dir cpmp_runs/pampa_final \
    --lambdas balanced --force_field MMFF --num_conformers 5 --repeats 3 --epochs 100


______________________________


python train.py \
  --train splits/PAMPA_train.csv \
  --val splits/PAMPA_val.csv \
  --test splits/PAMPA_test.csv \
  --out_dir cpmp_runs/pampa_final \
  --lambdas balanced \
  --force_field MMFF \
  --num_conformers 5 \
  --repeats 3 \
  --epochs 100



____________________

python train.py --train splits/PAMPA_train.csv --val splits/PAMPA_val.csv --test splits/PAMPA_test.csv --out_dir cpmp_runs/pampa_final --lambdas balanced --force_field MMFF --num_conformers 5 --repeats 3 --epochs 100

































