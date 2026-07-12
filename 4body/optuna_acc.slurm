#!/bin/bash
#SBATCH -J acc_optuna
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH -o /rds/user/zx342/hpc-work/acc_optuna_%A_%a.log

module load miniconda/3
source activate tfagpu
module load cuda/11.2 cudnn/8.1_cuda-11.2

hostname
python -c "import optuna, tensorflow as tf; print('optuna', optuna.__version__, 'GPUs', tf.config.list_physical_devices('GPU'))"

cd /rds/user/zx342/hpc-work/4body
export TF_CPP_MIN_LOG_LEVEL=3
# 4 parallel workers cooperate on ONE Optuna study via a shared SQLite DB on RDS.
# Each worker runs ACC_OPTUNA_TRIALS trials; TPE sampler + median pruner shared.
export ACC_OPTUNA_DB=/rds/user/zx342/hpc-work/4body/optuna_acc.db
export ACC_OPTUNA_STUDY=acc6d_v1
export ACC_OPTUNA_TRIALS=15        # 4 workers x 15 = ~60 trials total
export ACC_SEARCH_EPOCHS=15000     # proxy epochs per trial (full run later for the winner)
export ACC_NTRAIN=2000000          # subsample for search speed
export ACC_NORM_SIZE=1000000

# one-time env setup (run once on a login node, NOT here):
#   module load miniconda/3; source activate tfagpu; pip install optuna
# scp first: transform_cut.py, train_acc_6d.py, optuna_acc.py, mlp.py, runs12_cut.npz
python optuna_acc.py
