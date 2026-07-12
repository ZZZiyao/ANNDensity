#!/bin/bash
#SBATCH -J acc_swish
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH -o /rds/user/zx342/hpc-work/acc_swish_%j.log

module load miniconda/3
source activate tfagpu
module load cuda/11.2 cudnn/8.1_cuda-11.2

hostname
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"

cd /rds/user/zx342/hpc-work/4body
export TF_CPP_MIN_LOG_LEVEL=3
# SWISH ablation -- IDENTICAL config to the sigmoid conv003 run (acc_full.slurm),
# only the activation differs (hidden=swish, output=softplus). Apples-to-apples.
export ACC_FLATTEN=1
export ACC_SEED=1
export ACC_ACT=swish
export ACC_OUT=softplus
export ACC_LAMBDA2=0.03
export ACC_LR=2e-3
export ACC_LR_FINAL=2e-4
export ACC_NTRAIN=5000000
export ACC_NORM_SIZE=2000000
export ACC_MAX_EPOCHS=50000
export ACC_CHECK_EVERY=100
export ACC_PATIENCE=600       # never early-stop; best-tracking keeps the min
export ACC_OUTFILE=/rds/user/zx342/hpc-work/4body/acc_5d_swish003

# scp first: mlp.py, train_acc_swish.py, runs12_flat.npz, phasespace_transform.py
python train_acc_swish.py
