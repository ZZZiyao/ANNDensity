#!/bin/bash
#SBATCH -J acc_swish_scan
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --array=0-5
#SBATCH -o /rds/user/zx342/hpc-work/acc_swish_scan_%A_%a.log

module load miniconda/3
source activate tfagpu
module load cuda/11.2 cudnn/8.1_cuda-11.2

LAMBDAS=(0.003 0.01 0.03 0.1 0.3 1.0)
L=${LAMBDAS[$SLURM_ARRAY_TASK_ID]}
echo "=== array task $SLURM_ARRAY_TASK_ID : swish, lambda2 = $L ==="
hostname

cd /rds/user/zx342/hpc-work/4body
export TF_CPP_MIN_LOG_LEVEL=3
# lambda2 scan on the SWISH architecture (now that swish is the chosen activation and
# converges -> this scan is conclusive, unlike the earlier sigmoid/undertrained one).
export ACC_FLATTEN=1
export ACC_SEED=1
export ACC_ACT=swish
export ACC_OUT=softplus
export ACC_LAMBDA2=$L
export ACC_LR=2e-3
export ACC_LR_FINAL=2e-4
export ACC_NTRAIN=5000000
export ACC_NORM_SIZE=2000000
export ACC_MAX_EPOCHS=40000   # swish converged ~31600; 40k gives margin
export ACC_CHECK_EVERY=100
export ACC_PATIENCE=600       # run full, best-tracking keeps the min (fair across lambda)
export ACC_OUTFILE=/rds/user/zx342/hpc-work/4body/acc_5d_swish_lam${L}

# scp first: mlp.py, train_acc_swish.py, runs12_flat.npz, phasespace_transform.py
python train_acc_swish.py
