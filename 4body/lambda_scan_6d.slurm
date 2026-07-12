#!/bin/bash
#SBATCH -J acc6d_lam
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --array=0-5
#SBATCH -o /rds/user/zx342/hpc-work/acc6d_lam_%A_%a.log

module load miniconda/3
source activate tfagpu
module load cuda/11.2 cudnn/8.1_cuda-11.2

LAMBDAS=(0.001 0.003 0.01 0.03 0.1 0.3)
L=${LAMBDAS[$SLURM_ARRAY_TASK_ID]}
echo "=== array task $SLURM_ARRAY_TASK_ID : lambda2 = $L (cut+logit 6d) ==="
hostname

cd /rds/user/zx342/hpc-work/4body
export TF_CPP_MIN_LOG_LEVEL=3
# lambda2 scan on the cut+logit 6-feature ANN (fixed lr schedule + paper arch).
# Full data, full epochs; best-tracking keeps the global min (PATIENCE huge -> no early stop).
export ACC_LAMBDA2=$L
export ACC_LR=2e-3
export ACC_LR_FINAL=2e-4
export ACC_HIDDEN=32,64,32,8
export ACC_NTRAIN=0            # all cut events
export ACC_NORM_SIZE=2000000
export ACC_MAX_EPOCHS=40000
export ACC_CHECK_EVERY=100
export ACC_SEED=1
export ACC_OUTFILE=/rds/user/zx342/hpc-work/4body/acc_6d_lam${L}

# scp first: transform_cut.py, train_acc_6d.py, mlp.py, runs12_cut.npz
python train_acc_6d.py
