#!/bin/bash
#SBATCH -J acc_scan
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --array=0-5
#SBATCH -o /rds/user/zx342/hpc-work/acc_scan_%A_%a.log

module load miniconda/3
source activate tfagpu
module load cuda/11.2 cudnn/8.1_cuda-11.2

LAMBDAS=(0.003 0.01 0.03 0.1 0.3 1.0)
L=${LAMBDAS[$SLURM_ARRAY_TASK_ID]}

echo "=== array task $SLURM_ARRAY_TASK_ID : lambda2 = $L ==="
hostname
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"

cd /rds/user/zx342/hpc-work/4body
export TF_CPP_MIN_LOG_LEVEL=3
export ACC_FLATTEN=1          # phase-space-flattened coords; reads validated runs12_flat.npz
export ACC_SEED=1            # same split across all lambda -> fair comparison
export ACC_LAMBDA2=$L         # scan brackets the paper values: eff=5e-3, bg=1.0
export ACC_NORM_SIZE=2000000
export ACC_NTRAIN=5000000     # FULL data: optimal lambda2 depends on N (eff.reg ~ lambda2/N),
                              # so scan at the N we deploy. ~720k left disjoint for eval.
                              # => the best scan model IS the final model (no separate full run).
export ACC_MAX_EPOCHS=30000
export ACC_CHECK_EVERY=100
export ACC_PATIENCE=400       # > MAX_EPOCHS/CHECK_EVERY(=300) -> early-stop NEVER fires.
                              # Run full 30000 for every lambda (fair comparison) and let
                              # best-tracking keep each run's global min -- avoids the eff
                              # non-monotonic-val trap where early-stop quit at a false plateau.
export ACC_OUTFILE=/rds/user/zx342/hpc-work/4body/acc_5d_lam${L}

# scp to /rds/user/zx342/hpc-work/4body/ first:
#   runs12_flat.npz, phasespace_transform.py, train_acc_5d.py
python train_acc_5d.py
