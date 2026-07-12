#!/bin/bash
#SBATCH -J acc_full
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH -o /rds/user/zx342/hpc-work/acc_full_%j.log

module load miniconda/3
source activate tfagpu
module load cuda/11.2 cudnn/8.1_cuda-11.2

hostname
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"

cd /rds/user/zx342/hpc-work/4body
export TF_CPP_MIN_LOG_LEVEL=3
export ACC_FLATTEN=1          # train in phase-space-flattened coords; reads validated runs12_flat.npz
export ACC_SEED=1
export ACC_LAMBDA2=0.03       # CONVERGENCE TEST at a middle lambda; lr-schedule + more epochs
export ACC_LR=2e-3            # higher initial lr ...
export ACC_LR_FINAL=2e-4     # ... exp-decayed to 2e-4 over the run (faster, deeper convergence)
export ACC_NTRAIN=5000000     # 5M for train+val; ~720k left disjoint for eval
export ACC_NORM_SIZE=2000000
export ACC_MAX_EPOCHS=50000
export ACC_CHECK_EVERY=100
export ACC_PATIENCE=600       # > MAX_EPOCHS/CHECK_EVERY -> never early-stop; best-tracking keeps the min
export ACC_OUTFILE=/rds/user/zx342/hpc-work/4body/acc_5d_conv003

# Goal: does proper convergence (lr schedule + 50k epochs) fix the cos wiggles + m1/m2
# asymmetry, and how much does the upper-mass edge improve?  If yes -> rescan lambda converged.
# scp first: runs12_flat.npz, phasespace_transform.py, train_acc_5d.py, eval_acc_5d.py
python train_acc_5d.py
