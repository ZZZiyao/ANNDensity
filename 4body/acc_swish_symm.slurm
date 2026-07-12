#!/bin/bash
#SBATCH -J acc_swish_symm
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH -o /rds/user/zx342/hpc-work/acc_swish_symm_%j.log

module load miniconda/3
source activate tfagpu
module load cuda/11.2 cudnn/8.1_cuda-11.2

hostname
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"

cd /rds/user/zx342/hpc-work/4body
export TF_CPP_MIN_LOG_LEVEL=3
# SYMMETRISED swish: same as swish003 but enforce the m1<->m2 exchange symmetry by augmenting
# the train set with swapped copies. Tests whether that fixes the u2 fit + helps the mass edge.
# (eval uses the ORIGINAL data -> no leakage. The data is symmetric to ~1%, so this is justified.)
export ACC_FLATTEN=1
export ACC_SEED=1
export ACC_ACT=swish
export ACC_OUT=softplus
export ACC_LAMBDA2=0.01        # lambda2-scan optimum on swish
export ACC_LR=2e-3
export ACC_LR_FINAL=2e-4
export ACC_NTRAIN=5000000      # -> 10M training points after swap augmentation (~1.7x epoch cost)
export ACC_NORM_SIZE=2000000
export ACC_MAX_EPOCHS=35000    # swish converged ~31600; best-tracking keeps the min
export ACC_CHECK_EVERY=100
export ACC_PATIENCE=600        # never early-stop; run full, best-tracking keeps the global min
export ACC_OUTFILE=/rds/user/zx342/hpc-work/4body/acc_5d_swish_symm

# scp first: train_acc_swish_symm.py, mlp.py, runs12_flat.npz, phasespace_transform.py
python train_acc_swish_symm.py
