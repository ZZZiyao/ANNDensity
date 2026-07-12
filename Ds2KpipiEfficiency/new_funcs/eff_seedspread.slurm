#!/bin/bash
#SBATCH -J eff_seed
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH -o /rds/user/zx342/hpc-work/eff_seed_%A_%a.log

module load miniconda/3
source activate tfagpu
module load cuda/11.2 cudnn/8.1_cuda-11.2

SEEDS=(11 22 33 44)
S=${SEEDS[$SLURM_ARRAY_TASK_ID]}
echo "=== task $SLURM_ARRAY_TASK_ID : joint ANN seed $S (2M toy) ==="
hostname
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"

cd /rds/user/zx342/hpc-work/Ds2KpipiEfficiency/new_funcs
export TF_CPP_MIN_LOG_LEVEL=3
# Retrain the 2M joint ANN with different seeds -> map the DEGENERACY spread of the fitted
# latent params. Everything else fixed (same toy, hyperparams as eff_gpu_2e6).
export EFF_TOY_FILE=eff_joint_2e6.root
export EFF_SEED=$S
export EFF_NORM_SIZE=8000000
export EFF_MAX_EPOCHS=50000
export EFF_PATIENCE=600
export EFF_OUTFILE=/rds/user/zx342/hpc-work/Ds2KpipiEfficiency/new_funcs/eff_joint_2e6_seed${S}

# scp first: train_eff_joint.py, eff_joint_2e6.root, ../DistributionModel.py
python train_eff_joint.py
