#!/bin/bash
#SBATCH -J eff_2e6
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH -o /rds/user/zx342/hpc-work/eff_2e6_%j.log

module load miniconda/3
source activate tfagpu
module load cuda/11.2 cudnn/8.1_cuda-11.2

hostname
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"

cd /rds/user/zx342/hpc-work/Ds2KpipiEfficiency/new_funcs
export TF_CPP_MIN_LOG_LEVEL=3
# Retrain the §8.2 joint ANN on the 2M toy (aligns paper which uses 2e6; was 1e6).
# Same hyperparams as the 1e6 run (eff_gpu.slurm) -- only the input toy changes.
export EFF_TOY_FILE=eff_joint_2e6.root
export EFF_SEED=4
export EFF_NORM_SIZE=8000000
export EFF_MAX_EPOCHS=50000
export EFF_PATIENCE=600       # > 50000/CHECK_EVERY -> run full, best-tracking keeps the global min
export EFF_OUTFILE=/rds/user/zx342/hpc-work/Ds2KpipiEfficiency/new_funcs/eff_joint_train_2e6_gpu

# scp first to .../Ds2KpipiEfficiency/new_funcs/ :  eff_joint_2e6.root, train_eff_joint.py
# and ../DistributionModel.py if changed
python train_eff_joint.py
