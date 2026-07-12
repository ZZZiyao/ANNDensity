#!/bin/bash
#SBATCH -J eff2d_p025
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH -o /rds/user/zx342/hpc-work/eff2d_p025_%j.log

module load miniconda/3
source activate tfagpu
module load cuda/11.2 cudnn/8.1_cuda-11.2

hostname
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"

cd /rds/user/zx342/hpc-work/Ds2KpipiEfficiency/new_funcs
export TF_CPP_MIN_LOG_LEVEL=3
# Sec-6 2D ANN trained on the ptcut=0.25 calibration sample (the only heavy retrain).
# EFF_PTCUT only matters at GENERATION time; here we just read the regenerated p025 file.
export EFF_TOY_LO=eff_toy_1e5_p025.root
export EFF_MODEL_PREFIX=/rds/user/zx342/hpc-work/Ds2KpipiEfficiency/new_funcs/eff_train_2d_p025

# scp first to /rds/user/zx342/hpc-work/Ds2KpipiEfficiency/new_funcs/ :
#   TrainNN2D.py, eff_toy_1e5_p025.root
# and ../DistributionModel.py (the EFF_PTCUT edit)
python TrainNN2D.py
