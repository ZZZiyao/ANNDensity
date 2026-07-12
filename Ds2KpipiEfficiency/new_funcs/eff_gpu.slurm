#!/bin/bash
#SBATCH -J eff_gpu
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH -o /rds/user/zx342/hpc-work/eff_gpu_%j.log

module load miniconda/3
source activate tfagpu
module load cuda/11.2 cudnn/8.1_cuda-11.2

echo "=== hostname ==="; hostname
echo "=== nvidia-smi ==="; nvidia-smi
echo "=== TF GPU detection ==="
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"

cd /rds/user/zx342/hpc-work/Ds2KpipiEfficiency/new_funcs
export TF_CPP_MIN_LOG_LEVEL=3
export EFF_SEED=4             # seeds numpy (train/val split) + amplitf/TF (weight init) -> reproducible
export EFF_NORM_SIZE=8000000
export EFF_MAX_EPOCHS=50000
export EFF_PATIENCE=600       # > 50000/CHECK_EVERY -> early-stop never fires -> run full 50000, best-tracking keeps the global min (~ep 37600)
export EFF_OUTFILE=/rds/user/zx342/hpc-work/Ds2KpipiEfficiency/new_funcs/eff_joint_train_gpu

python train_eff_joint.py
