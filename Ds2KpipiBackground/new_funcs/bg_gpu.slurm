#!/bin/bash
#SBATCH -J bg_gpu
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH -o /rds/user/zx342/hpc-work/bg_gpu_%j.log

module load miniconda/3
source activate tfagpu
module load cuda/11.2 cudnn/8.1_cuda-11.2

echo "=== hostname ==="; hostname
echo "=== nvidia-smi ==="; nvidia-smi
echo "=== TF GPU detection ==="
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"

cd /rds/user/zx342/hpc-work/Ds2KpipiBackground/new_funcs
export TF_CPP_MIN_LOG_LEVEL=3
export BG_NORM_SIZE=4000000
export BG_MAX_EPOCHS=50000
export BG_OUTFILE=/rds/user/zx342/hpc-work/Ds2KpipiBackground/new_funcs/bg_joint_train_gpu

python train_bg_joint.py
