#!/bin/bash
#SBATCH -J gp_torch_scan
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --array=0-5
#SBATCH -o /rds/user/zx342/hpc-work/gp_torch_scan_%A_%a.log

module load miniconda/3
source activate gptorch

KERNELS=(rbf matern12 matern32 matern52 rq spectral)
K=${KERNELS[$SLURM_ARRAY_TASK_ID]}
echo "=== array task $SLURM_ARRAY_TASK_ID : kernel = $K ==="
hostname
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

cd /rds/user/zx342/hpc-work/4body
export GP_BINS=10,10,8,8,6     # 38400 bins -> ~5.9 GB kernel, safe on a 40GB A100
export GP_KERNEL=$K
export GP_ITERS=400
export ACC_NTRAIN=5000000

# scp first: gp_torch_5d.py, runs12_flat.npz (already there)
python gp_torch_5d.py

# After all 6 finish, compare:   cat /rds/user/zx342/hpc-work/4body/gp_torch_*.txt
# (one-time env setup -- run once on a login node, NOT here:)
#   module load miniconda/3; conda create -n gptorch python=3.10 -y; source activate gptorch
#   pip install torch --index-url https://download.pytorch.org/whl/cu118
#   pip install gpytorch numpy matplotlib
