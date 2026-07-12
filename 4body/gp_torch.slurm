#!/bin/bash
#SBATCH -J gp_torch
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH -o /rds/user/zx342/hpc-work/gp_torch_%j.log

module load miniconda/3
source activate gptorch          # one-time setup below; torch pip wheel bundles its own CUDA

hostname
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

cd /rds/user/zx342/hpc-work/4body
export GP_BINS=10,10,8,8,6        # 38400 bins -> ~5.9 GB kernel, safe on a 40GB A100 (bump if room)
export GP_KERNEL=matern52
export GP_ITERS=400
export ACC_NTRAIN=5000000

# scp first: gp_torch_5d.py, runs12_flat.npz (already there)
python gp_torch_5d.py

# ---------------------------------------------------------------------------
# ONE-TIME env setup (run once on a login node, NOT in this batch job):
#   module load miniconda/3
#   conda create -n gptorch python=3.10 -y
#   source activate gptorch
#   pip install torch --index-url https://download.pytorch.org/whl/cu118
#   pip install gpytorch numpy matplotlib
# ---------------------------------------------------------------------------
