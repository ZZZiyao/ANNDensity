#!/bin/bash
#SBATCH -J gp_cut_scan
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --array=0-5
#SBATCH -o /rds/user/zx342/hpc-work/gp_cut_scan_%A_%a.log

module load miniconda/3
source activate gptorch

KERNELS=(rbf matern12 matern32 matern52 rq spectral)
K=${KERNELS[$SLURM_ARRAY_TASK_ID]}
echo "=== array task $SLURM_ARRAY_TASK_ID : kernel = $K (cut 5d GP) ==="
hostname
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

cd /rds/user/zx342/hpc-work/4body
export GP_BINS=10,10,8,8,6     # 38400 bins -> ~5.9 GB kernel, safe on a 40GB A100
export GP_KERNEL=$K
export GP_ITERS=400
export ACC_NTRAIN=4500000     # cut data ~5.19M; leave ~690k disjoint for eval
export ACC_NTEST=600000

# scp first: gp_torch_cut.py, runs12_cut.npz  (env 'gptorch' = torch cu118 + gpytorch)
python gp_torch_cut.py
