#!/bin/bash
#SBATCH -J veto_gp
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --array=0-12
#SBATCH -o /rds/user/zx342/hpc-work/veto_gp_%A_%a.log

module load miniconda/3
source activate gptorch

source veto_matrix.sh
IFS='|' read -r NAME SPEC <<< "${VETO_CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "=== task $SLURM_ARRAY_TASK_ID : $NAME  veto=$SPEC (GP) ==="
hostname

cd /rds/user/zx342/hpc-work/4body
export GP_BINS=10,10,8,8,6
export GP_KERNEL=rbf           # winning kernel from the no-veto cut GP scan (best cos1/cos2 + cos2D)
export GP_ITERS=400
export ACC_NTRAIN=4500000
export ACC_NTEST=600000
export ACC_VETO=$SPEC
export GP_OUT=/rds/user/zx342/hpc-work/4body/veto_gp_${NAME}

# scp first: gp_torch_cut.py, veto.py, runs12_cut.npz, veto_matrix.sh
python gp_torch_cut.py
