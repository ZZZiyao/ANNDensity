#!/bin/bash
#SBATCH -J eff_fit
#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH -o /rds/user/zx342/hpc-work/eff_fit_%j.log

module load miniconda/3
source activate tfagpu

cd /rds/user/zx342/hpc-work/Ds2KpipiEfficiency/new_funcs
export TF_CPP_MIN_LOG_LEVEL=3
export EFF_INITFILE=eff_joint_train_gpu.npy

python fit_eff_joint.py
