#!/bin/bash
#SBATCH -J acc_eval
#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH -o /rds/user/zx342/hpc-work/acc_eval_%j.log

module load miniconda/3
source activate tfagpu          # CPU node -> TF runs on CPU (cudart warning is harmless)

cd /rds/user/zx342/hpc-work/4body
export TF_CPP_MIN_LOG_LEVEL=3
export ACC_FLATTEN=1
export ACC_NTRAIN=5000000

echo "=== swish lambda2 scan ==="
ACC_SCAN_PREFIX=acc_5d_swish_lam python eval_scan.py

echo "=== swish003 single eval (1D + cos2D chi2) ==="
python eval_acc_5d.py acc_5d_swish003.npy swish003
