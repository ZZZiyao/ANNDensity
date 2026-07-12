#!/bin/bash
#SBATCH -J acc_baseline
#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH -o /rds/user/zx342/hpc-work/acc_baseline_%A_%a.log

module load miniconda/3
source activate tfagpu   # CSD3 env with numpy/scipy/uproot; CPU only (no GPU needed here)

cd /rds/user/zx342/hpc-work/4body
export ACC_NTEST=600000
export ACC_EVAL_UNIF=2000000
export ACC_SEED=1

# the two traditional baselines, in parallel array tasks
case $SLURM_ARRAY_TASK_ID in
  0)
    echo "=== 5D Legendre baseline ==="
    export LEG_ORDER=4       # 5^5 = 3125 coefficients
    export LEG_BINS=8        # 8^5 = 32768 histogram bins
    export LEG_LAMBDA2=1.0
    python density_legendre_5d.py
    ;;
  1)
    echo "=== 5D cubic spline baseline ==="
    export SPL_KNOTS=8       # 8^5 = 32768 nodes
    python density_spline_5d.py
    ;;
esac

# scp first: density_legendre_5d.py, density_spline_5d.py, transform_cut.py, runs12_cut.npz
