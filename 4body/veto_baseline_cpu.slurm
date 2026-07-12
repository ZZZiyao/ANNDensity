#!/bin/bash
#SBATCH -J veto_base
#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --array=0-25
#SBATCH -o /rds/user/zx342/hpc-work/veto_base_%A_%a.log

module load miniconda/3
source activate tfagpu   # numpy/scipy/uproot; CPU only

source veto_matrix.sh    # VETO_CONFIGS (13 entries)
NCFG=${#VETO_CONFIGS[@]}                      # 13
IDX=$(( SLURM_ARRAY_TASK_ID % NCFG ))         # which veto config
METH=$(( SLURM_ARRAY_TASK_ID / NCFG ))        # 0 = Legendre, 1 = spline
IFS='|' read -r NAME SPEC <<< "${VETO_CONFIGS[$IDX]}"

cd /rds/user/zx342/hpc-work/4body
export ACC_VETO=$SPEC
export ACC_NTEST=600000
export ACC_EVAL_UNIF=2000000
export ACC_SEED=1

if [ $METH -eq 0 ]; then
  echo "=== Legendre veto: $NAME  ($SPEC) ==="
  export LEG_ORDER=4          # 3125 coeffs; global polynomial extrapolates into the hole
  export LEG_BINS=8
  export LEG_LAMBDA2=1.0
  export BASE_OUT=/rds/user/zx342/hpc-work/4body/veto_leg_${NAME}
  python density_legendre_5d.py
else
  echo "=== spline veto: $NAME  ($SPEC) ==="
  export SPL_KNOTS=8          # hole nodes filled by nearest non-veto node, then cubic
  export BASE_OUT=/rds/user/zx342/hpc-work/4body/veto_spl_${NAME}
  python density_spline_5d.py
fi

# scp first: density_legendre_5d.py, density_spline_5d.py, veto_recovery.py, veto.py,
#            transform_cut.py, veto_matrix.sh, runs12_cut.npz
