#!/bin/bash
#SBATCH -J veto_ann
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:30:00
#SBATCH --array=0-12
#SBATCH -o /rds/user/zx342/hpc-work/veto_ann_%A_%a.log

module load miniconda/3
source activate tfagpu
module load cuda/11.2 cudnn/8.1_cuda-11.2

source veto_matrix.sh   # defines VETO_CONFIGS
IFS='|' read -r NAME SPEC <<< "${VETO_CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "=== task $SLURM_ARRAY_TASK_ID : $NAME  veto=$SPEC ==="
hostname

cd /rds/user/zx342/hpc-work/4body

# FIXED hyperparameters (controlled experiment -- SAME for every veto). From the no-veto Optuna
# winner: arch=wide, lr=5e-4, lr_final_ratio=0.03. lambda2 fixed at 0.3 (Optuna+scan show it is
# insensitive; 0.3 = strongest reg -> smoothest model -> most reliable extrapolation into the hole).
export TF_CPP_MIN_LOG_LEVEL=3
export ACC_LAMBDA2=0.3
export ACC_LR=5e-4
export ACC_LR_FINAL=1.5e-5
export ACC_HIDDEN=64,128,64,16
export ACC_NORM_SIZE=2000000
export ACC_MAX_EPOCHS=30000
export ACC_CHECK_EVERY=100
export ACC_SEED=1
export ACC_NTRAIN=0
export ACC_VETO=$SPEC
export ACC_OUTFILE=/rds/user/zx342/hpc-work/4body/veto_ann_${NAME}

# scp first: transform_cut.py, train_acc_6d.py, eval_veto.py, veto.py, mlp.py,
#            runs12_cut.npz, veto_matrix.sh
python train_acc_6d.py
# recovery evaluation
python eval_veto.py veto_ann_${NAME}.npy "$SPEC" veto_ann_${NAME}_rec
