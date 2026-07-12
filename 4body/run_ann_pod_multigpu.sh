#!/bin/bash
# Multi-GPU ANN veto runner: dispatch the 13 veto configs across N GPUs (one config per GPU,
# round-robin). Each GPU runs its share serially; all GPUs run in parallel.
# Usage:  bash run_ann_pod_multigpu.sh [NGPU]     (default: auto-detect)
cd "$(dirname "$0")" || exit 1
source veto_matrix.sh
mkdir -p plots_veto

NGPU=${1:-$(nvidia-smi -L | wc -l)}
NCFG=${#VETO_CONFIGS[@]}
echo "=== ANN multi-GPU START $(date)  |  $NGPU GPUs, $NCFG configs ===" | tee -a plots_veto/ann_runlog.txt

# fixed hyperparameters (Optuna winner + lambda2=0.3); ACC_MAX_EPOCHS overridable via env
export ACC_LAMBDA2=0.3
export ACC_LR=5e-4
export ACC_LR_FINAL=1.5e-5
export ACC_HIDDEN=64,128,64,16
export ACC_NORM_SIZE=2000000
export ACC_MAX_EPOCHS=${ACC_MAX_EPOCHS:-30000}
export ACC_CHECK_EVERY=100
export ACC_NTRAIN=0
export ACC_SEED=1

for g in $(seq 0 $((NGPU - 1))); do
  (
    for i in $(seq $g $NGPU $((NCFG - 1))); do
      IFS='|' read -r NAME SPEC <<< "${VETO_CONFIGS[$i]}"
      echo "[GPU $g] $NAME  $SPEC  $(date +%T)" | tee -a plots_veto/ann_runlog.txt
      CUDA_VISIBLE_DEVICES=$g ACC_VETO="$SPEC" BASE_OUT="plots_veto/veto_ann_${NAME}" \
        python train_acc_pod.py > "plots_veto/ann_${NAME}.log" 2>&1
      grep -iE "recovery ratio=" "plots_veto/veto_ann_${NAME}_rec.txt" 2>/dev/null | tee -a plots_veto/ann_runlog.txt
    done
  ) &
done
wait
echo "=== ANN ALL DONE $(date) ===" | tee -a plots_veto/ann_runlog.txt
