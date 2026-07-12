#!/bin/bash
# RunPod ANN veto runner: loop over veto_matrix, train_acc_pod.py (pure TF, GPU) per config.
# Fixed hyperparams = Optuna winner (wide, lr 5e-4) + lambda2=0.3. 15000 epochs (halved for cost).
cd "$(dirname "$0")" || exit 1
source veto_matrix.sh
mkdir -p plots_veto
export ACC_LAMBDA2=0.3
export ACC_LR=5e-4
export ACC_LR_FINAL=1.5e-5
export ACC_HIDDEN=64,128,64,16
export ACC_NORM_SIZE=2000000
export ACC_MAX_EPOCHS=15000
export ACC_CHECK_EVERY=100
export ACC_NTRAIN=0
export ACC_SEED=1
LOG=plots_veto/ann_runlog.txt
echo "=== ANN START $(date) ===" | tee -a "$LOG"
for cfg in "${VETO_CONFIGS[@]}"; do
  IFS='|' read -r NAME SPEC <<< "$cfg"
  echo "[ANN] $NAME  $SPEC  $(date +%T)" | tee -a "$LOG"
  ACC_VETO="$SPEC" BASE_OUT="plots_veto/veto_ann_${NAME}" \
    python train_acc_pod.py > "plots_veto/ann_${NAME}.log" 2>&1
  grep -iE "recovery ratio=" "plots_veto/veto_ann_${NAME}_rec.txt" 2>/dev/null | tee -a "$LOG"
done
echo "=== ANN ALL DONE $(date) ===" | tee -a "$LOG"
