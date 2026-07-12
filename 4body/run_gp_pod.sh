#!/bin/bash
# RunPod GP veto runner: loop over veto_matrix, run gp_torch_cut.py (rbf) per config.
cd "$(dirname "$0")" || exit 1
source veto_matrix.sh
mkdir -p plots_veto
export GP_BINS=10,10,8,8,6
export GP_KERNEL=rbf
export GP_ITERS=400
export ACC_NTRAIN=4500000
export ACC_NTEST=600000
export ACC_EVAL_UNIF=2000000
LOG=plots_veto/gp_runlog.txt
echo "=== GP START $(date) ===" | tee -a "$LOG"
for cfg in "${VETO_CONFIGS[@]}"; do
  IFS='|' read -r NAME SPEC <<< "$cfg"
  echo "[GP] $NAME  $SPEC  $(date +%T)" | tee -a "$LOG"
  ACC_VETO="$SPEC" GP_OUT="plots_veto/veto_gp_${NAME}" \
    python gp_torch_cut.py > "plots_veto/gp_${NAME}.log" 2>&1
  grep -iE "recovery ratio=" "plots_veto/veto_gp_${NAME}_rec.txt" 2>/dev/null | tee -a "$LOG"
done
echo "=== GP ALL DONE $(date) ===" | tee -a "$LOG"
