#!/bin/bash
# Local overnight runner: Legendre + spline veto recovery over ALL veto_matrix configs.
# spline first (fast ~17s each), then Legendre (~15-23 min each). Outputs to plots_veto/.
cd "$(dirname "$0")" || exit 1
source activate pj26 2>/dev/null
mkdir -p plots_veto
source veto_matrix.sh

export ACC_NTEST=600000
export ACC_EVAL_UNIF=2000000
export ACC_SEED=1
LOG=plots_veto/runlog.txt
echo "=== START $(date) ===" | tee -a "$LOG"

# ---- spline: all 13 (fast, quick wins) ----
for cfg in "${VETO_CONFIGS[@]}"; do
  IFS='|' read -r NAME SPEC <<< "$cfg"
  echo "[spline]   $NAME  $SPEC  $(date +%T)" | tee -a "$LOG"
  ACC_VETO="$SPEC" SPL_KNOTS=8 BASE_OUT="plots_veto/veto_spl_${NAME}" \
    python density_spline_5d.py > "plots_veto/spl_${NAME}.log" 2>&1
  grep -iE "ratio=" "plots_veto/veto_spl_${NAME}_rec.txt" 2>/dev/null | tee -a "$LOG"
done
echo "=== spline done $(date) ===" | tee -a "$LOG"

# ---- Legendre: all 13 (slow) ----
for cfg in "${VETO_CONFIGS[@]}"; do
  IFS='|' read -r NAME SPEC <<< "$cfg"
  echo "[legendre] $NAME  $SPEC  $(date +%T)" | tee -a "$LOG"
  ACC_VETO="$SPEC" LEG_ORDER=4 LEG_BINS=8 LEG_LAMBDA2=1.0 BASE_OUT="plots_veto/veto_leg_${NAME}" \
    python density_legendre_5d.py > "plots_veto/leg_${NAME}.log" 2>&1
  grep -iE "ratio=" "plots_veto/veto_leg_${NAME}_rec.txt" 2>/dev/null | tee -a "$LOG"
done
echo "=== ALL DONE $(date) ===" | tee -a "$LOG"
