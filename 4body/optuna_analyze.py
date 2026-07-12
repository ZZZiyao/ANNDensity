"""
Analyse the cut+6d ANN Optuna study: best config, optimization history, per-hyperparameter
behaviour, architecture comparison. Pure matplotlib (no plotly). Prints the winning config
to fix for the controlled veto study.

Usage: python optuna_analyze.py [db_path] [study_name]
"""
import os
import sys
import numpy as np
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DB    = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "optuna_acc.db")
STUDY = sys.argv[2] if len(sys.argv) > 2 else "acc6d_v1"

study = optuna.load_study(study_name=STUDY, storage=f"sqlite:///{DB}")
done = [t for t in study.trials if t.state.name == "COMPLETE"]
pruned = [t for t in study.trials if t.state.name == "PRUNED"]
print(f"study '{STUDY}': {len(study.trials)} trials  ({len(done)} complete, {len(pruned)} pruned)")
print(f"\n=== BEST (lowest val NLL) ===\n  value = {study.best_value:.2f}")
for k, v in study.best_params.items():
    print(f"  {k:16s} {v}")

# top-5 completed trials
print("\n=== top 5 completed trials ===")
for t in sorted(done, key=lambda t: t.value)[:5]:
    print(f"  #{t.number:3d}  val={t.value:10.1f}  lam2={t.params.get('lambda2'):.4g}  "
          f"lr={t.params.get('lr'):.2g}  lrf={t.params.get('lr_final_ratio'):.3g}  arch={t.params.get('arch')}")

vals = np.array([t.value for t in done])
fig, ax = plt.subplots(2, 3, figsize=(18, 10))
# (0,0) optimization history
order = sorted(done, key=lambda t: t.number)
nums = [t.number for t in order]; vv = [t.value for t in order]
running = np.minimum.accumulate(vv)
ax[0, 0].scatter(nums, vv, s=25, alpha=0.5, label="trial")
ax[0, 0].plot(nums, running, "r-", lw=2, label="best so far")
ax[0, 0].set_xlabel("trial #"); ax[0, 0].set_ylabel("val NLL"); ax[0, 0].set_title("optimization history"); ax[0, 0].legend()
# (0,1) lambda2, (0,2) lr, (1,0) lr_final_ratio  -- value vs objective (log x)
for a, key in [(ax[0, 1], "lambda2"), (ax[0, 2], "lr"), (ax[1, 0], "lr_final_ratio")]:
    xs = [t.params[key] for t in done]
    a.scatter(xs, vals, s=25, alpha=0.6)
    a.set_xscale("log"); a.set_xlabel(key); a.set_ylabel("val NLL"); a.set_title(f"val NLL vs {key}")
    bx = study.best_params[key]; a.axvline(bx, color="r", ls="--", label=f"best={bx:.3g}"); a.legend(fontsize=8)
# (1,1) architecture strip
archs = sorted(set(t.params["arch"] for t in done))
for i, ar in enumerate(archs):
    yv = [t.value for t in done if t.params["arch"] == ar]
    ax[1, 1].scatter([i] * len(yv), yv, s=30, alpha=0.6)
    ax[1, 1].scatter([i], [np.median(yv)], marker="_", s=400, color="r")
ax[1, 1].set_xticks(range(len(archs))); ax[1, 1].set_xticklabels(archs, rotation=20)
ax[1, 1].set_ylabel("val NLL"); ax[1, 1].set_title("val NLL by architecture (red=median)")
# (1,2) param importances (if available)
try:
    imp = optuna.importance.get_param_importances(study)
    ax[1, 2].barh(list(imp.keys())[::-1], list(imp.values())[::-1], color="tab:purple")
    ax[1, 2].set_title("hyperparameter importance"); ax[1, 2].set_xlabel("importance")
except Exception as e:
    ax[1, 2].text(0.1, 0.5, f"importance N/A\n{str(e)[:40]}", fontsize=9)
fig.suptitle(f"Optuna cut+6d ANN ({len(done)} trials)  best val NLL={study.best_value:.1f}", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97]); plt.savefig(os.path.join(HERE, "optuna_analysis.png"), dpi=130)
print(f"\nSaved optuna_analysis.png")

# emit the winning config as slurm-ready exports
bp = study.best_params
ARCHS = {"small": "32,16,8", "paper": "32,64,32,8", "wide": "64,128,64,16", "deep": "32,64,64,32,8"}
print("\n=== fix THESE for the controlled veto study (slurm exports) ===")
print(f"export ACC_LAMBDA2={bp['lambda2']:.4g}")
print(f"export ACC_LR={bp['lr']:.4g}")
print(f"export ACC_LR_FINAL={bp['lr'] * bp['lr_final_ratio']:.4g}")
print(f"export ACC_HIDDEN={ARCHS[bp['arch']]}")
