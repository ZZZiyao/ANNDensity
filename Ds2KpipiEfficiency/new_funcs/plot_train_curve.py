"""
Plot the training curve from a train_*_joint .txt log (epoch / train NLL / val NLL).

Usage:  python plot_train_curve.py [logfile] [outpng]
Default: eff_joint_train_gpu.txt -> eff_train_curve.png
"""
import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
logfile = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "eff_joint_train_gpu.txt")
outpng = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "eff_train_curve.png")

pat = re.compile(r"Epoch\s+(\d+)\s+train\s+(-?[\d.]+)\s+val\s+(-?[\d.]+)\s+best_val\s+(-?[\d.eE+]+)")
ep, tr, va = [], [], []
for line in open(logfile):
    m = pat.search(line)
    if m:
        ep.append(int(m.group(1)))
        tr.append(float(m.group(2)))
        va.append(float(m.group(3)))

ep = np.array(ep); tr = np.array(tr); va = np.array(va)
best_i = int(np.argmin(va))
print(f"{len(ep)} checkpoints, epochs {ep.min()}-{ep.max()}")
print(f"best val NLL = {va[best_i]:.3f} at epoch {ep[best_i]}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# (a) both curves, full range
ax = axes[0]
ax.plot(ep, tr, label="train NLL", color="tab:blue")
ax.plot(ep, va, label="val NLL", color="tab:orange")
ax.axvline(ep[best_i], color="green", ls="--", lw=1, label=f"best val (ep {ep[best_i]})")
ax.set_xlabel("epoch"); ax.set_ylabel("NLL")
ax.set_title("(a) train & val NLL"); ax.legend(frameon=False)
ax.minorticks_on(); ax.tick_params(which="both", top=True, right=True)

# (b) val NLL zoom on the tail (last 60%) to see the plateau
ax = axes[1]
k = int(0.4 * len(ep))
ax.plot(ep[k:], va[k:], color="tab:orange", marker=".", ms=3)
ax.axhline(va[best_i], color="green", ls="--", lw=1, label=f"best = {va[best_i]:.1f}")
ax.axvline(ep[best_i], color="green", ls=":", lw=1)
ax.set_xlabel("epoch"); ax.set_ylabel("val NLL")
ax.set_title("(b) val NLL (tail, convergence)"); ax.legend(frameon=False)
ax.minorticks_on(); ax.tick_params(which="both", top=True, right=True)

plt.tight_layout()
plt.savefig(outpng, dpi=150)
print(f"Saved {outpng}")
