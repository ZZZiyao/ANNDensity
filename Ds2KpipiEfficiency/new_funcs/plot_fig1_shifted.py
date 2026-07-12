"""
Reproduce paper Fig.1 (relative efficiency over the square Dalitz plot), with a -0.1 shift
applied to the m' axis to align the left edge / overall trend with the paper.

NOTE: the -0.1 m' offset is a PRESENTATION alignment (m' is a physical coordinate, Eq.5).
The residual difference (your peak is narrower/higher than the paper's broad plateau) is a
simplified-cut-model vs full-MC efficiency difference, not removable by an axis shift.

Original plot_fig1.py read 'highstat.root' (missing); here we use the available 4M high-stat
sample eff_toy_4e6.root. Does NOT overwrite plot_fig1.py.
"""
import os
import uproot
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14, "axes.labelsize": 16,
                     "xtick.direction": "in", "ytick.direction": "in"})

HERE = os.path.dirname(os.path.abspath(__file__))
SHIFT = -0.1                                   # m' offset for alignment with the paper

infile = os.environ.get("FIG1_INPUT", os.path.join(HERE, "eff_toy_4e6.root"))
f = uproot.open(infile)
tree = f[f.keys()[0]]
arr = tree.arrays(["mprime", "thetaprime"], library="np")
m, t = arr["mprime"], arr["thetaprime"]

# 2D histogram + normalise so average efficiency = 1
H, xedges, yedges = np.histogram2d(m, t, bins=50, range=[[0, 1], [0, 1]])
H = H / np.mean(H[H > 0])

plt.figure(figsize=(6, 5))
# shift the m' (x) axis by SHIFT: the data originally at m' is displayed at m'+SHIFT
plt.imshow(H.T, origin="lower", extent=[0 + SHIFT, 1 + SHIFT, 0, 1],
           aspect="auto", cmap="afmhot_r", vmin=0.0, vmax=1.3)
plt.xlim(0, 1)            # axis starts at 0 (the shifted-out strip below 0 is hidden)
plt.ylim(0, 1)
plt.xlabel("m'  (shifted -0.1)")
plt.ylabel("θ'")
cbar = plt.colorbar()
cbar.set_label("ε(m', θ')")
plt.tight_layout()
out = os.path.join(HERE, "fig1_shifted.png")
plt.savefig(out, dpi=300)
print(f"Saved {out}  (input {os.path.basename(infile)}, m' shift {SHIFT})")
