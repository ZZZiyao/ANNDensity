"""
Paper Fig.1 reproduction for the ptcut=0.25 sample: relative efficiency ε(m',θ') over the
square Dalitz plot, normalised so the average over phase space = 1.

NO m' shift (the ptcut=0.25 fix makes the m' edge match the paper naturally; the old
plot_fig1_shifted.py with its -0.1 offset is obsolete). High-stat 4M sample after selection,
same as the paper's Fig.1 (which also uses 4e6 events). Matches paper style: afmhot_r, vmax≈1.3.
"""
import os
import uproot
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14, "axes.labelsize": 16,
                     "xtick.direction": "in", "ytick.direction": "in"})

HERE = os.path.dirname(os.path.abspath(__file__))
infile = os.environ.get("FIG1_INPUT", os.path.join(HERE, "eff_toy_4e6_p025.root"))
f = uproot.open(infile)
arr = f[f.keys()[0]].arrays(["mprime", "thetaprime"], library="np")
m, t = arr["mprime"], arr["thetaprime"]

H, xe, ye = np.histogram2d(m, t, bins=50, range=[[0, 1], [0, 1]])
H = H / np.mean(H[H > 0])           # normalise: average efficiency = 1

plt.figure(figsize=(6, 5))
plt.imshow(H.T, origin="lower", extent=[0, 1, 0, 1], aspect="auto",
           cmap="afmhot_r", vmin=0.0, vmax=1.3)
plt.xlabel("m'")
plt.ylabel("θ'")
cbar = plt.colorbar()
cbar.set_label("ε(m', θ')")
plt.tight_layout()
out = os.path.join(HERE, "fig1_p025.png")
plt.savefig(out, dpi=300)
print(f"Saved {out}  (input {os.path.basename(infile)}, N={len(m)}, ptcut=0.25, no shift)")
