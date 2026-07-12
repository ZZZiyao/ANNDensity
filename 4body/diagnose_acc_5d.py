"""
STEP 0 -- diagnostic for the 4-body B->K*K* acceptance.

Reads the 5 phase-space variables (m1, m2, cos1, cos2, phi) + weight_detJ from
runs12_MagBoth_L0Both.root and shows:
  Fig1 (1D): each variable, RAW (phasespace x eps) vs weight_detJ-WEIGHTED (= eps),
             both area-normalised -> see what the phase-space factor removes and
             whether eps has structure worth modelling.
  Fig2 (2D): weight_detJ-weighted heatmaps (m1-m2, cos1-cos2, m1-cos1, cos1-phi)
             -> see correlations (does a factorised model suffice?).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot

HERE = os.path.dirname(os.path.abspath(__file__))
f = uproot.open(os.path.join(HERE, "runs12_MagBoth_L0Both.root"))["DecayTree"]
a = f.arrays(["m1", "m2", "cos1", "cos2", "phi", "weight_detJ"], library="np")
m1, m2 = np.asarray(a["m1"], float), np.asarray(a["m2"], float)
c1, c2 = np.asarray(a["cos1"], float), np.asarray(a["cos2"], float)
phi = np.asarray(a["phi"], float)
w = np.asarray(a["weight_detJ"], float)
print(f"{len(m1)} events")

VARS = [
    ("m1", m1, (0.63, 1.04)),
    ("m2", m2, (0.63, 1.04)),
    ("cos1", c1, (-1, 1)),
    ("cos2", c2, (-1, 1)),
    ("phi", phi, (-np.pi, np.pi)),
]

# ---------- Fig 1: 1D raw vs weighted ----------
fig, axes = plt.subplots(1, 5, figsize=(24, 4.2))
for ax, (name, x, rng) in zip(axes, VARS):
    bins = np.linspace(rng[0], rng[1], 51)
    Hr, e = np.histogram(x, bins=bins)
    Hw, _ = np.histogram(x, bins=bins, weights=w)
    ctr = 0.5 * (e[:-1] + e[1:])
    Hr = Hr / Hr.sum(); Hw = Hw / Hw.sum()
    ax.step(ctr, Hr, where="mid", color="gray", label="raw (PS x eps)")
    ax.step(ctr, Hw, where="mid", color="tab:red", lw=1.8, label="weight_detJ (= eps)")
    ax.set_xlabel(name); ax.set_ylim(bottom=0)
    ax.minorticks_on(); ax.tick_params(which="both", top=True, right=True)
    if name == "m1":
        ax.set_ylabel("normalised")
        ax.legend(frameon=False, fontsize=9)
fig.suptitle("4-body acceptance: 1D projections (raw vs weight_detJ-weighted = eps)", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(HERE, "acc_diag_1d.png"), dpi=140)
print("Saved acc_diag_1d.png")
plt.close()

# ---------- Fig 2: 2D weighted heatmaps ----------
pairs = [("m1", m1, (0.63, 1.04), "m2", m2, (0.63, 1.04)),
         ("cos1", c1, (-1, 1), "cos2", c2, (-1, 1)),
         ("m1", m1, (0.63, 1.04), "cos1", c1, (-1, 1)),
         ("cos1", c1, (-1, 1), "phi", phi, (-np.pi, np.pi))]
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
for ax, (nx, X, rx, ny, Y, ry) in zip(axes, pairs):
    # weighted SUM per bin = integral of eps over other dims = eps projection
    H, xe, ye = np.histogram2d(X, Y, bins=40, range=[rx, ry], weights=w)
    dens = H / H[H > 0].mean()                       # average eps = 1
    pcm = ax.pcolormesh(xe, ye, dens.T, shading="auto", cmap="afmhot_r", vmin=0, vmax=2)
    ax.set_xlabel(nx); ax.set_ylabel(ny)
    fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
fig.suptitle("4-body acceptance: 2D weighted structure (eps, avg=1)", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(HERE, "acc_diag_2d.png"), dpi=140)
print("Saved acc_diag_2d.png")
