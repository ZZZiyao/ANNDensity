"""
4-body acceptance -- evaluation / goodness-of-fit (the tuning metric for lambda2).

Loads a trained acc_5d model and a HELD-OUT test set (disjoint from training via
the same seed), then for each projection compares:
  - data:  weight_detJ-weighted histogram of the test events   (= eps, data side)
  - model: ANN eps marginalised over the other dims (uniform 5D sample weighted by eps)
and reports a weighted chi2 (denominator = sum of squared weights = proper error of
a weighted histogram). chi2/nbins ~ 1 => model statistically consistent with data.

Projections: m1, m2, cos1, cos2, phi (1D) + cos1-cos2 (2D, where the structure is).
Output: <outprefix>_eval.png  +  printed chi2 per projection.

Usage: python eval_acc_5d.py [model.npy] [outprefix]
"""
import os
import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import amplitf.interface as atfi
atfi.set_single_precision()

import tensorflow as tf
import uproot

from amplitf.phasespace.rectangular_phasespace import RectangularPhaseSpace
import tfa.neural_nets as tfn

HERE = os.path.dirname(os.path.abspath(__file__))

def _i(n, d): return int(float(os.environ.get(n, d)))

ROOT_FILE = os.environ.get("ACC_ROOT", os.path.join(HERE, "runs12_MagBoth_L0Both.root"))
initfile  = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "acc_5d_train.npy")
outprefix = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(initfile)[0]
NTRAIN = _i("ACC_NTRAIN", 500000)     # how many were used for training (to skip them)
NTEST  = _i("ACC_NTEST", 500000)
SEED   = _i("ACC_SEED", 1)
N_UNIF = _i("ACC_EVAL_UNIF", 2000000)  # uniform sample for model marginalisation
FLATTEN = _i("ACC_FLATTEN", 0)

RANGES = [(0.63, 1.04), (0.63, 1.04), (-1.0, 1.0), (-1.0, 1.0), (-math.pi, math.pi)]
VARS = ["m1", "m2", "cos1", "cos2", "phi"]
if FLATTEN:
    import phasespace_transform as pst
    RANGES = pst.RANGES_FLAT
    VARS = ["u1", "u2", "cos1", "cos2", "phi"]

# ---- model (activation read from the .npy; legacy 4-element models -> sigmoid) ----
import mlp
ann = np.load(initfile, allow_pickle=True)
model = mlp.model_fn_from_npy(ann)

# ---- held-out test data (disjoint from training) ----
if FLATTEN:
    flat_npz = os.environ.get("ACC_FLATFILE", os.path.join(HERE, "runs12_flat.npz"))
    if os.path.exists(flat_npz):
        z = np.load(flat_npz)
        X = z["X"].astype(np.float64); W = z["W"].astype(np.float64)
    else:
        d = pst.load_flattened(ROOT_FILE)
        X = d["X"].astype(np.float64); W = d["W"].astype(np.float64)
else:
    arr = uproot.open(ROOT_FILE)["DecayTree"].arrays(VARS + ["weight_detJ"], library="np")
    X = np.stack([arr[v] for v in VARS], axis=1).astype(np.float64)
    W = np.asarray(arr["weight_detJ"], np.float64)
np.random.seed(SEED)
perm = np.random.permutation(len(X))
test_idx = perm[NTRAIN:NTRAIN + NTEST]
Xt, Wt = X[test_idx], W[test_idx]
print(f"test events: {len(Xt)} (disjoint from first {NTRAIN} used in training)")

# ---- uniform 5D sample, eps at each point (for model marginalisation) ----
U = RectangularPhaseSpace(RANGES).uniform_sample(N_UNIF).numpy().astype(np.float64)
epsU = model(U)

def chi2_1d(i, nb=40):
    lo, hi = RANGES[i]
    bins = np.linspace(lo, hi, nb + 1)
    Hd, _ = np.histogram(Xt[:, i], bins=bins, weights=Wt)
    Vd, _ = np.histogram(Xt[:, i], bins=bins, weights=Wt**2)        # weighted-hist variance
    Hm, _ = np.histogram(U[:, i], bins=bins, weights=epsU)
    Hm *= Hd.sum() / Hm.sum()
    ok = Vd > 0
    chi2 = np.sum((Hd[ok] - Hm[ok])**2 / Vd[ok])
    return bins, Hd, np.sqrt(Vd), Hm, chi2, ok.sum()

def chi2_2d(i, j, nb=25):
    bx = np.linspace(*RANGES[i], nb + 1); by = np.linspace(*RANGES[j], nb + 1)
    Hd, _, _ = np.histogram2d(Xt[:, i], Xt[:, j], bins=[bx, by], weights=Wt)
    Vd, _, _ = np.histogram2d(Xt[:, i], Xt[:, j], bins=[bx, by], weights=Wt**2)
    Hm, _, _ = np.histogram2d(U[:, i], U[:, j], bins=[bx, by], weights=epsU)
    Hm *= Hd.sum() / Hm.sum()
    ok = Vd > 0
    chi2 = np.sum((Hd[ok] - Hm[ok])**2 / Vd[ok])
    return chi2, ok.sum()

# ---- 1D projections + plot ----
fig, axes = plt.subplots(1, 5, figsize=(24, 4.2))
print("\nchi2 per projection (chi2 / nbins):")
for ax, i in zip(axes, range(5)):
    bins, Hd, Ed, Hm, c2, nb = chi2_1d(i)
    ctr = 0.5 * (bins[:-1] + bins[1:])
    ax.errorbar(ctr, Hd, yerr=Ed, fmt="k.", ms=3, capsize=0, label="data (weighted)")
    ax.plot(ctr, Hm, "r-", lw=1.6, label="ANN eps")
    ax.set_xlabel(VARS[i]); ax.set_ylim(bottom=0)
    ax.set_title(f"{VARS[i]}: chi2/nb = {c2/nb:.2f}")
    if FLATTEN and i in (0, 1):                       # relabel uniform-u axis with physical mass (GeV)
        ut = np.linspace(0, 1, 6)
        ax.set_xticks(ut); ax.set_xticklabels([f"{pst.u_to_m(t):.2f}" for t in ut])
        ax.set_xlabel(VARS[i] + "  (m, GeV)")
    if i == 0:
        ax.legend(frameon=False, fontsize=9)
    print(f"  {VARS[i]:6s}  chi2 = {c2:8.1f} / {nb:3d} = {c2/nb:.2f}")
c2cc, nbcc = chi2_2d(2, 3)
print(f"  cos1-cos2 (2D)  chi2 = {c2cc:8.1f} / {nbcc:4d} = {c2cc/nbcc:.2f}")
fig.suptitle(f"4-body acceptance eval ({os.path.basename(initfile)})   "
             f"cos1-cos2 2D chi2/nb = {c2cc/nbcc:.2f}", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(outprefix + "_eval.png", dpi=140)
print(f"\nSaved {outprefix}_eval.png")
