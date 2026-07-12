"""
4-body acceptance -- FULL 2D correlation validation (Data vs Model).

The 1D projections + a single cos1-cos2 2D miss any correlation hidden in the other
pairs (e.g. a mass-dependent angular acceptance u-cos).  This plots ALL 10 variable
pairs as Data | Model heatmaps with a per-pair chi2/nb, and prints a summary table so
the worst correlations are obvious.

  Data  = held-out test events, weighted by W (= eps marginalised onto the 2 vars).
  Model = uniform 5D sample weighted by the ANN eps, normalised to the data integral.
2D normalised so average = 1; afmhot_r, shared colour scale per Data|Model pair.

Usage: python plot_acc_2d.py [model.npy] [outprefix]
       (env: ACC_FLATTEN=1 ACC_NTRAIN=5000000 to match the training split)
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
initfile  = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "acc_5d_conv003.npy")
outprefix = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(initfile)[0]
NTRAIN = _i("ACC_NTRAIN", 5000000)
NTEST  = _i("ACC_NTEST", 1000000)
SEED   = _i("ACC_SEED", 1)
N_UNIF = _i("ACC_EVAL_UNIF", 2000000)
FLATTEN = _i("ACC_FLATTEN", 0)
NB = _i("ACC_2D_BINS", 30)

RANGES = [(0.63, 1.04), (0.63, 1.04), (-1.0, 1.0), (-1.0, 1.0), (-math.pi, math.pi)]
VARS = ["m1", "m2", "cos1", "cos2", "phi"]
pst = None
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
    z = np.load(os.environ.get("ACC_FLATFILE", os.path.join(HERE, "runs12_flat.npz")))
    X = z["X"].astype(np.float64); W = z["W"].astype(np.float64)
else:
    arr = uproot.open(ROOT_FILE)["DecayTree"].arrays(VARS + ["weight_detJ"], library="np")
    X = np.stack([arr[v] for v in VARS], axis=1).astype(np.float64)
    W = np.asarray(arr["weight_detJ"], np.float64)
np.random.seed(SEED)
perm = np.random.permutation(len(X))
ti = perm[NTRAIN:NTRAIN + NTEST]
Xt, Wt = X[ti], W[ti]
U = RectangularPhaseSpace(RANGES).uniform_sample(N_UNIF).numpy().astype(np.float64)
epsU = model(U)
print(f"test {len(Xt)} events, uniform {len(U)}")

# all 10 pairs, ordered: most informative first
PAIRS = [(2, 3), (0, 2), (1, 3), (0, 1), (0, 3), (1, 2), (2, 4), (3, 4), (0, 4), (1, 4)]


def hist2d(src, w, i, j):
    bx = np.linspace(*RANGES[i], NB + 1); by = np.linspace(*RANGES[j], NB + 1)
    H, _, _ = np.histogram2d(src[:, i], src[:, j], bins=[bx, by], weights=w)
    return H, bx, by


def pair_chi2_and_maps(i, j):
    Hd, bx, by = hist2d(Xt, Wt, i, j)
    Vd, _, _ = hist2d(Xt, Wt ** 2, i, j)          # weighted-hist variance
    Hm, _, _ = hist2d(U, epsU, i, j)
    Hm *= Hd.sum() / Hm.sum()
    ok = Vd > 0
    chi2 = np.sum((Hd[ok] - Hm[ok]) ** 2 / Vd[ok]) / ok.sum()
    # display-normalised (avg = 1 over filled bins)
    Dd = Hd / Hd[Hd > 0].mean()
    Dm = Hm / Hm[Hm > 0].mean()
    return chi2, Dd, Dm, bx, by


def axis(ax, i, j):
    ax.set_xlabel(VARS[i]); ax.set_ylabel(VARS[j])
    if FLATTEN:
        for k, sett, setl, setlab in [(i, ax.set_xticks, ax.set_xticklabels, ax.set_xlabel),
                                      (j, ax.set_yticks, ax.set_yticklabels, ax.set_ylabel)]:
            if k in (0, 1):
                t = np.linspace(0, 1, 5)
                sett(t); setl([f"{pst.u_to_m(v):.2f}" for v in t]); setlab(VARS[k] + " (m,GeV)")


# ---- compute + print table ----
results = []
print(f"\n{'pair':>12s}  chi2/nb")
print("-" * 24)
for (i, j) in PAIRS:
    c2, Dd, Dm, bx, by = pair_chi2_and_maps(i, j)
    results.append((i, j, c2, Dd, Dm, bx, by))
    print(f"{VARS[i]+'-'+VARS[j]:>12s}  {c2:6.2f}")

# ---- figure: Data | Model for every pair, 4 cols wide ----
ncols = 4
nrows = (2 * len(PAIRS) + ncols - 1) // ncols
fig = plt.figure(figsize=(20, 4 * nrows))
gs = fig.add_gridspec(nrows, ncols)
slot = 0
for (i, j, c2, Dd, Dm, bx, by) in results:
    vmax = max(2.0, float(np.percentile(Dd[Dd > 0], 99)))
    for D, tag in [(Dd, "Data"), (Dm, "Model")]:
        ax = fig.add_subplot(gs[slot // ncols, slot % ncols])
        pm = ax.pcolormesh(bx, by, D.T, shading="auto", cmap="afmhot_r", vmin=0, vmax=vmax)
        axis(ax, i, j)
        ax.set_title(f"{tag} {VARS[i]}-{VARS[j]}" + (f"  chi2/nb={c2:.2f}" if tag == "Data" else ""), fontsize=8)
        fig.colorbar(pm, ax=ax, fraction=0.046, pad=0.04)
        slot += 1
fig.suptitle(f"4-body acceptance: all 2D correlations Data vs Model ({os.path.basename(initfile)})", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig(outprefix + "_2d.png", dpi=110)
print(f"\nSaved {outprefix}_2d.png")
