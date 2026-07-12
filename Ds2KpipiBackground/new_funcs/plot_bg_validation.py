"""
Reproduce Fig.13-15 (Appendix B): model-assisted BACKGROUND joint ANN validation.
Layout matches the paper (split BY OBSERVABLE):
  Fig.13 : top row = 1D m', theta', mD; then m' vs (theta', mD, 8 params), each Data|Fit.
  Fig.14 : theta' vs (mD, 8 params), each Data|Fit.
  Fig.15 : mD vs (8 params), each Data|Fit.
2D correlations packed 4 columns wide (= 2 correlations per row, each a Data|Fit pair).
"Data" = training toy; "Fit" = ANN density (uniform sample weighted by eps); 2D avg=1.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import amplitf.interface as atfi
atfi.set_single_precision()
import tensorflow as tf
import tfa.rootio as tfr
import tfa.neural_nets as tfn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from DistributionModel import parameters_list, observables_toys

MODEL = os.path.join(HERE, "bg_joint_train_gpu.npy")
TOY   = os.path.join(HERE, "toy_1e6.root")
N_UNIF = int(float(os.environ.get("VAL_UNIF", 2000000)))

ann = np.load(MODEL, allow_pickle=True)
ranges = ann[1]
weights = [tf.constant(w, dtype=atfi.fptype()) for w in ann[2]]
biases = [tf.constant(b, dtype=atfi.fptype()) for b in ann[3]]
def density(x):
    return tfn.multilayer_perceptron(np.asarray(x, np.float32), ranges, weights, biases).numpy()

VARS = observables_toys + [p[0] for p in parameters_list]
LABELS = [r"$m'$", r"$\theta'$", r"$m_D$"] + [p[1] for p in parameters_list]
NOBS = len(observables_toys)
NPAR = len(parameters_list)
print(f"{len(VARS)} dims:", VARS)

data = np.asarray(tfr.read_tuple(TOY, VARS))
U = np.column_stack([np.random.uniform(lo, hi, N_UNIF) for (lo, hi) in ranges]).astype(np.float32)
dU = density(U)
print(f"data {len(data)}, uniform {len(U)}")


def hist2d_norm(src, wts, i, j, nb=41):
    bx = np.linspace(*ranges[i], nb); by = np.linspace(*ranges[j], nb)
    H, _, _ = np.histogram2d(src[:, i], src[:, j], bins=[bx, by], weights=wts)
    return H / H[H > 0].mean(), bx, by

def draw2d(ax, H, bx, by, vmax, i, j, title, fig):
    pm = ax.pcolormesh(bx, by, H.T, shading="auto", cmap="afmhot_r", vmin=0, vmax=vmax)
    ax.set_xlabel(LABELS[i]); ax.set_ylabel(LABELS[j]); ax.set_title(title, fontsize=8)
    fig.colorbar(pm, ax=ax, fraction=0.046, pad=0.04)

def draw1d(ax, k):
    b = np.linspace(*ranges[k], 51)
    Hd, e = np.histogram(data[:, k], bins=b)
    Hf, _ = np.histogram(U[:, k], bins=b, weights=dU); Hf = Hf * Hd.sum() / Hf.sum()
    c = 0.5 * (e[:-1] + e[1:])
    ax.errorbar(c, Hd, yerr=np.sqrt(Hd), fmt="k.", ms=3, capsize=0, label="Data")
    ax.plot(c, Hf, "r-", lw=1.5, label="Fit")
    ax.set_xlabel(LABELS[k]); ax.set_ylabel("Entries"); ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9)

NCOLS = 4

def make_fig(pairs, fname, title, top1d=None, floor=2.0):
    npan = 2 * len(pairs)
    corr_rows = (npan + NCOLS - 1) // NCOLS
    nrows = corr_rows + (1 if top1d else 0)
    fig = plt.figure(figsize=(20, 4 * nrows))
    gs = fig.add_gridspec(nrows, NCOLS)
    r0 = 0
    if top1d:
        for c, k in enumerate(top1d):
            draw1d(fig.add_subplot(gs[0, c]), k)
        r0 = 1
    slot = 0
    for (i, j) in pairs:
        Hd, bx, by = hist2d_norm(data, None, i, j)
        Hf, _, _ = hist2d_norm(U, dU, i, j)
        vmax = max(floor, float(np.percentile(Hd[Hd > 0], 99)))
        for H, tag in [(Hd, "Data"), (Hf, "Fit")]:
            ax = fig.add_subplot(gs[r0 + slot // NCOLS, slot % NCOLS])
            draw2d(ax, H, bx, by, vmax, i, j, f"{tag} {LABELS[i]}-{LABELS[j]}", fig)
            slot += 1
    fig.suptitle(title, fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(os.path.join(HERE, fname), dpi=110); plt.close()
    print(f"Saved {fname}")

# Split cleanly BY OBSERVABLE (clearer than the paper's sequential page-packing):
#   Fig.13 = m' vs all; Fig.14 = theta' vs all; Fig.15 = mD vs all.
# Fig.13: m' vs (theta', mD, params), with 1D obs on top
make_fig([(0, 1), (0, 2)] + [(0, NOBS + p) for p in range(NPAR)],
         "bg_fig13.png",
         "Fig.13 -- Bg joint ANN: m', theta', mD (1D) + m' correlations (Data vs Fit)",
         top1d=[0, 1, 2])
# Fig.14: theta' vs (mD, params)
make_fig([(1, 2)] + [(1, NOBS + p) for p in range(NPAR)],
         "bg_fig14.png",
         "Fig.14 -- Bg joint ANN: theta' correlations (Data vs Fit)", floor=1.5)
# Fig.15: mD vs params
make_fig([(2, NOBS + p) for p in range(NPAR)],
         "bg_fig15.png",
         "Fig.15 -- Bg joint ANN: mD correlations (Data vs Fit)", floor=1.5)
