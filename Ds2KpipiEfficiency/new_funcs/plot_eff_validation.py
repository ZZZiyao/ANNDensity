"""
Reproduce Fig.12 (Appendix A): model-assisted EFFICIENCY joint ANN validation,
P(m', theta', Theta_5). Layout matches the paper:
  top row : 1D m', 1D theta', then m'-theta' 2D as Data | Fit;
  rows 1-5: m' and theta' vs each of the 5 model parameters, each as Data | Fit.
"Data" = training toy; "Fit" = ANN density (uniform 7D sample weighted by eps).
2D normalised so average = 1; colour scale auto per pair (matched Data/Fit).
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

MODEL = os.path.join(HERE, "eff_joint_train_gpu.npy")
TOY   = os.path.join(HERE, "eff_joint_1e6.root")
N_UNIF = int(float(os.environ.get("VAL_UNIF", 2000000)))

ann = np.load(MODEL, allow_pickle=True)
ranges = ann[1]
weights = [tf.constant(w, dtype=atfi.fptype()) for w in ann[2]]
biases = [tf.constant(b, dtype=atfi.fptype()) for b in ann[3]]
def density(x):
    return tfn.multilayer_perceptron(np.asarray(x, np.float32), ranges, weights, biases).numpy()

VARS = observables_toys + [p[0] for p in parameters_list]      # 7 names
LABELS = [r"$m'$", r"$\theta'$"] + [p[1] for p in parameters_list]
NOBS = len(observables_toys)                                   # 2
NPAR = len(parameters_list)                                    # 5
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

def pair2d(ax_d, ax_f, i, j, fig, floor=1.5):
    Hd, bx, by = hist2d_norm(data, None, i, j)
    Hf, _, _ = hist2d_norm(U, dU, i, j)
    vmax = max(floor, float(np.percentile(Hd[Hd > 0], 99)))
    draw2d(ax_d, Hd, bx, by, vmax, i, j, f"Data {LABELS[i]}-{LABELS[j]}", fig)
    draw2d(ax_f, Hf, bx, by, vmax, i, j, f"Fit {LABELS[i]}-{LABELS[j]}", fig)

def draw1d(ax, k):
    b = np.linspace(*ranges[k], 51)
    Hd, e = np.histogram(data[:, k], bins=b)
    Hf, _ = np.histogram(U[:, k], bins=b, weights=dU); Hf = Hf * Hd.sum() / Hf.sum()
    c = 0.5 * (e[:-1] + e[1:])
    ax.errorbar(c, Hd, yerr=np.sqrt(Hd), fmt="k.", ms=3, capsize=0, label="Data")
    ax.plot(c, Hf, "r-", lw=1.5, label="Fit")
    ax.set_xlabel(LABELS[k]); ax.set_ylabel("Entries"); ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9)

# ---------- Fig.12: (1+NPAR) rows x 4 cols ----------
fig = plt.figure(figsize=(20, 4 * (1 + NPAR)))
gs = fig.add_gridspec(1 + NPAR, 4)
# top row: 1D m', 1D theta', m'-theta' Data, m'-theta' Fit
draw1d(fig.add_subplot(gs[0, 0]), 0)
draw1d(fig.add_subplot(gs[0, 1]), 1)
pair2d(fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 3]), 0, 1, fig, floor=1.5)
# rows 1..NPAR: m', theta' vs each param, Data|Fit
for p in range(NPAR):
    pj = NOBS + p
    r = 1 + p
    pair2d(fig.add_subplot(gs[r, 0]), fig.add_subplot(gs[r, 1]), 0, pj, fig)   # m' vs param
    pair2d(fig.add_subplot(gs[r, 2]), fig.add_subplot(gs[r, 3]), 1, pj, fig)   # theta' vs param
fig.suptitle("Fig.12 -- Eff joint ANN: m',theta' (1D) + m'-theta' and obs-parameter 2D (Data vs Fit)", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.985]); plt.savefig(os.path.join(HERE, "eff_fig12.png"), dpi=110); plt.close()
print("Saved eff_fig12.png")
