"""
Plot ANN density estimation result (Fig 5 style).

Loads trained weights from eff_train_2d.npy, evaluates the ANN density,
and plots:
  (a) 2D contour of density in (m', theta')
  (b) m' projection: test data errorbar + ANN curve
  (c) theta' projection: test data errorbar + ANN curve

Also computes chi2 on both train and test sets (50x50 bins).
"""
import numpy as np
import uproot
import matplotlib.pyplot as plt
import tensorflow as tf

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import amplitf.interface as atfi
atfi.set_single_precision()

import tfa.neural_nets as tfn

# -------- style ----------
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "xtick.direction": "in",
    "ytick.direction": "in",
})


def load_ann(npyfile):
    """Load trained ANN weights and return (model_func, scale, ranges)."""
    saved = np.load(npyfile, allow_pickle=True)
    scale = saved[0]
    ranges = saved[1]
    weights_np = saved[2]
    biases_np = saved[3]

    weights = [tf.constant(w, dtype=atfi.fptype()) for w in weights_np]
    biases = [tf.constant(b, dtype=atfi.fptype()) for b in biases_np]

    def model(x):
        return tfn.multilayer_perceptron(x, ranges, weights, biases) + 1e-20

    return model, scale, ranges


def eval_on_grid(model, scale, fine=200):
    """Evaluate normalised ANN density on a fine grid."""
    mf = np.linspace(0, 1, fine).astype(np.float32)
    tf_arr = np.linspace(0, 1, fine).astype(np.float32)
    Mg, Tg = np.meshgrid(mf, tf_arr, indexing="ij")
    grid = np.stack([Mg.ravel(), Tg.ravel()], axis=1)

    pdf = model(tf.constant(grid)).numpy()
    pdf_2d = pdf.reshape(fine, fine)

    # Normalise to average density = 1
    pdf_2d = pdf_2d / pdf_2d.mean()

    return mf, tf_arr, pdf_2d


def compute_chi2(model, scale, m, t, bins=50):
    """Compute Pearson chi2 on 50x50 histogram."""
    H, xedges, yedges = np.histogram2d(m, t, bins=bins, range=[[0, 1], [0, 1]])
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])

    Mc, Tc = np.meshgrid(xc, yc, indexing="ij")
    grid = np.stack([Mc.ravel().astype(np.float32),
                     Tc.ravel().astype(np.float32)], axis=1)

    mu = model(tf.constant(grid)).numpy()
    mu *= H.sum() / mu.sum()
    mu_2d = mu.reshape(H.shape)

    valid = mu_2d > 0
    chi2 = np.sum((H[valid] - mu_2d[valid])**2 / mu_2d[valid])
    return chi2


def main():
    bins_proj = 100

    # ---- load ANN ----
    model, scale, ranges = load_ann("eff_train_2d.npy")

    # ---- load data ----
    f_train = uproot.open("eff_toy_4e6.root")
    tree_train = f_train[f_train.keys()[0]]
    m_train = tree_train["mprime"].array(library="np")
    t_train = tree_train["thetaprime"].array(library="np")

    f_test = uproot.open("eff_toy_1e5.root")
    tree_test = f_test[f_test.keys()[0]]
    m_test = tree_test["mprime"].array(library="np")
    t_test = tree_test["thetaprime"].array(library="np")

    # ---- chi2 ----
    chi2_train = compute_chi2(model, scale, m_train, t_train)
    chi2_test = compute_chi2(model, scale, m_test, t_test)
    print(f"chi2 (train, 50x50) = {chi2_train:.1f}")
    print(f"chi2 (test,  50x50) = {chi2_test:.1f}")

    # ---- evaluate on fine grid ----
    mf, tf_arr, pdf_2d = eval_on_grid(model, scale)

    # ---- 1D projections (test set, 100 bins) ----
    H_m_test, edges_m = np.histogram(m_test, bins=bins_proj, range=(0, 1))
    H_t_test, edges_t = np.histogram(t_test, bins=bins_proj, range=(0, 1))
    centers_m = 0.5 * (edges_m[:-1] + edges_m[1:])
    centers_t = 0.5 * (edges_t[:-1] + edges_t[1:])
    width_m = edges_m[1] - edges_m[0]
    width_t = edges_t[1] - edges_t[0]

    # ANN projection: average fine 2D grid along the other axis
    pdf_m_proj = pdf_2d.mean(axis=1)   # average over theta'
    pdf_t_proj = pdf_2d.mean(axis=0)   # average over m'

    eps_m_interp = np.interp(centers_m, mf, pdf_m_proj)
    eps_t_interp = np.interp(centers_t, tf_arr, pdf_t_proj)

    scale_m = H_m_test.sum() / eps_m_interp.sum()
    scale_t = H_t_test.sum() / eps_t_interp.sum()
    fit_m_line = eps_m_interp * scale_m
    fit_t_line = eps_t_interp * scale_t

    # ---- plot ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) 2D density (contour)
    ax = axes[0]
    vmax = np.ceil(pdf_2d.max() * 10) / 10
    levels = np.linspace(0.0, vmax, 21)
    cf = ax.contourf(mf, tf_arr, pdf_2d.T, levels=levels, cmap="afmhot_r")
    ax.set_xlabel("m'")
    ax.set_ylabel("\u03b8'")
    ax.set_title("(a)")
    fig.colorbar(cf, ax=ax, label="\u03b5(m', \u03b8')")
    ax.minorticks_on()
    ax.tick_params(which="both", top=True, right=True)

    # (b) m' projection
    ax = axes[1]
    ax.errorbar(centers_m, H_m_test, yerr=np.sqrt(H_m_test),
                fmt="k.", ms=3, capsize=0, label="Test data")
    ax.plot(centers_m, fit_m_line, "r-", lw=1.5, label="ANN fit")
    ax.set_xlabel("m'")
    ax.set_ylabel(f"Entries / ({width_m:.2f})")
    ax.set_title("(b)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=10)
    ax.minorticks_on()
    ax.tick_params(which="both", top=True, right=True)

    # (c) theta' projection
    ax = axes[2]
    ax.errorbar(centers_t, H_t_test, yerr=np.sqrt(H_t_test),
                fmt="k.", ms=3, capsize=0, label="Test data")
    ax.plot(centers_t, fit_t_line, "r-", lw=1.5, label="ANN fit")
    ax.set_xlabel("\u03b8'")
    ax.set_ylabel(f"Entries / ({width_t:.2f})")
    ax.set_title("(c)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=10)
    ax.minorticks_on()
    ax.tick_params(which="both", top=True, right=True)

    fig.suptitle(
        f"ANN density estimation   "
        f"\u03c7\u00b2(train) = {chi2_train:.0f}   "
        f"\u03c7\u00b2(test) = {chi2_test:.0f}",
        fontsize=14,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("fig5_ann.png", dpi=300)
    print(f"Saved fig5_ann.png")
    plt.close()


if __name__ == "__main__":
    main()
