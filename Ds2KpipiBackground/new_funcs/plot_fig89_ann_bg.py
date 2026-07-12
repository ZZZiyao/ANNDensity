"""
Plot Fig 8 (sideband fit) and Fig 9 (signal region extrapolation)
for 3D ANN background density estimation.

Loads trained weights from train_3d.npy.
"""
import numpy as np
import uproot
import matplotlib.pyplot as plt
import tensorflow as tf

import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import amplitf.interface as atfi
atfi.set_single_precision()

import tfa.neural_nets as tfn

# -------- style ----------
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

MD_LO, MD_S1, MD_S2, MD_HI = 1.77, 1.92, 2.02, 2.17


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


def eval_2d_at_md(model, md_val, fine=50):
    """Evaluate ANN at fixed mD on a fine (m', theta') grid."""
    mf = np.linspace(0, 1, fine).astype(np.float32)
    tf_arr = np.linspace(0, 1, fine).astype(np.float32)
    Mg, Tg = np.meshgrid(mf, tf_arr, indexing="ij")
    md_col = np.full(Mg.size, md_val, dtype=np.float32)
    grid = np.stack([Mg.ravel(), Tg.ravel(), md_col], axis=1)

    pdf = model(tf.constant(grid)).numpy()
    return mf, tf_arr, pdf.reshape(fine, fine)


def eval_3d_sideband(model, fine_dp=50, fine_md=15):
    """Evaluate ANN on sideband mD grid."""
    mf = np.linspace(0, 1, fine_dp).astype(np.float32)
    tf_arr = np.linspace(0, 1, fine_dp).astype(np.float32)
    md_lower = np.linspace(MD_LO, MD_S1, fine_md).astype(np.float32)
    md_upper = np.linspace(MD_S2, MD_HI, fine_md).astype(np.float32)
    md_fine = np.concatenate([md_lower, md_upper])

    mu_3d = np.zeros((len(md_fine), fine_dp, fine_dp))
    for k, md_val in enumerate(md_fine):
        _, _, mu_2d = eval_2d_at_md(model, md_val, fine_dp)
        mu_3d[k] = mu_2d

    return mf, tf_arr, md_lower, md_upper, md_fine, mu_3d


def main():
    test_file = os.path.join(HERE, "0.20.41e5_test.root")

    # ---- load ANN ----
    model, scale, ranges = load_ann(os.path.join(HERE, "train_3d.npy"))

    # ---- load test data ----
    f_test = uproot.open(test_file)
    tree_test = f_test[f_test.keys()[0]]
    arr = tree_test.arrays(["mprime", "thetaprime", "md"], library="np")
    m_all = arr["mprime"]
    t_all = arr["thetaprime"]
    md_all = arr["md"]

    # Sideband masks
    lower_sb = (md_all >= MD_LO) & (md_all < MD_S1)
    upper_sb = (md_all > MD_S2) & (md_all <= MD_HI)
    sb_mask = lower_sb | upper_sb
    sig_mask = (md_all >= MD_S1) & (md_all <= MD_S2)

    m_sb = m_all[sb_mask]
    t_sb = t_all[sb_mask]
    md_sb = md_all[sb_mask]

    # ---- evaluate on sideband grid ----
    print("Evaluating on sideband grid ...")
    mf, tf_arr, md_lower, md_upper, md_fine, mu_3d = eval_3d_sideband(model)

    n_lower = len(md_lower)
    mu_low = mu_3d[:n_lower]
    mu_up = mu_3d[n_lower:]

    # ========== Fig 8: sideband fit ==========
    fig, axes = plt.subplots(4, 3, figsize=(18, 22))
    fig.suptitle("ANN background estimation (sidebands)", fontsize=16, y=0.98)

    # Row 1: mD vs m'
    ax = axes[0, 0]
    ax.hist2d(m_sb, md_sb, bins=[50, 30],
              range=[[0, 1], [MD_LO, MD_HI]], cmap="afmhot_r")
    ax.set_xlabel("m'"); ax.set_ylabel("$m_D$ (GeV)")
    ax.set_title("Data: $m_D$ vs $m'$")

    ax = axes[0, 1]
    proj_lo = mu_low.sum(axis=2)
    proj_up = mu_up.sum(axis=2)
    ax.pcolormesh(mf, md_lower, proj_lo, shading="auto", cmap="afmhot_r")
    ax.pcolormesh(mf, md_upper, proj_up, shading="auto", cmap="afmhot_r")
    ax.set_xlim(0, 1); ax.set_ylim(MD_LO, MD_HI)
    ax.set_xlabel("m'"); ax.set_ylabel("$m_D$ (GeV)")
    ax.set_title("ANN: $m_D$ vs $m'$")

    axes[0, 2].axis("off")

    # Row 2: theta' vs m'
    ax = axes[1, 0]
    ax.hist2d(m_sb, t_sb, bins=[50, 50],
              range=[[0, 1], [0, 1]], cmap="afmhot_r")
    ax.set_xlabel("m'"); ax.set_ylabel("$\\theta'$")
    ax.set_title("Data: $\\theta'$ vs $m'$")

    ax = axes[1, 1]
    proj_t_m = mu_3d.sum(axis=0)
    ax.pcolormesh(mf, tf_arr, proj_t_m.T, shading="auto", cmap="afmhot_r")
    ax.set_xlabel("m'"); ax.set_ylabel("$\\theta'$")
    ax.set_title("ANN: $\\theta'$ vs $m'$")

    axes[1, 2].axis("off")

    # Row 3: mD vs theta'
    ax = axes[2, 0]
    ax.hist2d(t_sb, md_sb, bins=[50, 30],
              range=[[0, 1], [MD_LO, MD_HI]], cmap="afmhot_r")
    ax.set_xlabel("$\\theta'$"); ax.set_ylabel("$m_D$ (GeV)")
    ax.set_title("Data: $m_D$ vs $\\theta'$")

    ax = axes[2, 1]
    proj_lo_t = mu_low.sum(axis=1)
    proj_up_t = mu_up.sum(axis=1)
    ax.pcolormesh(tf_arr, md_lower, proj_lo_t, shading="auto", cmap="afmhot_r")
    ax.pcolormesh(tf_arr, md_upper, proj_up_t, shading="auto", cmap="afmhot_r")
    ax.set_xlim(0, 1); ax.set_ylim(MD_LO, MD_HI)
    ax.set_xlabel("$\\theta'$"); ax.set_ylabel("$m_D$ (GeV)")
    ax.set_title("ANN: $m_D$ vs $\\theta'$")

    axes[2, 2].axis("off")

    # Row 4: 1D projections
    # m'
    ax = axes[3, 0]
    H_m, edges_m = np.histogram(m_sb, bins=100, range=(0, 1))
    centers_m = 0.5 * (edges_m[:-1] + edges_m[1:])
    ax.errorbar(centers_m, H_m, yerr=np.sqrt(H_m),
                fmt="k.", ms=3, capsize=0, label="Data")
    gp_m = mu_3d.sum(axis=(0, 2))
    gp_m_interp = np.interp(centers_m, mf, gp_m)
    gp_m_interp *= H_m.sum() / gp_m_interp.sum()
    ax.plot(centers_m, gp_m_interp, "r-", lw=1.5, label="ANN")
    ax.set_xlabel("m'"); ax.set_ylabel("Entries / (0.01)")
    ax.legend(frameon=False); ax.set_ylim(bottom=0)

    # theta'
    ax = axes[3, 1]
    H_t, edges_t = np.histogram(t_sb, bins=100, range=(0, 1))
    centers_t = 0.5 * (edges_t[:-1] + edges_t[1:])
    ax.errorbar(centers_t, H_t, yerr=np.sqrt(H_t),
                fmt="k.", ms=3, capsize=0, label="Data")
    gp_t = mu_3d.sum(axis=(0, 1))
    gp_t_interp = np.interp(centers_t, tf_arr, gp_t)
    gp_t_interp *= H_t.sum() / gp_t_interp.sum()
    ax.plot(centers_t, gp_t_interp, "r-", lw=1.5, label="ANN")
    ax.set_xlabel("$\\theta'$"); ax.set_ylabel("Entries / (0.01)")
    ax.legend(frameon=False); ax.set_ylim(bottom=0)

    # mD: fine-binned errorbar in sidebands + ANN curve on sidebands only
    ax = axes[3, 2]
    md_bin_width = 0.005
    md_edges_lo = np.arange(MD_LO, MD_S1 + 1e-12, md_bin_width)
    md_edges_hi = np.arange(MD_S2, MD_HI + 1e-12, md_bin_width)
    H_md_lo, _ = np.histogram(md_sb, bins=md_edges_lo)
    H_md_hi, _ = np.histogram(md_sb, bins=md_edges_hi)
    cen_lo = 0.5 * (md_edges_lo[:-1] + md_edges_lo[1:])
    cen_hi = 0.5 * (md_edges_hi[:-1] + md_edges_hi[1:])
    ax.errorbar(cen_lo, H_md_lo, yerr=np.sqrt(H_md_lo),
                fmt="k.", ms=3, capsize=0, label="Data")
    ax.errorbar(cen_hi, H_md_hi, yerr=np.sqrt(H_md_hi),
                fmt="k.", ms=3, capsize=0)

    # ANN evaluated at each fine bin center; sum over (m', θ') 20x20 grid
    fine_dp = 20
    mf20 = np.linspace(0, 1, fine_dp).astype(np.float32)
    tf20 = np.linspace(0, 1, fine_dp).astype(np.float32)
    Mg20, Tg20 = np.meshgrid(mf20, tf20, indexing="ij")
    mt_grid = np.stack([Mg20.ravel(), Tg20.ravel()], axis=1)

    def ann_md_curve(md_vals):
        out = np.zeros(len(md_vals))
        for idx, md_val in enumerate(md_vals):
            grid_3d = np.column_stack([
                mt_grid,
                np.full(mt_grid.shape[0], md_val, dtype=np.float32),
            ])
            pdf = model(tf.constant(grid_3d)).numpy()
            out[idx] = np.clip(pdf, 0, None).sum()
        return out

    ann_lo = ann_md_curve(cen_lo.astype(np.float32))
    ann_hi = ann_md_curve(cen_hi.astype(np.float32))

    # Single scale: match total sideband counts (ANN PDF normalisation is
    # not directly comparable to event counts, so we calibrate empirically).
    total_data = H_md_lo.sum() + H_md_hi.sum()
    total_ann = ann_lo.sum() + ann_hi.sum()
    scale = total_data / total_ann if total_ann > 0 else 1.0
    ann_lo *= scale
    ann_hi *= scale

    ax.plot(cen_lo, ann_lo, "r-", lw=1.5, label="ANN")
    ax.plot(cen_hi, ann_hi, "r-", lw=1.5)
    ax.axvline(MD_S1, color="red", ls="--", lw=1)
    ax.axvline(MD_S2, color="red", ls="--", lw=1)
    ax.set_xlabel("$m_D$ (GeV)")
    ax.set_ylabel(f"Entries / ({md_bin_width} GeV)")
    ax.set_xlim(MD_LO, MD_HI)
    ax.legend(frameon=False); ax.set_ylim(bottom=0)

    for row in axes:
        for a in row:
            if not a.axison:
                continue
            a.minorticks_on()
            a.tick_params(which="both", top=True, right=True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(HERE, "fig8_ann_bg.png"), dpi=200)
    print("Saved fig8_ann_bg.png")
    plt.close()

    # ========== Fig 9: signal region extrapolation ==========
    md_signal = 1.97
    print(f"Evaluating signal region (mD = {md_signal}) ...")
    mf_fine, tf_fine, mu_sig_2d = eval_2d_at_md(model, md_signal, fine=100)
    mu_sig_2d = np.clip(mu_sig_2d, 0, None)
    mu_sig_2d = mu_sig_2d / mu_sig_2d.mean()

    # Test data in signal region
    m_sig = m_all[sig_mask]
    t_sig = t_all[sig_mask]

    # chi2 on 50x50
    H_sig_50, xe, ye = np.histogram2d(m_sig, t_sig, bins=50, range=[[0, 1], [0, 1]])
    _, _, mu_chi2 = eval_2d_at_md(model, md_signal, fine=50)
    mu_chi2 = np.clip(mu_chi2, 0, None)
    mu_chi2 *= H_sig_50.sum() / mu_chi2.sum()
    valid = mu_chi2 > 0
    chi2 = np.sum((H_sig_50[valid] - mu_chi2[valid])**2 / mu_chi2[valid])
    print(f"chi2 (signal, 50x50) = {chi2:.1f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"ANN signal region extrapolation ($m_D$ = {md_signal} GeV)   "
        f"\u03c7\u00b2(50\u00d750) = {chi2:.1f}",
        fontsize=14)

    # (a) 2D
    ax = axes[0]
    vmax = np.ceil(mu_sig_2d.max() * 10) / 10
    levels = np.linspace(0, vmax, 21)
    cf = ax.contourf(mf_fine, tf_fine, mu_sig_2d.T, levels=levels, cmap="afmhot_r")
    ax.set_xlabel("m'"); ax.set_ylabel("$\\theta'$")
    ax.set_title("(a)")
    fig.colorbar(cf, ax=ax)
    ax.minorticks_on(); ax.tick_params(which="both", top=True, right=True)

    # Sideband projection (scaled to signal total) for comparison
    H_m_sb, _ = np.histogram(m_sb, bins=100, range=(0, 1))
    H_t_sb, _ = np.histogram(t_sb, bins=100, range=(0, 1))

    # (b) m' projection
    ax = axes[1]
    H_m_sig, edges_m = np.histogram(m_sig, bins=100, range=(0, 1))
    centers_m = 0.5 * (edges_m[:-1] + edges_m[1:])
    ax.errorbar(centers_m, H_m_sig, yerr=np.sqrt(H_m_sig),
                fmt="k.", ms=3, capsize=0, label="Signal data")
    # Sideband scaled
    H_m_sb_scaled = H_m_sb.astype(float) * H_m_sig.sum() / H_m_sb.sum()
    ax.plot(centers_m, H_m_sb_scaled, color="m", ls="--", lw=1.5, label="Sidebands")
    # ANN extrapolation
    ann_m = mu_sig_2d.sum(axis=1)
    ann_m_interp = np.interp(centers_m, mf_fine, ann_m)
    ann_m_interp *= H_m_sig.sum() / ann_m_interp.sum()
    ax.plot(centers_m, ann_m_interp, "g-", lw=2, label="ANN extrap.")
    ax.set_xlabel("m'"); ax.set_ylabel("Entries / (0.01)")
    ax.set_title("(b)"); ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=10)
    ax.minorticks_on(); ax.tick_params(which="both", top=True, right=True)

    # (c) theta' projection
    ax = axes[2]
    H_t_sig, edges_t = np.histogram(t_sig, bins=100, range=(0, 1))
    centers_t = 0.5 * (edges_t[:-1] + edges_t[1:])
    ax.errorbar(centers_t, H_t_sig, yerr=np.sqrt(H_t_sig),
                fmt="k.", ms=3, capsize=0, label="Signal data")
    # Sideband scaled
    H_t_sb_scaled = H_t_sb.astype(float) * H_t_sig.sum() / H_t_sb.sum()
    ax.plot(centers_t, H_t_sb_scaled, color="m", ls="--", lw=1.5, label="Sidebands")
    # ANN extrapolation
    ann_t = mu_sig_2d.sum(axis=0)
    ann_t_interp = np.interp(centers_t, tf_fine, ann_t)
    ann_t_interp *= H_t_sig.sum() / ann_t_interp.sum()
    ax.plot(centers_t, ann_t_interp, "g-", lw=2, label="ANN extrap.")
    ax.set_xlabel("$\\theta'$"); ax.set_ylabel("Entries / (0.01)")
    ax.set_title("(c)"); ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=10)
    ax.minorticks_on(); ax.tick_params(which="both", top=True, right=True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(HERE, "fig9_ann_bg_signal.png"), dpi=200)
    print("Saved fig9_ann_bg_signal.png")
    plt.close()


if __name__ == "__main__":
    main()
