"""
Plot Fig 6: GP background density estimation in sideband regions.

Row 1: 2D mD vs m'  (left: data, right: GP prediction)
Row 2: 2D theta' vs m'  (left: data, right: GP prediction)
Row 3: 2D mD vs theta'  (left: data, right: GP prediction)
Row 4: 1D projections of m', theta', mD

Also plots Fig 7: signal region extrapolation at mD = 1.97 GeV.
"""
import os
import numpy as np
import uproot
import GPy
import pickle
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# -------- style ----------
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

# ---- sideband definitions ----
MD_LO, MD_S1, MD_S2, MD_HI = 1.77, 1.92, 2.02, 2.17


def rebuild_model(pkl_file):
    """Load saved GP data and rebuild model."""
    with open(pkl_file, "rb") as fin:
        saved = pickle.load(fin)

    X = saved["X"]
    Y = saved["Y"]

    kern = GPy.kern.Matern52(input_dim=3, ARD=True)
    kern_const = GPy.kern.Bias(input_dim=3)
    kern_linear = GPy.kern.Linear(input_dim=1, active_dims=[2])

    model = GPy.models.GPRegression(X, Y, kernel=kern + kern_const + kern_linear)
    model.update_model(False)
    model[:] = saved["model_param_array"]
    model.update_model(True)

    return model, saved


def main():
    test_file = os.path.join(HERE, "0.20.41e5_test.root")

    # ---- load GP model ----
    model, saved = rebuild_model(os.path.join(HERE, "gp_3d_bg_model.pkl"))
    md_bin_centers = saved["md_bin_centers"]
    dp_centers_m = saved["dp_centers_m"]
    dp_centers_t = saved["dp_centers_t"]

    # ---- load test data (sidebands only) ----
    f = uproot.open(test_file)
    tree = f[f.keys()[0]]
    arr = tree.arrays(["mprime", "thetaprime", "md"], library="np")
    m = arr["mprime"]
    t = arr["thetaprime"]
    md = arr["md"]

    lower_sb = (md >= MD_LO) & (md < MD_S1)
    upper_sb = (md > MD_S2) & (md <= MD_HI)
    sb_mask = lower_sb | upper_sb

    m_sb = m[sb_mask]
    t_sb = t[sb_mask]
    md_sb = md[sb_mask]

    # ---- GP prediction on fine grid in sideband mD range ----
    fine_dp = 50
    fine_md = 30
    mf = np.linspace(0, 1, fine_dp)
    tf_arr = np.linspace(0, 1, fine_dp)

    # mD values covering both sidebands
    md_lower = np.linspace(MD_LO, MD_S1, fine_md // 2)
    md_upper = np.linspace(MD_S2, MD_HI, fine_md // 2)
    md_fine = np.concatenate([md_lower, md_upper])

    # Build prediction grid (fine_md x fine_dp x fine_dp)
    Xpred_list = []
    for md_val in md_fine:
        for m_val in mf:
            for t_val in tf_arr:
                Xpred_list.append([m_val, t_val, md_val])
    Xpred = np.array(Xpred_list)

    print("Predicting on sideband grid ...")
    mu_pred, _ = model.predict(Xpred)
    mu_pred = np.clip(mu_pred.ravel(), 0, None)
    mu_3d = mu_pred.reshape(len(md_fine), fine_dp, fine_dp)

    # ========== Fig 6 ==========
    fig, axes = plt.subplots(4, 3, figsize=(18, 22))
    fig.suptitle("GP background estimation (sidebands)", fontsize=16, y=0.98)

    # Split predictions into lower/upper sideband
    n_lower = fine_md // 2
    mu_lower = mu_3d[:n_lower]
    mu_upper = mu_3d[n_lower:]

    # --- Row 3: mD vs m' ---
    # Data
    ax = axes[2, 0]
    ax.hist2d(m_sb, md_sb, bins=[50, 30],
              range=[[0, 1], [MD_LO, MD_HI]], cmap="afmhot_r")
    ax.set_xlabel("m'"); ax.set_ylabel("$m_D$ (GeV)")
    ax.set_title("Data: $m_D$ vs $m'$")

    # GP prediction: project over theta', plot lower and upper separately
    ax = axes[2, 1]
    proj_lower = mu_lower.sum(axis=2)
    proj_upper = mu_upper.sum(axis=2)
    ax.pcolormesh(mf, md_lower, proj_lower, shading="auto", cmap="afmhot_r")
    ax.pcolormesh(mf, md_upper, proj_upper, shading="auto", cmap="afmhot_r")
    ax.set_xlim(0, 1); ax.set_ylim(MD_LO, MD_HI)
    ax.set_xlabel("m'"); ax.set_ylabel("$m_D$ (GeV)")
    ax.set_title("GP: $m_D$ vs $m'$")

    axes[2, 2].axis("off")

    # --- Row 2: theta' vs m' ---
    ax = axes[1, 0]
    ax.hist2d(m_sb, t_sb, bins=[50, 50],
              range=[[0, 1], [0, 1]], cmap="afmhot_r")
    ax.set_xlabel("m'"); ax.set_ylabel("$\\theta'$")
    ax.set_title("Data: $\\theta'$ vs $m'$")

    ax = axes[1, 1]
    proj_t_m = mu_3d.sum(axis=0)  # sum over mD
    ax.pcolormesh(mf, tf_arr, proj_t_m.T, shading="auto", cmap="afmhot_r")
    ax.set_xlabel("m'"); ax.set_ylabel("$\\theta'$")
    ax.set_title("GP: $\\theta'$ vs $m'$")

    axes[1, 2].axis("off")

    # --- Row 1: mD vs theta' ---
    ax = axes[0, 0]
    ax.hist2d(t_sb, md_sb, bins=[50, 30],
              range=[[0, 1], [MD_LO, MD_HI]], cmap="afmhot_r")
    ax.set_xlabel("$\\theta'$"); ax.set_ylabel("$m_D$ (GeV)")
    ax.set_title("Data: $m_D$ vs $\\theta'$")

    ax = axes[0, 1]
    proj_lower_t = mu_lower.sum(axis=1)
    proj_upper_t = mu_upper.sum(axis=1)
    ax.pcolormesh(tf_arr, md_lower, proj_lower_t, shading="auto", cmap="afmhot_r")
    ax.pcolormesh(tf_arr, md_upper, proj_upper_t, shading="auto", cmap="afmhot_r")
    ax.set_xlim(0, 1); ax.set_ylim(MD_LO, MD_HI)
    ax.set_xlabel("$\\theta'$"); ax.set_ylabel("$m_D$ (GeV)")
    ax.set_title("GP: $m_D$ vs $\\theta'$")

    axes[0, 2].axis("off")

    # --- Row 4: 1D projections (errorbar style) ---
    # m'
    ax = axes[3, 0]
    H_m_data, edges_m = np.histogram(m_sb, bins=100, range=(0, 1))
    centers_m_1d = 0.5 * (edges_m[:-1] + edges_m[1:])
    ax.errorbar(centers_m_1d, H_m_data, yerr=np.sqrt(H_m_data),
                fmt="k.", ms=3, capsize=0, label="Data")
    gp_m = mu_3d.sum(axis=(0, 2))
    gp_m_interp = np.interp(centers_m_1d, mf, gp_m)
    gp_m_interp *= H_m_data.sum() / gp_m_interp.sum()
    ax.plot(centers_m_1d, gp_m_interp, "r-", lw=1.5, label="GP")
    ax.set_xlabel("m'"); ax.set_ylabel("Entries / (0.01)")
    ax.legend(frameon=False)
    ax.set_ylim(bottom=0)

    # theta'
    ax = axes[3, 1]
    H_t_data, edges_t = np.histogram(t_sb, bins=100, range=(0, 1))
    centers_t_1d = 0.5 * (edges_t[:-1] + edges_t[1:])
    ax.errorbar(centers_t_1d, H_t_data, yerr=np.sqrt(H_t_data),
                fmt="k.", ms=3, capsize=0, label="Data")
    gp_t = mu_3d.sum(axis=(0, 1))
    gp_t_interp = np.interp(centers_t_1d, tf_arr, gp_t)
    gp_t_interp *= H_t_data.sum() / gp_t_interp.sum()
    ax.plot(centers_t_1d, gp_t_interp, "r-", lw=1.5, label="GP")
    ax.set_xlabel("$\\theta'$"); ax.set_ylabel("Entries / (0.01)")
    ax.legend(frameon=False)
    ax.set_ylim(bottom=0)

    # mD: fine-binned errorbar in sidebands + GP curve on sidebands only
    ax = axes[3, 2]
    md_bin_width = 0.005
    md_train_width = 0.05  # mD bin width used in GP training (fit_gp_3d_bg.py)
    md_edges_lo = np.arange(MD_LO, MD_S1 + 1e-12, md_bin_width)
    md_edges_hi = np.arange(MD_S2, MD_HI + 1e-12, md_bin_width)

    H_md_lo, _ = np.histogram(md_sb, bins=md_edges_lo)
    H_md_hi, _ = np.histogram(md_sb, bins=md_edges_hi)
    centers_lo = 0.5 * (md_edges_lo[:-1] + md_edges_lo[1:])
    centers_hi = 0.5 * (md_edges_hi[:-1] + md_edges_hi[1:])

    ax.errorbar(centers_lo, H_md_lo, yerr=np.sqrt(H_md_lo),
                fmt="k.", ms=3, capsize=0, label="Data")
    ax.errorbar(centers_hi, H_md_hi, yerr=np.sqrt(H_md_hi),
                fmt="k.", ms=3, capsize=0)

    # GP curve at each fine bin center. sum_(m',θ') GP ≈ 0.05 * rho_1d(mD)
    # (expected count per 0.05 GeV mD slice). Multiply by md_bin_width/md_train_width
    # to convert to "entries per md_bin_width GeV".
    dp_centers_m = saved["dp_centers_m"]
    dp_centers_t = saved["dp_centers_t"]
    scale = md_bin_width / md_train_width  # = 0.1

    def gp_md_curve(md_vals):
        out = np.zeros(len(md_vals))
        for idx, md_val in enumerate(md_vals):
            Xq = np.array([[mc, tc, md_val]
                           for mc in dp_centers_m for tc in dp_centers_t])
            mu_q, _ = model.predict(Xq)
            out[idx] = np.clip(mu_q, 0, None).sum()
        return out * scale

    gp_lo = gp_md_curve(centers_lo)
    gp_hi = gp_md_curve(centers_hi)

    ax.plot(centers_lo, gp_lo, "r-", lw=1.5, label="GP")
    ax.plot(centers_hi, gp_hi, "r-", lw=1.5)
    ax.axvline(MD_S1, color="red", ls="--", lw=1)
    ax.axvline(MD_S2, color="red", ls="--", lw=1)
    ax.set_xlabel("$m_D$ (GeV)")
    ax.set_ylabel(f"Entries / ({md_bin_width} GeV)")
    ax.set_xlim(MD_LO, MD_HI)
    ax.legend(frameon=False)
    ax.set_ylim(bottom=0)

    for row in axes:
        for ax in row:
            if not ax.axison:
                continue
            ax.minorticks_on()
            ax.tick_params(which="both", top=True, right=True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(HERE, "fig6_gp_bg.png"), dpi=200)
    print("Saved fig6_gp_bg.png")
    plt.close()

    # ========== Fig 7: signal region extrapolation ==========
    md_signal = 1.97  # signal region center

    Xsig = []
    for m_val in mf:
        for t_val in tf_arr:
            Xsig.append([m_val, t_val, md_signal])
    Xsig = np.array(Xsig)

    print("Predicting signal region ...")
    mu_sig, _ = model.predict(Xsig)
    mu_sig = np.clip(mu_sig.ravel(), 0, None)
    mu_sig_2d = mu_sig.reshape(fine_dp, fine_dp)
    mu_sig_2d = mu_sig_2d / mu_sig_2d.mean()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"GP signal region extrapolation ($m_D$ = {md_signal} GeV)", fontsize=14)

    # (a) 2D
    ax = axes[0]
    vmax = np.ceil(mu_sig_2d.max() * 10) / 10
    levels = np.linspace(0, vmax, 21)
    cf = ax.contourf(mf, tf_arr, mu_sig_2d.T, levels=levels, cmap="afmhot_r")
    ax.set_xlabel("m'"); ax.set_ylabel("$\\theta'$")
    ax.set_title("(a)")
    fig.colorbar(cf, ax=ax)
    ax.minorticks_on()
    ax.tick_params(which="both", top=True, right=True)

    # ---- load test data: signal region AND sidebands (for comparison) ----
    f_test = uproot.open(test_file)
    tree_test = f_test[f_test.keys()[0]]
    arr_test = tree_test.arrays(["mprime", "thetaprime", "md"], library="np")

    sig_mask = (arr_test["md"] >= MD_S1) & (arr_test["md"] <= MD_S2)
    sb_mask_test = ((arr_test["md"] >= MD_LO) & (arr_test["md"] < MD_S1)) | \
                   ((arr_test["md"] > MD_S2) & (arr_test["md"] <= MD_HI))
    m_sig_data = arr_test["mprime"][sig_mask]
    t_sig_data = arr_test["thetaprime"][sig_mask]
    m_sb_data  = arr_test["mprime"][sb_mask_test]
    t_sb_data  = arr_test["thetaprime"][sb_mask_test]

    # (b) m' projection: Signal region data + Sidebands (shape ref) + GP extrap
    ax = axes[1]
    H_m, edges_m = np.histogram(m_sig_data, bins=100, range=(0, 1))
    centers_m = 0.5 * (edges_m[:-1] + edges_m[1:])

    # GP extrapolation at m_D = m_signal, scaled to signal-region total
    gp_m_sig = mu_sig_2d.sum(axis=1)
    gp_m_sig_interp = np.interp(centers_m, mf, gp_m_sig)
    gp_m_sig_interp *= H_m.sum() / gp_m_sig_interp.sum()

    # Sideband m' shape, scaled to signal-region total (the "naive" reference)
    H_m_sb, _ = np.histogram(m_sb_data, bins=edges_m)
    sb_scale_m = H_m.sum() / max(H_m_sb.sum(), 1)
    H_m_sb_s = H_m_sb * sb_scale_m
    err_m_sb_s = np.sqrt(H_m_sb) * sb_scale_m

    ax.errorbar(centers_m, H_m, yerr=np.sqrt(H_m),
                fmt="k.", ms=3, capsize=0, label="Signal region")
    ax.errorbar(centers_m, H_m_sb_s, yerr=err_m_sb_s,
                fmt="none", ecolor="blue", elinewidth=0.6,
                marker="s", markersize=3, mfc="none", mec="blue",
                label="Sidebands")
    ax.plot(centers_m, gp_m_sig_interp, "r-", lw=1.5, label="Fit result")
    ax.set_xlabel("m'"); ax.set_ylabel("Entries / (0.01)")
    ax.set_title("(b)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=10)
    ax.minorticks_on()
    ax.tick_params(which="both", top=True, right=True)

    # (c) theta' projection: Signal region data + Sidebands + GP extrap
    ax = axes[2]
    H_t, edges_t = np.histogram(t_sig_data, bins=100, range=(0, 1))
    centers_t = 0.5 * (edges_t[:-1] + edges_t[1:])

    gp_t_sig = mu_sig_2d.sum(axis=0)
    gp_t_sig_interp = np.interp(centers_t, tf_arr, gp_t_sig)
    gp_t_sig_interp *= H_t.sum() / gp_t_sig_interp.sum()

    H_t_sb, _ = np.histogram(t_sb_data, bins=edges_t)
    sb_scale_t = H_t.sum() / max(H_t_sb.sum(), 1)
    H_t_sb_s = H_t_sb * sb_scale_t
    err_t_sb_s = np.sqrt(H_t_sb) * sb_scale_t

    ax.errorbar(centers_t, H_t, yerr=np.sqrt(H_t),
                fmt="k.", ms=3, capsize=0, label="Signal region")
    ax.errorbar(centers_t, H_t_sb_s, yerr=err_t_sb_s,
                fmt="none", ecolor="blue", elinewidth=0.6,
                marker="s", markersize=3, mfc="none", mec="blue",
                label="Sidebands")
    ax.plot(centers_t, gp_t_sig_interp, "r-", lw=1.5, label="Fit result")
    ax.set_xlabel("$\\theta'$"); ax.set_ylabel("Entries / (0.01)")
    ax.set_title("(c)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=10)
    ax.minorticks_on()
    ax.tick_params(which="both", top=True, right=True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(HERE, "fig7_gp_bg_signal.png"), dpi=200)
    print("Saved fig7_gp_bg_signal.png")
    plt.close()


if __name__ == "__main__":
    main()
