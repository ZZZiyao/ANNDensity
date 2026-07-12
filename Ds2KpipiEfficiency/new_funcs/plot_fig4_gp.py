"""
Gaussian process fit to efficiency using GPy with Matern kernel + constant mean.

Train on eff_toy_4e6.root (50x50 histogram), evaluate chi2 on eff_toy_1e5.root.
nDoF = 50*50 - 5 = 2495.
"""
import numpy as np
import uproot
import matplotlib.pyplot as plt
import GPy

# -------- style ----------
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "xtick.direction": "in",
    "ytick.direction": "in",
})


def main():
    bins_fit = 50
    bins_proj = 100
    n_params = 5

    # ---- load data ----
    import os
    _TAG = os.environ.get("EFF_TAG", "")
    f_train = uproot.open(os.environ.get("EFF_TOY_HI", "eff_toy_4e6.root"))
    tree_train = f_train[f_train.keys()[0]]
    m_train = tree_train["mprime"].array(library="np")
    t_train = tree_train["thetaprime"].array(library="np")

    f_test = uproot.open(os.environ.get("EFF_TOY_LO", "eff_toy_1e5.root"))
    tree_test = f_test[f_test.keys()[0]]
    m_test = tree_test["mprime"].array(library="np")
    t_test = tree_test["thetaprime"].array(library="np")

    print(f"Training: {len(m_train)} events")
    print(f"Test:     {len(m_test)} events")

    # ---- training histogram (50x50) ----
    H_train, xedges, yedges = np.histogram2d(
        m_train, t_train, bins=bins_fit, range=[[0, 1], [0, 1]]
    )
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])

    # GP input: bin centers (2500 x 2)
    Mc, Tc = np.meshgrid(xc, yc, indexing="ij")
    X = np.stack([Mc.ravel(), Tc.ravel()], axis=1)
    Y = H_train.ravel().reshape(-1, 1)

    # ---- GP model: Matern kernel + constant mean ----
    # Matern 3/2 kernel with ARD (separate length scale per dimension)
    kern = GPy.kern.Matern32(input_dim=2, ARD=True)

    # Constant mean via bias kernel
    kern_mean = GPy.kern.Bias(input_dim=2)

    model = GPy.models.GPRegression(X, Y, kernel=kern + kern_mean)

    # ---- optimise hyperparameters ----
    print("Optimising GP hyperparameters ...")
    model.optimize(messages=False, max_iters=1000)
    model.optimize_restarts(num_restarts=5, verbose=False)

    print("\nGP parameters:")
    print(model)

    # ---- predict on fine grid (for 2D plot) ----
    fine = 200
    mf = np.linspace(0, 1, fine)
    tf_arr = np.linspace(0, 1, fine)
    Mg, Tg = np.meshgrid(mf, tf_arr, indexing="ij")
    X_fine = np.stack([Mg.ravel(), Tg.ravel()], axis=1)

    mu_fine, _ = model.predict(X_fine)
    mu_fine_2d = mu_fine.reshape(fine, fine)

    # Normalise to average density = 1
    eps_fine = mu_fine_2d / mu_fine_2d.mean()

    # ---- chi2 on test set (50x50) ----
    H_test_50, _, _ = np.histogram2d(
        m_test, t_test, bins=bins_fit, range=[[0, 1], [0, 1]]
    )

    mu_test, _ = model.predict(X)
    mu_test = mu_test.ravel()
    mu_test *= H_test_50.sum() / mu_test.sum()
    mu_test_2d = mu_test.reshape(H_test_50.shape)

    valid = mu_test_2d > 0
    chi2 = np.sum((H_test_50[valid] - mu_test_2d[valid])**2 / mu_test_2d[valid])
    ndf = bins_fit**2 - n_params
    red_chi2 = chi2 / ndf

    print(f"\nchi2/nDoF = {chi2:.0f}/{ndf} = {red_chi2:.2f}")

    # ---- 1D projections (test set, 100 bins) ----
    H_m_test, edges_m = np.histogram(m_test, bins=bins_proj, range=(0, 1))
    H_t_test, edges_t = np.histogram(t_test, bins=bins_proj, range=(0, 1))
    centers_m = 0.5 * (edges_m[:-1] + edges_m[1:])
    centers_t = 0.5 * (edges_t[:-1] + edges_t[1:])
    width_m = edges_m[1] - edges_m[0]
    width_t = edges_t[1] - edges_t[0]

    # GP projection: average fine grid along the other axis
    eps_m_proj = eps_fine.mean(axis=1)
    eps_t_proj = eps_fine.mean(axis=0)

    eps_m_interp = np.interp(centers_m, mf, eps_m_proj)
    eps_t_interp = np.interp(centers_t, tf_arr, eps_t_proj)

    scale_m = H_m_test.sum() / eps_m_interp.sum()
    scale_t = H_t_test.sum() / eps_t_interp.sum()
    fit_m_line = eps_m_interp * scale_m
    fit_t_line = eps_t_interp * scale_t

    # ---- plot ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) 2D efficiency (contour)
    ax = axes[0]
    vmax = np.ceil(eps_fine.max() * 10) / 10
    levels = np.linspace(0.0, vmax, 21)
    cf = ax.contourf(mf, tf_arr, eps_fine.T, levels=levels, cmap="afmhot_r")
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
    ax.plot(centers_m, fit_m_line, "r-", lw=1.5, label="GP fit")
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
    ax.plot(centers_t, fit_t_line, "r-", lw=1.5, label="GP fit")
    ax.set_xlabel("\u03b8'")
    ax.set_ylabel(f"Entries / ({width_t:.2f})")
    ax.set_title("(c)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=10)
    ax.minorticks_on()
    ax.tick_params(which="both", top=True, right=True)

    fig.suptitle(
        f"Gaussian process (Mat\u00e9rn 3/2)   "
        f"\u03c7\u00b2/nDoF = {chi2:.0f}/{ndf} = {red_chi2:.2f}",
        fontsize=14,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"fig4_gp{_TAG}.png", dpi=300)
    print(f"Saved fig4_gp.png")
    plt.close()


if __name__ == "__main__":
    main()
