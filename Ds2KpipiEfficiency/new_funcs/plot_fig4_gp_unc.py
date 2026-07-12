"""
Gaussian process fit to the three-body efficiency (GPy, Matern 3/2 + constant mean),
WITH posterior uncertainty shown everywhere:
  (a) mean efficiency eps(m',theta')
  (b) relative posterior uncertainty sigma_eps/eps  [the "background" map's uncertainty]
  (c) m'     projection: GP mean +/- 1 sigma band
  (d) theta' projection: GP mean +/- 1 sigma band

The projection band is NOT a naive average of per-point sigma: neighbouring grid points
are strongly correlated, so for each projection abscissa we predict the whole slice with
full_cov=True and propagate  sigma_proj^2 = a^T Sigma a  with a = 1/K (the averaging weights).
Latent-function uncertainty is used (include_likelihood=False), i.e. the uncertainty on the
smooth efficiency itself, not on a single noisy bin count.
"""
import os
import pickle
import numpy as np
import uproot
import matplotlib.pyplot as plt
import GPy

plt.rcParams.update({"font.size": 13, "axes.labelsize": 15,
                     "xtick.direction": "in", "ytick.direction": "in"})


def latent_predict(model, X, full_cov=False):
    """GP posterior of the latent function (no per-observation noise)."""
    return model.predict(X, full_cov=full_cov, include_likelihood=False)


def main():
    bins_fit, bins_proj, n_params = 50, 100, 5
    _TAG = os.environ.get("EFF_TAG", "")

    f_train = uproot.open(os.environ.get("EFF_TOY_HI", "eff_toy_4e6.root"))
    tr = f_train[f_train.keys()[0]]
    m_train = tr["mprime"].array(library="np"); t_train = tr["thetaprime"].array(library="np")
    f_test = uproot.open(os.environ.get("EFF_TOY_LO", "eff_toy_1e5.root"))
    te = f_test[f_test.keys()[0]]
    m_test = te["mprime"].array(library="np"); t_test = te["thetaprime"].array(library="np")
    _ntest = int(os.environ.get("EFF_TEST_N", "0"))   # subsample the test to an independent N (e.g. 1e5)
    if _ntest and _ntest < len(m_test):
        _idx = np.random.RandomState(0).choice(len(m_test), _ntest, replace=False)
        m_test, t_test = m_test[_idx], t_test[_idx]
    print(f"Training: {len(m_train)}   Test: {len(m_test)}")

    # ---- training histogram (50x50) -> GP targets ----
    H_train, xe, ye = np.histogram2d(m_train, t_train, bins=bins_fit, range=[[0, 1], [0, 1]])
    xc = 0.5 * (xe[:-1] + xe[1:]); yc = 0.5 * (ye[:-1] + ye[1:])
    Mc, Tc = np.meshgrid(xc, yc, indexing="ij")
    X = np.stack([Mc.ravel(), Tc.ravel()], axis=1); Y = H_train.ravel().reshape(-1, 1)

    kern = GPy.kern.Matern52(input_dim=2, ARD=True) + GPy.kern.Bias(input_dim=2)
    model = GPy.models.GPRegression(X, Y, kernel=kern)
    pkl = f"gp_acc_model{_TAG}.pkl"                       # persist the fit: never re-optimise
    if os.path.exists(pkl):
        with open(pkl, "rb") as fh:
            model.update_model(False); model[:] = pickle.load(fh)["param_array"]; model.update_model(True)
        print(f"loaded {pkl}")
    else:
        print("Optimising GP ...")
        model.optimize(messages=False, max_iters=1000)
        model.optimize_restarts(num_restarts=5, verbose=False)
        with open(pkl, "wb") as fh:
            pickle.dump({"param_array": model.param_array.copy(), "kern": "Matern52(ARD)+Bias"}, fh)
        print(f"fitted + saved {pkl}")
    print(model)
    print("ACC Matern52 ARD lengthscales [m', theta']:", np.round(np.asarray(model.sum.Mat52.lengthscale), 4))

    # ---- fine grid: mean + latent variance for the 2D maps ----
    fine = 200
    mf = np.linspace(0, 1, fine); tf = np.linspace(0, 1, fine)
    Mg, Tg = np.meshgrid(mf, tf, indexing="ij")
    Xf = np.stack([Mg.ravel(), Tg.ravel()], axis=1)
    mu_f, var_f = latent_predict(model, Xf)                 # (40000,1) each
    mu_2d = mu_f.reshape(fine, fine)
    c = mu_2d.mean()                                        # normalisation to <eps>=1
    eps_2d = mu_2d / c
    relsig_2d = np.sqrt(np.clip(var_f, 0, None)).reshape(fine, fine) / mu_2d  # sigma/eps (norm-free)
    print(f"relative sigma_eps/eps : median {np.median(relsig_2d)*100:.2f}%  max {relsig_2d.max()*100:.2f}%")

    # ---- chi2 on the 50x50 test histogram ----
    H_te50, _, _ = np.histogram2d(m_test, t_test, bins=bins_fit, range=[[0, 1], [0, 1]])
    mu_te, _ = latent_predict(model, X); mu_te = mu_te.ravel()
    mu_te *= H_te50.sum() / mu_te.sum(); mu_te2d = mu_te.reshape(H_te50.shape)
    v = mu_te2d > 0
    chi2 = np.sum((H_te50[v] - mu_te2d[v]) ** 2 / mu_te2d[v]); ndf = bins_fit ** 2 - n_params
    print(f"chi2/nDoF = {chi2:.0f}/{ndf} = {chi2/ndf:.2f}")

    # ---- projections with propagated band (per-slice full_cov) ----
    def projection(fixed_axis, n_ab=120, n_sl=80):
        """Average eps over the OTHER axis; a=1/n_sl weights; sigma^2 = a^T Sigma a."""
        ab = np.linspace(0, 1, n_ab); sl = np.linspace(0, 1, n_sl); a = np.full(n_sl, 1.0 / n_sl)
        eps = np.empty(n_ab); sig = np.empty(n_ab)
        for i, av in enumerate(ab):
            pts = np.stack([np.full(n_sl, av), sl], axis=1) if fixed_axis == 0 else np.stack([sl, np.full(n_sl, av)], axis=1)
            mean_s, cov_s = latent_predict(model, pts, full_cov=True)
            eps[i] = a @ mean_s.ravel() / c
            sig[i] = np.sqrt(max(a @ cov_s @ a, 0.0)) / c
        return ab, eps, sig

    ab_m, eps_m, sig_m = projection(0)      # m' projection (average over theta')
    ab_t, eps_t, sig_t = projection(1)      # theta' projection (average over m')

    H_m, em = np.histogram(m_test, bins=bins_proj, range=(0, 1))
    H_t, et = np.histogram(t_test, bins=bins_proj, range=(0, 1))
    cm = 0.5 * (em[:-1] + em[1:]); ct = 0.5 * (et[:-1] + et[1:]); wm = em[1] - em[0]; wt = et[1] - et[0]

    def to_entries(centers, ab, eps, sig, Hdata):
        e_i = np.interp(centers, ab, eps); s_i = np.interp(centers, ab, sig)
        scale = Hdata.sum() / e_i.sum()
        return e_i * scale, s_i * scale

    line_m, band_m = to_entries(cm, ab_m, eps_m, sig_m, H_m)
    line_t, band_t = to_entries(ct, ab_t, eps_t, sig_t, H_t)

    # ---- plot 2x2 ----
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    # (a) mean efficiency
    a0 = ax[0, 0]; vmax = np.ceil(eps_2d.max() * 10) / 10
    cf = a0.contourf(mf, tf, eps_2d.T, levels=np.linspace(0, vmax, 21), cmap="afmhot_r")
    a0.set_xlabel("m'"); a0.set_ylabel("θ'"); a0.set_title("(a) mean  ε(m', θ')")
    fig.colorbar(cf, ax=a0, label="ε"); a0.minorticks_on(); a0.tick_params(which="both", top=True, right=True)

    # (b) relative uncertainty
    a1 = ax[0, 1]
    cf2 = a1.contourf(mf, tf, 100 * relsig_2d.T, levels=21, cmap="viridis")
    a1.set_xlabel("m'"); a1.set_ylabel("θ'"); a1.set_title("(b) relative uncertainty  σ$_\\varepsilon$/ε")
    fig.colorbar(cf2, ax=a1, label="%"); a1.minorticks_on(); a1.tick_params(which="both", top=True, right=True)

    # (c) m' projection + band
    a2 = ax[1, 0]
    a2.errorbar(cm, H_m, yerr=np.sqrt(H_m), fmt="k.", ms=3, capsize=0, label="Test data")
    a2.fill_between(cm, line_m - band_m, line_m + band_m, color="r", alpha=0.30, lw=0, label="GP $\\pm1\\sigma$")
    a2.plot(cm, line_m, "r-", lw=1.4, label="GP mean")
    a2.set_xlabel("m'"); a2.set_ylabel(f"Entries / ({wm:.2f})"); a2.set_title("(c) m' projection")
    a2.set_ylim(bottom=0); a2.legend(frameon=False, fontsize=10); a2.minorticks_on(); a2.tick_params(which="both", top=True, right=True)

    # (d) theta' projection + band
    a3 = ax[1, 1]
    a3.errorbar(ct, H_t, yerr=np.sqrt(H_t), fmt="k.", ms=3, capsize=0, label="Test data")
    a3.fill_between(ct, line_t - band_t, line_t + band_t, color="r", alpha=0.30, lw=0, label="GP $\\pm1\\sigma$")
    a3.plot(ct, line_t, "r-", lw=1.4, label="GP mean")
    a3.set_xlabel("θ'"); a3.set_ylabel(f"Entries / ({wt:.2f})"); a3.set_title("(d) θ' projection")
    a3.set_ylim(bottom=0); a3.legend(frameon=False, fontsize=10); a3.minorticks_on(); a3.tick_params(which="both", top=True, right=True)

    fig.suptitle(f"Gaussian process (Matérn 5/2)   χ²/nDoF = {chi2:.0f}/{ndf} = {chi2/ndf:.2f}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = f"fig4_gp_unc{_TAG}.png"; plt.savefig(out, dpi=200); print(f"Saved {out}"); plt.close()


if __name__ == "__main__":
    main()
