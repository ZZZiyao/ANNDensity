"""
3D Gaussian process fit to combinatorial background in sideband regions.

Input: 6 x 20 x 20 = 2400 bins from (m', theta', mD) sideband data.
Kernel: Matern 5/2 (ARD) + constant mean in (m', theta') + linear mean in mD.
Saves fitted GP model to gp_3d_bg_model.pkl.
"""
import os
import numpy as np
import uproot
import GPy
import pickle

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    train_file = os.path.join(HERE, "0.20.41e5_train.root")

    # ---- sideband definitions ----
    md_lo, md_s1, md_s2, md_hi = 1.77, 1.92, 2.02, 2.17
    bins_dp = 20    # bins in m' and theta' (width 0.05)
    bins_md = 3     # bins per sideband in mD (width 0.05 GeV)

    # ---- load data ----
    f = uproot.open(train_file)
    tree = f[f.keys()[0]]
    arr = tree.arrays(["mprime", "thetaprime", "md"], library="np")
    m = arr["mprime"]
    t = arr["thetaprime"]
    md = arr["md"]

    # Select sidebands only
    lower_sb = (md >= md_lo) & (md < md_s1)
    upper_sb = (md > md_s2) & (md <= md_hi)
    sb_mask = lower_sb | upper_sb

    m_sb = m[sb_mask]
    t_sb = t[sb_mask]
    md_sb = md[sb_mask]

    print(f"Sideband events: {sb_mask.sum()}")

    # ---- 3D binning: mD(6) x m'(20) x theta'(20) ----
    # 3 bins per sideband, each 0.05 GeV wide
    md_edges_lower = np.linspace(md_lo, md_s1, bins_md + 1)
    md_edges_upper = np.linspace(md_s2, md_hi, bins_md + 1)

    md_bin_edges = (list(zip(md_edges_lower[:-1], md_edges_lower[1:])) +
                    list(zip(md_edges_upper[:-1], md_edges_upper[1:])))
    md_bin_centers = [0.5 * (lo + hi) for lo, hi in md_bin_edges]

    dp_edges_m = np.linspace(0, 1, bins_dp + 1)
    dp_edges_t = np.linspace(0, 1, bins_dp + 1)
    dp_centers_m = 0.5 * (dp_edges_m[:-1] + dp_edges_m[1:])
    dp_centers_t = 0.5 * (dp_edges_t[:-1] + dp_edges_t[1:])

    # Build 3D histogram
    H = np.zeros((len(md_bin_edges), bins_dp, bins_dp))
    for k, (md_lo_k, md_hi_k) in enumerate(md_bin_edges):
        mask_md = (md_sb >= md_lo_k) & (md_sb < md_hi_k)
        H_2d, _, _ = np.histogram2d(
            m_sb[mask_md], t_sb[mask_md],
            bins=[dp_edges_m, dp_edges_t]
        )
        H[k] = H_2d

    print(f"3D histogram: {H.shape}, total entries: {H.sum():.0f}")

    # ---- GP input (2400 x 3) / output (2400 x 1) ----
    Xlist = []
    Ylist = []
    for k, md_c in enumerate(md_bin_centers):
        for i, m_c in enumerate(dp_centers_m):
            for j, t_c in enumerate(dp_centers_t):
                Xlist.append([m_c, t_c, md_c])
                Ylist.append(H[k, i, j])

    X = np.array(Xlist)
    Y = np.array(Ylist).reshape(-1, 1)

    # ---- GP model: Matern 5/2 + constant mean + linear mean in mD ----
    kern = GPy.kern.Matern52(input_dim=3, ARD=True)
    kern_const = GPy.kern.Bias(input_dim=3)
    kern_linear = GPy.kern.Linear(input_dim=1, active_dims=[2])

    model = GPy.models.GPRegression(X, Y, kernel=kern + kern_const + kern_linear)

    # ---- optimise hyperparameters (multi-start to avoid local minima) ----
    print("Optimising (3 restarts) ...")
    model.optimize_restarts(num_restarts=3, max_iters=1000,
                            verbose=True, robust=True)

    print("\nGP parameters:")
    print(model)

    # ---- save ----
    outfile = os.path.join(HERE, "gp_3d_bg_model.pkl")
    with open(outfile, "wb") as fout:
        pickle.dump({
            "model_param_array": model.param_array.copy(),
            "X": X,
            "Y": Y,
            "kern_class": "Matern52+Bias+Linear",
            "md_bin_edges": md_bin_edges,
            "md_bin_centers": md_bin_centers,
            "dp_centers_m": dp_centers_m,
            "dp_centers_t": dp_centers_t,
            "H": H,
        }, fout)
    print(f"Saved {outfile}")


if __name__ == "__main__":
    main()
