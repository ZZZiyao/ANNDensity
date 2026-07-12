"""
Diagnose: print fitted GP hyperparameters and compare with paper Table 2.

Paper Table 2 (Mathad et al., arXiv:1902.01452v2):
    sigma^2_GP        = 279   events^2     (Matern variance)
    sigma^2_mean_mD   = 43.0  events^2     (linear-mean variance in mD)
    sigma^2_mean_DP   = 33.6  events^2     (constant-mean variance in m', theta')
    rho_mD            = 0.27  GeV          (Matern length scale in mD)
    rho_m'            = 0.12                (Matern length scale in m')
    rho_theta'        = 0.07                (Matern length scale in theta')
    epsilon           = 22.8  events       (noise std, = sqrt(noise variance))
"""
import os
import pickle
import numpy as np
import GPy

HERE = os.path.dirname(os.path.abspath(__file__))


def rebuild_model(pkl_file):
    with open(pkl_file, "rb") as fin:
        saved = pickle.load(fin)
    X, Y = saved["X"], saved["Y"]
    kern = GPy.kern.Matern52(input_dim=3, ARD=True)
    kern_const = GPy.kern.Bias(input_dim=3)
    kern_linear = GPy.kern.Linear(input_dim=1, active_dims=[2])
    model = GPy.models.GPRegression(X, Y, kernel=kern + kern_const + kern_linear)
    model.update_model(False)
    model[:] = saved["model_param_array"]
    model.update_model(True)
    return model, saved


def main():
    pkl = os.path.join(HERE, "gp_3d_bg_model.pkl")
    model, saved = rebuild_model(pkl)

    # GPy nicely formatted summary
    print("=" * 70)
    print("GPy model.__str__():")
    print("=" * 70)
    print(model)

    # Raw param array (order depends on kernel sum)
    print("\n" + "=" * 70)
    print("Raw param_array:")
    print("=" * 70)
    pa = np.asarray(model.param_array).ravel()
    for i, v in enumerate(pa):
        print(f"  [{i}] {v:.6g}")

    # Best-effort named extraction
    print("\n" + "=" * 70)
    print("Side-by-side vs paper Table 2:")
    print("=" * 70)

    # In our construction (Matern52 ARD + Bias + Linear, Gaussian noise):
    # [0]      Matern52.variance         -> sigma^2_GP
    # [1..3]   Matern52.lengthscale[3]   -> rho_m', rho_theta', rho_mD  (input order)
    # [4]      Bias.variance             -> sigma^2_mean_DP
    # [5]      Linear.variances          -> sigma^2_mean_mD  (slope variance)
    # [6]      Gaussian_noise.variance   -> epsilon^2
    try:
        sigma2_GP   = float(pa[0])
        rho_m       = float(pa[1])
        rho_t       = float(pa[2])
        rho_mD      = float(pa[3])
        sigma2_DP   = float(pa[4])
        sigma2_mD   = float(pa[5])
        eps         = float(np.sqrt(pa[6]))

        rows = [
            ("sigma^2_GP        (events^2)", sigma2_GP, 279.0),
            ("sigma^2_mean_DP   (events^2)", sigma2_DP, 33.6),
            ("sigma^2_mean_mD   (events^2)", sigma2_mD, 43.0),
            ("rho_m'                       ", rho_m, 0.12),
            ("rho_theta'                   ", rho_t, 0.07),
            ("rho_mD            (GeV)      ", rho_mD, 0.27),
            ("epsilon = sqrt(noise) (ev)   ", eps, 22.8),
        ]
        print(f"  {'param':<32s}  {'reproduced':>12s}  {'paper':>12s}  {'ratio':>10s}")
        print("  " + "-" * 70)
        for name, mine, paper in rows:
            ratio = mine / paper if paper else float("nan")
            print(f"  {name:<32s}  {mine:>12.4g}  {paper:>12.4g}  {ratio:>10.2f}")
    except Exception as e:
        print(f"(Could not auto-map params: {e})")
        print("Inspect raw array above against GPy summary.")


if __name__ == "__main__":
    main()
