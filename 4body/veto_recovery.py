"""
Shared veto-recovery evaluation: given a model callable eps(X5)->array and a veto spec, compare
the model's eps to the held-out truth INSIDE the vetoed region (recovery ratio + bias chi2 +
plot). Used by the Legendre/spline veto baselines so every method reports the SAME metric as
eval_veto.py (ANN). Projects on the widest-window veto variable, like eval_veto.py.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import transform_cut as tl
import veto as vt

HERE = os.path.dirname(os.path.abspath(__file__))
VARS = ["m1", "m2", "cos1", "cos2", "phi"]


def recovery(model, spec_str, outprefix, seed=1, n_unif=2000000, label="model"):
    spec = vt.parse(spec_str)
    z = np.load(os.path.join(HERE, "runs12_cut.npz"))
    X5, W = z["X5"].astype(np.float64), z["W"].astype(np.float64)
    Fu, X5u = tl.uniform_norm_sample(n_unif, np.random.RandomState(seed + 11))
    epsU = np.asarray(model(X5u), float)

    vvar = max(spec, key=lambda s: s[2] - s[1])[0]           # widest-window var = projection axis
    # restrict to the veto "column" (in every OTHER veto window) so that for a 2D/3D veto the
    # projection onto vvar isolates the hole instead of mixing in non-vetoed events
    others = [(v, lo, hi) for (v, lo, hi) in spec if v != vvar]
    def colmask(A):
        m = np.ones(len(A), bool)
        for (v, lo, hi) in others:
            m &= (A[:, v] > lo) & (A[:, v] < hi)
        return m
    cd, cu = colmask(X5), colmask(X5u)
    lo, hi = tl.ORIG_RANGES[vvar]
    b = np.linspace(lo, hi, 41); c = 0.5 * (b[:-1] + b[1:])
    Hd, _ = np.histogram(X5[cd, vvar], bins=b, weights=W[cd])         # truth in column (incl. hole)
    Vd, _ = np.histogram(X5[cd, vvar], bins=b, weights=W[cd] ** 2)
    Hm, _ = np.histogram(X5u[cu, vvar], bins=b, weights=epsU[cu])
    Hd_n = Hd / Hd.mean(); Ed_n = np.sqrt(Vd) / Hd.mean()
    Hm_n = Hm / Hm.mean(); Hm_n *= Hd_n.mean() / Hm_n.mean()

    vlo, vhi = [(s[1], s[2]) for s in spec if s[0] == vvar][0]
    vbins = (c > vlo) & (c < vhi)
    ratio = Hm_n[vbins].mean() / Hd_n[vbins].mean()
    ok = (Vd > 0) & vbins
    chi2 = np.sum((Hd_n[ok] - Hm_n[ok]) ** 2 / Ed_n[ok] ** 2) / max(ok.sum(), 1)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.errorbar(c, Hd_n, yerr=Ed_n, fmt="k.", ms=4, capsize=0, label="TRUE eps (held-out)")
    ax.plot(c, Hm_n, "r-", lw=2, label=f"{label} (fit WITHOUT veto)")
    ax.axvspan(vlo, vhi, color="tab:blue", alpha=0.15, label="VETO region")
    ax.set_xlabel(VARS[vvar]); ax.set_ylabel("eps (avg=1)"); ax.set_ylim(bottom=0); ax.legend()
    ax.set_title(f"veto recovery [{label}]: {vt.describe(spec)}\nratio={ratio:.3f}  chi2/nb={chi2:.2f}")
    plt.tight_layout(); plt.savefig(outprefix + ".png", dpi=130)
    with open(outprefix + ".txt", "w") as f:
        f.write(f"method={label} veto='{spec_str}' axis={VARS[vvar]} ratio={ratio:.4f} "
                f"chi2nb={chi2:.4f} nbins={int(vbins.sum())}\n")
    print(f"  [{label}] recovery ratio={ratio:.3f}  chi2/nb={chi2:.2f}  saved {outprefix}.png")
    return ratio, chi2
