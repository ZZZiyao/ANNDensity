"""
Evaluate the cut+6d lambda2 scan: for each acc_6d_lam*.npy, compute chi2/nb per
projection (m1,m2,cos1,cos2,phi) + the cos1-cos2 2D, using the SAME recipe as
eval_acc_6d.py (uniform norm sample in orig cut coords -> features -> eps; weighted
variance). Also read best_val from each .txt. Prints a table and saves a figure.

Env: ACC_NTEST=1000000, ACC_SEED=1, ACC_EVAL_UNIF=2000000.
Output: lambda_scan_6d.png  (+ printed table for the thesis).
"""
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import amplitf.interface as atfi
atfi.set_single_precision()
import tensorflow as tf
import mlp
import transform_cut as tl

HERE = os.path.dirname(os.path.abspath(__file__))
def _i(n, d): return int(float(os.environ.get(n, d)))
NTEST = _i("ACC_NTEST", 1000000); SEED = _i("ACC_SEED", 1); N_UNIF = _i("ACC_EVAL_UNIF", 2000000)
LAMBDAS = ["0.001", "0.003", "0.01", "0.03", "0.1", "0.3"]
VARS = ["m1", "m2", "cos1", "cos2", "phi"]
ORIG = tl.ORIG_RANGES

z = np.load(os.path.join(HERE, "runs12_cut.npz"))
X5all, Wall = z["X5"].astype(np.float64), z["W"].astype(np.float64)
np.random.seed(SEED); perm = np.random.permutation(len(X5all)); ti = perm[len(X5all) - NTEST:]
Xt, Wt = X5all[ti], Wall[ti]
Ften, X5u = tl.uniform_norm_sample(N_UNIF, np.random.RandomState(SEED + 1))
print(f"test {len(Xt)}, uniform {N_UNIF}")


def chi2_1d(epsU, i, nb=40):
    lo, hi = ORIG[i]; b = np.linspace(lo, hi, nb + 1)
    Hd, _ = np.histogram(Xt[:, i], bins=b, weights=Wt)
    Vd, _ = np.histogram(Xt[:, i], bins=b, weights=Wt ** 2)
    Hm, _ = np.histogram(X5u[:, i], bins=b, weights=epsU); Hm *= Hd.sum() / Hm.sum()
    ok = Vd > 0
    return np.sum((Hd[ok] - Hm[ok]) ** 2 / Vd[ok]) / ok.sum()


def chi2_cc(epsU, nb=30):
    b = np.linspace(-1, 1, nb + 1)
    Hd, _, _ = np.histogram2d(Xt[:, 2], Xt[:, 3], bins=[b, b], weights=Wt)
    Vd, _, _ = np.histogram2d(Xt[:, 2], Xt[:, 3], bins=[b, b], weights=Wt ** 2)
    Hm, _, _ = np.histogram2d(X5u[:, 2], X5u[:, 3], bins=[b, b], weights=epsU); Hm *= Hd.sum() / Hm.sum()
    ok = Vd > 0
    return np.sum((Hd[ok] - Hm[ok]) ** 2 / Vd[ok]) / ok.sum()


def best_val(txt):
    if not os.path.exists(txt):
        return None
    m = None
    for line in open(txt):
        mm = re.search(r"best_val\s+(-?\d+\.?\d*)", line)
        if mm:
            m = float(mm.group(1))
    return m


rows, labs, bvs = [], [], []
print(f"\n{'lambda2':>8s} | {'m1':>6s} {'m2':>6s} {'cos1':>6s} {'cos2':>6s} {'phi':>6s} | {'cc-2D':>6s} {'mean1D':>7s} | {'best_val':>10s}")
for L in LAMBDAS:
    npy = os.path.join(HERE, f"acc_6d_lam{L}.npy")
    if not os.path.exists(npy):
        print(f"{L:>8s} | (missing)"); continue
    epsU = mlp.model_fn_from_npy(np.load(npy, allow_pickle=True))(Ften).astype(np.float64)
    r = [chi2_1d(epsU, i) for i in range(5)]
    cc = chi2_cc(epsU); mean1d = float(np.mean(r))
    bv = best_val(os.path.join(HERE, f"acc_6d_lam{L}.txt"))
    rows.append(r + [cc]); labs.append(L); bvs.append(bv)
    bvs_s = f"{bv:10.1f}" if bv is not None else f"{'--':>10s}"
    print(f"{L:>8s} | " + " ".join(f"{v:6.2f}" for v in r) + f" | {cc:6.2f} {mean1d:7.2f} | {bvs_s}")

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
names = VARS + ["cos1-cos2"]
xl = [float(L) for L in labs]
for k, nm in enumerate(names):
    ax[0].plot(xl, [r[k] for r in rows], "o-", label=nm)
ax[0].axhline(1.0, color="gray", ls="--", lw=1)
ax[0].set_xscale("log"); ax[0].set_xlabel(r"$\lambda_2$"); ax[0].set_ylabel(r"$\chi^2/n_b$")
ax[0].set_title("performance vs $\\lambda_2$ (cut + 6-feature, 40-bin)"); ax[0].legend(fontsize=9)
ax[1].plot(xl, [np.mean(r[:5]) for r in rows], "ks-", label="mean over 5 projections")
ax[1].axhline(1.0, color="gray", ls="--", lw=1)
ax[1].set_xscale("log"); ax[1].set_xlabel(r"$\lambda_2$"); ax[1].set_ylabel(r"mean $\chi^2/n_b$")
ax[1].set_title("mean $\\chi^2/n_b$ vs $\\lambda_2$"); ax[1].legend(fontsize=9)
fig.suptitle("lambda2 scan (cut + 6-feature ANN, paper topology, swish)", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(HERE, "lambda_scan_6d.png"), dpi=130)
print("\nSaved lambda_scan_6d.png")
