"""
7-panel version of eval_acc_6d.py: top row m1,m2,cos1,cos2; bottom row phi, cos1-cos2 data,
cos1-cos2 ANN fit (+ blank). Same chi2 recipe as eval_acc_6d.py.
Usage: python eval_acc_6d_7p.py [model.npy] [outprefix]
"""
import os, sys
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
initfile = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "acc_6d_lam0.1.npy")
outprefix = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(initfile)[0] + "_7p"
NTRAIN = _i("ACC_NTRAIN", 0); NTEST = _i("ACC_NTEST", 1000000)
SEED = _i("ACC_SEED", 1); N_UNIF = _i("ACC_EVAL_UNIF", 2000000)
VARS = ["m1", "m2", "cos1", "cos2", "phi"]; ORIG = tl.ORIG_RANGES

model = mlp.model_fn_from_npy(np.load(initfile, allow_pickle=True))
z = np.load(os.path.join(HERE, "runs12_cut.npz"))
X5all, Wall = z["X5"].astype(np.float64), z["W"].astype(np.float64)
np.random.seed(SEED); perm = np.random.permutation(len(X5all))
ti = perm[len(X5all) - NTEST:] if not NTRAIN else perm[NTRAIN:NTRAIN + NTEST]
Xt, Wt = X5all[ti], Wall[ti]
Ften, X5u = tl.uniform_norm_sample(N_UNIF, np.random.RandomState(SEED + 1))
epsU = model(Ften).astype(np.float64)
print(f"model {os.path.basename(initfile)}  test {len(Xt)}  uniform {N_UNIF}")


def chi2_1d(i, nb=40):
    lo, hi = ORIG[i]; b = np.linspace(lo, hi, nb + 1); c = 0.5 * (b[:-1] + b[1:])
    Hd, _ = np.histogram(Xt[:, i], bins=b, weights=Wt)
    Vd, _ = np.histogram(Xt[:, i], bins=b, weights=Wt ** 2)
    Hm, _ = np.histogram(X5u[:, i], bins=b, weights=epsU); Hm *= Hd.sum() / Hm.sum()
    ok = Vd > 0
    return c, Hd, np.sqrt(Vd), Hm, np.sum((Hd[ok] - Hm[ok]) ** 2 / Vd[ok]) / ok.sum()


def cc_maps(nb=30):
    ex = np.linspace(-1, 1, nb + 1); ey = np.linspace(-1, 1, nb + 1)
    Hd, _, _ = np.histogram2d(Xt[:, 2], Xt[:, 3], bins=[ex, ey], weights=Wt)
    Vd, _, _ = np.histogram2d(Xt[:, 2], Xt[:, 3], bins=[ex, ey], weights=Wt ** 2)
    Hm, _, _ = np.histogram2d(X5u[:, 2], X5u[:, 3], bins=[ex, ey], weights=epsU); Hm *= Hd.sum() / Hm.sum()
    ok = Vd > 0; c2 = np.sum((Hd[ok] - Hm[ok]) ** 2 / Vd[ok]) / ok.sum()
    Dn = Hd / Hd[Hd > 0].mean(); Mn = Hm / Hm[Hm > 0].mean()
    return ex, ey, Dn, Mn, c2


fig, ax = plt.subplots(2, 4, figsize=(20, 9)); ax = ax.ravel()
print("chi2/nb per projection:")
for k, i in enumerate([0, 1, 2, 3]):                       # top row: m1, m2, cos1, cos2
    c, Hd, Ed, Hm, c2 = chi2_1d(i)
    ax[k].errorbar(c, Hd, yerr=Ed, fmt="k.", ms=3, capsize=0, label="data")
    ax[k].plot(c, Hm, "r-", lw=1.6, label="ANN")
    ax[k].set_xlabel(VARS[i]); ax[k].set_ylim(bottom=0); ax[k].set_title(f"{VARS[i]}: $\\chi^2/n_b$={c2:.2f}")
    if k == 0: ax[k].legend(frameon=False)
    print(f"  {VARS[i]:5s} {c2:.2f}")
c, Hd, Ed, Hm, c2 = chi2_1d(4)                             # bottom-left: phi
ax[4].errorbar(c, Hd, yerr=Ed, fmt="k.", ms=3, capsize=0)
ax[4].plot(c, Hm, "r-", lw=1.6)
ax[4].set_xlabel("phi"); ax[4].set_ylim(bottom=0); ax[4].set_title(f"phi: $\\chi^2/n_b$={c2:.2f}")
print(f"  phi   {c2:.2f}")
ex, ey, Dn, Mn, c2cc = cc_maps()                           # cos1-cos2 data + ANN fit
vmax = max(Dn.max(), Mn.max())
pm = ax[5].pcolormesh(ex, ey, Dn.T, shading="auto", cmap="afmhot_r", vmin=0, vmax=vmax)
ax[5].set_xlabel("cos1"); ax[5].set_ylabel("cos2"); ax[5].set_title("cos1-cos2: data"); fig.colorbar(pm, ax=ax[5])
pm = ax[6].pcolormesh(ex, ey, Mn.T, shading="auto", cmap="afmhot_r", vmin=0, vmax=vmax)
ax[6].set_xlabel("cos1"); ax[6].set_ylabel("cos2"); ax[6].set_title(f"cos1-cos2: ANN fit ($\\chi^2/n_b$={c2cc:.2f})"); fig.colorbar(pm, ax=ax[6])
ax[7].axis("off")
print(f"  cc2D  {c2cc:.2f}")
fig.suptitle("Four-body acceptance: neural-network fit", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(outprefix + "_eval.png", dpi=140)
print(f"Saved {outprefix}_eval.png")
