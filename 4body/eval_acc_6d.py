"""
Evaluate a cut+logit 6-feature acceptance model: chi2 per projection (in ORIGINAL cut
coords) + Data|Model projection plot.  Same chi2 recipe as eval_acc_5d.py (uniform sample
in orig cut coords -> features -> eps; weighted-hist variance), so numbers are comparable.

Usage: python eval_acc_6d.py [model.npy] [outprefix]
"""
import os
import sys
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
initfile  = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "acc_6d.npy")
outprefix = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(initfile)[0]
NTRAIN = _i("ACC_NTRAIN", 0); NTEST = _i("ACC_NTEST", 1000000)
SEED = _i("ACC_SEED", 1); N_UNIF = _i("ACC_EVAL_UNIF", 2000000)
VARS = ["m1", "m2", "cos1", "cos2", "phi"]
ORIG = tl.ORIG_RANGES

model = mlp.model_fn_from_npy(np.load(initfile, allow_pickle=True))   # takes 6-feature input

z = np.load(os.path.join(HERE, "runs12_cut.npz"))
X5all, Wall = z["X5"].astype(np.float64), z["W"].astype(np.float64)
np.random.seed(SEED); perm = np.random.permutation(len(X5all))
ntr = NTRAIN if NTRAIN else int(0.9 * len(X5all)) * 0   # default: all-but-test disjoint
ti = perm[len(X5all) - NTEST:] if not NTRAIN else perm[NTRAIN:NTRAIN + NTEST]
Xt, Wt = X5all[ti], Wall[ti]

# uniform norm sample in orig cut coords -> features -> eps
Ften, X5u = tl.uniform_norm_sample(N_UNIF, np.random.RandomState(SEED + 1))
epsU = model(Ften).astype(np.float64)
print(f"test {len(Xt)}, uniform {N_UNIF}")


def chi2_1d(i, nb=40):
    lo, hi = ORIG[i]; b = np.linspace(lo, hi, nb + 1); c = 0.5 * (b[:-1] + b[1:])
    Hd, _ = np.histogram(Xt[:, i], bins=b, weights=Wt)
    Vd, _ = np.histogram(Xt[:, i], bins=b, weights=Wt ** 2)
    Hm, _ = np.histogram(X5u[:, i], bins=b, weights=epsU); Hm *= Hd.sum() / Hm.sum()
    ok = Vd > 0
    return c, Hd, np.sqrt(Vd), Hm, np.sum((Hd[ok] - Hm[ok]) ** 2 / Vd[ok]) / ok.sum()


def chi2_cc(nb=30):
    bx = np.linspace(-1, 1, nb + 1); by = np.linspace(-1, 1, nb + 1)
    Hd, _, _ = np.histogram2d(Xt[:, 2], Xt[:, 3], bins=[bx, by], weights=Wt)
    Vd, _, _ = np.histogram2d(Xt[:, 2], Xt[:, 3], bins=[bx, by], weights=Wt ** 2)
    Hm, _, _ = np.histogram2d(X5u[:, 2], X5u[:, 3], bins=[bx, by], weights=epsU); Hm *= Hd.sum() / Hm.sum()
    ok = Vd > 0
    return np.sum((Hd[ok] - Hm[ok]) ** 2 / Vd[ok]) / ok.sum()


fig, ax = plt.subplots(1, 5, figsize=(24, 4.2))
print("\nchi2/nb per projection (cut+logit 6d):")
for a, i in zip(ax, range(5)):
    c, Hd, Ed, Hm, c2 = chi2_1d(i)
    a.errorbar(c, Hd, yerr=Ed, fmt="k.", ms=3, capsize=0, label="data")
    a.plot(c, Hm, "r-", lw=1.6, label="ANN eps")
    a.set_xlabel(VARS[i]); a.set_ylim(bottom=0); a.set_title(f"{VARS[i]}: chi2/nb={c2:.2f}")
    if i == 0: a.legend(frameon=False)
    print(f"  {VARS[i]:5s}  chi2/nb = {c2:.2f}")
c2cc = chi2_cc()
print(f"  cos1-cos2 2D  chi2/nb = {c2cc:.2f}")
fig.suptitle(f"cut+logit 6d acceptance ({os.path.basename(initfile)})  cos2D chi2/nb={c2cc:.2f}", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig(outprefix + "_eval.png", dpi=140)
print(f"\nSaved {outprefix}_eval.png")
