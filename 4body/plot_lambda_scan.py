"""
Plot a lambda2 scan: (a) training curves (val NLL vs epoch, from the .txt logs) and
(b) performance vs lambda2 (chi2/nb per projection, computed from each saved model).

Works for the swish scan (default) or the sigmoid scan via ACC_SCAN_PREFIX.
Env: ACC_SCAN_PREFIX (default acc_5d_swish_lam), ACC_NTRAIN=5000000, ACC_NTEST, ACC_SEED, ACC_EVAL_UNIF.
Output: lambda_scan.png
"""
import os
import re
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import amplitf.interface as atfi
atfi.set_single_precision()
import tensorflow as tf
from amplitf.phasespace.rectangular_phasespace import RectangularPhaseSpace
import phasespace_transform as pst
import mlp

HERE = os.path.dirname(os.path.abspath(__file__))
def _i(n, d): return int(float(os.environ.get(n, d)))
PREFIX = os.environ.get("ACC_SCAN_PREFIX", "acc_5d_swish_lam")
LAMBDAS = ["0.003", "0.01", "0.03", "0.1", "0.3", "1.0"]
NTRAIN = _i("ACC_NTRAIN", 5000000); NTEST = _i("ACC_NTEST", 1000000)
SEED = _i("ACC_SEED", 1); N_UNIF = _i("ACC_EVAL_UNIF", 2000000)
RANGES = pst.RANGES_FLAT
VARS = ["u1", "u2", "cos1", "cos2", "phi"]

# ---- test data (disjoint) + uniform sample ----
z = np.load(os.path.join(HERE, "runs12_flat.npz"))
X = z["X"].astype(np.float64); W = z["W"].astype(np.float64)
np.random.seed(SEED); perm = np.random.permutation(len(X)); ti = perm[NTRAIN:NTRAIN + NTEST]
Xt, Wt = X[ti], W[ti]
U = RectangularPhaseSpace(RANGES).uniform_sample(N_UNIF).numpy().astype(np.float64)


def chi2_1d(epsU, i, nb=40):
    b = np.linspace(*RANGES[i], nb + 1)
    Hd, _ = np.histogram(Xt[:, i], bins=b, weights=Wt); Vd, _ = np.histogram(Xt[:, i], bins=b, weights=Wt ** 2)
    Hm, _ = np.histogram(U[:, i], bins=b, weights=epsU); Hm *= Hd.sum() / Hm.sum()
    ok = Vd > 0; return np.sum((Hd[ok] - Hm[ok]) ** 2 / Vd[ok]) / ok.sum()


def chi2_cc(epsU, nb=30):
    b = np.linspace(-1, 1, nb + 1)
    Hd, _, _ = np.histogram2d(Xt[:, 2], Xt[:, 3], bins=[b, b], weights=Wt)
    Vd, _, _ = np.histogram2d(Xt[:, 2], Xt[:, 3], bins=[b, b], weights=Wt ** 2)
    Hm, _, _ = np.histogram2d(U[:, 2], U[:, 3], bins=[b, b], weights=epsU); Hm *= Hd.sum() / Hm.sum()
    ok = Vd > 0; return np.sum((Hd[ok] - Hm[ok]) ** 2 / Vd[ok]) / ok.sum()


def parse_curve(txt):
    ep, val = [], []
    for line in open(txt):
        m = re.match(r"Epoch\s+(\d+)\s+train\s+\S+\s+val\s+(\S+)", line)
        if m:
            ep.append(int(m.group(1))); val.append(float(m.group(2)))
    return np.array(ep), np.array(val)


rows, curves, labs = [], [], []
print(f"{'lambda2':>8s} | {'u1':>6s} {'u2':>6s} {'cos1':>6s} {'cos2':>6s} {'phi':>6s} | {'cc-2D':>6s}")
for L in LAMBDAS:
    npy = os.path.join(HERE, f"{PREFIX}{L}.npy"); txt = os.path.join(HERE, f"{PREFIX}{L}.txt")
    if not os.path.exists(npy):
        print(f"{L:>8s} | (missing)"); continue
    epsU = mlp.model_fn_from_npy(np.load(npy, allow_pickle=True))(U)
    r = [chi2_1d(epsU, i) for i in range(5)] + [chi2_cc(epsU)]
    rows.append(r); labs.append(L)
    curves.append(parse_curve(txt) if os.path.exists(txt) else None)
    print(f"{L:>8s} | " + " ".join(f"{v:6.2f}" for v in r[:5]) + f" | {r[5]:6.2f}")

# ---- plot ----
fig, ax = plt.subplots(1, 2, figsize=(16, 5.5))
for L, cv in zip(labs, curves):
    if cv is None:
        continue
    ep, val = cv
    line, = ax[0].plot(ep, val, lw=1.2, label=f"$\\lambda_2$={L}")
    bi = int(np.argmin(val)); ax[0].plot(ep[bi], val[bi], "o", color=line.get_color(), ms=6)
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("validation NLL"); ax[0].set_title("training curves (dot = best)")
ax[0].legend(fontsize=9)

names = VARS + ["cos1-cos2"]
xl = [float(L) for L in labs]
for k, nm in enumerate(names):
    ax[1].plot(xl, [r[k] for r in rows], "o-", label=nm)
ax[1].axhline(1.0, color="gray", ls="--", lw=1)
ax[1].set_xscale("log"); ax[1].set_xlabel("$\\lambda_2$"); ax[1].set_ylabel("$\\chi^2$/nb")
ax[1].set_title("performance vs $\\lambda_2$ (40-bin)"); ax[1].legend(fontsize=9)
fig.suptitle(f"lambda2 scan: {PREFIX}*", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(HERE, "lambda_scan.png"), dpi=130)
print("\nSaved lambda_scan.png")
