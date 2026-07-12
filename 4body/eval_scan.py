"""
4-body acceptance -- compare lambda2 scan results.

Loads the test data + a uniform 5D sample ONCE, then evaluates each lambda2 model
(acc_5d_lam<L>.npy) and prints a chi2/nbins comparison table across projections.
Pick the lambda2 whose chi2/nb is closest to 1 across the board (esp. cos1-cos2 2D)
without over-fitting.

Usage: python eval_scan.py
"""
import os
import math
import numpy as np

import amplitf.interface as atfi
atfi.set_single_precision()
import tensorflow as tf
import uproot
from amplitf.phasespace.rectangular_phasespace import RectangularPhaseSpace
import tfa.neural_nets as tfn

HERE = os.path.dirname(os.path.abspath(__file__))

def _i(n, d): return int(float(os.environ.get(n, d)))
ROOT_FILE = os.environ.get("ACC_ROOT", os.path.join(HERE, "runs12_MagBoth_L0Both.root"))
NTRAIN = _i("ACC_NTRAIN", 1000000)
NTEST  = _i("ACC_NTEST", 1000000)
SEED   = _i("ACC_SEED", 1)
N_UNIF = _i("ACC_EVAL_UNIF", 2000000)
FLATTEN = _i("ACC_FLATTEN", 0)
LAMBDAS = ["0.003", "0.01", "0.03", "0.1", "0.3", "1.0"]
PREFIX = os.environ.get("ACC_SCAN_PREFIX", "acc_5d_lam")   # "acc_5d_swish_lam" for the swish scan

RANGES = [(0.63, 1.04), (0.63, 1.04), (-1.0, 1.0), (-1.0, 1.0), (-math.pi, math.pi)]
VARS = ["m1", "m2", "cos1", "cos2", "phi"]
if FLATTEN:
    import phasespace_transform as pst
    RANGES = pst.RANGES_FLAT
    VARS = ["u1", "u2", "cos1", "cos2", "phi"]

# ---- test data (disjoint from training) ----
if FLATTEN:
    flat_npz = os.environ.get("ACC_FLATFILE", os.path.join(HERE, "runs12_flat.npz"))
    z = np.load(flat_npz)
    X = z["X"].astype(np.float64); W = z["W"].astype(np.float64)
else:
    arr = uproot.open(ROOT_FILE)["DecayTree"].arrays(VARS + ["weight_detJ"], library="np")
    X = np.stack([arr[v] for v in VARS], axis=1).astype(np.float64)
    W = np.asarray(arr["weight_detJ"], np.float64)
np.random.seed(SEED)
perm = np.random.permutation(len(X))
ti = perm[NTRAIN:NTRAIN + NTEST]
Xt, Wt = X[ti], W[ti]
U = RectangularPhaseSpace(RANGES).uniform_sample(N_UNIF).numpy().astype(np.float64)
print(f"test {len(Xt)} events, uniform sample {len(U)}")


import mlp
def load_model(npy):
    return mlp.model_fn_from_npy(np.load(npy, allow_pickle=True))


def chi2_1d(model_eps_U, i, nb=40):
    bins = np.linspace(*RANGES[i], nb + 1)
    Hd, _ = np.histogram(Xt[:, i], bins=bins, weights=Wt)
    Vd, _ = np.histogram(Xt[:, i], bins=bins, weights=Wt**2)
    Hm, _ = np.histogram(U[:, i], bins=bins, weights=model_eps_U)
    Hm *= Hd.sum() / Hm.sum()
    ok = Vd > 0
    return np.sum((Hd[ok] - Hm[ok])**2 / Vd[ok]) / ok.sum()


def chi2_2d(model_eps_U, i, j, nb=25):
    bx = np.linspace(*RANGES[i], nb + 1); by = np.linspace(*RANGES[j], nb + 1)
    Hd, _, _ = np.histogram2d(Xt[:, i], Xt[:, j], bins=[bx, by], weights=Wt)
    Vd, _, _ = np.histogram2d(Xt[:, i], Xt[:, j], bins=[bx, by], weights=Wt**2)
    Hm, _, _ = np.histogram2d(U[:, i], U[:, j], bins=[bx, by], weights=model_eps_U)
    Hm *= Hd.sum() / Hm.sum()
    ok = Vd > 0
    return np.sum((Hd[ok] - Hm[ok])**2 / Vd[ok]) / ok.sum()


print(f"\n{'lambda2':>8s} | {'m1':>6s} {'m2':>6s} {'cos1':>6s} {'cos2':>6s} {'phi':>6s} | {'cc-2D':>6s}")
print("-" * 60)
for L in LAMBDAS:
    npy = os.path.join(HERE, f"{PREFIX}{L}.npy")
    if not os.path.exists(npy):
        print(f"{L:>8s} | (missing)")
        continue
    model = load_model(npy)
    epsU = model(U)
    c = [chi2_1d(epsU, i) for i in range(5)]
    cc = chi2_2d(epsU, 2, 3)
    print(f"{L:>8s} | {c[0]:6.2f} {c[1]:6.2f} {c[2]:6.2f} {c[3]:6.2f} {c[4]:6.2f} | {cc:6.2f}")
print("\nPick lambda2 with chi2/nb closest to 1 across projections (esp. cc-2D),")
print("not the lowest (too-low everywhere can mean over-smoothing missed structure).")
