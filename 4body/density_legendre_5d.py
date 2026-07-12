"""
5D Legendre baseline for the 4-body acceptance -- the "traditional" method whose tensor-product
basis blows up with dimension: (order+1)^5 coefficients.

eps(x) = sum_{k1..k5} c_{k} L_k1(x1)...L_k5(x5),  variables normalised to [-1,1].
Fit by weighted Poisson NLL to the 5D weight_detJ-weighted histogram (+ L2 reg), then evaluate
with the SAME chi2 recipe as the ANN/GP (uniform sample -> eps -> 40-bin projections).

CPU only (numpy). Env: LEG_ORDER (per-dim order, default 4), LEG_BINS (per-dim hist bins,
default 8), LEG_LAMBDA2, ACC_NTEST, ACC_SEED, ACC_EVAL_UNIF.
Output: leg5d.npz (coeffs + meta), leg5d_eval.png + chi2 printed.
"""
import os
import numpy as np
from numpy.polynomial.legendre import legval
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import transform_cut as tl

HERE = os.path.dirname(os.path.abspath(__file__))
def _i(n, d): return int(float(os.environ.get(n, d)))
def _f(n, d): return float(os.environ.get(n, d))
ORDER = _i("LEG_ORDER", 4)
NB    = _i("LEG_BINS", 8)
LAM2  = _f("LEG_LAMBDA2", 1.0)
SEED  = _i("ACC_SEED", 1); NTEST = _i("ACC_NTEST", 1000000); N_UNIF = _i("ACC_EVAL_UNIF", 2000000)
ITERS = _i("LEG_ITERS", 4000)
RANGES = tl.ORIG_RANGES; VARS = ["m1", "m2", "cos1", "cos2", "phi"]
nc1 = ORDER + 1
NCOEFF = nc1 ** 5
print(f"5D Legendre: order={ORDER}/dim -> {NCOEFF} coefficients; hist {NB}^5={NB**5} bins; lambda2={LAM2}")

# ---- data ----
z = np.load(os.path.join(HERE, "runs12_cut.npz"))
X5, W = z["X5"].astype(np.float64), z["W"].astype(np.float64)
rng = np.random.RandomState(SEED); perm = rng.permutation(len(X5))
te = perm[len(X5) - NTEST:]; tr = perm[:len(X5) - NTEST]
Xtr, Wtr = X5[tr], W[tr]; Xte, Wte = X5[te], W[te]


def to_unit(X):  # map each var to [-1,1]
    return np.stack([2 * (X[:, i] - RANGES[i][0]) / (RANGES[i][1] - RANGES[i][0]) - 1 for i in range(5)], axis=1)


def design(U):
    """(N,5) in [-1,1] -> (N, NCOEFF) tensor-product Legendre design matrix."""
    # per-dim Legendre values L_0..L_ORDER  (N, nc1) each
    Ls = []
    eye = np.eye(nc1)
    for d in range(5):
        Ls.append(np.stack([legval(U[:, d], eye[k]) for k in range(nc1)], axis=1))  # (N,nc1)
    # outer product across the 5 dims -> (N, nc1^5)
    out = Ls[0]
    for d in range(1, 5):
        out = (out[:, :, None] * Ls[d][:, None, :]).reshape(len(U), -1)
    return out


# ---- 5D weighted histogram (training target for Poisson fit) ----
edges = [np.linspace(*RANGES[i], NB + 1) for i in range(5)]
ctrs = [0.5 * (e[:-1] + e[1:]) for e in edges]
H, _ = np.histogramdd(Xtr, bins=edges, weights=Wtr)
H = (H / H[H > 0].mean()).ravel()                       # normalise avg=1
mesh = np.meshgrid(*ctrs, indexing="ij")
Xc = np.stack([m.ravel() for m in mesh], axis=1)
Xdes = design(to_unit(Xc))                              # (NB^5, NCOEFF)
print(f"design matrix {Xdes.shape}")

# ---- VETO: drop veto-region bins from the fit; the global polynomial extrapolates into the hole ----
import veto as vt
SPEC = vt.parse(os.environ.get("ACC_VETO", ""))
if SPEC:
    keepbin = ~vt.mask(Xc, SPEC)
    print(f"VETO [{vt.describe(SPEC)}]: fitting {keepbin.sum()} of {len(H)} bins (hole extrapolated)")
    H = H[keepbin]; Xdes = Xdes[keepbin]


# ---- weighted Poisson NLL fit (gradient descent + backtracking) ----
def nll(c):
    mu = np.clip(Xdes @ c, 1e-10, None)
    return -2 * np.sum(H * np.log(mu) - mu) + LAM2 * np.sum(c ** 2)
def grad(c):
    mu = np.clip(Xdes @ c, 1e-10, None)
    return -2 * Xdes.T @ (H / mu - 1.0) + 2 * LAM2 * c

c = np.zeros(NCOEFF); c[0] = H.mean()
step = 1e-4; f = nll(c)
for it in range(ITERS):
    g = grad(c); ok = False
    for _ in range(30):
        cn = c - step * g
        fn = nll(cn)
        if fn < f:
            c, f, step, ok = cn, fn, step * 1.1, True; break
        step *= 0.5
    if not ok:
        break
    if it % 500 == 0:
        print(f"  iter {it}  nll {f:.1f}  step {step:.1e}")
print(f"fit done: nll={f:.1f}")
np.savez(os.path.join(HERE, "leg5d.npz"), c=c, order=ORDER, nb=NB, lam2=LAM2)


def model(Xphys):
    # chunked: design(N,3125) would be ~50 GB for N=2e6; process in batches (~2.5 GB each)
    out = np.empty(len(Xphys))
    CH = 100000
    for a in range(0, len(Xphys), CH):
        out[a:a + CH] = np.clip(design(to_unit(Xphys[a:a + CH])) @ c, 0, None)
    return out


if SPEC:   # veto run: report recovery in the hole and stop (skip the full-range eval)
    import sys, veto_recovery as vr
    vr.recovery(model, os.environ["ACC_VETO"],
                os.environ.get("BASE_OUT", os.path.join(HERE, "leg5d")) + "_rec",
                seed=SEED, label="Legendre")
    sys.exit(0)


# ---- eval: same chi2 recipe as ANN/GP (5 1D projections + cos1-cos2 2D) ----
Fu, X5u = tl.uniform_norm_sample(N_UNIF, np.random.RandomState(SEED + 1))
epsU = model(X5u)
fig, axs = plt.subplots(2, 4, figsize=(22, 9)); axs = axs.ravel()
print("\nchi2/nb per projection (5D Legendre):")
chis = []
for i in range(5):
    a = axs[i]
    b = np.linspace(*RANGES[i], 41); cc = 0.5 * (b[:-1] + b[1:])
    Hd, _ = np.histogram(Xte[:, i], bins=b, weights=Wte)
    Vd, _ = np.histogram(Xte[:, i], bins=b, weights=Wte ** 2)
    Hm, _ = np.histogram(X5u[:, i], bins=b, weights=epsU); Hm *= Hd.sum() / Hm.sum()
    okm = Vd > 0; c2 = np.sum((Hd[okm] - Hm[okm]) ** 2 / Vd[okm]) / okm.sum(); chis.append(c2)
    a.errorbar(cc, Hd / Hd.mean(), yerr=np.sqrt(Vd) / Hd.mean(), fmt="k.", ms=3, capsize=0, label="data")
    a.plot(cc, Hm / Hm.mean(), "r-", lw=1.6, label="Legendre")
    a.set_xlabel(VARS[i]); a.set_ylim(bottom=0); a.set_title(f"{VARS[i]}: chi2/nb={c2:.2f}")
    if i == 0: a.legend(frameon=False)
    print(f"  {VARS[i]:5s}  chi2/nb = {c2:.2f}")
# cos1-cos2 2D projection (indices 2,3): data map + model map side by side, shared colour scale
NB2 = 30
ex = np.linspace(-1, 1, NB2 + 1)
Hd2, _, _ = np.histogram2d(Xte[:, 2], Xte[:, 3], bins=[ex, ex], weights=Wte)
Vd2, _, _ = np.histogram2d(Xte[:, 2], Xte[:, 3], bins=[ex, ex], weights=Wte ** 2)
Hm2, _, _ = np.histogram2d(X5u[:, 2], X5u[:, 3], bins=[ex, ex], weights=epsU); Hm2 *= Hd2.sum() / Hm2.sum()
ok2 = Vd2 > 0; c2cc = np.sum((Hd2[ok2] - Hm2[ok2]) ** 2 / Vd2[ok2]) / ok2.sum()
Dn = Hd2 / Hd2[Hd2 > 0].mean(); Mn = Hm2 / Hm2[Hm2 > 0].mean(); vmax = max(Dn.max(), Mn.max())
a = axs[5]
pm = a.pcolormesh(ex, ex, Dn.T, shading="auto", cmap="afmhot_r", vmin=0, vmax=vmax)
a.set_xlabel("cos1"); a.set_ylabel("cos2"); a.set_title("cos1-cos2: data"); fig.colorbar(pm, ax=a)
a = axs[6]
pm = a.pcolormesh(ex, ex, Mn.T, shading="auto", cmap="afmhot_r", vmin=0, vmax=vmax)
a.set_xlabel("cos1"); a.set_ylabel("cos2"); a.set_title(f"cos1-cos2: Legendre fit (chi2/nb={c2cc:.2f})"); fig.colorbar(pm, ax=a)
axs[7].axis("off")
print(f"  {'cc2D':5s}  chi2/nb = {c2cc:.2f}")
fig.suptitle(f"5D Legendre baseline (order={ORDER}, {NCOEFF} coeffs)  "
             f"mean(1D) chi2/nb={np.mean(chis):.2f}  cos1-cos2={c2cc:.2f}", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(os.path.join(HERE, "leg5d_eval.png"), dpi=130)
with open(os.path.join(HERE, "leg5d.txt"), "w") as f2:
    f2.write(f"method=legendre order={ORDER} ncoeff={NCOEFF} bins={NB} lambda2={LAM2} "
             + " ".join(f"{VARS[i]}={chis[i]:.3f}" for i in range(5))
             + f" cc2D={c2cc:.3f} mean={np.mean(chis):.3f}\n")
print(f"\nSaved leg5d_eval.png  (ncoeff={NCOEFF}, mean chi2/nb={np.mean(chis):.2f}, cos1-cos2={c2cc:.2f})")
