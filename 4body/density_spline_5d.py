"""
5D cubic-spline baseline for the 4-body acceptance -- the other "traditional" method whose
grid blows up with dimension: knots^5 nodes.

Bin the weight_detJ-weighted data into a 5D grid (knots per dim), then cubic-interpolate
between the bin centres (scipy RegularGridInterpolator, method='cubic') to get a continuous
eps. Evaluate with the SAME chi2 recipe as ANN/GP.

CPU only. Env: SPL_KNOTS (per-dim, default 8), ACC_NTEST, ACC_SEED, ACC_EVAL_UNIF.
Output: spl5d_eval.png + chi2 printed. Highlights: edge/empty bins -> interpolation rings.
"""
import os
import numpy as np
from scipy.ndimage import map_coordinates   # order=3 -> cubic B-spline on a regular grid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import transform_cut as tl

HERE = os.path.dirname(os.path.abspath(__file__))
def _i(n, d): return int(float(os.environ.get(n, d)))
K = _i("SPL_KNOTS", 8)
SEED = _i("ACC_SEED", 1); NTEST = _i("ACC_NTEST", 1000000); N_UNIF = _i("ACC_EVAL_UNIF", 2000000)
RANGES = tl.ORIG_RANGES; VARS = ["m1", "m2", "cos1", "cos2", "phi"]
print(f"5D cubic spline: {K} knots/dim -> {K**5} nodes")

z = np.load(os.path.join(HERE, "runs12_cut.npz"))
X5, W = z["X5"].astype(np.float64), z["W"].astype(np.float64)
rng = np.random.RandomState(SEED); perm = rng.permutation(len(X5))
te = perm[len(X5) - NTEST:]; tr = perm[:len(X5) - NTEST]
Xtr, Wtr = X5[tr], W[tr]; Xte, Wte = X5[te], W[te]

# 5D weighted histogram on the knot grid. Sum-of-W per (uniform-volume) bin is proportional to
# eps at that bin (events sample PS, weight=weight_detJ removes it) -- SAME convention as Legendre
# and the eval. Empty bins -> 0; the spline must interpolate over them (the dimensionality story).
edges = [np.linspace(*RANGES[i], K + 1) for i in range(5)]
ctrs = [0.5 * (e[:-1] + e[1:]) for e in edges]
Hs, _ = np.histogramdd(Xtr, bins=edges, weights=Wtr)
Hc, _ = np.histogramdd(Xtr, bins=edges)
dens = Hs / Hs[Hs > 0].mean()                               # normalise avg=1 (over filled bins)
nempty = int((Hc == 0).sum())
print(f"grid {K}^5={K**5} bins; empty bins: {nempty} ({100*nempty/K**5:.1f}%)  <- spline interpolates over these")

# ---- VETO: overwrite hole nodes with the nearest non-veto node (a local interpolator can only
#      carry surrounding info into the hole; this is the spline's honest hole-filling attempt) ----
import veto as vt
SPEC = vt.parse(os.environ.get("ACC_VETO", ""))
if SPEC:
    from scipy.spatial import cKDTree
    mesh = np.meshgrid(*ctrs, indexing="ij")
    Xnode = np.stack([m.ravel() for m in mesh], axis=1)
    vmask = vt.mask(Xnode, SPEC)
    flat = dens.ravel().copy()
    _, idx = cKDTree(Xnode[~vmask]).query(Xnode[vmask])
    flat[vmask] = flat[~vmask][idx]                        # nearest non-veto node -> hole
    dens = flat.reshape(dens.shape)
    print(f"VETO [{vt.describe(SPEC)}]: {int(vmask.sum())} hole nodes filled by nearest non-veto node")

def model(Xphys):
    # map physical coords -> fractional grid index, then cubic (order=3) B-spline interpolation
    coords = np.stack([(Xphys[:, i] - ctrs[i][0]) / (ctrs[i][1] - ctrs[i][0]) for i in range(5)], axis=0)
    return np.clip(map_coordinates(dens, coords, order=3, mode="nearest"), 0, None)

if SPEC:   # veto run: report recovery in the hole and stop
    import sys, veto_recovery as vr
    vr.recovery(model, os.environ["ACC_VETO"],
                os.environ.get("BASE_OUT", os.path.join(HERE, "spl5d")) + "_rec",
                seed=SEED, label="spline")
    sys.exit(0)

Fu, X5u = tl.uniform_norm_sample(N_UNIF, np.random.RandomState(SEED + 1))
epsU = model(X5u)   # map_coordinates(mode='nearest') extrapolates beyond the grid centres

fig, axs = plt.subplots(2, 4, figsize=(22, 9)); axs = axs.ravel()
print("\nchi2/nb per projection (5D cubic spline):")
chis = []
for i in range(5):
    a = axs[i]
    b = np.linspace(*RANGES[i], 41); cc = 0.5 * (b[:-1] + b[1:])
    Hd, _ = np.histogram(Xte[:, i], bins=b, weights=Wte)
    Vd, _ = np.histogram(Xte[:, i], bins=b, weights=Wte ** 2)
    Hm, _ = np.histogram(X5u[:, i], bins=b, weights=epsU); Hm *= Hd.sum() / Hm.sum()
    okm = Vd > 0; c2 = np.sum((Hd[okm] - Hm[okm]) ** 2 / Vd[okm]) / okm.sum(); chis.append(c2)
    a.errorbar(cc, Hd / Hd.mean(), yerr=np.sqrt(Vd) / Hd.mean(), fmt="k.", ms=3, capsize=0, label="data")
    a.plot(cc, Hm / Hm.mean(), "r-", lw=1.6, label="spline")
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
a.set_xlabel("cos1"); a.set_ylabel("cos2"); a.set_title(f"cos1-cos2: spline fit (chi2/nb={c2cc:.2f})"); fig.colorbar(pm, ax=a)
axs[7].axis("off")
print(f"  {'cc2D':5s}  chi2/nb = {c2cc:.2f}")
fig.suptitle(f"5D cubic spline baseline ({K} knots/dim, {K**5} nodes, {nempty} empty)  "
             f"mean(1D) chi2/nb={np.mean(chis):.2f}  cos1-cos2={c2cc:.2f}", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(os.path.join(HERE, "spl5d_eval.png"), dpi=130)
with open(os.path.join(HERE, "spl5d.txt"), "w") as f2:
    f2.write(f"method=spline knots={K} nodes={K**5} empty={nempty} "
             + " ".join(f"{VARS[i]}={chis[i]:.3f}" for i in range(5))
             + f" cc2D={c2cc:.3f} mean={np.mean(chis):.3f}\n")
print(f"\nSaved spl5d_eval.png  (nodes={K**5}, empty={nempty}, mean chi2/nb={np.mean(chis):.2f}, cos1-cos2={c2cc:.2f})")
