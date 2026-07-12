"""
Veto-recovery evaluation: how well does a model (ANN or GP) reconstruct the acceptance eps
INSIDE the vetoed region, having never seen data there?

Ground truth = the weight_detJ-weighted data that WAS vetoed (held out). We compare, inside
the veto region, the model's eps to this truth and report:
  - recovery ratio   = <eps_model> / <eps_true>     (1 = perfect)
  - recovery chi2/nb = bin-by-bin weighted chi2 in the veto region (in the vetoed variable(s))
  - (GP only) coverage = fraction of veto-region bins where |model-true| < 1*sigma_GP band

Usage: python eval_veto.py <model.npy> "<veto spec>" [outprefix]
       (ANN .npy from train_acc_6d; GP handled by eval_veto_gp via --gp, see eval_veto_gp.py)
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
import veto as vt

HERE = os.path.dirname(os.path.abspath(__file__))
def _i(n, d): return int(float(os.environ.get(n, d)))

modelfile = sys.argv[1]
spec_str  = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("ACC_VETO", "")
outprefix = sys.argv[3] if len(sys.argv) > 3 else os.path.splitext(modelfile)[0] + "_veto"
spec = vt.parse(spec_str)
N_UNIF = _i("ACC_EVAL_UNIF", 2000000)

model = mlp.model_fn_from_npy(np.load(modelfile, allow_pickle=True))
z = np.load(os.path.join(HERE, "runs12_cut.npz"))
X5, W = z["X5"].astype(np.float64), z["W"].astype(np.float64)
inveto = vt.mask(X5, spec)
print(f"veto [{vt.describe(spec)}]: {inveto.sum()} truth events in the hole")

# uniform sample over full domain -> features -> model eps; keep those inside the veto
Fu, X5u = tl.uniform_norm_sample(N_UNIF, np.random.RandomState(11))
epsU = model(Fu).astype(np.float64)
uin = vt.mask(X5u, spec)

# global normalisation: scale model eps so its mean over the FULL domain matches data
# (data weighted hist over full domain, normalised avg=1)
gnorm = W.mean()  # not used directly; we normalise both to avg=1 below

# --- recovery: project onto the vetoed variable(s) ---
# pick the variable with the widest veto window as the projection axis
vvar = max(spec, key=lambda s: s[2] - s[1])[0] if spec else 0
ORIG = tl.ORIG_RANGES; VARS = ["m1", "m2", "cos1", "cos2", "phi"]
# restrict to the veto "column" (every OTHER veto window) so a 2D/3D projection isolates the hole
others = [(v, lo, hi) for (v, lo, hi) in spec if v != vvar]
def _col(A):
    m = np.ones(len(A), bool)
    for (v, lo, hi) in others:
        m &= (A[:, v] > lo) & (A[:, v] < hi)
    return m
cd, cu = _col(X5), _col(X5u)
lo, hi = ORIG[vvar]; b = np.linspace(lo, hi, 41); c = 0.5 * (b[:-1] + b[1:])
# data (true) eps in the column, normalised avg=1
Hd, _ = np.histogram(X5[cd, vvar], bins=b, weights=W[cd])
Vd, _ = np.histogram(X5[cd, vvar], bins=b, weights=W[cd] ** 2)
Hd_n = Hd / Hd.mean(); Ed_n = np.sqrt(Vd) / Hd.mean()
# model eps in the column (uniform sample weighted by model), normalised avg=1
Hm, _ = np.histogram(X5u[cu, vvar], bins=b, weights=epsU[cu]); Hm_n = Hm / Hm.mean()
Hm_n *= Hd_n.mean() / Hm_n.mean()

# veto-region bins along this axis
vlo, vhi = [(s[1], s[2]) for s in spec if s[0] == vvar][0]
vbins = (c > vlo) & (c < vhi)
ratio = Hm_n[vbins].mean() / Hd_n[vbins].mean()
ok = (Vd > 0) & vbins
chi2 = np.sum((Hd_n[ok] - Hm_n[ok]) ** 2 / (Ed_n[ok] ** 2)) / max(ok.sum(), 1)
print(f"\nveto-region recovery ({VARS[vvar]} axis, {vbins.sum()} bins):")
print(f"  recovery ratio = {ratio:.3f}   (1.0 = perfect)")
print(f"  recovery chi2/nb = {chi2:.2f}")

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.errorbar(c, Hd_n, yerr=Ed_n, fmt="k.", ms=4, capsize=0, label="TRUE eps (held-out)")
ax.plot(c, Hm_n, "r-", lw=2, label="model (trained WITHOUT veto)")
ax.axvspan(vlo, vhi, color="tab:blue", alpha=0.15, label="VETO region")
ax.set_xlabel(VARS[vvar]); ax.set_ylabel("eps (avg=1)"); ax.set_ylim(bottom=0); ax.legend()
ax.set_title(f"veto recovery: {vt.describe(spec)}\nratio={ratio:.3f}  chi2/nb={chi2:.2f}")
plt.tight_layout(); plt.savefig(outprefix + ".png", dpi=130)
print(f"Saved {outprefix}.png")
# machine-readable line for aggregation
with open(outprefix + ".txt", "w") as f:
    f.write(f"veto='{spec_str}' axis={VARS[vvar]} ratio={ratio:.4f} chi2nb={chi2:.4f} "
            f"nbins={int(vbins.sum())} ntruth={int(inveto.sum())}\n")
