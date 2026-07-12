"""
Full 5D acceptance GP on GPU via GPyTorch (GPy is CPU-only).

Exact GP on a binned 5D grid: bin the validated flattened data into a regular 5D grid; on
a 5D grid SUM(W) per bin is proportional to the density eps at that bin centre, so the GP
regresses eps(u1,u2,cos1,cos2,phi) directly.  GPyTorch runs the exact GP on GPU with
conjugate gradients (no explicit O(N^3) Cholesky), and -- unlike GPy's grid model -- uses
PER-BIN heteroscedastic noise from the weighted-histogram variance (SUM(W^2)), which is the
statistically correct error for weighted data.

Kernel configurable (GP_KERNEL = rbf | matern52 | matern32), ARD (length scale per dim),
ScaleKernel + ConstantMean.  Inputs normalised to [0,1].

Env: GP_BINS="10,10,8,8,5" (per-dim grid), GP_KERNEL=matern52, GP_ITERS=300,
     ACC_NTRAIN=5000000, ACC_NTEST, ACC_SEED.  Needs torch + gpytorch (env 'gptorch').
Output: gp_torch_5d.png (5 1D projections + cos1-cos2 2D, Data vs GP) + chi2 table; saves
        gp_torch_5d_model.pt.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import gpytorch

HERE = os.path.dirname(os.path.abspath(__file__))

def _i(n, d): return int(float(os.environ.get(n, d)))
NTRAIN = _i("ACC_NTRAIN", 5000000)
NTEST  = _i("ACC_NTEST", 1000000)
SEED   = _i("ACC_SEED", 1)
ITERS  = _i("GP_ITERS", 300)
BINS = [int(b) for b in os.environ.get("GP_BINS", "10,10,8,8,5").split(",")]
KERNEL = os.environ.get("GP_KERNEL", "matern52").lower()
OUT = os.environ.get("GP_OUT", os.path.join(HERE, f"gp_torch_cut_{KERNEL}"))
# CUT representation (matches the cut+6d ANN): mass cut (0.65,1.03), original coords,
# weight_detJ. GP grids the 5 cut coords; phi as a plain dim (eps is flat in phi).
RANGES = [(0.65, 1.03), (0.65, 1.03), (-1.0, 1.0), (-1.0, 1.0), (-np.pi, np.pi)]
VARS = ["m1", "m2", "cos1", "cos2", "phi"]

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {dev}  grid {BINS} = {int(np.prod(BINS))} bins  kernel {KERNEL}")

# ---- data + split (cut sample, original coords + weight_detJ) ----
z = np.load(os.path.join(HERE, "runs12_cut.npz"))
X5, W = z["X5"].astype(np.float64), z["W"].astype(np.float64)
rng = np.random.RandomState(SEED)
perm = rng.permutation(len(X5))
tr, te = perm[:NTRAIN], perm[NTRAIN:NTRAIN + NTEST]
Xtr, Wtr = X5[tr], W[tr]
Xte, Wte = X5[te], W[te]

edges = [np.linspace(*RANGES[i], BINS[i] + 1) for i in range(5)]
ctrs = [0.5 * (e[:-1] + e[1:]) for e in edges]
lo = np.array([RANGES[i][0] for i in range(5)]); span = np.array([RANGES[i][1] - RANGES[i][0] for i in range(5)])

# ---- 5D weighted histogram: Y (density) + heteroscedastic noise (SUM W^2) ----
Hsum, _ = np.histogramdd(Xtr, bins=edges, weights=Wtr)
Hsq, _ = np.histogramdd(Xtr, bins=edges, weights=Wtr ** 2)
ymean = Hsum.mean()
Y = (Hsum / ymean).ravel()
noise = (Hsq / ymean ** 2).ravel()
noise[noise <= 0] = noise[noise > 0].mean()           # floor empty bins (Y=0, moderate uncertainty)

mesh = np.meshgrid(*ctrs, indexing="ij")
Xg = np.stack([(mesh[i].ravel() - lo[i]) / span[i] for i in range(5)], axis=1)  # normalised [0,1]
Xg_orig = np.stack([mesh[i].ravel() for i in range(5)], axis=1)                 # orig coords for veto

# ---- VETO: drop grid bins inside the veto region from TRAINING (GP fills the hole by kernel
#      interpolation, with LARGE posterior uncertainty there -- that band is the GP's selling point)
import veto as vt
SPEC = vt.parse(os.environ.get("ACC_VETO", ""))
if SPEC:
    keepbin = ~vt.mask(Xg_orig, SPEC)
    print(f"VETO [{vt.describe(SPEC)}]: training on {keepbin.sum()} of {len(Y)} grid bins")
    Xg, Y, noise = Xg[keepbin], Y[keepbin], noise[keepbin]

Xt = torch.tensor(Xg, dtype=torch.float32, device=dev)
Yt = torch.tensor(Y, dtype=torch.float32, device=dev)
Nt = torch.tensor(noise, dtype=torch.float32, device=dev)
print(f"grid points {len(Y)} (kernel matrix ~{len(Y)**2*4/1e9:.1f} GB)")


def build_covar():
    K = gpytorch.kernels
    if KERNEL == "spectral":
        sm = K.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=5)
        sm.initialize_from_data(Xt, Yt)                 # SMK has its own scale; don't wrap
        return sm
    base = {
        "rbf": lambda: K.RBFKernel(ard_num_dims=5),
        "matern12": lambda: K.MaternKernel(nu=0.5, ard_num_dims=5),
        "matern32": lambda: K.MaternKernel(nu=1.5, ard_num_dims=5),
        "matern52": lambda: K.MaternKernel(nu=2.5, ard_num_dims=5),
        "rq": lambda: K.RQKernel(ard_num_dims=5),
    }[KERNEL]()
    return K.ScaleKernel(base)


class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, X, Y, lik):
        super().__init__(X, Y, lik)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = build_covar()

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


lik = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=Nt, learn_additional_noise=True).to(dev)
model = ExactGP(Xt, Yt, lik).to(dev)

# ---- train (maximise marginal likelihood) ----
model.train(); lik.train()
opt = torch.optim.Adam(model.parameters(), lr=0.05)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
with gpytorch.settings.max_cg_iterations(2000), gpytorch.settings.cg_tolerance(1e-2):
    for i in range(ITERS):
        opt.zero_grad()
        loss = -mll(model(Xt), Yt)
        loss.backward(); opt.step()
        if i % 50 == 0 or i == ITERS - 1:
            print(f"iter {i:4d}  loss {loss.item():.4f}")
try:
    bk = model.covar_module.base_kernel if hasattr(model.covar_module, "base_kernel") else model.covar_module
    ls = bk.lengthscale.detach().cpu().numpy().ravel()
    print("ARD length scales (normalised):", np.round(ls, 3))
except Exception:
    ls = np.array([]); print("(per-dim length scales N/A for this kernel)")
final_loss = float(loss.item())

# ---- evaluate EXACTLY like the ANN: predict GP on a uniform 5D sample, then 40-bin chi2 ----
# (training grid stays coarse -- exact-GP cost -- but the GP is continuous, so we marginalise
#  via a uniform sample weighted by eps, identical to eval_acc_5d.py -> directly comparable.)
model.eval(); lik.eval()
NU = _i("ACC_EVAL_UNIF", 2000000)
ru = np.random.RandomState(SEED + 1)
Uph = np.column_stack([ru.uniform(*RANGES[k], NU) for k in range(5)])          # physical coords
Un = ((Uph - lo) / span).astype(np.float32)                                    # normalised for GP
epsU = np.empty(NU); sdU = np.empty(NU)
CH = _i("GP_PRED_CHUNK", 15000)              # Kmn batch = CH x N_grid; keep small to avoid GPU OOM
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    for a in range(0, NU, CH):
        p = model(torch.tensor(Un[a:a + CH], device=dev))
        epsU[a:a + CH] = p.mean.cpu().numpy()
        sdU[a:a + CH] = p.stddev.cpu().numpy()
epsU = np.clip(epsU, 1e-12, None)

# ---- VETO run: recovery ratio + chi2 + GP coverage in the hole (same metric as the other
#      methods, plus coverage = fraction of hole bins with truth within the GP +-1 sigma band) ----
if SPEC:
    import sys
    vvar = max(SPEC, key=lambda s: s[2] - s[1])[0]
    others = [(v, lo2, hi2) for (v, lo2, hi2) in SPEC if v != vvar]   # restrict to veto column
    def _col(A):
        m = np.ones(len(A), bool)
        for (v, lo2, hi2) in others:
            m &= (A[:, v] > lo2) & (A[:, v] < hi2)
        return m
    cd, cu = _col(X5), _col(Uph)
    e = np.linspace(*RANGES[vvar], 41); cc = 0.5 * (e[:-1] + e[1:])
    Hd, _ = np.histogram(X5[cd, vvar], bins=e, weights=W[cd])     # truth in column (incl. hole)
    Vd, _ = np.histogram(X5[cd, vvar], bins=e, weights=W[cd] ** 2)
    Hm_raw, _ = np.histogram(Uph[cu, vvar], bins=e, weights=epsU[cu])
    bd_raw, _ = np.histogram(Uph[cu, vvar], bins=e, weights=sdU[cu])   # marginalised GP uncertainty
    Hd_n = Hd / Hd.mean(); Ed_n = np.sqrt(Vd) / Hd.mean()
    ms = 1.0 / Hm_raw.mean(); Hm_n = Hm_raw * ms; band_n = bd_raw * ms
    vlo, vhi = [(t[1], t[2]) for t in SPEC if t[0] == vvar][0]
    vb = (cc > vlo) & (cc < vhi); ok = (Vd > 0) & vb
    ratio = Hm_n[vb].mean() / Hd_n[vb].mean()
    chi2 = np.sum((Hd_n[ok] - Hm_n[ok]) ** 2 / Ed_n[ok] ** 2) / max(ok.sum(), 1)
    cov = float(np.mean(np.abs(Hd_n[ok] - Hm_n[ok]) < band_n[ok]))   # within 1 sigma_GP
    fig, axr = plt.subplots(figsize=(9, 5.5))
    axr.errorbar(cc, Hd_n, yerr=Ed_n, fmt="k.", ms=4, capsize=0, label="TRUE eps (held-out)")
    axr.fill_between(cc, Hm_n - band_n, Hm_n + band_n, color="r", alpha=0.25, label="GP ±1σ")
    axr.plot(cc, Hm_n, "r-", lw=2, label="GP (fit WITHOUT veto)")
    axr.axvspan(vlo, vhi, color="tab:blue", alpha=0.15, label="VETO region")
    axr.set_xlabel(VARS[vvar]); axr.set_ylabel("eps (avg=1)"); axr.set_ylim(bottom=0); axr.legend()
    axr.set_title(f"veto recovery [GP {KERNEL}]: {vt.describe(SPEC)}\n"
                  f"ratio={ratio:.3f}  chi2/nb={chi2:.2f}  coverage={cov:.2f}")
    plt.tight_layout(); plt.savefig(OUT + "_rec.png", dpi=130)
    with open(OUT + "_rec.txt", "w") as f:
        f.write(f"method=GP_{KERNEL} veto='{os.environ.get('ACC_VETO','')}' axis={VARS[vvar]} "
                f"ratio={ratio:.4f} chi2nb={chi2:.4f} coverage={cov:.4f} nbins={int(vb.sum())}\n")
    print(f"  [GP {KERNEL}] recovery ratio={ratio:.3f} chi2/nb={chi2:.2f} coverage={cov:.2f}  saved {OUT}_rec.png")
    # save the trained GP so any projection can be re-plotted locally later. Before this the veto
    # branch exited here and the model was discarded (only _rec survived). ~1 MB: normalised grid +
    # targets + noise + kernel state -- enough to rebuild ExactGP(Xt,Yt,lik) and load_state_dict.
    torch.save({"state": model.state_dict(),
                "Xg": Xt.cpu().numpy().astype("float32"),
                "Y": Yt.cpu().numpy().astype("float32"),
                "noise": Nt.cpu().numpy().astype("float32"),
                "bins": BINS, "kernel": KERNEL, "ranges": RANGES,
                "lo": np.asarray(lo), "span": np.asarray(span),
                "lengthscales": ls, "spec": os.environ.get("ACC_VETO", "")},
               OUT + "_model.pt")
    print(f"  saved {OUT}_model.pt")
    sys.exit(0)

NB1, NB2 = 40, 30

def chi2_1d(i):
    e = np.linspace(*RANGES[i], NB1 + 1); c = 0.5 * (e[:-1] + e[1:])
    Hd, _ = np.histogram(Xte[:, i], bins=e, weights=Wte)
    Vd, _ = np.histogram(Xte[:, i], bins=e, weights=Wte ** 2)
    Hm, _ = np.histogram(Uph[:, i], bins=e, weights=epsU); s = Hd.sum() / Hm.sum(); Hm *= s
    band, _ = np.histogram(Uph[:, i], bins=e, weights=sdU); band *= s            # marginalised GP uncertainty
    ok = Vd > 0
    return c, Hd, np.sqrt(Vd), Hm, band, np.sum((Hd[ok] - Hm[ok]) ** 2 / Vd[ok]) / ok.sum()

def chi2_cc():
    ex = np.linspace(-1, 1, NB2 + 1); ey = np.linspace(-1, 1, NB2 + 1)
    Hd, _, _ = np.histogram2d(Xte[:, 2], Xte[:, 3], bins=[ex, ey], weights=Wte)
    Vd, _, _ = np.histogram2d(Xte[:, 2], Xte[:, 3], bins=[ex, ey], weights=Wte ** 2)
    Hm, _, _ = np.histogram2d(Uph[:, 2], Uph[:, 3], bins=[ex, ey], weights=epsU); Hm *= Hd.sum() / Hm.sum()
    ok = Vd > 0
    return ex, ey, Hm / Hm[Hm > 0].mean(), np.sum((Hd[ok] - Hm[ok]) ** 2 / Vd[ok]) / ok.sum()

print("\nchi2/nb per projection (GPyTorch 5D, {}, 40-bin uniform eval = same as ANN):".format(KERNEL))
proj = [chi2_1d(i) for i in range(5)]
for i in range(5):
    print(f"  {VARS[i]:5s}  chi2/nb = {proj[i][5]:.2f}")
ccx, ccy, ccmap, c2cc = chi2_cc()
print(f"  cos1-cos2 2D  chi2/nb = {c2cc:.2f}   (swish ANN ~1.12)")

# one-line summary per kernel (for cross-kernel comparison after the array finishes)
with open(OUT + ".txt", "w") as f:
    f.write(f"kernel={KERNEL} bins={BINS} loss={final_loss:.4f} "
            + " ".join(f"{VARS[i]}={proj[i][5]:.3f}" for i in range(5))
            + f" cc2D={c2cc:.3f} ls={np.round(ls,3).tolist()}\n")
torch.save({"state": model.state_dict(), "bins": BINS, "kernel": KERNEL, "ranges": RANGES,
            "lengthscales": ls}, OUT + "_model.pt")

# ---- plot ----
fig, ax = plt.subplots(2, 3, figsize=(18, 9))
for i in range(5):
    a = ax[i // 3][i % 3]
    c, Hd, Ed, Hm, band, c2 = proj[i]
    a.errorbar(c, Hd, yerr=Ed, fmt="k.", ms=3, capsize=0, label="Data")
    a.fill_between(c, Hm - band, Hm + band, color="r", alpha=0.25, label="GP ±1σ")
    a.plot(c, Hm, "r-", lw=1.6, label="GP fit")
    a.set_xlabel(VARS[i]); a.set_ylim(bottom=0); a.set_title(f"{VARS[i]}: chi2/nb={c2:.2f}")
    if i == 0:
        a.legend(frameon=False)
a6 = ax[1][2]
pm = a6.pcolormesh(ccx, ccy, ccmap.T, shading="auto", cmap="afmhot_r", vmin=0)
a6.set_xlabel("cos1"); a6.set_ylabel("cos2"); a6.set_title(f"GP eps(cos1,cos2)  chi2/nb={c2cc:.2f}")
fig.colorbar(pm, ax=a6)
fig.suptitle(f"4-body acceptance: GPyTorch 5D GP ({KERNEL} ARD, grid {BINS}, {dev})", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(OUT + ".png", dpi=120)
print(f"\nSaved {OUT}.png + {OUT}_model.pt + {OUT}.txt")
