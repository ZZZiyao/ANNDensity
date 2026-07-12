"""
Chapter 8.3 (model-assisted ANN, BACKGROUND) -- STEPS 2 & 3 + Fig.11.

Faithful to the original FitSample.py:
  STEP 2: freeze the joint ANN P(x, Theta); maximum-likelihood fit the 8 latent
          params to the SIDEBAND test sample (signal region vetoed via exp_phase_space).
  STEP 3: extrapolate B(m',theta') = P(m',theta', mD=1.97 | Theta_pred), compare
          to signal-region data + sideband shape, report Theta vs true + chi2, plot Fig.11.

model(x, pars): pad the 3 observables with the 8 optimisable latent params,
then scale * MLP -- identical to the original.

Usage:  python fit_bg_joint.py [calibfile] [outprefix] [seed]
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import amplitf.interface as atfi
import amplitf.likelihood as atfl

import tfa.optimisation as tfo
import tfa.rootio as tfr
import tfa.neural_nets as tfn

import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from DistributionModel import (
    parameters_list, exp_phase_space, observables_data, observables_titles,
    observables_phase_space, sqdlz_phsp, md, true_cuts,
)

MD_SIGNAL = 1.97
MD_S1, MD_S2 = 1.92, 2.02     # signal region bounds for projections

# ---------------- config ----------------
initfile  = os.path.join(HERE, os.environ.get("BG_INITFILE", "bg_joint_train.npy"))
calibfile = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "0.20.41e5_test.root")
outprefix = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "bg_joint_fit")
seed      = int(sys.argv[3]) if len(sys.argv) > 3 else 1
n_restart = int(float(os.environ.get("BG_FIT_RESTARTS", 5)))

plt.rcParams.update({"font.size": 13, "axes.labelsize": 15,
                     "xtick.direction": "in", "ytick.direction": "in"})

# ---------------- load frozen ANN ----------------
ann = np.load(initfile, allow_pickle=True)
(scale, ranges) = ann[:2]
(weights, biases) = tfn.init_fixed_weights_biases(ann[2:])

ndim = exp_phase_space.dimensionality()           # 3
observables_bounds = exp_phase_space.bounds()


def model(x, pars):
    vec = tf.reshape(
        tf.concat([tf.constant(ndim * [0.], dtype=atfi.fptype()), pars], axis=0),
        [1, ndim + len(pars)])
    x2 = tf.pad(x, [[0, 0], [0, len(pars)]], 'CONSTANT') + vec
    return scale * tfn.multilayer_perceptron(x2, ranges, weights, biases)


atfi.set_seed(seed)
np.random.seed(seed)

# ---------------- data (sidebands only) + normalisation ----------------
data_all = tfr.read_tuple(calibfile, branches=observables_data)[:100000, :]
data_sample = exp_phase_space.filter(data_all)            # sidebands (signal vetoed)
norm_sample = exp_phase_space.uniform_sample(int(float(os.environ.get("BG_FIT_NORM", 1000000))))


@atfi.function
def nll(pars):
    parslist = [pars[i[0]] for i in parameters_list]
    return atfl.unbinned_nll(model(data_sample, parslist),
                             atfl.integral(model(norm_sample, parslist)))


pars = [tfo.FitParameter(p[0], (p[2][0] + p[2][1]) / 2., p[2][0], p[2][1])
        for p in parameters_list]

print(f"Sideband data sample size = {len(data_sample)}")
print(f"Normalisation sample size = {len(norm_sample)}")

# ---------------- STEP 2: minuit fit ----------------
best_nll = 1e10
best_result = None
for i in range(n_restart):
    for p in pars:
        p.update(np.random.uniform(p.lower_limit, p.upper_limit))
    result = tfo.run_minuit(nll, pars)
    print(f"restart {i}: loglh = {result['loglh']:.3f}")
    if result['loglh'] < best_nll:
        best_nll = result['loglh']
        best_result = result

print("Optimization Finished!")
parslist = [best_result["params"][i[0]][0] for i in parameters_list]

true_map = {parameters_list[i][0]: float(true_cuts[i]) for i in range(len(parameters_list))}
print("\nFitted latent parameters (reproduced vs true):")
for p in parameters_list:
    name = p[0]
    fit = best_result["params"][name][0]
    err = best_result["params"][name][1]
    print(f"  {name:12s}  fit = {fit:8.4f} +/- {err:7.4f}   true = {true_map[name]:7.4f}   range {p[2]}")


# ---------------- STEP 3: signal-region extrapolation B(m',theta') = P(.,mD=1.97) ----------------
def bg_density_2d(grid_mt, md_val):
    """Evaluate model at fixed mD over a (m', theta') grid, at fitted Theta."""
    x = np.column_stack([grid_mt, np.full(grid_mt.shape[0], md_val)]).astype(np.float64)
    return model(x, parslist).numpy()


fine = 100
mf = np.linspace(0, 1, fine)
tf_arr = np.linspace(0, 1, fine)
Mg, Tg = np.meshgrid(mf, tf_arr, indexing="ij")
grid_mt = np.stack([Mg.ravel(), Tg.ravel()], axis=1)
B2d = bg_density_2d(grid_mt, MD_SIGNAL).reshape(fine, fine)
B2d = B2d / B2d.mean()        # average density = 1

# ---- test data: signal region & sidebands ----
arr = np.asarray(tfr.read_tuple(calibfile, branches=observables_data))
mO, tO, mdO = arr[:, 0], arr[:, 1], arr[:, 2]
sig = (mdO >= MD_S1) & (mdO <= MD_S2)
sb = ((mdO >= 1.77) & (mdO < MD_S1)) | ((mdO > MD_S2) & (mdO <= 2.17))
m_sig, t_sig = mO[sig], tO[sig]
m_sb, t_sb = mO[sb], tO[sb]

bins = 100
Hm_sig, em = np.histogram(m_sig, bins=bins, range=(0, 1))
Ht_sig, et = np.histogram(t_sig, bins=bins, range=(0, 1))
cm = 0.5 * (em[:-1] + em[1:]); ct = 0.5 * (et[:-1] + et[1:])

# fit projections (scaled to signal-region totals)
fit_m = np.interp(cm, mf, B2d.mean(axis=1)); fit_m *= Hm_sig.sum() / fit_m.sum()
fit_t = np.interp(ct, tf_arr, B2d.mean(axis=0)); fit_t *= Ht_sig.sum() / fit_t.sum()
# sideband shape (scaled to signal-region totals) -- the naive proxy
Hm_sb, _ = np.histogram(m_sb, bins=em); Hm_sb = Hm_sb * Hm_sig.sum() / max(Hm_sb.sum(), 1)
Ht_sb, _ = np.histogram(t_sb, bins=et); Ht_sb = Ht_sb * Ht_sig.sum() / max(Ht_sb.sum(), 1)

# chi2 on 50x50 (signal region)
H50, xe, ye = np.histogram2d(m_sig, t_sig, bins=(50, 50), range=[[0, 1], [0, 1]])
xc = 0.5 * (xe[:-1] + xe[1:]); yc = 0.5 * (ye[:-1] + ye[1:])
Mc, Tc = np.meshgrid(xc, yc, indexing="ij")
g50 = np.stack([Mc.ravel(), Tc.ravel()], axis=1)
mu = bg_density_2d(g50, MD_SIGNAL); mu *= H50.sum() / mu.sum()
mu50 = mu.reshape(H50.shape)
valid = mu50 > 0
chi2 = float(np.sum((H50[valid] - mu50[valid]) ** 2 / mu50[valid]))
print(f"\nChi2 (signal region, 50x50) = {chi2:.1f}")

# ---- Fig.11 ----
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
vmax = np.ceil(B2d.max() * 10) / 10
cf = axes[0].contourf(mf, tf_arr, B2d.T, levels=np.linspace(0, vmax, 21), cmap="afmhot_r")
axes[0].set_xlabel("m'"); axes[0].set_ylabel("θ'"); axes[0].set_title("(a)")
fig.colorbar(cf, ax=axes[0], label="B(m', θ')")

axes[1].errorbar(cm, Hm_sig, yerr=np.sqrt(Hm_sig), fmt="k.", ms=3, capsize=0, label="Signal region")
axes[1].plot(cm, Hm_sb, color="m", ls="--", lw=1.3, label="Sidebands")
axes[1].plot(cm, fit_m, "r-", lw=1.8, label="Fit result")
axes[1].set_xlabel("m'"); axes[1].set_ylabel(f"Entries / ({em[1]-em[0]:.2f})")
axes[1].set_title("(b)"); axes[1].set_ylim(bottom=0); axes[1].legend(frameon=False, fontsize=9)

axes[2].errorbar(ct, Ht_sig, yerr=np.sqrt(Ht_sig), fmt="k.", ms=3, capsize=0, label="Signal region")
axes[2].plot(ct, Ht_sb, color="m", ls="--", lw=1.3, label="Sidebands")
axes[2].plot(ct, fit_t, "r-", lw=1.8, label="Fit result")
axes[2].set_xlabel("θ'"); axes[2].set_ylabel(f"Entries / ({et[1]-et[0]:.2f})")
axes[2].set_title("(c)"); axes[2].set_ylim(bottom=0); axes[2].legend(frameon=False, fontsize=9)

for ax in axes:
    ax.minorticks_on(); ax.tick_params(which="both", top=True, right=True)

fig.suptitle(f"Model-assisted ANN background (signal region, $m_D$={MD_SIGNAL})   "
             f"χ²(50×50) = {chi2:.1f}", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(outprefix + "_fig11.png", dpi=300)
print(f"Saved {outprefix}_fig11.png")
