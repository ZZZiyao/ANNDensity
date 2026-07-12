"""
Chapter 8.2 (model-assisted ANN, EFFICIENCY) -- STEPS 2 & 3 + Fig.10.

Faithful to the original FitSample.py:
  STEP 2: freeze the joint ANN P(x, Theta); maximum-likelihood fit the 5 latent
          cut parameters to the test sample (the "real data" at true_cuts).
  STEP 3: evaluate P(x | Theta_pred), report Theta vs true + chi2, plot Fig.10.

model(x, pars): pad the 2D observables with the 5 optimisable latent params,
then scale * MLP  -- identical to the original.

Usage:  python fit_eff_joint.py [calibfile] [outprefix] [seed]
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import amplitf.interface as atfi
# NOTE: do NOT set_single_precision here -- init_fixed_weights_biases casts the
# saved float32 weights up to fptype (float64), so the whole graph runs float64.
import amplitf.likelihood as atfl

import tfa.optimisation as tfo
import tfa.rootio as tfr
import tfa.neural_nets as tfn

import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from DistributionModel import (
    parameters_list, observables_phase_space, observables_data, true_cuts,
)

# ---------------- config ----------------
initfile  = os.path.join(HERE, os.environ.get("EFF_INITFILE", "eff_joint_train.npy"))
calibfile = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "eff_toy_1e5.root")
outprefix = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "eff_joint_fit")
seed      = int(sys.argv[3]) if len(sys.argv) > 3 else 1
n_restart = int(float(os.environ.get("EFF_FIT_RESTARTS", 5)))

plt.rcParams.update({"font.size": 14, "axes.labelsize": 16,
                     "xtick.direction": "in", "ytick.direction": "in"})

# ---------------- load frozen ANN ----------------
ann = np.load(initfile, allow_pickle=True)
(scale, ranges) = ann[:2]
(weights, biases) = tfn.init_fixed_weights_biases(ann[2:])

ndim = observables_phase_space.dimensionality()           # 2
observables_bounds = observables_phase_space.bounds()


def model(x, pars):
    # pad 2D observables with the constant (optimisable) latent params, then MLP
    vec = tf.reshape(
        tf.concat([tf.constant(ndim * [0.], dtype=atfi.fptype()), pars], axis=0),
        [1, ndim + len(pars)])
    x2 = tf.pad(x, [[0, 0], [0, len(pars)]], 'CONSTANT') + vec
    return scale * tfn.multilayer_perceptron(x2, ranges, weights, biases)


atfi.set_seed(seed)
np.random.seed(seed)

# ---------------- data + normalisation ----------------
data_sample = tfr.read_tuple(calibfile, branches=observables_data)[:100000, :]
data_sample = observables_phase_space.filter(data_sample)
NG = int(float(os.environ.get("EFF_FIT_NORMGRID", 200)))   # normalisation grid NGxNG
norm_sample = observables_phase_space.rectangular_grid_sample((NG, NG))


@atfi.function
def nll(pars):
    parslist = [pars[i[0]] for i in parameters_list]
    return atfl.unbinned_nll(model(data_sample, parslist),
                             atfl.integral(model(norm_sample, parslist)))


pars = [tfo.FitParameter(p[0], (p[2][0] + p[2][1]) / 2., p[2][0], p[2][1])
        for p in parameters_list]

print(f"Data sample size = {len(data_sample)}")
print(f"Normalisation grid size = {len(norm_sample)}")

# ---------------- STEP 2: minuit fit (restart, keep best) ----------------
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
    print(f"  {name:10s}  fit = {fit:8.4f} +/- {err:7.4f}   true = {true_map[name]:7.4f}   range {p[2]}")

# ---------------- STEP 3: density at Theta_pred, chi2, Fig.10 ----------------
def fit_model_2d(x):
    return model(x, parslist)


# fine grid for the 2D density and projections
fine = 200
mf = np.linspace(0, 1, fine).astype(np.float64)
tf_arr = np.linspace(0, 1, fine).astype(np.float64)
Mg, Tg = np.meshgrid(mf, tf_arr, indexing="ij")
grid = np.stack([Mg.ravel(), Tg.ravel()], axis=1)
pdf_2d = fit_model_2d(tf.constant(grid)).numpy().reshape(fine, fine)
pdf_2d = pdf_2d / pdf_2d.mean()          # normalise: average density = 1

# test-data projections
m_test = data_sample[:, 0].numpy() if hasattr(data_sample, "numpy") else np.asarray(data_sample)[:, 0]
t_test = np.asarray(data_sample)[:, 1]
bins_proj = 100
H_m, edm = np.histogram(m_test, bins=bins_proj, range=(0, 1))
H_t, edt = np.histogram(t_test, bins=bins_proj, range=(0, 1))
cm = 0.5 * (edm[:-1] + edm[1:]); ct = 0.5 * (edt[:-1] + edt[1:])
fit_m = np.interp(cm, mf, pdf_2d.mean(axis=1)); fit_m *= H_m.sum() / fit_m.sum()
fit_t = np.interp(ct, tf_arr, pdf_2d.mean(axis=0)); fit_t *= H_t.sum() / fit_t.sum()

# chi2 on 50x50
grid50_m = np.linspace(0, 1, 50);
H50, xe, ye = np.histogram2d(m_test, t_test, bins=(50, 50), range=[[0, 1], [0, 1]])
xc = 0.5 * (xe[:-1] + xe[1:]); yc = 0.5 * (ye[:-1] + ye[1:])
Mc, Tc = np.meshgrid(xc, yc, indexing="ij")
g50 = np.stack([Mc.ravel().astype(np.float64), Tc.ravel().astype(np.float64)], axis=1)
mu = fit_model_2d(tf.constant(g50)).numpy(); mu *= H50.sum() / mu.sum()
mu50 = mu.reshape(H50.shape)
valid = mu50 > 0
chi2 = float(np.sum((H50[valid] - mu50[valid]) ** 2 / mu50[valid]))
print(f"\nChi2 (test, 50x50) = {chi2:.1f}")

# ---- Fig.10 ----
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
vmax = np.ceil(pdf_2d.max() * 10) / 10
cf = axes[0].contourf(mf, tf_arr, pdf_2d.T, levels=np.linspace(0, vmax, 21), cmap="afmhot_r")
axes[0].set_xlabel("m'"); axes[0].set_ylabel("θ'"); axes[0].set_title("(a)")
fig.colorbar(cf, ax=axes[0], label="ε(m', θ')")

axes[1].errorbar(cm, H_m, yerr=np.sqrt(H_m), fmt="k.", ms=3, capsize=0, label="Simulation")
axes[1].plot(cm, fit_m, "r-", lw=1.5, label="Fit result")
axes[1].set_xlabel("m'"); axes[1].set_ylabel(f"Entries / ({edm[1]-edm[0]:.2f})")
axes[1].set_title("(b)"); axes[1].set_ylim(bottom=0); axes[1].legend(frameon=False, fontsize=10)

axes[2].errorbar(ct, H_t, yerr=np.sqrt(H_t), fmt="k.", ms=3, capsize=0, label="Simulation")
axes[2].plot(ct, fit_t, "r-", lw=1.5, label="Fit result")
axes[2].set_xlabel("θ'"); axes[2].set_ylabel(f"Entries / ({edt[1]-edt[0]:.2f})")
axes[2].set_title("(c)"); axes[2].set_ylim(bottom=0); axes[2].legend(frameon=False, fontsize=10)

for ax in axes:
    ax.minorticks_on(); ax.tick_params(which="both", top=True, right=True)

fig.suptitle(f"Model-assisted ANN efficiency   χ²(50×50) = {chi2:.1f}", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(outprefix + "_fig10.png", dpi=300)
print(f"Saved {outprefix}_fig10.png")
