"""
Chapter 8.3 (model-assisted ANN, BACKGROUND) -- STEP 1 training.

Trains the JOINT density ANN P(x, Theta) over the 11D combined phase space
(3 observables m',theta',mD + 8 latent params: track pT/p cuts, mean pT of
K/pi/K*/rho, K* and rho fractions -- paper Table 4). Custom loop with val-based
EARLY STOPPING (mirror of train_eff_joint.py).

Paper §8.3: topology [32,64,32,8], lambda2 = 1, 50000 epochs, norm 4e6.

Output (estimate_density-compatible, loadable by fit_bg_joint.py):
  bg_joint_train.npy = [scale, ranges, weights, biases]

Config is env-overridable: BG_NORM_SIZE, BG_MAX_EPOCHS, BG_LR, BG_LAMBDA2,
BG_PATIENCE, BG_CHECK_EVERY, BG_VAL_FRAC, BG_OUTFILE, BG_TOY_FILE, BG_SEED.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")

import amplitf.interface as atfi
atfi.set_single_precision()

import tensorflow as tf

from amplitf.phasespace.rectangular_phasespace import RectangularPhaseSpace
from amplitf.phasespace.combined_phasespace import CombinedPhaseSpace

import tfa.rootio as tfr
import tfa.neural_nets as tfn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from DistributionModel import (
    parameters_list, observables_phase_space, observables_toys,
)


def _f(name, default):
    return float(os.environ.get(name, default))
def _i(name, default):
    return int(float(os.environ.get(name, default)))

# ----------------------- config (env-overridable) -----------------------
# Defaults match the original TrainNN.py (estimate_density call) for §8.3 background.
TOY_FILE      = os.environ.get("BG_TOY_FILE", os.path.join(HERE, "toy_1e6.root"))
OUTFILE       = os.environ.get("BG_OUTFILE", os.path.join(HERE, "bg_joint_train"))
N_HIDDEN      = [32, 64, 32, 8]
LEARNING_RATE = _f("BG_LR", 2e-4)              # original: 0.0002
LAMBDA2       = _f("BG_LAMBDA2", 1.0)          # original: 1
NORM_SIZE     = _i("BG_NORM_SIZE", 4000000)    # original: 4e6
MAX_EPOCHS    = _i("BG_MAX_EPOCHS", 50000)     # original: 50000
VAL_FRAC      = _f("BG_VAL_FRAC", 0.1)         # ADDED for early stopping
CHECK_EVERY   = _i("BG_CHECK_EVERY", 100)
PATIENCE      = _i("BG_PATIENCE", 20)
MIN_DELTA     = _f("BG_MIN_DELTA", 0.0)        # absolute min improvement in val NLL (sign-safe)
SEED          = _i("BG_SEED", 4)               # original: 4


def main():
    atfi.set_seed(SEED)
    np.random.seed(SEED)

    # ---- combined 11D phase space + ranges ----
    variables = observables_toys + [p[0] for p in parameters_list]
    parameters_bounds = [p[2] for p in parameters_list]
    ranges = observables_phase_space.bounds() + parameters_bounds
    phsp = CombinedPhaseSpace(observables_phase_space,
                              RectangularPhaseSpace(parameters_bounds))
    n_input = len(ranges)

    # ---- load + filter data, split train / validation ----
    data = atfi.const(tfr.read_tuple(TOY_FILE, variables))
    data = atfi.cast_real(data[phsp.inside(data)])
    n_tot = int(data.shape[0])
    perm = np.random.permutation(n_tot)
    n_val = int(VAL_FRAC * n_tot)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_data = tf.gather(data, train_idx)
    val_data = tf.gather(data, val_idx)
    print(f"Input dims = {n_input}; total {n_tot}, train {len(train_idx)}, val {len(val_idx)}")

    # ---- model (fresh, or resume from a saved checkpoint .npy) ----
    RESUME = os.environ.get("BG_RESUME", "")
    if RESUME and os.path.exists(RESUME):
        print(f"Resuming from checkpoint {RESUME}")
        ckpt = np.load(RESUME, allow_pickle=True)
        weights, biases = tfn.init_weights_biases(ckpt[2:])   # trainable Variables
    else:
        print("Starting from fresh random weights")
        weights, biases = tfn.create_weights_biases(n_input, N_HIDDEN)
    params = weights + biases

    def fitmodel(x):
        return tfn.multilayer_perceptron(x, ranges, weights, biases) + 1e-20

    norm_sample = phsp.uniform_sample(NORM_SIZE)

    @tf.function
    def integral():
        return tf.reduce_mean(fitmodel(norm_sample))

    @tf.function
    def train_loss():
        I = integral()
        nll = -tf.reduce_sum(atfi.log(fitmodel(train_data) / I))
        return nll + LAMBDA2 * tfn.l2_regularisation(weights)

    @tf.function
    def val_nll():
        I = integral()
        return -tf.reduce_sum(atfi.log(fitmodel(val_data) / I))

    opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # ---- training loop with early stopping ----
    best_val = np.inf
    best_epoch = -1
    stale = 0
    logf = open(OUTFILE + ".txt", "w")

    def snapshot():
        w = fitmodel(norm_sample)
        scale = 1.0 / float(np.mean(w.numpy()))
        return [scale, ranges,
                [v.numpy() for v in weights],
                [b.numpy() for b in biases]]

    for epoch in range(MAX_EPOCHS):
        opt.minimize(train_loss, params)

        if epoch % CHECK_EVERY == 0:
            v = float(val_nll())
            t = float(train_loss())
            improved = v < best_val - MIN_DELTA
            msg = f"Epoch {epoch:6d}  train {t:14.3f}  val {v:14.3f}" \
                  f"  best_val {best_val:14.3f}  stale {stale}"
            print(msg + ("  *" if improved else ""))
            logf.write(msg + ("  *\n" if improved else "\n"))
            logf.flush()

            if improved:
                best_val = v
                best_epoch = epoch
                np.save(OUTFILE, np.array(snapshot(), dtype=object))
                stale = 0
            else:
                stale += 1
                if stale >= PATIENCE:
                    print(f"Early stop at epoch {epoch} "
                          f"(best epoch {best_epoch}, best val {best_val:.3f})")
                    break

    logf.write(f"best_epoch {best_epoch}  best_val {best_val:.6f}\n")
    logf.close()
    print(f"Saved {OUTFILE}.npy  (best epoch {best_epoch}, val NLL {best_val:.3f})")


if __name__ == "__main__":
    main()
