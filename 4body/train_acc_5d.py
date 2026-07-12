"""
4-body B->K*K* acceptance -- 5D generic ANN density estimation.

Learns eps(m1, m2, cos1, cos2, phi) directly from the phase-space sample, using
weight_detJ as a per-event weight (which removes the phase-space factor, so the
weighted density = acceptance eps). No latent params (not model-assisted), no phi
transform (eps is flat in phi). Custom loop with val-based EARLY STOPPING.

Output (estimate_density-compatible): acc_5d_train.npy = [scale, ranges, weights, biases]

Config env-overridable: ACC_ROOT, ACC_OUTFILE, ACC_LR, ACC_LAMBDA2, ACC_NORM_SIZE,
ACC_NTRAIN, ACC_MAX_EPOCHS, ACC_VAL_FRAC, ACC_CHECK_EVERY, ACC_PATIENCE, ACC_SEED.
"""
import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")

import amplitf.interface as atfi
atfi.set_single_precision()

import tensorflow as tf
import uproot

from amplitf.phasespace.rectangular_phasespace import RectangularPhaseSpace
import tfa.neural_nets as tfn

HERE = os.path.dirname(os.path.abspath(__file__))

def _f(n, d): return float(os.environ.get(n, d))
def _i(n, d): return int(float(os.environ.get(n, d)))

# ----------------------- config -----------------------
ROOT_FILE  = os.environ.get("ACC_ROOT", os.path.join(HERE, "runs12_MagBoth_L0Both.root"))
OUTFILE    = os.environ.get("ACC_OUTFILE", os.path.join(HERE, "acc_5d_train"))
N_HIDDEN   = [32, 64, 32, 8]
LEARN_RATE = _f("ACC_LR", 1e-3)
LR_FINAL   = _f("ACC_LR_FINAL", 0.0)          # >0 -> exponential decay LEARN_RATE -> LR_FINAL over MAX_EPOCHS
LAMBDA2    = _f("ACC_LAMBDA2", 0.1)
NORM_SIZE  = _i("ACC_NORM_SIZE", 1000000)     # uniform 5D normalisation sample
NTRAIN     = _i("ACC_NTRAIN", 500000)         # subsample of the 5.7M events (0 = all)
MAX_EPOCHS = _i("ACC_MAX_EPOCHS", 50000)
VAL_FRAC   = _f("ACC_VAL_FRAC", 0.1)
CHECK_EVERY = _i("ACC_CHECK_EVERY", 100)
PATIENCE   = _i("ACC_PATIENCE", 20)
MIN_DELTA  = _f("ACC_MIN_DELTA", 0.0)
SEED       = _i("ACC_SEED", 1)
FLATTEN    = _i("ACC_FLATTEN", 0)             # 1 = train in phase-space-flattened coords (u1,u2,...)

# 5D phase space ranges (m1, m2, cos1, cos2, phi)
RANGES = [(0.63, 1.04), (0.63, 1.04), (-1.0, 1.0), (-1.0, 1.0), (-math.pi, math.pi)]
VARS = ["m1", "m2", "cos1", "cos2", "phi"]
if FLATTEN:
    import phasespace_transform as pst
    RANGES = pst.RANGES_FLAT                  # u1,u2 in [0,1]; angles unchanged


def main():
    atfi.set_seed(SEED)
    np.random.seed(SEED)

    # ---- load data, subsample, split ----
    if FLATTEN:
        # flattened coords (u1,u2,cos1,cos2,phi) + tame weight (weighted density = eps).
        # Prefer the pre-materialised, validated runs12_flat.npz; else compute on the fly.
        flat_npz = os.environ.get("ACC_FLATFILE", os.path.join(HERE, "runs12_flat.npz"))
        if os.path.exists(flat_npz):
            z = np.load(flat_npz)
            X, W = z["X"].astype(np.float32), z["W"].astype(np.float32)
            print(f"FLATTEN on: loaded {os.path.basename(flat_npz)} (validated)")
        else:
            d = pst.load_flattened(ROOT_FILE)
            X, W = d["X"], d["W"]
            print("FLATTEN on: computed coords on the fly (npz not found)")
    else:
        # original coords (m1,m2,cos1,cos2,phi) + weight_detJ
        arr = uproot.open(ROOT_FILE)["DecayTree"].arrays(VARS + ["weight_detJ"], library="np")
        X = np.stack([arr[v] for v in VARS], axis=1).astype(np.float32)
        W = np.asarray(arr["weight_detJ"], np.float32)
    n_all = len(X)
    perm = np.random.permutation(n_all)
    if NTRAIN and NTRAIN < n_all:
        perm = perm[:NTRAIN]
    X, W = X[perm], W[perm]
    n_val = int(VAL_FRAC * len(X))
    Xv, Wv = X[:n_val], W[:n_val]
    Xt, Wt = X[n_val:], W[n_val:]
    print(f"events: {n_all} total -> using {len(X)} (train {len(Xt)}, val {len(Xv)})")

    train_data, train_w = atfi.const(Xt), atfi.const(Wt)
    val_data, val_w = atfi.const(Xv), atfi.const(Wv)

    phsp = RectangularPhaseSpace(RANGES)
    norm_sample = phsp.uniform_sample(NORM_SIZE)

    # ---- model ----
    weights, biases = tfn.create_weights_biases(len(RANGES), N_HIDDEN)
    params = weights + biases

    def fitmodel(x):
        return tfn.multilayer_perceptron(x, RANGES, weights, biases) + 1e-20

    @tf.function
    def integral():
        return tf.reduce_mean(fitmodel(norm_sample))

    @tf.function
    def train_loss():
        I = integral()
        nll = -tf.reduce_sum(train_w * atfi.log(fitmodel(train_data) / I))
        return nll + LAMBDA2 * tfn.l2_regularisation(weights)

    @tf.function
    def val_nll():
        I = integral()
        return -tf.reduce_sum(val_w * atfi.log(fitmodel(val_data) / I))

    if LR_FINAL > 0.0:
        # exponential decay from LEARN_RATE to LR_FINAL over MAX_EPOCHS (per-epoch step)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            LEARN_RATE, decay_steps=MAX_EPOCHS, decay_rate=LR_FINAL / LEARN_RATE, staircase=False)
        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        print(f"LR schedule: {LEARN_RATE:.1e} -> {LR_FINAL:.1e} (exp decay over {MAX_EPOCHS} epochs)")
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=LEARN_RATE)

    def snapshot():
        sc = 1.0 / float(np.mean(fitmodel(norm_sample).numpy()))
        return [sc, RANGES, [v.numpy() for v in weights], [b.numpy() for b in biases]]

    # ---- training loop with early stopping ----
    best_val, best_epoch, stale = np.inf, -1, 0
    logf = open(OUTFILE + ".txt", "w")
    for epoch in range(MAX_EPOCHS):
        opt.minimize(train_loss, params)
        if epoch % CHECK_EVERY == 0:
            v = float(val_nll()); t = float(train_loss())
            improved = v < best_val - MIN_DELTA
            msg = f"Epoch {epoch:6d}  train {t:14.3f}  val {v:14.3f}  best_val {best_val:14.3f}  stale {stale}"
            print(msg + ("  *" if improved else "")); logf.write(msg + ("  *\n" if improved else "\n")); logf.flush()
            if improved:
                best_val, best_epoch, stale = v, epoch, 0
                np.save(OUTFILE, np.array(snapshot(), dtype=object))
            else:
                stale += 1
                if stale >= PATIENCE:
                    print(f"Early stop at epoch {epoch} (best {best_epoch}, val {best_val:.3f})"); break
    logf.write(f"best_epoch {best_epoch}  best_val {best_val:.6f}\n"); logf.close()
    print(f"Saved {OUTFILE}.npy  (best epoch {best_epoch}, val NLL {best_val:.3f})")


if __name__ == "__main__":
    main()
