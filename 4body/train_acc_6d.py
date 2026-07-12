"""
4-body acceptance -- 6-feature ANN on the CUT + LOGIT representation (teacher's approach).

Coords: mass cut (0.65,1.03) + logit(m1,m2,cos1,cos2) + (cos phi, sin phi)  -> 6 inputs.
Weight: weight_detJ directly (cut tames it; no q1*q2, no CDF coord). See transform_logit.py.

Exposes train_model(cfg, ...) so the Optuna driver (optuna_acc.py) can call it per trial,
and a standalone env-driven __main__ for a single run / slurm.

cfg keys (all optional, with defaults): lambda2, lr, lr_final, n_hidden, max_epochs,
norm_size, ntrain, val_frac, check_every, patience, seed, outfile.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")

import amplitf.interface as atfi
atfi.set_single_precision()
import tensorflow as tf

import tfa.neural_nets as tfn
import mlp
import transform_cut as tl
import veto as vt

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULTS = dict(
    lambda2=0.03, lr=2e-3, lr_final=2e-4, n_hidden=[32, 64, 32, 8],
    max_epochs=30000, norm_size=2000000, ntrain=0, val_frac=0.1,
    check_every=100, patience=10**9, seed=1, hidden_act="swish", out_act="softplus",
    veto="", outfile=None,
)


def _load(cfg):
    z = np.load(os.environ.get("ACC_FLATFILE", os.path.join(HERE, "runs12_cut.npz")))
    F, W, X5 = z["F"].astype(np.float32), z["W"].astype(np.float32), z["X5"]
    spec = vt.parse(cfg.get("veto", "") or "")
    if spec:
        keep = ~vt.mask(X5, spec)        # drop events inside the veto (they "don't exist")
        F, W = F[keep], W[keep]
        print(f"VETO [{vt.describe(spec)}]: removed {(~keep).sum()} events, {keep.sum()} remain")
    rng = np.random.RandomState(cfg["seed"])
    perm = rng.permutation(len(F))
    if cfg["ntrain"]:
        perm = perm[:cfg["ntrain"]]
    F, W = F[perm], W[perm]
    nval = int(cfg["val_frac"] * len(F))
    return F[nval:], W[nval:], F[:nval], W[:nval], spec


def train_model(cfg=None, verbose=True, prune_cb=None):
    """Train one model; return dict(best_val, best_epoch). Saves to cfg['outfile'] if set.
    prune_cb(epoch, val) -> True to stop early (Optuna pruning)."""
    c = dict(DEFAULTS); c.update(cfg or {})
    atfi.set_seed(c["seed"]); np.random.seed(c["seed"])
    Ft, Wt, Fv, Wv, spec = _load(c)
    if verbose:
        print(f"train {len(Ft)}, val {len(Fv)}  | hp: lam2={c['lambda2']:.4g} lr={c['lr']:.2g}"
              f"->{c['lr_final']:.2g} hidden={c['n_hidden']}")

    RANGES = tl.RANGES_6
    train_data, train_w = atfi.const(Ft), atfi.const(Wt)
    val_data, val_w = atfi.const(Fv), atfi.const(Wv)
    # norm sample: uniform in ORIGINAL cut coords -> 6 features.
    # CRUCIAL for veto: exclude the veto region from the norm sample, else the unbinned NLL
    # rewards pushing eps->0 there. With it excluded, eps in the hole is set by smooth extrapolation.
    normF, normX5 = tl.uniform_norm_sample(int(c["norm_size"] * (1.3 if spec else 1.0)),
                                           np.random.RandomState(c["seed"] + 7))
    if spec:
        normF = normF[~vt.mask(normX5, spec)]
    norm_sample = atfi.const(normF)

    weights, biases = mlp.create_weights_biases_he(6, c["n_hidden"])
    params = weights + biases

    def fitmodel(x):
        return mlp.mlp_forward(x, RANGES, weights, biases, c["hidden_act"], c["out_act"]) + 1e-20

    @tf.function
    def integral():
        return tf.reduce_mean(fitmodel(norm_sample))

    @tf.function
    def train_loss():
        I = integral()
        nll = -tf.reduce_sum(train_w * atfi.log(fitmodel(train_data) / I))
        return nll + c["lambda2"] * tfn.l2_regularisation(weights)

    @tf.function
    def val_nll():
        I = integral()
        return -tf.reduce_sum(val_w * atfi.log(fitmodel(val_data) / I))

    if c["lr_final"] > 0:
        sched = tf.keras.optimizers.schedules.ExponentialDecay(
            c["lr"], decay_steps=c["max_epochs"], decay_rate=c["lr_final"] / c["lr"], staircase=False)
        opt = tf.keras.optimizers.Adam(learning_rate=sched)
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=c["lr"])

    def snapshot():
        sc = 1.0 / float(np.mean(fitmodel(norm_sample).numpy()))
        return [sc, RANGES, [v.numpy() for v in weights], [b.numpy() for b in biases],
                c["hidden_act"], c["out_act"]]

    best_val, best_epoch, stale = np.inf, -1, 0
    logf = open(c["outfile"] + ".txt", "w") if c["outfile"] else None
    for epoch in range(c["max_epochs"]):
        opt.minimize(train_loss, params)
        if epoch % c["check_every"] == 0:
            v = float(val_nll())
            improved = v < best_val
            if verbose and epoch % (c["check_every"] * 10) == 0:
                print(f"  epoch {epoch:6d}  val {v:12.2f}  best {best_val:12.2f}")
            if logf:
                logf.write(f"Epoch {epoch:6d}  val {v:14.3f}  best_val {best_val:14.3f}"
                           + ("  *\n" if improved else "\n")); logf.flush()
            if improved:
                best_val, best_epoch, stale = v, epoch, 0
                if c["outfile"]:
                    np.save(c["outfile"], np.array(snapshot(), dtype=object))
            else:
                stale += 1
                if stale >= c["patience"]:
                    break
            if prune_cb and prune_cb(epoch, v):
                break
    if logf:
        logf.write(f"best_epoch {best_epoch}  best_val {best_val:.6f}\n"); logf.close()
    if verbose:
        print(f"  done: best_val={best_val:.3f} @ epoch {best_epoch}")
    return {"best_val": best_val, "best_epoch": best_epoch}


def _cfg_from_env():
    def f(n, d): return float(os.environ.get(n, d))
    def i(n, d): return int(float(os.environ.get(n, d)))
    nh = os.environ.get("ACC_HIDDEN")
    return dict(
        lambda2=f("ACC_LAMBDA2", 0.03), lr=f("ACC_LR", 2e-3), lr_final=f("ACC_LR_FINAL", 2e-4),
        n_hidden=[int(x) for x in nh.split(",")] if nh else [32, 64, 32, 8],
        max_epochs=i("ACC_MAX_EPOCHS", 30000), norm_size=i("ACC_NORM_SIZE", 2000000),
        ntrain=i("ACC_NTRAIN", 0), patience=i("ACC_PATIENCE", 10**9), seed=i("ACC_SEED", 1),
        check_every=i("ACC_CHECK_EVERY", 100), veto=os.environ.get("ACC_VETO", ""),
        outfile=os.environ.get("ACC_OUTFILE", os.path.join(HERE, "acc_6d")),
    )


if __name__ == "__main__":
    train_model(_cfg_from_env())
