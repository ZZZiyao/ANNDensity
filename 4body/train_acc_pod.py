"""
Self-contained ANN veto trainer for a RunPod/Colab GPU (modern TF, NO amplitf/tfa).
Same math as train_acc_6d.py + mlp.py: wide MLP, He init, swish hidden, softplus out,
weighted unbinned NLL with the norm sample EXCLUDING the veto region, exp lr decay,
best-tracking. Computes the 6 features from X5 (so an X5+W npz is enough -- no F needed).
After training, runs the shared veto recovery (ratio + chi2). Pure TF 2.x GradientTape
so it works on Keras 3 / TF 2.16+.

Env: ACC_VETO, ACC_LAMBDA2, ACC_LR, ACC_LR_FINAL, ACC_HIDDEN, ACC_NORM_SIZE,
     ACC_MAX_EPOCHS, ACC_NTRAIN, ACC_SEED, ACC_CHECK_EVERY, BASE_OUT.
"""
import os
import numpy as np
import tensorflow as tf
import transform_cut as tl
import veto as vt
import veto_recovery as vr

HERE = os.path.dirname(os.path.abspath(__file__))
F32 = tf.float32
def _f(n, d): return float(os.environ.get(n, d))
def _i(n, d): return int(float(os.environ.get(n, d)))

VETO   = os.environ.get("ACC_VETO", "")
LAM2   = _f("ACC_LAMBDA2", 0.3)
LR     = _f("ACC_LR", 5e-4)
LRF    = _f("ACC_LR_FINAL", 1.5e-5)
NH     = [int(x) for x in os.environ.get("ACC_HIDDEN", "64,128,64,16").split(",")]
NORM   = _i("ACC_NORM_SIZE", 2000000)
EPOCHS = _i("ACC_MAX_EPOCHS", 15000)
NTRAIN = _i("ACC_NTRAIN", 0)
SEED   = _i("ACC_SEED", 1)
CHECK  = _i("ACC_CHECK_EVERY", 100)
OUT    = os.environ.get("BASE_OUT", os.path.join(HERE, "acc_pod"))
RANGES = tl.RANGES_6
print(f"GPUs: {tf.config.list_physical_devices('GPU')}  | veto='{VETO}' hidden={NH} lam2={LAM2} lr={LR}->{LRF} epochs={EPOCHS}")


def swish(z): return z * tf.nn.sigmoid(z)

def normalise(x):
    return tf.stack([(x[:, i] - RANGES[i][0]) / (RANGES[i][1] - RANGES[i][0]) for i in range(6)], axis=1)

def forward(x, W, B):
    layer = normalise(tf.cast(x, F32))
    n = len(W)
    for i in range(n):
        z = tf.matmul(layer, W[i]) + B[i]
        layer = tf.nn.softplus(z) if i == n - 1 else swish(z)
    return layer[:, 0] + 1e-20

def he_init(n_in, layers):
    sizes = [n_in] + list(layers) + [1]; W, B = [], []
    for i in range(len(sizes) - 1):
        s = np.sqrt(2.0 / sizes[i])
        W.append(tf.Variable(s * np.random.normal(size=[sizes[i], sizes[i + 1]]), dtype=F32))
        B.append(tf.Variable(np.zeros([sizes[i + 1]]), dtype=F32))
    return W, B


# ---- data: load X5,W; compute the 6 features from X5; apply veto ----
z = np.load(os.path.join(HERE, "runs12_cut.npz"))
X5, W = z["X5"].astype(np.float32), z["W"].astype(np.float32)
spec = vt.parse(VETO)
if spec:
    keep = ~vt.mask(X5, spec)
    X5, W = X5[keep], W[keep]
    print(f"VETO [{vt.describe(spec)}]: {int((~keep).sum())} events removed, {keep.sum()} remain")
tf.random.set_seed(SEED); np.random.seed(SEED)
perm = np.random.permutation(len(X5))
if NTRAIN: perm = perm[:NTRAIN]
X5, W = X5[perm], W[perm]
nval = int(0.1 * len(X5))
Ft = tl.features_from_orig(X5[nval:]); Wt = W[nval:]
Fv = tl.features_from_orig(X5[:nval]); Wv = W[:nval]
# norm sample uniform in ORIGINAL cut coords -> features, EXCLUDING the veto region
normF, normX5 = tl.uniform_norm_sample(int(NORM * (1.3 if spec else 1.0)), np.random.RandomState(SEED + 7))
if spec: normF = normF[~vt.mask(normX5, spec)]
print(f"train {len(Ft)}, val {len(Fv)}, norm {len(normF)}")

train_x, train_w = tf.constant(Ft), tf.constant(Wt.astype(np.float32))
val_x, val_w = tf.constant(Fv), tf.constant(Wv.astype(np.float32))
norm_x = tf.constant(normF)
Wnet, Bnet = he_init(6, NH); params = Wnet + Bnet

@tf.function
def integral(): return tf.reduce_mean(forward(norm_x, Wnet, Bnet))
@tf.function
def train_loss():
    I = integral()
    nll = -tf.reduce_sum(train_w * tf.math.log(forward(train_x, Wnet, Bnet) / I))
    return nll + LAM2 * tf.add_n([tf.reduce_sum(w ** 2) for w in Wnet])
@tf.function
def val_nll():
    I = integral()
    return -tf.reduce_sum(val_w * tf.math.log(forward(val_x, Wnet, Bnet) / I))
@tf.function
def step():
    with tf.GradientTape() as tape:
        L = train_loss()
    g = tape.gradient(L, params); opt.apply_gradients(zip(g, params)); return L

sched = tf.keras.optimizers.schedules.ExponentialDecay(LR, EPOCHS, LRF / LR, staircase=False)
opt = tf.keras.optimizers.Adam(learning_rate=sched)

best, best_ep = np.inf, -1; bestW = bestB = None
for e in range(EPOCHS):
    step()
    if e % CHECK == 0:
        v = float(val_nll())
        if v < best:
            best, best_ep = v, e
            bestW = [w.numpy() for w in Wnet]; bestB = [b.numpy() for b in Bnet]
        if e % (CHECK * 20) == 0:
            print(f"  epoch {e:6d}  val {v:12.1f}  best {best:12.1f}")
print(f"done: best_val={best:.1f} @ epoch {best_ep}")

# ---- recovery with the best weights (model takes ORIGINAL X5 -> features -> eps) ----
Wc = [tf.constant(w) for w in bestW]; Bc = [tf.constant(b) for b in bestB]
def model_x5(X5arr):
    return forward(tl.features_from_orig(np.asarray(X5arr, np.float32)), Wc, Bc).numpy()
if spec:
    vr.recovery(model_x5, VETO, OUT + "_rec", seed=SEED, label="ANN")
# save model (mlp-compatible 6-element format)
sc = 1.0 / float(np.mean(model_x5(tl.uniform_norm_sample(200000, np.random.RandomState(3))[1])))
np.save(OUT + ".npy", np.array([sc, RANGES, bestW, bestB, "swish", "softplus"], dtype=object))
print(f"saved {OUT}.npy" + (f" + {OUT}_rec.png" if spec else ""))
