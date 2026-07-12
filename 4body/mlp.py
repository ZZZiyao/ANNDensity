"""
Configurable MLP for the 4-body acceptance -- shared by training and evaluation.

The original tfa.neural_nets.multilayer_perceptron hard-codes SIGMOID on every layer
(including the output).  Sigmoid saturates -> vanishing gradients (slow convergence)
and difficulty representing sharp features (mass-window edge).  This module lets the
HIDDEN activation be non-saturating (swish/relu/...) while keeping the OUTPUT strictly
POSITIVE (density must be > 0): softplus / sigmoid / exp.

Backward compatible: mlp_forward(..., hidden="sigmoid", out="sigmoid") reproduces the
original tfa MLP exactly, so legacy 4-element .npy models evaluate identically.
"""
import numpy as np
import tensorflow as tf
import amplitf.interface as atfi


def _swish(z):
    return z * tf.nn.sigmoid(z)

HIDDEN_ACTS = {
    "sigmoid": tf.nn.sigmoid,
    "tanh": tf.nn.tanh,
    "relu": tf.nn.relu,
    "leaky_relu": tf.nn.leaky_relu,
    "swish": _swish,
    "softplus": tf.nn.softplus,
}
# output must be strictly positive (density > 0)
OUT_ACTS = {
    "sigmoid": tf.nn.sigmoid,
    "softplus": tf.nn.softplus,
    "exp": tf.exp,
}


def _normalise(x, ranges):
    x = tf.cast(x, atfi.fptype())
    cols = [(x[:, i] - atfi.const(ranges[i][0])) / (atfi.const(ranges[i][1]) - atfi.const(ranges[i][0]))
            for i in range(len(ranges))]
    return tf.stack(cols, axis=1)


@tf.function
def mlp_forward(x, ranges, weights, biases, hidden, out):
    """Fully-connected MLP: `hidden` activation on hidden layers, `out` on the output."""
    h = HIDDEN_ACTS[str(hidden)]
    o = OUT_ACTS[str(out)]
    layer = _normalise(x, ranges)
    n = len(weights)
    for i in range(n):
        z = tf.add(tf.matmul(layer, weights[i]), biases[i])
        layer = o(z) if i == n - 1 else h(z)
    return layer[:, 0]


def create_weights_biases_he(n_input, layers, n_output=1):
    """He/Glorot-scaled init (sigma = sqrt(2/fan_in) per layer) -- stable for non-saturating
    activations (swish/relu), unlike the original fixed sigma=1 which can explode over 4 layers."""
    sizes = [n_input] + list(layers) + [n_output]
    weights, biases = [], []
    for i in range(len(sizes) - 1):
        sigma = np.sqrt(2.0 / sizes[i])
        weights.append(tf.Variable(sigma * np.random.normal(size=[sizes[i], sizes[i + 1]]), dtype=atfi.fptype()))
        biases.append(tf.Variable(np.zeros([sizes[i + 1]]), dtype=atfi.fptype()))
    return weights, biases


def model_fn_from_npy(ann):
    """Build a numpy-callable eps(x) from a loaded .npy model.
    Format: [scale, ranges, weights, biases] (legacy sigmoid) or
            [scale, ranges, weights, biases, hidden_act, out_act] (new)."""
    ranges = ann[1]
    weights = [tf.constant(w, dtype=atfi.fptype()) for w in ann[2]]
    biases = [tf.constant(b, dtype=atfi.fptype()) for b in ann[3]]
    hidden = ann[4] if len(ann) > 4 else "sigmoid"
    out = ann[5] if len(ann) > 5 else "sigmoid"
    return lambda x: mlp_forward(np.asarray(x, np.float32), ranges, weights, biases, hidden, out).numpy()
