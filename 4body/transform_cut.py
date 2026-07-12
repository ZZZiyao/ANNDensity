"""
Cut representation for the 4-body Bs->K*K* acceptance (teacher's approach, no logit).

Mass cut (0.65, 1.03) kills the threshold region where detJ->0 (source of large weights),
so weight_detJ is tame on the cut sample and is used directly as the per-event weight.

The 6 ANN INPUT features keep the ORIGINAL coords for m1,m2,cos1,cos2 (eps is already smooth
in them -- a logit would only distort the gentle parabola), and encode phi periodicity with
(cos phi, sin phi):
    m1, m2, cos1, cos2, cos(phi), sin(phi)
The ANN learns eps(x); the norm sample is drawn uniform in the ORIGINAL cut coords and passed
through the same (m,cos -> identity; phi -> cos/sin) map.
"""
import numpy as np

M_LO, M_HI = 0.65, 1.03          # mass cut (kills the threshold region where detJ->0)

# ANN-input ranges for the 6 features (for mlp range-normalisation)
RANGES_6 = [(M_LO, M_HI), (M_LO, M_HI), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
FEATURES = ["m1", "m2", "cos1", "cos2", "cos_phi", "sin_phi"]
# original cut-coordinate ranges (for drawing the uniform norm sample)
ORIG_RANGES = [(M_LO, M_HI), (M_LO, M_HI), (-1.0, 1.0), (-1.0, 1.0), (-np.pi, np.pi)]


def to_features(m1, m2, cos1, cos2, phi):
    """(m1,m2,cos1,cos2,phi) original cut coords -> (N,6): m,cos kept, phi -> cos/sin."""
    return np.stack([
        np.asarray(m1, float), np.asarray(m2, float),
        np.asarray(cos1, float), np.asarray(cos2, float),
        np.cos(phi), np.sin(phi),
    ], axis=1).astype(np.float32)


def features_from_orig(X5):
    return to_features(X5[:, 0], X5[:, 1], X5[:, 2], X5[:, 3], X5[:, 4])


def orig_from_features(F):
    """Inverse (for plots): m,cos identity; phi = atan2(sin,cos)."""
    return np.stack([F[:, 0], F[:, 1], F[:, 2], F[:, 3], np.arctan2(F[:, 5], F[:, 4])], axis=1)


def uniform_norm_sample(n, rng=None):
    """Uniform sample in ORIGINAL cut coords -> (N,6) features + (N,5) orig."""
    rng = rng or np.random
    X5 = np.column_stack([rng.uniform(lo, hi, n) for (lo, hi) in ORIG_RANGES]).astype(np.float32)
    return features_from_orig(X5), X5


def load_cut(root_file, extra=()):
    """Read tree, apply mass cut, return X5 (orig cut coords), W (weight_detJ), F (6 features)."""
    import uproot
    need = ["m1", "m2", "cos1", "cos2", "phi", "weight_detJ"] + list(extra)
    a = uproot.open(root_file)["DecayTree"].arrays(need, library="np")
    m1 = np.asarray(a["m1"], float); m2 = np.asarray(a["m2"], float)
    cut = (m1 > M_LO) & (m1 < M_HI) & (m2 > M_LO) & (m2 < M_HI)
    X5 = np.stack([m1[cut], m2[cut], np.asarray(a["cos1"], float)[cut],
                   np.asarray(a["cos2"], float)[cut], np.asarray(a["phi"], float)[cut]], axis=1).astype(np.float32)
    W = np.asarray(a["weight_detJ"], float)[cut].astype(np.float32)
    out = {"X5": X5, "W": W, "F": features_from_orig(X5)}
    for b in extra:
        out[b] = np.asarray(a[b])[cut]
    return out


if __name__ == "__main__":
    import os
    HERE = os.path.dirname(os.path.abspath(__file__))
    d = load_cut(os.path.join(HERE, "runs12_MagBoth_L0Both.root"))
    F, W = d["F"], d["W"]
    neff = W.sum() ** 2 / (W ** 2).sum()
    print(f"cut sample: N={len(W)}  Neff={neff:.0f}  Neff/N={neff/len(W):.3f}")
    for i, nm in enumerate(FEATURES):
        print(f"  {nm:9s} [{F[:,i].min():7.3f}, {F[:,i].max():7.3f}]")
