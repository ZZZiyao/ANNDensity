"""
Cut + logit transform for the 4-body Bs->K*K* acceptance (teacher's approach).

Instead of absorbing detJ via the phase-space-CDF coordinate (see phasespace_transform.py),
we simply CUT away the threshold region where detJ->0 (the source of the large weights):
    m1, m2 in (0.65, 1.03)
On the cut sample, weight_detJ is already tame (max/median ~330x vs 1560x), so we weight
directly by weight_detJ -- no q1*q2 factor, no CDF coordinate.

The 6 ANN INPUT features (logit spreads the bounded vars onto (-inf,inf); phi uses cos/sin
to respect periodicity, avoiding the +-pi discontinuity):
    logit((m1  -0.65)/(1.03-0.65)),  logit((m2  -0.65)/(1.03-0.65)),
    logit((cos1+1)/2),               logit((cos2+1)/2),
    cos(phi),                        sin(phi)
The ANN learns eps as a function of these 6 features; eps is a scalar field, so the logit is
just a reparametrisation of the ANN input (no Jacobian enters -- the norm sample is drawn
uniform in the ORIGINAL cut coords and passed through the same transform).
"""
import numpy as np

M_LO, M_HI = 0.65, 1.03          # mass cut (kills the threshold region where detJ->0)
EPS = 1e-4                        # logit boundary clip -> logit range ~[-9.2, 9.2]

# ANN-input ranges for the 6 features (for mlp range-normalisation)
LOGIT_LIM = float(np.log((1 - EPS) / EPS))     # ~9.21
RANGES_6 = [(-LOGIT_LIM, LOGIT_LIM)] * 4 + [(-1.0, 1.0), (-1.0, 1.0)]
FEATURES = ["logit_m1", "logit_m2", "logit_cos1", "logit_cos2", "cos_phi", "sin_phi"]
# original cut-coordinate ranges (for drawing the uniform norm sample)
ORIG_RANGES = [(M_LO, M_HI), (M_LO, M_HI), (-1.0, 1.0), (-1.0, 1.0), (-np.pi, np.pi)]


def _logit(x, lo, hi):
    u = np.clip((np.asarray(x, float) - lo) / (hi - lo), EPS, 1 - EPS)
    return np.log(u / (1.0 - u))


def _expit(z, lo, hi):
    u = 1.0 / (1.0 + np.exp(-np.asarray(z, float)))
    return lo + u * (hi - lo)


def to_features(m1, m2, cos1, cos2, phi):
    """(m1,m2,cos1,cos2,phi) in original cut coords -> (N,6) logit/cos-sin features."""
    return np.stack([
        _logit(m1, M_LO, M_HI), _logit(m2, M_LO, M_HI),
        _logit(cos1, -1.0, 1.0), _logit(cos2, -1.0, 1.0),
        np.cos(phi), np.sin(phi),
    ], axis=1).astype(np.float32)


def features_from_orig(X5):
    """X5 = (N,5) original cut coords -> (N,6) features."""
    return to_features(X5[:, 0], X5[:, 1], X5[:, 2], X5[:, 3], X5[:, 4])


def orig_from_features(F):
    """Inverse (for physical-axis plots). phi recovered via atan2(sin,cos)."""
    m1 = _expit(F[:, 0], M_LO, M_HI); m2 = _expit(F[:, 1], M_LO, M_HI)
    cos1 = _expit(F[:, 2], -1.0, 1.0); cos2 = _expit(F[:, 3], -1.0, 1.0)
    phi = np.arctan2(F[:, 5], F[:, 4])
    return np.stack([m1, m2, cos1, cos2, phi], axis=1)


def uniform_norm_sample(n, rng=None):
    """Uniform sample in the ORIGINAL cut coords, returned as (N,6) features + (N,5) orig."""
    rng = rng or np.random
    X5 = np.column_stack([rng.uniform(lo, hi, n) for (lo, hi) in ORIG_RANGES]).astype(np.float32)
    return features_from_orig(X5), X5


def load_cut(root_file, extra=()):
    """Read the tree, apply the mass cut, return original cut coords + weight_detJ.
    Returns dict: X5 (N,5 orig cut coords), W (weight_detJ), F (N,6 features)."""
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
    print(f"feature ranges (should sit inside RANGES_6={[tuple(round(v,1) for v in r) for r in RANGES_6]}):")
    for i, nm in enumerate(FEATURES):
        print(f"  {nm:11s} [{F[:,i].min():7.2f}, {F[:,i].max():7.2f}]")
    # round-trip check
    Xr = orig_from_features(F)
    print("round-trip max|dm1|=%.2e  max|dphi|=%.2e" % (
        np.abs(Xr[:, 0] - d["X5"][:, 0]).max(), np.abs(Xr[:, 4] - d["X5"][:, 4]).max()))
