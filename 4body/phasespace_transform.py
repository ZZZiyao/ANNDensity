"""
Phase-space-flattening transform for the 4-body Bs->K*K* acceptance.

The ROOT events carry a per-event Jacobian detJ which we verified analytically to be
    detJ(m1,m2) = q(m1) * q(m2) * pB(m1,m2)          (to 0.03%)
where
    q(m)      = Kpi breakup momentum in the K* rest frame  (-> 0 at the Kpi threshold)
    pB(m1,m2) = K* momentum in the Bs rest frame           (varies only ~5% over the box)

The large weight_detJ = weight/detJ comes entirely from 1/(q1 q2) blowing up at the
mass thresholds.  We remove it at the source by changing the MASS variables to the
phase-space CDF coordinates
    u_i = ( int_{mthr}^{m_i} q dm' ) / ( int_{mthr}^{mmax} q dm' )  in [0,1]
which makes the q1*q2 part of the measure EXACTLY uniform (residual = the ~1% pB ripple).
The angles cos1,cos2,phi need no transform (phase space is already flat in them).

In the flattened coordinates (u1,u2,cos1,cos2,phi) the per-event weight that makes the
weighted density equal the acceptance eps is
    w_new = weight_detJ * q(m1) * q(m2)
i.e. the validated stored eps-weight (weight_detJ) times the EXACT Jacobian of THIS
transform (du_i/dm_i = q(m_i)/N_i).  This is exact regardless of how closely the stored
detJ matches the analytic q1*q2*pB.  (Using weight/pB instead is only an approximation:
it assumes detJ == q1*q2*pB, which fails by up to ~50% for rare near-threshold events.)
w_new is tame (N_eff ~ 2.3M vs 1.22M, max/median ~80x vs 1560x).

This module is pure deterministic preprocessing (CPU, instant) -- no fitting, no HPC.
"""
import numpy as np

# PDG masses (GeV)
M_K, M_PI, M_BS = 0.493677, 0.139570, 5.36682
# Kpi mass range present in the sample (K* window)
M_LO, M_HI = 0.6333, 1.042

# flattened-coordinate ranges: u1,u2 in [0,1]; angles unchanged
RANGES_FLAT = [(0.0, 1.0), (0.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-np.pi, np.pi)]
VARS = ["m1", "m2", "cos1", "cos2", "phi"]


def q(m):
    """Kpi breakup momentum in the K* rest frame."""
    m = np.asarray(m, float)
    s = m * m
    return np.sqrt(np.clip((s - (M_K + M_PI) ** 2) * (s - (M_K - M_PI) ** 2), 0.0, None)) / (2.0 * m)


def pB(m1, m2):
    """K* momentum in the Bs rest frame (function of both masses)."""
    m1 = np.asarray(m1, float); m2 = np.asarray(m2, float)
    s = M_BS * M_BS
    return np.sqrt(np.clip((s - (m1 + m2) ** 2) * (s - (m1 - m2) ** 2), 0.0, None)) / (2.0 * M_BS)


# ---- u(m) CDF lookup (built once) ----
_grid = np.linspace(M_LO, M_HI, 16001)
_Q = q(_grid)
_cum = np.concatenate([[0.0], np.cumsum(0.5 * (_Q[1:] + _Q[:-1]) * np.diff(_grid))])
_cdf = _cum / _cum[-1]


def m_to_u(m):
    """Forward transform mass -> uniform phase-space coordinate in [0,1]."""
    return np.interp(np.asarray(m, float), _grid, _cdf)


def u_to_m(u):
    """Inverse transform u -> physical mass (GeV); monotonic, for axis relabelling/plots."""
    return np.interp(np.asarray(u, float), _cdf, _grid)


def load_flattened(root_file, extra_branches=()):
    """
    Read the tree and return events in flattened coordinates with the tame weight.

    Returns dict with:
      X  : (N,5) float32  = (u1, u2, cos1, cos2, phi)
      W  : (N,)   float32 = weight_detJ * q(m1) * q(m2)   (tame; weighted density = eps)
      m1, m2 : raw physical masses (for reference / physical-axis plots)
      plus any names listed in extra_branches.
    """
    import uproot
    need = ["m1", "m2", "cos1", "cos2", "phi", "weight_detJ"] + list(extra_branches)
    a = uproot.open(root_file)["DecayTree"].arrays(need, library="np")
    m1 = np.asarray(a["m1"], float); m2 = np.asarray(a["m2"], float)
    wdJ = np.asarray(a["weight_detJ"], float)
    X = np.stack([m_to_u(m1), m_to_u(m2),
                  np.asarray(a["cos1"], float), np.asarray(a["cos2"], float),
                  np.asarray(a["phi"], float)], axis=1).astype(np.float32)
    W = (wdJ * q(m1) * q(m2)).astype(np.float32)   # exact: stored eps-weight x transform Jacobian
    out = {"X": X, "W": W, "m1": m1, "m2": m2}
    for b in extra_branches:
        out[b] = np.asarray(a[b])
    return out


if __name__ == "__main__":
    # self-test on the bundled sample
    import os
    HERE = os.path.dirname(os.path.abspath(__file__))
    d = load_flattened(os.path.join(HERE, "runs12_MagBoth_L0Both.root"))
    X, W = d["X"], d["W"]
    neff = W.sum() ** 2 / (W ** 2).sum()
    print(f"N={len(W)}  Neff={neff:.0f}  Neff/N={neff/len(W):.3f}  "
          f"W: median={np.median(W):.3f} max={W.max():.2f} max/med={W.max()/np.median(W):.0f}x")
    print(f"u1 in [{X[:,0].min():.3f},{X[:,0].max():.3f}]  u2 in [{X[:,1].min():.3f},{X[:,1].max():.3f}]")
    # check uniformity of u under PURE phase space (sanity): histogram of u should be ~flat
    # for events weighted by detJ/ (q1 q2) ... here just report u-marginal shape of raw events
    h, _ = np.histogram(X[:, 0], bins=10, range=(0, 1))
    print("u1 raw-event histogram (10 bins):", (h / h.mean()).round(2))
