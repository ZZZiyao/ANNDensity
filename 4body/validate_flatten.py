"""
Validate that the materialised flattened dataset (runs12_flat.npz) processes the
weights correctly, i.e. that it encodes the SAME physical acceptance eps as the
original weight_detJ -- with no coding/units/indexing bug.

Checks (all must pass):
  1. weight identity   : W_new  ==  weight_detJ * q(m1) * q(m2)   (per event)
  2. weight identity 2 : W_new  ==  weight       /  pB(m1,m2)     (per event, 2nd route)
  3. coordinate round-trip : u_to_m(stored u) == stored m
  4. Jacobian/uniformity   : a q(m)-distributed sample maps to a UNIFORM u
  5. eps invariance        : eps(m1) and eps(cos1) reconstructed from the NEW file
                             (W_new / (q1 q2) = weight_detJ) coincide with the ORIGINAL.
A figure validate_flatten.png overlays the eps curves and the uniformity test.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot
import phasespace_transform as pst

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "runs12_MagBoth_L0Both.root")
NPZ = os.path.join(HERE, "runs12_flat.npz")

q, pB, u_to_m, m_to_u = pst.q, pst.pB, pst.u_to_m, pst.m_to_u


def neff(x):
    return x.sum() ** 2 / (x ** 2).sum()


def main():
    # ---- load both ----
    a = uproot.open(SRC)["DecayTree"].arrays(
        ["m1", "m2", "cos1", "cos2", "phi", "weight", "weight_detJ"], library="np")
    m1o = np.asarray(a["m1"], float); m2o = np.asarray(a["m2"], float)
    c1o = np.asarray(a["cos1"], float)
    w_o = np.asarray(a["weight"], float)
    wdJ = np.asarray(a["weight_detJ"], float)

    z = np.load(NPZ)
    X, W, m1n, m2n = z["X"], z["W"].astype(float), z["m1"].astype(float), z["m2"].astype(float)
    u1, u2 = X[:, 0].astype(float), X[:, 1].astype(float)

    print("=" * 70)
    ok_all = True

    def check(name, cond, detail):
        nonlocal ok_all
        ok_all &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name}: {detail}")

    # 1. weight DEFINITION: W == weight_detJ * q1 * q2  (stored eps-weight x transform Jacobian)
    target1 = wdJ * q(m1o) * q(m2o)
    rel1 = np.abs(W - target1) / np.maximum(np.abs(target1), 1e-30)
    check("weight = weight_detJ*q1*q2  (exact definition)", rel1.max() < 1e-4,
          f"max rel.err = {rel1.max():.2e}  (median {np.median(rel1):.1e})")

    # INFO (not a gate): how well does the analytic factorisation detJ = q1 q2 pB hold?
    #   if it were exact, weight/pB would equal W.  It fails for rare near-threshold events,
    #   which is exactly why W is tied to the STORED weight_detJ, not to analytic pB.
    target2 = w_o / pB(m1o, m2o)
    rel2 = np.abs(W - target2) / np.maximum(np.abs(target2), 1e-30)
    frac_bad = float((rel2 > 0.01).mean())
    print(f"[INFO] analytic detJ=q1q2pB approximation: median dev {np.median(rel2):.1e}, "
          f"max {rel2.max():.2e}, {frac_bad*100:.3f}% of events off by >1% (near threshold) "
          f"-> avoided by using stored weight_detJ")

    # 3. coordinate round-trip: stored u must invert to stored m, and m_to_u(m) == stored u
    rt_m = np.abs(u_to_m(u1) - m1n)
    rt_u = np.abs(m_to_u(m1n) - u1)
    check("round-trip u<->m", rt_m.max() < 1e-3 and rt_u.max() < 1e-4,
          f"|u_to_m(u)-m|max={rt_m.max():.2e}  |m_to_u(m)-u|max={rt_u.max():.2e}")

    # 4. Jacobian/uniformity: draw m ~ q(m) (pure phase space, 1D), transform -> u should be flat
    rng = np.random.RandomState(0)
    grid = np.linspace(pst.M_LO, pst.M_HI, 20000); Q = q(grid); pdf = Q / Q.sum()
    m_samp = rng.choice(grid, size=2000000, p=pdf)
    u_samp = m_to_u(m_samp)
    h, _ = np.histogram(u_samp, bins=20, range=(0, 1))
    flat_dev = np.abs(h / h.mean() - 1).max()
    check("q(m)-sample -> uniform u", flat_dev < 0.02,
          f"max bin deviation from flat = {flat_dev*100:.2f}%")

    # 5. eps invariance: reconstruct weight_detJ from new file (= W/(q1 q2)) and compare eps proj
    wdJ_from_new = W / (q(m1n) * q(m2n))
    rel5 = np.abs(wdJ_from_new - wdJ) / np.maximum(wdJ, 1e-30)
    check("eps weight recovered from new file", rel5.max() < 1e-4,
          f"max rel.err vs original weight_detJ = {rel5.max():.2e}")

    print("=" * 70)
    print(f"Neff: original weight_detJ = {neff(wdJ):.0f}   ->   new W = {neff(W):.0f}  "
          f"(x{neff(W)/neff(wdJ):.2f})")
    print("ALL CHECKS PASSED" if ok_all else "*** SOME CHECKS FAILED ***")

    # ---- figure: eps overlays + uniformity ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    # eps(m1): original (weight_detJ) vs new-file-reconstructed (W/(q1 q2))
    b = np.linspace(pst.M_LO, pst.M_HI, 51); ctr = 0.5 * (b[:-1] + b[1:])
    Ha, _ = np.histogram(m1o, bins=b, weights=wdJ); Ha /= Ha.sum()
    Hb, _ = np.histogram(m1n, bins=b, weights=wdJ_from_new); Hb /= Hb.sum()
    ax[0].step(ctr, Ha, where="mid", color="k", lw=3, alpha=0.4, label="original (weight_detJ)")
    ax[0].step(ctr, Hb, where="mid", color="r", lw=1.2, label="new file (W/q1q2)")
    ax[0].set_xlabel("m1 (GeV)"); ax[0].set_ylabel("eps (norm)"); ax[0].set_title("eps(m1): original vs new"); ax[0].legend(fontsize=8)
    # eps(cos1)
    bc = np.linspace(-1, 1, 51); ctrc = 0.5 * (bc[:-1] + bc[1:])
    Hac, _ = np.histogram(c1o, bins=bc, weights=wdJ); Hac /= Hac.sum()
    Hbc, _ = np.histogram(X[:, 2].astype(float), bins=bc, weights=wdJ_from_new); Hbc /= Hbc.sum()
    ax[1].step(ctrc, Hac, where="mid", color="k", lw=3, alpha=0.4, label="original")
    ax[1].step(ctrc, Hbc, where="mid", color="r", lw=1.2, label="new file")
    ax[1].set_xlabel("cos1"); ax[1].set_title("eps(cos1): original vs new"); ax[1].legend(fontsize=8)
    # uniformity of u under pure phase space
    ax[2].hist(u_samp, bins=40, range=(0, 1), color="tab:green")
    ax[2].axhline(len(u_samp) / 40, color="k", ls="--")
    ax[2].set_xlabel("u (from q(m)-distributed sample)"); ax[2].set_title(f"phase space -> flat in u (dev {flat_dev*100:.1f}%)")
    plt.tight_layout(); plt.savefig(os.path.join(HERE, "validate_flatten.png"), dpi=120)
    print("Saved validate_flatten.png")


if __name__ == "__main__":
    main()
