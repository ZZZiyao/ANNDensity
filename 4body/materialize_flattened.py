"""
Materialise a flattened training dataset from runs12_MagBoth_L0Both.root.

Writes runs12_flat.npz with the training-ready arrays (already in phase-space-flattened
coordinates, with the tame weight baked in):
    X  : (N,5) float32  = (u1, u2, cos1, cos2, phi)        u_i = phase-space CDF of m_i
    W  : (N,)   float32 = weight / pB(m1,m2)               weighted density = acceptance eps
    m1, m2 : (N,) float32  physical masses (kept for reference / physical-axis plots)

The transform is deterministic (see phasespace_transform.py). Run validate_flatten.py
afterwards to confirm the weight handling is correct.
"""
import os
import numpy as np
import phasespace_transform as pst

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "runs12_MagBoth_L0Both.root")
OUT = os.path.join(HERE, "runs12_flat.npz")


def neff(x):
    return x.sum() ** 2 / (x ** 2).sum()


def main():
    d = pst.load_flattened(SRC)
    X, W, m1, m2 = d["X"], d["W"], d["m1"].astype(np.float32), d["m2"].astype(np.float32)
    np.savez(OUT, X=X, W=W, m1=m1, m2=m2,
             columns=np.array(["u1", "u2", "cos1", "cos2", "phi"]))
    sz = os.path.getsize(OUT) / 1e6
    print(f"Wrote {OUT}  ({sz:.0f} MB)")
    print(f"  N={len(W)}  Neff={neff(W):.0f}  Neff/N={neff(W)/len(W):.3f}")
    print(f"  W: median={np.median(W):.3f}  p99.9={np.percentile(W,99.9):.2f}  max={W.max():.2f}")
    print(f"  u1 in [{X[:,0].min():.4f},{X[:,0].max():.4f}]  u2 in [{X[:,1].min():.4f},{X[:,1].max():.4f}]")


if __name__ == "__main__":
    main()
