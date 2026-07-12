"""
Veto specification for the acceptance-recovery extension.

A veto removes every event whose ORIGINAL cut coords fall simultaneously inside a set of
per-variable windows -- so a 1D spec carves a slab, a 2D spec a column, a 3D spec a block,
etc.  The SAME spec is used to (a) drop training events, (b) exclude the norm sample from
the veto region (crucial: otherwise the unbinned ANN collapses eps->0 there), and (c) define
the held-out ground-truth region for recovery evaluation.

Spec string (env ACC_VETO):  "m1:0.80:0.88"           (1D mass slab)
                              "m1:0.80:0.88,cos1:0.3:0.6"   (2D block)
                              "cos1:0.2:0.6,cos2:0.2:0.6,m1:0.80:0.92"  (3D)
Empty / unset -> no veto.
"""
import numpy as np

VAR = {"m1": 0, "m2": 1, "cos1": 2, "cos2": 3, "phi": 4}


def parse(spec):
    """'m1:0.80:0.88,cos1:0.3:0.6' -> [(0,0.80,0.88),(2,0.3,0.6)]; '' -> []."""
    if not spec:
        return []
    out = []
    for part in spec.split(","):
        v, lo, hi = part.split(":")
        out.append((VAR[v] if v in VAR else int(v), float(lo), float(hi)))
    return out


def mask(X5, spec):
    """Boolean array, True = event is INSIDE the veto (to be removed). X5 = (N,5) orig coords."""
    if isinstance(spec, str):
        spec = parse(spec)
    m = np.ones(len(X5), bool)
    for (v, lo, hi) in spec:
        m &= (X5[:, v] > lo) & (X5[:, v] < hi)
    return m


def describe(spec):
    if isinstance(spec, str):
        spec = parse(spec)
    inv = {v: k for k, v in VAR.items()}
    return " AND ".join(f"{inv.get(v, v)} in ({lo:g},{hi:g})" for (v, lo, hi) in spec) or "none"
