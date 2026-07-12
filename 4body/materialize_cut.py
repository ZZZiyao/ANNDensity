"""Materialise the cut dataset (no logit): runs12_cut.npz (F=6 features m,cos+cos/sin phi)."""
import os
import numpy as np
import transform_cut as tc

HERE = os.path.dirname(os.path.abspath(__file__))
d = tc.load_cut(os.path.join(HERE, "runs12_MagBoth_L0Both.root"))
np.savez(os.path.join(HERE, "runs12_cut.npz"),
         F=d["F"], W=d["W"], X5=d["X5"], features=np.array(tc.FEATURES))
neff = d["W"].sum() ** 2 / (d["W"] ** 2).sum()
print(f"Wrote runs12_cut.npz  N={len(d['W'])}  Neff={neff:.0f}  "
      f"({os.path.getsize(os.path.join(HERE, 'runs12_cut.npz'))/1e6:.0f} MB)")
