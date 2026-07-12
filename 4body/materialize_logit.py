"""Materialise the cut+logit dataset: runs12_cut_logit.npz (F=6 features, W=weight_detJ)."""
import os
import numpy as np
import transform_logit as tl

HERE = os.path.dirname(os.path.abspath(__file__))
d = tl.load_cut(os.path.join(HERE, "runs12_MagBoth_L0Both.root"))
np.savez(os.path.join(HERE, "runs12_cut_logit.npz"),
         F=d["F"], W=d["W"], X5=d["X5"], features=np.array(tl.FEATURES))
neff = d["W"].sum() ** 2 / (d["W"] ** 2).sum()
print(f"Wrote runs12_cut_logit.npz  N={len(d['W'])}  Neff={neff:.0f}  "
      f"({os.path.getsize(os.path.join(HERE, 'runs12_cut_logit.npz'))/1e6:.0f} MB)")
