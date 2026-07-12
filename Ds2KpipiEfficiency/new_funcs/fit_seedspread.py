"""
Aggregate the seed-spread study: fit the §8.2 latent params for each joint-ANN seed, then
report the MEAN +/- SPREAD (degeneracy uncertainty) per parameter and compare to the paper.

The spread across seeds is the degeneracy uncertainty -- different joint ANNs put the minuit
minimum at different points along the flat (degenerate) direction. The paper's value is one
draw from the same family; we check whether it falls within our spread.

Usage: EFF_PTCUT=0.4 python fit_seedspread.py "<glob of seed models>" <calibfile>
   e.g. EFF_PTCUT=0.4 python fit_seedspread.py "eff_joint_2e6_seed*.npy" eff_toy_1e5.root
Runs fit_eff_joint.py as a subprocess per model (reuses the exact same fit), parses its output.
"""
import os
import sys
import glob
import subprocess
import re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
models = sorted(glob.glob(sys.argv[1])) if len(sys.argv) > 1 else sorted(glob.glob("eff_joint_2e6_seed*.npy"))
calib  = sys.argv[2] if len(sys.argv) > 2 else "eff_toy_1e5.root"
# also include the original single model if present
for extra in ["eff_joint_train_2e6_gpu.npy"]:
    if os.path.exists(os.path.join(HERE, extra)) and extra not in models:
        models = [extra] + models
print(f"models: {models}\ncalib: {calib}\n")

# paper Table 3 (true=0.4): reco values
PAPER = {"ptcut": 0.408, "pcut": 4.5, "dptcut": 1.82, "maxptcut": 1.23, "sumptcut": 3.19}
TRUE  = {"ptcut": float(os.environ.get("EFF_PTCUT", 0.4)), "pcut": 3.0, "dptcut": 2.0,
         "maxptcut": 1.0, "sumptcut": 3.0}
ORDER = ["ptcut", "pcut", "dptcut", "maxptcut", "sumptcut"]

vals = {p: [] for p in ORDER}; chis = []
env = dict(os.environ); env.setdefault("EFF_FIT_RESTARTS", "15")
for m in models:
    out = subprocess.run([sys.executable, os.path.join(HERE, "fit_eff_joint.py"), calib,
                          os.path.splitext(m)[0] + "_ss"],
                         env={**env, "EFF_INITFILE": m}, capture_output=True, text=True).stdout
    for p in ORDER:
        mt = re.search(rf"{p}\s+fit =\s+([\-0-9.]+)", out)
        if mt: vals[p].append(float(mt.group(1)))
    mc = re.search(r"Chi2 .*?=\s+([0-9.]+)", out)
    if mc: chis.append(float(mc.group(1)))
    print(f"  {os.path.basename(m):32s} done"
          + ("" if mc is None else f"  chi2={float(mc.group(1)):.0f}"))

print(f"\n{'param':9s} {'true':>6s} {'mean':>7s} {'spread':>7s} {'min':>7s} {'max':>7s} {'paper':>7s}  paper in range?")
print("-" * 70)
for p in ORDER:
    a = np.array(vals[p]);
    if len(a) == 0: continue
    inrange = a.min() - a.std() <= PAPER[p] <= a.max() + a.std()
    print(f"{p:9s} {TRUE[p]:6.2f} {a.mean():7.3f} {a.std():7.3f} {a.min():7.3f} {a.max():7.3f} "
          f"{PAPER[p]:7.2f}  {'YES' if inrange else 'no'}")
print(f"\nchi2/ndof: {np.mean(chis)/2495:.3f} +/- {np.std(chis)/2495:.3f}  (paper 1.01)")
print(f"n seeds = {len(models)}")
