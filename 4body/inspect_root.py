import uproot, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
t = uproot.open(os.path.join(HERE, "runs12_MagBoth_L0Both.root"))["DecayTree"]
a = t.arrays(["m1", "detJ", "weight", "weight_detJ"], library="np")
m1 = np.asarray(a["m1"], float)
detJ = np.asarray(a["detJ"], float)
w = np.asarray(a["weight"], float)
wdJ = np.asarray(a["weight_detJ"], float)

def hist(x, ww, lbl, nb=14, lo=0.63, hi=1.05):
    H, e = np.histogram(x, bins=nb, range=(lo, hi), weights=ww)
    print(f"\n{lbl}:")
    mx = H.max()
    for i in range(nb):
        bar = "#" * int(55 * H[i] / mx) if mx > 0 else ""
        print(f"  [{e[i]:.3f},{e[i+1]:.3f}] {bar}")

hist(m1, None, "m1 raw (unweighted)")
hist(m1, detJ, "m1 * detJ")
hist(m1, 1.0/detJ, "m1 / detJ")
hist(m1, w, "m1 * weight")
hist(m1, wdJ, "m1 * weight_detJ")
print("\nweight==1 fraction:", np.mean(np.isclose(w, 1.0)))
print("weight integer-like fraction:", np.mean(np.isclose(w, np.round(w))))
