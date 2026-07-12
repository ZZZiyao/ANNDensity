"""
1x3 sideband-diagnostic figure for one (kstarfrac, rhofrac): the bottom row of the 6-panel
diagnose_components figure (mD with SB windows; m' and theta' Signal vs Lower/Upper SB
projections), relabelled (a)(b)(c). Reuses the exact helpers so the style matches.

Usage:  python plot_sideband_abc.py [nev]     (env BG_KF, BG_RF; default 0.2, 0.4)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # parent has DistributionModel
import numpy as np
import matplotlib.pyplot as plt
import amplitf.interface as atfi

from diagnose_components import (
    generate_mix, define_regions_hard, hist_errorbar,
    MD_LO, MD_S1, MD_S2, MD_HI,
)

plt.rcParams.update({"font.size": 12, "axes.labelsize": 14,
                     "xtick.direction": "in", "ytick.direction": "in"})

kf = float(os.environ.get("BG_KF", 0.2))
rf = float(os.environ.get("BG_RF", 0.4))
nev = int(float(sys.argv[1])) if len(sys.argv) > 1 else 1000000

atfi.set_seed(42)
data = generate_mix(kf, rf, nev)

n = min(100000, len(data["mprime"]))
idx = np.random.choice(len(data["mprime"]), n, replace=False)
m, t, md = data["mprime"][idx], data["thetaprime"][idx], data["md"][idx]
signal, lowSB, upSB, full = define_regions_hard(md)
Ns = int(signal.sum())
print(f"kf={kf}, rf={rf}: subsample {n}, signal {Ns}, lowSB {int(lowSB.sum())}, upSB {int(upSB.sum())}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

# (a) mD with sideband window lines
ax = axes[0]
bins_md = np.arange(MD_LO, MD_HI + 1e-12, 0.004)
hist_errorbar(ax, md, bins_md, mask=full, fmt="k.", ls="none")
ax.axvline(MD_S1, color="red", linestyle="--", linewidth=1.5)
ax.axvline(MD_S2, color="red", linestyle="--", linewidth=1.5)
ax.set_xlabel(r"$m_D$ (GeV)"); ax.set_ylabel("Entries / (0.004 GeV)")
ax.set_title("(a) $m_D$"); ax.set_ylim(bottom=0)
ax.minorticks_on(); ax.tick_params(which="both", top=True, right=True)

# (b) m' projections
ax = axes[1]
bins_m = np.arange(0, 1.0 + 1e-12, 0.01)
hist_errorbar(ax, m, bins_m, mask=signal, label="Signal", fmt="k.", ls="none")
hist_errorbar(ax, m, bins_m, mask=lowSB, label="Lower SB", fmt="b", ls="--", lw=1.5, scale_to=Ns)
hist_errorbar(ax, m, bins_m, mask=upSB, label="Upper SB", fmt="r", ls="--", lw=1.5, scale_to=Ns)
ax.set_xlabel(r"$m'$"); ax.set_ylabel("Entries / (0.01)")
ax.set_title("(b) $m'$ projections"); ax.set_ylim(bottom=0)
ax.legend(frameon=False, fontsize=10)
ax.minorticks_on(); ax.tick_params(which="both", top=True, right=True)

# (c) theta' projections
ax = axes[2]
bins_t = np.arange(0, 1.0 + 1e-12, 0.01)
hist_errorbar(ax, t, bins_t, mask=signal, label="Signal", fmt="k.", ls="none")
hist_errorbar(ax, t, bins_t, mask=lowSB, label="Lower SB", fmt="b", ls="--", lw=1.5, scale_to=Ns)
hist_errorbar(ax, t, bins_t, mask=upSB, label="Upper SB", fmt="r", ls="--", lw=1.5, scale_to=Ns)
ax.set_xlabel(r"$\theta'$"); ax.set_ylabel("Entries / (0.01)")
ax.set_title("(c) $\\theta'$ projections"); ax.set_ylim(bottom=0)
ax.legend(frameon=False, fontsize=10)
ax.minorticks_on(); ax.tick_params(which="both", top=True, right=True)

plt.tight_layout()
os.makedirs("plots", exist_ok=True)
out = os.path.join("plots", f"sideband_abc_kf{kf:.1f}_rf{rf:.1f}.png")
plt.savefig(out, dpi=200)
print(f"Saved {out}")
