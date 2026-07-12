"""
Thesis figure for sec:problem-weights: the per-event weight w = weight/detJ before and
after the tighter K* mass window. Top: full window (long divergent tail from q->0 at the
K-pi threshold). Bottom: cut window (0.65,1.03) GeV, tail tamed, N_eff recovered.
Saves weight_cut_compare.png next to the other 4body figures (on the thesis graphicspath).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
M_LO, M_HI = 0.65, 1.03   # tighter analysis window (transform_cut.py)


def neff(w):
    return w.sum() ** 2 / (w ** 2).sum()


def main():
    import uproot
    t = uproot.open(os.path.join(HERE, "runs12_MagBoth_L0Both.root"))["DecayTree"]
    a = t.arrays(["m1", "m2", "weight_detJ"], library="np")
    m1 = np.asarray(a["m1"], float)
    m2 = np.asarray(a["m2"], float)
    w = np.asarray(a["weight_detJ"], float)

    cut = (m1 > M_LO) & (m1 < M_HI) & (m2 > M_LO) & (m2 < M_HI)
    wc = w[cut]

    # normalise both weights to unit mean so the two panels are on a common footing
    w_n = w / w.mean()
    wc_n = wc / wc.mean()

    def stats(x, n_full):
        return dict(N=len(x), Neff=neff(x), frac_evt=len(x) / n_full,
                    ratio=x.max() / np.median(x))

    s_full = stats(w_n, len(w))
    s_cut = stats(wc_n, len(w))
    print("full:", {k: (f"{v:.3g}" if isinstance(v, float) else v) for k, v in s_full.items()})
    print("cut :", {k: (f"{v:.3g}" if isinstance(v, float) else v) for k, v in s_cut.items()})

    xmax = np.percentile(w_n, 99.999)          # common x-range, drop a few extreme outliers
    bins = np.linspace(0, xmax, 160)

    fig, ax = plt.subplots(2, 1, figsize=(6.4, 5.2), sharex=True)

    ax[0].hist(w_n, bins=bins, color="#3b6fb0", log=True)
    ax[0].set_ylabel("events")
    ax[0].text(0.97, 0.90,
               r"full window $0.63<m<1.04$ GeV" "\n"
               rf"$N_{{\rm eff}}/N={s_full['Neff']/s_full['N']:.2f}$,  "
               rf"$w_{{\max}}/\tilde w={s_full['ratio']:.0f}$",
               transform=ax[0].transAxes, ha="right", va="top", fontsize=10)

    ax[1].hist(wc_n, bins=bins, color="#b03b3b", log=True)
    ax[1].set_ylabel("events")
    ax[1].set_xlabel(r"per-event weight $w=w_{\rm gen}/\det J$  (unit mean)")
    ax[1].text(0.97, 0.90,
               rf"cut window ${M_LO}<m<{M_HI}$ GeV" "\n"
               rf"$N_{{\rm eff}}/N={s_cut['Neff']/s_cut['N']:.2f}$,  "
               rf"$w_{{\max}}/\tilde w={s_cut['ratio']:.0f}$",
               transform=ax[1].transAxes, ha="right", va="top", fontsize=10)

    ax[0].set_xlim(0, xmax)
    for a_ in ax:
        a_.margins(x=0)
    fig.tight_layout()
    out = os.path.join(HERE, "weight_cut_compare.png")
    fig.savefig(out, dpi=150)
    fig.savefig(out.replace(".png", ".pdf"))
    print("wrote", out)


if __name__ == "__main__":
    main()
