"""
Training-curve figure: train vs validation NLL for each of the three neural-network fits,
drawn as THREE separate panels (one per problem), with the early-stopping best epoch marked.

The logged NLL is a SUM over events (train over ~9x more events than val, and the 4-body
sum is weighted), so raw train/val sit far apart.  Each curve is divided by the number of
events in its sum (Sum w for the weighted 4-body) to give a per-event NLL, so train and val
overlay and the generalisation gap is readable.  Note the "train" curve is the training loss
and includes the small L2 penalty; "val" is the pure NLL used for early stopping.

Output: 4body/training_curves.png
"""
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"D:\DIS\project\MPhiiProject\ANNDensity"

# title, logfile, N_train, N_val  (sum of weights for the weighted 4-body)
PANELS = [
    ("3-body acceptance", os.path.join(ROOT, "Ds2KpipiEfficiency", "new_funcs", "eff_joint_train_gpu.txt"), 900000.0, 100000.0),
    ("3-body background", os.path.join(ROOT, "Ds2KpipiBackground", "new_funcs", "bg_joint_train_gpu.txt"), 900000.0, 100000.0),
    ("4-body acceptance", os.path.join(ROOT, "4body", "acc_5d_swish_lam0.01.txt"), 98413.4, 11004.0),
]

PAT = re.compile(r"Epoch\s+(\d+)\s+train\s+(-?[\d.]+)\s+val\s+(-?[\d.]+)")
LOGX = bool(int(os.environ.get("TC_LOGX", "0")))   # set TC_LOGX=1 for log epoch axis


def parse(logfile):
    ep, tr, va = [], [], []
    with open(logfile) as f:
        for line in f:
            m = PAT.search(line)
            if m:
                ep.append(int(m.group(1))); tr.append(float(m.group(2))); va.append(float(m.group(3)))
    return np.array(ep), np.array(tr), np.array(va)


def main():
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13,
                         "xtick.direction": "in", "ytick.direction": "in"})
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    C_TRAIN, C_VAL, C_BEST, C_LINE = "#4477AA", "#EE6677", "#228833", "#BBBBBB"

    # panels that get a zoomed inset around the best epoch: index -> (x_lo, x_hi)
    ZOOM = {} if LOGX else {0: (25000, 50000), 1: (38000, 50000)}
    XMAX = 50000
    EXTEND = {} if LOGX else {2: XMAX}   # 4-body converged early: hold its curve flat to XMAX

    for i, (ax, (title, logf, ntr, nval)) in enumerate(zip(axes, PANELS)):
        ep, tr, va = parse(logf)
        tr_pe, va_pe = tr / ntr, va / nval
        bi = int(np.argmin(va))
        be, bv = ep[bi], va_pe[bi]

        def draw(a, legend=False):
            a.axvline(be, color=C_LINE, ls="--", lw=1.1, zorder=1)
            a.plot(ep, tr_pe, color=C_TRAIN, lw=1.6, label="train" if legend else None)
            a.plot(ep, va_pe, color=C_VAL, lw=1.6, label="validation" if legend else None)
            a.plot(be, bv, "o", color=C_BEST, ms=7, mec="white", mew=0.9, zorder=5)

        draw(ax, legend=(i == 0))

        # hold the converged curve flat out to XMAX
        if i in EXTEND:
            xh = [ep[-1], EXTEND[i]]
            ax.plot(xh, [tr_pe[-1]] * 2, color=C_TRAIN, lw=1.6)
            ax.plot(xh, [va_pe[-1]] * 2, color=C_VAL, lw=1.6)

        # y-limits from the post-spike region (drop the epoch-0 pre-training value)
        m = ep >= 100
        y = np.concatenate([tr_pe[m], va_pe[m]])
        lo, hi = y.min(), y.max()
        pad = 0.06 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        if LOGX:
            ax.set_xscale("log"); ax.set_xlim(left=max(ep[m].min(), 50))
        else:
            ax.set_xlim(0, XMAX)

        ax.set_xlabel("epoch"); ax.set_ylabel("NLL / event")
        ax.set_title(f"{title}  –  best epoch {be}")
        if i == 0:
            ax.legend(frameon=False, fontsize=11, loc="center left")
        ax.minorticks_on(); ax.tick_params(which="both", top=True, right=True)

        # small zoomed inset ("magnifier") around the best epoch
        if i in ZOOM:
            x0, x1 = ZOOM[i]
            axins = ax.inset_axes([0.58, 0.50, 0.38, 0.42])
            draw(axins)
            wm = (ep >= x0) & (ep <= x1)
            yz = np.concatenate([tr_pe[wm], va_pe[wm]])
            zp = 0.15 * (yz.max() - yz.min()) or 5e-4
            axins.set_xlim(x0, x1); axins.set_ylim(yz.min() - zp, yz.max() + zp)
            axins.tick_params(labelsize=7); axins.minorticks_on()
            ax.indicate_inset_zoom(axins, edgecolor="0.6", lw=0.8)

        print(f"{title:20s}  best epoch {be:6d}  val/ev {bv:.4f}  train/ev {tr_pe[bi]:.4f}")

    fig.suptitle("Neural-network training curves (train vs. validation)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    name = "training_curves_logx.png" if LOGX else "training_curves.png"
    out = os.path.join(ROOT, "4body", name)
    plt.savefig(out, dpi=150)
    print("Saved", out)


if __name__ == "__main__":
    main()
