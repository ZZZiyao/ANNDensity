import amplitf.interface as atfi

atfi.set_single_precision()

import tfa.plotting as tfp
import tfa.rootio as tfr
import tfa.neural_nets as tfn

import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless (CSD3 batch); interactive calls become no-ops
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from DistributionModel import observables_phase_space, observables_toys, observables_titles

bounds = observables_phase_space.bounds()

# env-overridable I/O so the ptcut=0.25 sample trains a separate model
_TOY_LO  = os.environ.get("EFF_TOY_LO", "eff_toy_1e5.root")
_OUTNAME = os.environ.get("EFF_MODEL_PREFIX", "eff_train_2d")
data = atfi.const(tfr.read_tuple(_TOY_LO, observables_toys))

tfp.set_lhcb_style(size=9, usetex=False)
fig, ax = plt.subplots(nrows=len(observables_toys), ncols=len(observables_toys), figsize=(8, 6))

tfn.estimate_density(
    observables_phase_space,
    data,
    ranges=bounds,
    labels=observables_titles,
    learning_rate=0.001,
    weight_penalty=0.1,
    n_hidden=[32, 64, 32, 8],
    training_epochs=30000,
    norm_size=500000,
    print_step=50,
    display_step=500,
    initfile="init_2d.npy",
    outfile=_OUTNAME,
    seed=2,
    fig=fig,
    axes=ax,
)
