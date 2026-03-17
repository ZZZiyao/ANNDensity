import amplitf.interface as atfi

atfi.set_single_precision()

import tfa.plotting as tfp
import tfa.rootio as tfr
import tfa.neural_nets as tfn

import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from DistributionModel import observables_phase_space, observables_toys, observables_titles

bounds = observables_phase_space.bounds()

data = atfi.const(tfr.read_tuple("eff_toy_4e6.root", observables_toys))

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
    outfile="eff_train_2d",
    seed=2,
    fig=fig,
    axes=ax,
)
