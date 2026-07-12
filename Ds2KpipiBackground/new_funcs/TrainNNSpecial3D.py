import tensorflow as tf

import amplitf.interface as atfi
atfi.set_single_precision()

import tfa.plotting as tfp
import tfa.rootio as tfr
import tfa.neural_nets as tfn

import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from DistributionModel import observables_phase_space, observables_toys, observables_titles, exp_phase_space

variables = observables_toys
titles = observables_titles
bounds = exp_phase_space.bounds()

data = atfi.const(tfr.read_tuple("0.20.41e5_train.root", variables))
data = exp_phase_space.filter(data)

print(f"Data after sideband filter: {data.shape[0]}")

plt.ion()
tfp.set_lhcb_style(size=8, usetex=False)
fig, ax = plt.subplots(nrows=len(variables), ncols=len(variables), figsize=(10, 7))

mass_penalty_fraction = 10.
weight_penalty = 1.

def regularisation(weights):
    """
    L2 regularisation with extra penalty on first-layer mD weights.
    weights[0] shape = (3, 32): row 2 = mD input connections.
    """
    penalty = 0.
    for w in weights:
        penalty += tf.reduce_sum(tf.square(w))
    penalty += mass_penalty_fraction * tf.reduce_sum(tf.square(weights[0][2, :]))
    return weight_penalty * penalty

tfn.estimate_density(
    exp_phase_space,
    data,
    ranges=bounds,
    labels=titles,
    learning_rate=0.0002,
    n_hidden=[32, 64, 32, 8],
    training_epochs=50000,
    norm_size=500000,
    print_step=50,
    display_step=500,
    initfile="init_3d.npy",
    outfile="train_3d",
    seed=4,
    fig=fig,
    axes=ax,
    regularisation=regularisation,
)
