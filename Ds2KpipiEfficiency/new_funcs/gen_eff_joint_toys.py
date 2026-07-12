"""
Chapter 8.2 (model-assisted ANN, EFFICIENCY) -- STEP 1 data.

Generates the JOINT (x, Theta) training sample:
  x     = (m', theta')                                  2 observables
  Theta = (ptcut, pcut, dptcut, maxptcut, sumptcut)     5 latent params (paper Table 3)
The 5 cuts vary event-by-event via selection_with_random_cuts.
Output = 7 columns. Paper §8.2 uses 2e6 events (generation is cheap); default 1e6.

Usage:  python gen_eff_joint_toys.py [nev] [outfile]
"""
import os
import sys
import tensorflow as tf

import amplitf.interface as atfi
import tfa.rootio as tfr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from DistributionModel import (
    parameters_list, observables_toys, observables_phase_space,
    selection_with_random_cuts, random_array_size,
)


def main():
    nev = int(float(sys.argv[1])) if len(sys.argv) > 1 else 2000000   # paper §8.2 uses 2e6
    outfile = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "eff_joint_2e6.root")
    chunk_size = 500000

    atfi.set_seed(nev + 1)

    branches = observables_toys + [i[0] for i in parameters_list]

    n = 0
    arrays = []
    while True:
        unfiltered = observables_phase_space.unfiltered_sample(chunk_size)
        sample = observables_phase_space.filter(unfiltered)
        rnd = tf.random.uniform([sample.shape[0], random_array_size], dtype=atfi.fptype())
        array = atfi.stack(selection_with_random_cuts(sample, rnd), axis=1)
        arrays += [array]
        size = array.shape[0]
        n += size
        print(f"Selected size = {n}, last chunk = {size}")
        if n > nev:
            break

    tfr.write_tuple(outfile, atfi.concat(arrays, axis=0)[:nev, :], branches)
    print(f"Saved {outfile}  ({nev} events, {len(branches)} branches)")


if __name__ == "__main__":
    main()
