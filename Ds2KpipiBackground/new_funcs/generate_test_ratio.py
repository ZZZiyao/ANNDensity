"""
Generate test_ratio.root (train) and test_ratio_test.root (test)
with kstarfrac=0.2, rhofrac=0.4, 100k events each.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import tensorflow as tf
import amplitf.interface as atfi
import tfa.rootio as tfr

from DistributionModel import (
    true_cuts, generate_selection, generated_variables,
    observables_phase_space, random_array_size,
)

def main():
    nev = 100000
    kstarfrac = 0.2
    rhofrac = 0.4
    chunk_size = 500000

    atfi.set_seed(42)

    cuts = list(true_cuts[:6]) + [atfi.const(kstarfrac), atfi.const(rhofrac)]

    arrays = []
    n = 0
    while n < nev:
        rnd = tf.random.uniform([chunk_size, random_array_size], dtype=atfi.fptype())
        obs = generate_selection(cuts, rnd, constant_cuts=True)
        batch = atfi.stack(obs, axis=1)
        arrays.append(batch)
        n += batch.shape[0]
        print(f"  collected {n}/{nev}")

    data = np.concatenate(arrays, axis=0)[:nev]

    tfr.write_tuple("0.20.41e5_train.root", data, generated_variables)
    print(f"Saved test_ratio.root ({nev} events, kstar={kstarfrac}, rho={rhofrac})")

    # Generate independent test set with different seed
    atfi.set_seed(123)

    arrays = []
    n = 0
    while n < nev:
        rnd = tf.random.uniform([chunk_size, random_array_size], dtype=atfi.fptype())
        obs = generate_selection(cuts, rnd, constant_cuts=True)
        batch = atfi.stack(obs, axis=1)
        arrays.append(batch)
        n += batch.shape[0]
        print(f"  test: collected {n}/{nev}")

    data = np.concatenate(arrays, axis=0)[:nev]

    tfr.write_tuple("0.20.41e5_test.root", data, generated_variables)
    print(f"Saved test_ratio_test.root ({nev} events, kstar={kstarfrac}, rho={rhofrac})")


if __name__ == "__main__":
    main()
