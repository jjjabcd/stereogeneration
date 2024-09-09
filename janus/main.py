import os, sys
sys.path.append('..')

from stereogeneration import JANUS, docking, cd, fingerprint
from stereogeneration.janus import JANUS
from stereogeneration.filter import passes_filter

import selfies as sf
import selfies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.covariance import EllipticEnvelope

import rdkit.Chem as Chem
from rdkit.Chem import Draw

import multiprocessing
from functools import partial
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--target", action="store", type=str, default="1OYT", help="Protein target, defaults 1OYT.")
    parser.add_argument("--num_workers", action="store", type=int, default=1, help="Number of workers, defaults 1.")
    parser.add_argument("--stereo", action="store_true", dest="stereo", help="Toggle stereogeneration, defaults false.", default=False)
    parser.add_argument("--classifier", action="store_true", dest="use_classifier", help="Toggle classifier, defaults false", default=False)
    parser.add_argument("--use_fragments", action="store_true", help="Toggle using fragments, defaults false.", default=False)
    parser.add_argument("--starting_pop", action="store", type=str, default="best", help="Method to select starting population: random, worst, best. Defaults best.")
    parser.add_argument("--starting_size", action="store", type=int, default=5000, help="Number of starting smiles, must be larger than pop size")

    FLAGS = parser.parse_args()
    assert FLAGS.target in [
        '1OYT',             # docking to 1oyt
        '1SYH',             # docking to 1syh
        '6Y2F',             # docking to 6y2f
        'cd',               # circular dichroism target
        'fp-albuterol',     # fingerprints with albuterol
        'fp-mestranol'      # fingerprints with mestranol
    ], 'Invalid target fitness.'
    
    stereo = FLAGS.stereo
    print(f'Stereoisomers? : {stereo}')

    # if looking at fingerprint similarity, turn off the diverse topk option
    use_diverse_topk = 'fp' not in FLAGS.target

    # load isoalphabet from zinc dataset
    fname = '../data/isoalphabet.npz' if stereo else '../data/alphabet.npz'
    saved_alpha = np.load(fname, allow_pickle=True)
    alphabet = saved_alpha['alphabet'].tolist()
    # weights = saved_alpha['weights'].tolist() if stereo else None
    weights = None
    print(f'Alphabet contains: {len(alphabet)}')

    # function with specified target
    # use `partial` to allow multiprocessing over cpus
    if FLAGS.target in ['1OYT', '1SYH', '6Y2F']:
        tar_func = partial(docking.fitness_function, target=FLAGS.target)
    elif FLAGS.target == 'cd':
        tar_func = cd.fitness_function
    else:
        tar_func = partial(fingerprint.fitness_function, target=FLAGS.target)
    tar_func.__name__ = f'{FLAGS.target}_score'
    
    # all parameters to be set, below are defaults
    params_dict = {
        # Number of iterations that JANUS runs for
        "generations": 50,

        # The number of molecules for which fitness calculations are done, 
        # exploration and exploitation each have their own population
        # so the effective total budget per iteration at most 2*generation_size
        "generation_size": 100,
        
        # Number of molecules that are exchanged between the exploration and exploitation
        # this should be < 10% of generation size; this is the number of replaced species
        # if too high, too many molecules will be replaced
        "num_exchanges": 5,

        # Callable filtering function (None defaults to no filtering)
        "custom_filter": passes_filter,

        # Fragments from starting population used to extend alphabet for mutations
        "use_fragments": FLAGS.use_fragments,

        # An option to use a classifier as selection bias
        "use_classifier": FLAGS.use_classifier,

        # alphabet parameters
        "alphabet": alphabet,
        "alphabet_weights": weights,
        "num_sample_frags": 50,

        # exploration parameters
        "explr_num_random_samples": 64,
        "explr_num_mutations": 10,
        "crossover_num_random_samples": 3, # 5,

        # exploitation parameters
        "exploit_num_random_samples": 128,
        "exploit_num_mutations": 200,
        "top_mols": 5,

        # other parameters
        'stereo': stereo,
        "num_workers": FLAGS.num_workers,
        "verbose_out": True,

        "use_diverse_topk": use_diverse_topk

    }

    # get initial fitnesses from csv
    # read the dataset
    df = pd.read_csv(f'../../zinc.csv')
    if not stereo:
        df = df[['smiles', FLAGS.target]].rename(columns={FLAGS.target: 'fitness'})
    else:
        df = df[['isosmiles', FLAGS.target]].rename(
            columns={'isosmiles':'smiles', FLAGS.target: 'fitness'}
        )


    # get the starting smiles and write to file
    # threshold = params_dict['generation_size']
    threshold = FLAGS.starting_size
    if FLAGS.starting_pop == 'worst':
        start_df = df.sort_values(by = 'fitness', ascending = True)[:threshold]
    elif FLAGS.starting_pop == 'best':
        start_df = df.sort_values(by = 'fitness', ascending = False)[:threshold]
    elif FLAGS.starting_pop == 'random':
        start_df = df.sample(n=threshold)
    else:
        raise ValueError('Invalid method to selecting starting population.')
    start_df = start_df.drop_duplicates('smiles', keep=False)
    init_fitness = start_df['fitness'].tolist()

    # write the smiles file, this will be read by JANUS
    fname = f'{FLAGS.target}_starting_smiles.txt'
    with open(fname, 'w') as f:
        for smi in start_df['smiles']:
            f.write(smi+'\n')

    # create JANUS agent
    output_dir = 'RESULTS_stereo' if stereo else 'RESULTS_nonstereo'
    agent = JANUS(
        work_dir=output_dir,
        fitness_function = tar_func,
        start_population = fname,
        starting_fitness = init_fitness,
        **params_dict
    )

    agent.run()

