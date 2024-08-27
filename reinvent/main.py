#!/usr/bin/env python
import os, sys
sys.path.append('..')

# reinvent specific imports
from stereogeneration.reinvent.train_agent import train_agent
from stereogeneration.reinvent.train_prior import pretrain
from stereogeneration.reinvent.data_structs import construct_vocabulary
from stereogeneration.utils import normalize_score

from stereogeneration import utils, docking, cd, fingerprint
from rdkit.Chem import Crippen

import numpy as np
import pandas as pd

import rdkit.Chem as Chem

import multiprocessing
from functools import partial
import argparse
import time


parser = argparse.ArgumentParser(description="Main script for running the model")
parser.add_argument("--target", action="store", type=str, default="1OYT", 
                    help="Protein target, defaults 1OYT.")
parser.add_argument('--learning_rate', action='store', dest='learning_rate',
                    type=float, default=0.0005)
parser.add_argument('--num_steps', action='store', dest='n_steps', type=int,
                    default=50)
parser.add_argument('--batch_size', action='store', dest='batch_size', type=int,
                    default=200)
parser.add_argument('--sigma', action='store', dest='sigma', type=int,
                    default=50)
parser.add_argument('--experience', action='store', dest='experience_replay', type=int,
                    default=0, help='Number of experience sequences to sample each step. '\
                    '0 means no experience replay.')
parser.add_argument('--num_workers', action='store', dest='num_workers',
                    type=int, default=1,
                    help='Number of processes used to run the scoring function. "0" means ' \
                    'that the scoring function will be run in the main process.')
# parser.add_argument('--agent', action='store', dest='restore_agent_from',
#                     default='data/Prior.ckpt',
#                     help='Path to an RNN checkpoint file to use as a Agent.')
parser.add_argument('--lower_bound', action='store', dest='lower_bound', type=int, default=3,
                    help='Lower bound for score normalization.')
parser.add_argument('--upper_bound', action='store', dest='upper_bound', type=int, default=13,
                    help='Upper bound for score normalization.')
parser.add_argument("--stereo", action="store_true", dest="stereo", 
                    help="Toggle stereogeneration, defaults false.", default=False)
parser.add_argument("--starting_pop", action="store", type=str, default="best", help="Method to select starting population: random, worst, best. Defaults best.")
parser.add_argument("--starting_size", action="store", type=int, default=None, help="Number of "\
                    "starting smiles, must be larger than pop size")
parser.add_argument("--overwrite", action="store_true", dest="overwrite", 
                    help="Toggle processing/pretraining even if files exist, defaults false.", 
                    default=False)


if __name__ == "__main__":

    FLAGS = parser.parse_args()
    assert FLAGS.target in [
        '1OYT',             # docking to 1oyt
        '1SYH',             # docking to 1syh
        '6Y2F',             # docking to 6y2f
        'cd',               # circular dichroism target
        'fp-albuterol',     # fingerprints with albuterol
        'fp-mestranol'      # fingerprints with mestranol
    ], 'Invalid target fitness.'

    # generate target function
    # normalization functions should give a score between [-1, 1]
    if FLAGS.target in ['1OYT', '1SYH', '6Y2F']:
        tar_func = partial(docking.fitness_function, target=FLAGS.target)
        norm_func = lambda x: normalize_score(x, r=[FLAGS.lower_bound, FLAGS.upper_bound])
    elif FLAGS.target == 'cd':
        tar_func = cd.fitness_function
        norm_func = lambda x: normalize_score(x, r=[-5000, 5000])
    else:
        tar_func = partial(fingerprint.fitness_function, target=FLAGS.target)
        # norm_func = lambda x: np.multiply(2.0, np.maximum(x, -1.)) - 1.0
        k = 1.0
        norm_func = lambda x: np.minimum(np.maximum(x, 0.0), k) / k
    tar_func.__name__ = f'{FLAGS.target}_score'


    # read the dataset
    df = pd.read_csv(f'../../zinc.csv')
    print(f'Stereoinformation? : {FLAGS.stereo}')
    if not FLAGS.stereo: 
        starting_df = df[['smiles', FLAGS.target]].rename(columns={FLAGS.target: 'fitness'})
    else:
        starting_df = df[['isosmiles', FLAGS.target]].rename(columns={'isosmiles': 'smiles', FLAGS.target: 'fitness'})
    starting_df = starting_df.drop_duplicates('smiles')

    # process smiles (if required)
    # will save a Vocabulary object
    voc_path = '../data/Voc_stereo' if FLAGS.stereo else '../data/Voc_nonstereo'
    if not os.path.exists(voc_path) or FLAGS.overwrite:
        print('Generating vocabulary...')
        smi_list = starting_df['smiles'].tolist()
        voc_chars = construct_vocabulary(smi_list, path=voc_path)
    else:
        print(f'{voc_path} found. No need to generated vocabulary.')


    # train a prior and save (if required)
    # will save a ckpt file for pytorch
    prior_path = '../data/Prior_stereo.ckpt' if FLAGS.stereo else '../data/Prior_nonstereo.ckpt'
    if not os.path.exists(prior_path) or FLAGS.overwrite:
        pretrain_dict = {
            'num_epochs': 500,
            'train_ratio': 0.85,
            'verbose': True,
            'stereo': FLAGS.stereo,
            'starting_df': starting_df,
        }
        print('Start pretraining...')
        pretrain(**pretrain_dict)
        print('Petraining done')
    else:
        print(f'{prior_path} found. No pretraining needed.')

    arg_dict = vars(FLAGS)
    arg_dict.update(
        {
            'scoring_function': tar_func, 
            'restore_prior_from': prior_path,
            'restore_agent_from': prior_path,
            'normalize_score': norm_func,
            'voc_path': voc_path,
            'starting_df': starting_df,
            'starting_size': FLAGS.starting_size,
        }
    )
    train_agent(**arg_dict)



