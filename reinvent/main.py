#!/usr/bin/env python
import os, sys
sys.path.append('..')

# reinvent specific imports
from stereogeneration.reinvent.train_agent import train_agent
from stereogeneration.reinvent.train_prior import pretrain
from stereogeneration.reinvent.data_structs import construct_vocabulary

from stereogeneration import utils, docking
from rdkit.Chem import Crippen

import numpy as np
import pandas as pd

import rdkit.Chem as Chem

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
                    default=100)
parser.add_argument('--sigma', action='store', dest='sigma', type=int,
                    default=20)
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
parser.add_argument("--stereo", action="store_true", dest="stereo", 
                    help="Toggle stereogeneration, defaults false.", default=False)
parser.add_argument("--overwrite", action="store_true", dest="overwrite", 
                    help="Toggle processing/pretraining even if files exist, defaults false.", 
                    default=False)


def debug_fitness(smi):
    # logP fitness for quick evaluation and debugging :)
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return -999.0
    else:
        score = Crippen.MolLogP(m)
        return score

if __name__ == "__main__":

    FLAGS = parser.parse_args()
    assert FLAGS.target in ['1OYT', '1SYH', '6Y2F'], 'Invalid protein target'

    # read the dataset
    df = pd.read_csv(f'../data/{FLAGS.target}/starting_smiles.csv')
    print(f'Stereoinformation? : {FLAGS.stereo}')

    # write file of smiles (if required)
    # will save a mols_filtered smiles file
    fname = '../data/mols_filtered_stereo.smi' if FLAGS.stereo else '../data/mols_filtered_nonstereo.smi'
    if not os.path.exists(fname) or FLAGS.overwrite:
        print('Generating processed smiles file...')
        if not FLAGS.stereo:
            # remove stereo information
            df['smiles'] = df['smiles'].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x), isomericSmiles=False, canonical=True))
            df = df.drop_duplicates('smiles')
        with open(fname, 'w') as f:
            for smi in df['smiles']:
                f.write(smi+'\n')
    else:
        print(f'{fname} found. No need to generated vocabulary.')

    # process smiles (if required)
    # will save a Vocabulary object
    voc_path = '../data/Voc_stereo' if FLAGS.stereo else '../data/Voc_nonstereo'
    if not os.path.exists(voc_path) or FLAGS.overwrite:
        print('Generating vocabulary...')
        smi_list = df['smiles'].tolist()    
        voc_chars = construct_vocabulary(smi_list, path=voc_path)
    else:
        print(f'{voc_path} found. No need to generated vocabulary.')


    # train a prior and save (if required)
    # will save a ckpt file for pytorch
    prior_path = '../data/Prior_stereo.ckpt' if FLAGS.stereo else '../data/Prior_nonstereo.ckpt'
    if not os.path.exists(prior_path) or FLAGS.overwrite:
        pretrain_dict = {
            'num_epochs': 100,
            'train_ratio': 0.8,
            'verbose': True,
            'stereo': FLAGS.stereo,
        }
        print('Start pretraining...')
        pretrain(**pretrain_dict)
        print('Petraining done')
    else:
        print(f'{prior_path} found. No pretraining needed.')
    
    # train the agent
    arg_dict = vars(FLAGS)
    arg_dict.update(
        {
            'scoring_function': partial(docking.fitness_function, target=FLAGS.target), # debug_fitness
            'restore_prior_from': prior_path,
            'restore_agent_from': prior_path,
            'voc_path': voc_path,
        }
    )
    train_agent(**arg_dict)



