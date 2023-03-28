import os, sys
sys.path.append('..')

import csv
import time
import tempfile, inspect

from stereogeneration import JANUS, docking
from stereogeneration.utils import sanitize_smiles
from stereogeneration.filter import passes_filter

import selfies as sf
import selfies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.covariance import EllipticEnvelope

import rdkit.Chem as Chem
from rdkit.Chem import Draw
# from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
# from morfeus.conformer import ConformerEnsemble

import multiprocessing
from functools import partial
import subprocess
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--target", action="store", type=str, default="1OYT", help="Protein target, defaults 1OYT.")
    parser.add_argument("--num_workers", action="store", type=int, default=1, help="Number of workers, defaults 1.")
    parser.add_argument("--stereo", action="store_true", dest="stereo", help="Toggle stereogeneration, defaults false.", default=False)
    parser.add_argument("--classifier", action="store_true", dest="use_classifier", help="Toggle classifier, defaults false", default=False)
    parser.add_argument("--starting_pop", action="store", type=str, default="worst", help="Method to select starting population: random, worst, best.")
    parser.add_argument("--starting_size", action="store", type=int, default=5000, help="Number of starting smiles, must be larger than pop size")

    FLAGS = parser.parse_args()
    assert FLAGS.target in ['1OYT', '1SYH', '6Y2F'], 'Invalid protein target'
    
    fitness_function = docking.fitness_function
    stereo = FLAGS.stereo
    print(f'Stereoisomers? : {stereo}')

    # load isoalphabet from zinc dataset
    fname = '../data/isoalphabet.npz' if stereo else '../data/alphabet.npz'
    alphabet = np.load(fname, allow_pickle=True)['alphabet'].tolist()
    print(f'Alphabet contains: {len(alphabet)}')
    
    alphabet = list(set(alphabet))
    # print(f'Alphabet with required characters contains: {len(alphabet)}')

    # all parameters to be set, below are defaults
    params_dict = {
        # Number of iterations that JANUS runs for
        "generations": 50,

        # The number of molecules for which fitness calculations are done, 
        # exploration and exploitation each have their own population
        "generation_size": 100,
        
        # Number of molecules that are exchanged between the exploration and exploitation
        "num_exchanges": 10,

        # Callable filtering function (None defaults to no filtering)
        "custom_filter": passes_filter,

        # Fragments from starting population used to extend alphabet for mutations
        "use_fragments": True,

        # An option to use a classifier as selection bias
        "use_classifier": FLAGS.use_classifier,

        "num_workers": FLAGS.num_workers,

        # alphabet
        "alphabet": alphabet,

        "verbose_out": True,

        "num_sample_frags": 100,

        "exploit_num_random_samples": 400,
        "exploit_num_mutations": 400,

        "top_mols": 5,

        "explr_num_random_samples": 10,
        "explr_num_mutations": 10,
        "crossover_num_random_samples": 5,

        'stereo': stereo,

    }

    # Set your SELFIES constraints (below used for manuscript)
    # default_constraints = selfies.get_semantic_constraints()
    # new_constraints = default_constraints
    # new_constraints['S'] = 2
    # new_constraints['P'] = 3
    # selfies.set_semantic_constraints(new_constraints)  # update constraints

    # get initial fitnesses from csv
    df = pd.read_csv('../data/zinc.csv')
    df = df[['isosmiles', FLAGS.target]].rename(
        columns={'isosmiles':'smiles', FLAGS.target: 'fitness'}
    )

    # remove failed jobs
    df = df[df['fitness'] > -900.0] 

    # keep only molecules that pass the filter
    # df = df[df['smiles'].apply(passes_filter)]

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

    # remove stereo information if stereo set to False
    if not stereo:
        start_df['smiles'] = start_df['smiles'].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x), canonical=True, isomericSmiles=False))
    start_df = start_df.drop_duplicates('smiles')
    init_fitness = start_df['fitness'].tolist()

    # write the smiles file, this will be read by JANUS
    fname = f'{FLAGS.target}_starting_smiles.txt'
    with open(fname, 'w') as f:
        for smi in start_df['smiles']:
            f.write(smi+'\n')

    # function with specified target
    tar_func = partial(fitness_function, target=FLAGS.target)
    tar_func.__name__ = f'{FLAGS.target}_score'

    output_dir = 'RESULTS_stereo' if stereo else 'RESULTS_nonstereo'
    agent = JANUS(
        work_dir=output_dir,
        fitness_function = tar_func,
        start_population = fname,
        starting_fitness = init_fitness,
        **params_dict
    )

    agent.run()

    ### TESTING CODE ###

    # FITNESS
    # data = pd.read_csv('data/starting_smiles.csv')
    # import pdb;  pdb.set_trace()
    
    # with open('data/starting_smiles.txt', 'w') as f:
    #     for smi in data['smiles']:
    #         f.write(smi+'\n')

    # init_smiles = []
    # with open('data/starting_smiles.txt', 'r') as f:
    #     for line in f:
    #         line = sanitize_smiles(line.strip())
    #         if line is not None:
    #             init_smiles.append(line)
    # init_smiles = list(set(init_smiles))

    # with multiprocessing.Pool(64) as pool:
    #     results = pool.map(fitness_function, init_smiles)

    # df = pd.DataFrame({'smiles': init_smiles, 'fitness': results})
    # df.to_csv('starting_smiles.csv', index=False)

    # TESTING ISOMERS
    # from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers

    # start_smi = Chem.MolToSmiles(Chem.MolFromSmiles('CCOC(=O)[C@@H]1CCCN(C(=O)c2nc(-c3ccc(C)cc3)n3c2CCCCC3)C1'), isomericSmiles=False)
    # mol = Chem.MolFromSmiles(start_smi)
    # isomers = list(EnumerateStereoisomers(mol))
    # import pdb;  pdb.set_trace()
    # f1 = fitness_function(Chem.MolToSmiles(isomers[0], isomericSmiles=True))
    # f2 = fitness_function(Chem.MolToSmiles(isomers[1], isomericSmiles=True))

    # print(f1)
    # print(f2)

    # img = Draw.MolsToGridImage(isomers[0:2], legends=(str(f1), str(f2)))
    # img.save('isomers.png')

    # f = fitness_function('CCOC(=O)[C@@H]1CCCN(C(=O)c2nc(-c3ccc(C)cc3)n3c2CCCCC3)C1')
    # print(f)
    # import pdb; pdb.set_trace()

    # run code
    # zinc_df = pd.read_csv('zinc.csv')

    # smi_list = zinc_df['isosmiles'].tolist()
    # import pdb; pdb.set_trace()
    # print('running...')
    # score = fitness_function(smi_list[0])

    # import pdb; pdb.set_trace()
    

    # with multiprocessing.Pool(64) as pool:
    #     results = pool.map(fitness_function, smi_list)

    # np.savez('scores.npz', scores=results)

    # zinc_df['docking_score'] = results
    # zinc_df.to_csv('zinc_scores.csv', index=False)

    
    # some preprocessing
    # zinc_df['smiles'] = zinc_df['smiles'].apply(sanitize_smiles)
    # zinc_df = zinc_df.drop_duplicates().dropna()
            
    # # zinc_df['smiles'] = zinc_df['smiles'].apply()
    # zinc_df['selfies'] = zinc_df['smiles'].apply(sf.encoder)

    # isosmiles = []
    # is_iso = []
    # for i, r in zinc_df.iterrows():
    #     print(i)
    #     smi, iso = assign_stereo(r.smiles)
    #     isosmiles.append(smi)
    #     is_iso.append(iso)

    # zinc_df['isosmiles'] = isosmiles
    # zinc_df['is_iso'] = is_iso

    # zinc_df['isoselfies'] = zinc_df.isosmiles.apply(sf.encoder)
    # alphabet = sf.get_alphabet_from_selfies(zinc_df.isoselfies)


    # import pdb; pdb.set_trace()

    
    ### Plot outliers removal

    # df = pd.read_csv(f'data/{FLAGS.target}/starting_smiles.csv')

    # df = df[df['fitness'] > -900.0] 
    # _, bins, _ = plt.hist(df['fitness'].to_numpy(), bins=50, range=[-25, 10])
    # plt.close()

    # fig, ax = plt.subplots(2, 2, figsize=(15,15), sharex=True)
    # ax = ax.flatten()
    # # # remove failed jobs and outliers
    # ax[0].hist(df['fitness'].to_numpy(), bins=bins, density=True, label='original')
    # ax[0].legend()

    # mu, std = df['fitness'].mean(), df['fitness'].std(axis=0)
    # new_df = df[df['fitness'] > mu - 3.*std]
    # new_df = new_df[new_df['fitness'] < mu + 3.*std]
    # ax[1].hist(new_df['fitness'].to_numpy(), bins=bins, density=True, label=r'$\mu \pm 3\sigma$')
    # ax[1].legend()

    # q1 = np.quantile(df['fitness'], q = 0.25)
    # q3 = np.quantile(df['fitness'], q = 0.75)
    # iqr = q3 - q1
    # new_df = df[df['fitness'] > q1 - 1.5*iqr]
    # new_df = new_df[new_df['fitness'] < q3 + 1.5*iqr]
    # ax[2].hist(new_df['fitness'].to_numpy(), bins=bins, density=True, label=r'IQR')
    # ax[2].legend()

    # keep = EllipticEnvelope().fit_predict(df[['fitness']].to_numpy())
    # ax[3].hist(df[keep == 1]['fitness'].to_numpy(), bins=bins, density=True, label=r'Elliptic envelope')
    # ax[3].legend()

    # plt.savefig(f'data/{FLAGS.target}/histogram.png')
    # import pdb; pdb.set_trace()

    # keep = EllipticEnvelope().fit_predict(df[['fitness']].to_numpy())
    # df = df[keep==1]
    # mu, std = df['fitness'].mean(), df['fitness'].std()
    # df = df[df['fitness'] <= mu + 3*std]
    # df = df[df['fitness'] >= mu - 3*std]


