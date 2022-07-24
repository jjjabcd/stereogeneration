import os, sys
import csv
import time
import tempfile, inspect

from stereogeneration import JANUS
from stereogeneration.utils import sanitize_smiles
from stereogeneration.filter import passes_filter

import selfies as sf
import selfies
import numpy as np
import pandas as pd

import rdkit.Chem as Chem
from rdkit.Chem import Draw
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
from morfeus.conformer import ConformerEnsemble

import multiprocessing
from functools import partial
import subprocess
from argparse import ArgumentParser

def fitness_function(smi: str, target: str = '4LDE'):
    cwd = os.getcwd()
    tmp_dir = tempfile.TemporaryDirectory(dir='/tmp')
    os.chdir(tmp_dir.name)

    smina_path = os.path.join(cwd, 'docking')
    target_path = os.path.join(smina_path, target)

    # check the cache
    if os.path.isfile(os.path.join(target_path, 'cache.csv')):
        with open(os.path.join(target_path, 'cache.csv'), 'r') as f:
            reader = csv.reader(f)
            cache = {rows[0]: float(rows[1]) for rows in reader}
    else:
        cache = []

    # name = re.sub(r'[^\w]', '', smi)
    name = 'mol'
    if smi in cache:
        score = cache[smi]
    else:
        t0 = time.time()

        # do conformer search using morfeus
        try:
            ensemble_p = ConformerEnsemble.from_rdkit(smi, optimize="MMFF94", random_seed=25)
            ensemble_p.prune_rmsd()
            ensemble_p.sort()   
            ensemble_p[0:1].write_xyz(f'{name}.xyz')
            _ = subprocess.run(f'obabel -ixyz {str(name)}.xyz -O {name}.pdb --best', shell=True, capture_output=True)
        except:
            # generate ligand files directly from smiles using openbabel
            print('Default to openbabel embedding...')
            _ = subprocess.run(f'obabel -:"{smi}" --gen3d -h -O {name}.pdb --best', shell=True, capture_output=True)

        try:
            # run docking procedure
            output = subprocess.run(f"{smina_path}/smina.static -r {target_path}/receptor.pdb -l {name}.pdb --autobox_ligand \
                {target_path}/ligand.pdb --autobox_add 5 --exhaustiveness 16 --seed 42",
                shell=True, capture_output=True)

            if output.returncode != 0:
                # print('Job failed.')
                score = -999.0
            else:
                # extract result from output
                found = False
                for s in output.stdout.decode('utf-8').split():
                    if found:
                        # print(f'{s} kcal/mol')
                        break
                    if s == '1':
                        found = True

                score = -float(s)
        except:
            score = -1000.0
        
        # write to file
        with open(os.path.join(target_path, 'cache.csv'), 'a') as f:
            f.write(f'{smi},{score},{time.time() - t0}\n')

    with open(os.path.join(cwd, 'OUT_ALL.csv'), 'a') as f:
        f.write(f'{smi},{score}\n')

    os.chdir(cwd)
    tmp_dir.cleanup()
        
    return score



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--target", action="store", type=str, default="4LDE", help="Protein target, defaults 4LDE.")
    parser.add_argument("--num_workers", action="store", type=int, default=1, help="Number of workers, defaults 1.")
    parser.add_argument("--stereo", action="store_true", dest="stereo", help="Toggle stereogeneration, defaults false.", default=False)
    parser.add_argument("--classifier", action="store_true", dest="use_classifier", help="Toggle classifier, defaults false", default=False)

    FLAGS = parser.parse_args()
    assert FLAGS.target in ['4LDE', '1OYT', '1SYH'], 'Invalid protein target'

    stereo = FLAGS.stereo
    print(f'Stereoisomers? : {stereo}')

    # load isoalphabet from zinc dataset
    fname = 'data/isoalphabet.npz' if stereo else 'data/alphabet.npz'
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
    default_constraints = selfies.get_semantic_constraints()
    new_constraints = default_constraints
    new_constraints['S'] = 2
    new_constraints['P'] = 3
    selfies.set_semantic_constraints(new_constraints)  # update constraints

    # get initial fitnesses from csv
    df = pd.read_csv(f'data/{FLAGS.target}/starting_smiles.csv')
    init_fitness = df['fitness'].tolist()
    
    # function with specified target
    tar_func = partial(fitness_function, target=FLAGS.target)
    tar_func.__name__ = f'{FLAGS.target}_score'

    fname = f'data/starting_smiles.txt' if stereo else f'data/starting_smiles_noniso.txt'
    output_dir = 'RESULTS_stereo' if stereo else 'RESULTS_nonstereo'
    agent = JANUS(
        work_dir=output_dir,
        fitness_function = tar_func,
        start_population = fname,
        starting_fitness = init_fitness,
        **params_dict
    )

    agent.run()

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

    



