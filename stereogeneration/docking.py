import os, sys
import csv
import time
import tempfile, inspect

import selfies as sf
import selfies
import numpy as np
import pandas as pd

import rdkit.Chem as Chem
from morfeus.conformer import ConformerEnsemble

import multiprocessing
from functools import partial
import subprocess
from argparse import ArgumentParser

def fitness_function(smi: str, target: str = '1SYH', seed: int =30624700):
    ''' Docking score (maximize) for given target protein. Scored my SMINA.
    Select from available protein targets: 1OYT, 1SYH, 4LDE, and 6Y2F
    '''

    cwd = os.getcwd()
    tmp_dir = tempfile.TemporaryDirectory(dir='/tmp')
    os.chdir(tmp_dir.name)

    # smina_path = os.path.join(cwd, 'docking')
    smina_path = os.path.join(os.path.dirname(inspect.getfile(fitness_function)), 'docking')
    target_path = os.path.join(smina_path, target)

    name = 'mol'
    t0 = time.time()

    # do conformer search using morfeus
    try:
        ensemble_p = ConformerEnsemble.from_rdkit(smi, optimize="MMFF94", random_seed=seed)
        ensemble_p.prune_rmsd()
        ensemble_p.sort()   
        ensemble_p[0:1].write_xyz(f'{name}.xyz')
        _ = subprocess.run(f'obabel -ixyz {str(name)}.xyz -O {name}.pdb --best', shell=True, capture_output=True)
    except:
        # generate ligand files directly from smiles using openbabel
        print(f'Default to openbabel embedding... for : {smi}')
        _ = subprocess.run(f'obabel -:"{smi}" --gen3d -h -O {name}.pdb --best', shell=True, capture_output=True)

    try:
        # run docking procedure
        output = subprocess.run(f"{smina_path}/smina.static -r {target_path}/receptor.pdb -l {name}.pdb --autobox_ligand \
            {target_path}/ligand.pdb --autobox_add 3 --exhaustiveness 16 --seed {seed}",
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
        
    with open(os.path.join(cwd, 'OUT_ALL.csv'), 'a') as f:
        f.write(f'{smi},{score},{time.time() - t0}\n')

    os.chdir(cwd)
    tmp_dir.cleanup()
        
    return score
