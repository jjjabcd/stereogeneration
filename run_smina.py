import multiprocessing
import sys, os
from main import fitness_function
import pandas as pd
from functools import partial
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--file", action="store", type=str, help="Starting smiles", required=True)
parser.add_argument("--target", action="store", type=str, default="4LDE", help="Protein target, defaults 4LDE.")
parser.add_argument("--num_workers", action="store", type=int, default=1, help="Number of workers, defaults 1.")
FLAGS = parser.parse_args()

with open(FLAGS.file) as f:
    smiles = f.readlines()
    smiles = [smi.strip() for smi in smiles]
    with multiprocessing.Pool(FLAGS.num_workers) as pool:
        results = pool.map(
            partial(fitness_function, target=FLAGS.target), smiles)
pd.DataFrame({'smiles': smiles, 'fitness': results}).to_csv(
    os.path.join('data', FLAGS.target, 'starting_smiles.csv'), index=False)

