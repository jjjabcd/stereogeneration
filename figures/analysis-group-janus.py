import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import rdkit.Chem as Chem
from rdkit.Chem import Draw
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

from argparse import ArgumentParser
from tqdm import tqdm

RUN_TYPES = ['stereo', 'nonstereo']
CMAP = {n: sns.color_palette()[i]  for i, n in enumerate(RUN_TYPES)}

def has_stereo(smi):
    return len(Chem.FindMolChiralCenters(Chem.MolFromSmiles(smi))) > 0

def is_stereo(smi):
    """ Return boolean of whether the smiles is chiral or not.
    """
    mol = Chem.MolFromSmiles(smi)

    if len(Chem.FindMolChiralCenters(mol)) == 0:
        isomers = list(EnumerateStereoisomers(mol))
        if len(isomers) > 1:
            return True
        else:
            return False
    else:
        return True

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--target", action="store", type=str, default="1SYH")
    parser.add_argument("--property_name", action="store", type=str, default="1SYH docking score", help="Protein target, defaults 1OYT. \
        Is just important for the axis names.")
    
    FLAGS = parser.parse_args()

    df_top1 = []
    df_explt = []
    df_explr = []

    df = pd.read_csv('zinc.csv')
    best_in_dataset = df.nlargest(1, FLAGS.target)
    
    # load results
    fnames = glob.glob(os.path.join('.', '*stereo/group-janus/RESULTS_*/'))
    for fname in tqdm(fnames):
        # key = os.path.dirname(fname)
        run_type = 'stereo' if 'RESULTS_stereo' in fname else 'nonstereo'
        # use_classifier = 'JANUS+C' if prefix == 'classifier_stereogeneration' else 'JANUS'
        run = fname.split('_')[0].split('/')[-1]
        
        df = pd.read_csv(fname + 'generation_all_best.csv')
        df['run_type'] = run_type
        df['run'] = run
        df['generation'] += 1
        df.index += 1
        df.iloc[0] = [0, best_in_dataset['smiles'].values[0], best_in_dataset[FLAGS.target].values[0], run_type, run]
        df_top1.append(df)

        df = pd.read_csv(fname + 'exploitation_results.csv')
        df['run_type'] = run_type
        df['run'] = run
        df_explt.append(df)

        df = pd.read_csv(fname + 'exploration_results.csv')
        df['run_type'] = run_type
        df['run'] = run
        df_explr.append(df)
    
    df_top1 = pd.concat(df_top1, ignore_index=True).sort_values(['run', 'generation'])
    df_explt = pd.concat(df_explt, ignore_index=True).sort_values(['run', 'generation'])
    df_explr = pd.concat(df_explr, ignore_index=True).sort_values(['run', 'generation'])
    # results = pd.concat([df_explt, df_explr]).sort_values(['run', 'generation'])


    ### Performing analysis
    # plot the best molecules over ALL runs
    n_plot = 5
    df_best = df_top1.drop_duplicates('smiles').sort_values('fitness', ascending=False).groupby('run_type', as_index=False).head(n_plot)
    df_best['mols'] = df_best['smiles'].apply(Chem.MolFromSmiles)
    df_best['is_stereo'] = df_best['smiles'].apply(is_stereo)
    df_best = df_best.sort_values('run_type')
    labels = [
        r['run_type']+'  '+str(r['is_stereo'])+'\n'+str(r['fitness'])
        for i, r in df_best.iterrows()
    ]
    img = Draw.MolsToGridImage(df_best['mols'].tolist(), molsPerRow=n_plot, subImgSize=(300,300), legends=labels)
    img.save(f'gjanus_best.png')

    # plot the traces
    results = df_top1.rename(columns={'generation': 'Generation', 'fitness': f'{FLAGS.target} Score', 'run_type': 'Run type'})
    sns.lineplot(data=results, x='Generation', y=f'{FLAGS.target} Score', hue='Run type', palette=CMAP, hue_order=RUN_TYPES)
    plt.savefig('gjanus_traces.png', bbox_inches='tight')
    plt.close()




