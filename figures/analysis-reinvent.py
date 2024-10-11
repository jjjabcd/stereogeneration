import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import rdkit.Chem as Chem
from rdkit import RDLogger
from rdkit.Chem import Draw
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

from argparse import ArgumentParser
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')

RUN_TYPES = ['stereo', 'nonstereo']
CMAP = {n: sns.color_palette()[i]  for i, n in enumerate(RUN_TYPES)}

def has_stereo(smi):
    return len(Chem.FindMolChiralCenters(Chem.MolFromSmiles(smi))) > 0

def is_stereo(smi):
    """ Return boolean of whether the smiles is chiral or not.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False

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
    parser.add_argument("--target", action="store", type=str, default="1OYT", help="Protein target, defaults 1OYT.")
    
    FLAGS = parser.parse_args()
    # assert FLAGS.target in ['6Y2F', '1OYT', '1SYH'], 'Invalid protein target'

    # results = {'generation': [], 'fitness': [], 'run_type': [], 'smiles': []}
    results = []
    results_per_gen = []

    df = pd.read_csv('zinc.csv')
    best_in_dataset = df.nlargest(1, FLAGS.target)
    
    # search through result files
    fnames = glob.glob(os.path.join('.', '*stereo/reinvent/RESULTS/*/results.csv'))
    for fname in tqdm(fnames):
        # key = os.path.dirname(fname)
        key = fname
        run = key.split('_')[0].split('/')[-1]
        # run_type = 'stereo' if 'RESULTS_stereo' in key else 'nonstereo'
        run_type = 'nonstereo' if 'nonstereo' in key else 'stereo'
        # use_classifier = 'JANUS+C' if prefix == 'classifier_stereogeneration' else 'JANUS'

        df = pd.read_csv(fname)
        df = df[df['fitness'] > -900.0]
        df['run_type'] = [run_type]*len(df)
        results.append(df)

        new_df = {'generation': [], 'avg_fitness': [], 'best_fitness': [],  'run_type': [], 'run': []} #, 'is_stereo_percent': []}
        for gen, gdf in df.groupby('generation'):
            gdf = gdf[gdf['fitness'] > -200.0]
            new_df['run'].append(run)
            new_df['generation'].append(int(gen))
            new_df['avg_fitness'].append(gdf['fitness'].mean())
            new_df['run_type'].append(run_type)
            # frac = gdf['smiles'].apply(is_stereo).astype(int).sum() / len(gdf)
            # new_df['is_stereo_percent'].append(frac)

            # best traces
            if gen == 0:
                mem = gdf['fitness'].max()
            else:
                if gdf['fitness'].max() > mem:
                    mem = gdf['fitness'].max()
            new_df['best_fitness'].append(mem)
        new_df = pd.DataFrame(new_df)
        new_df['generation'] += 1
        new_df.index += 1
        new_df.iloc[0] = [0, np.nan, best_in_dataset[FLAGS.target].values[0], run_type, run]
        results_per_gen.append(new_df)

    # plot the best molecules over ALL runs
    n_plot = 5
    results = pd.concat(results)
    results['generation'] = results['generation'].astype(int)

    df_best = results.drop_duplicates('smiles').sort_values('fitness', ascending=False).groupby('run_type', as_index=False).head(n_plot)
    df_best['mols'] = df_best['smiles'].apply(Chem.MolFromSmiles)
    df_best['is_stereo'] = df_best['smiles'].apply(is_stereo)
    df_best = df_best.sort_values('run_type')
    labels = [
        r['run_type']+'  '+str(r['is_stereo'])+'\n'+str(r['fitness'])
        for i, r in df_best.iterrows()
    ]
    img = Draw.MolsToGridImage(df_best['mols'].tolist(), molsPerRow=n_plot, subImgSize=(300,300), legends=labels)
    img.save(f'reinvent_best.png')

    # plot the traces
    results_per_gen = pd.concat(results_per_gen).reset_index()
    results_per_gen = results_per_gen.rename(columns={'generation': 'Generation', 
        'best_fitness': f'{FLAGS.target} Score', 'run_type': 'Run type', 
        'avg_fitness': f'Average {FLAGS.target} score in population',})
        # 'is_stereo_percent': 'Percent of stereosmiles'})
    sns.lineplot(data=results_per_gen, x='Generation', y=f'{FLAGS.target} Score', hue='Run type', palette=CMAP)
    plt.savefig('reinvent_traces.png', bbox_inches='tight')
    plt.close()

    # sns.lineplot(data=results_per_gen, x='Generation', y=f'Average {FLAGS.target} score in population', hue='Run type', palette=CMAP)
    # plt.savefig('trace_average.png', bbox_inches='tight')
    # plt.close()

    # sns.lineplot(data=results_per_gen, x='Generation', y='Percent of stereosmiles')
    # plt.savefig('generation_stereo.png', bbox_inches='tight')
    # plt.close()

    




