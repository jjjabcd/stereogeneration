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
    parser.add_argument("--target", action="store", type=str, default="1OYT", help="Protein target, defaults 1OYT. \
        Is just important for the axis names.")
    
    FLAGS = parser.parse_args()
    assert FLAGS.target in ['6Y2F', '1OYT', '1SYH'], 'Invalid protein target'

    results = {'generation': [], 'fitness': [], 'run_type': [], 'smiles': []}
    
    # search through result files
    fnames = glob.glob(os.path.join('.', '*stereo/janus/*/generation_all_best.txt'))
    for fname in tqdm(fnames):
        key = os.path.dirname(fname)
        run_type = 'stereo' if 'RESULTS_stereo' in key else 'nonstereo'
        # use_classifier = 'JANUS+C' if prefix == 'classifier_stereogeneration' else 'JANUS'
        
        count = 0
        with open(fname, 'r') as f:
            for l in f:
                l = l.split()
                results['fitness'].append(float(l[-1]))
                results['generation'].append(count)
                results['run_type'].append(run_type)
                # results['model'].append(use_classifier)
                results['smiles'].append(l[1].strip(','))
                count += 1

    # plot the best molecules over ALL runs
    n_plot = 5
    results = pd.DataFrame(results)
    df_best = results.drop_duplicates('smiles').sort_values('fitness', ascending=False).groupby('run_type', as_index=False).head(n_plot)
    df_best['mols'] = df_best['smiles'].apply(Chem.MolFromSmiles)
    df_best['is_stereo'] = df_best['smiles'].apply(is_stereo)
    df_best = df_best.sort_values('run_type')
    labels = [
        r['run_type']+'  '+str(r['is_stereo'])+'\n'+str(r['fitness'])
        for i, r in df_best.iterrows()
    ]
    img = Draw.MolsToGridImage(df_best['mols'].tolist(), molsPerRow=n_plot, subImgSize=(300,300), legends=labels)
    img.save(f'{FLAGS.target}_best.png')

    # plot the traces
    results = results.rename(columns={'generation': 'Generation', 'fitness': f'{FLAGS.target} Docking Score', 'run_type': 'Run type'})
    sns.lineplot(data=results, x='Generation', y=f'{FLAGS.target} Docking Score', hue='Run type', palette=CMAP)
    plt.savefig('traces.png', bbox_inches='tight')
    plt.close()

    # population stuff    
    pop_results = {'generation': [], 'is_stereo_percent': [], 'avg_fitness': [], 'run_type': []}

    fnames = glob.glob(os.path.join('.', '*_*stereo/janus/RESULTS*/*_DATA/'))
    for fname in tqdm(fnames):
        smi_list, fit_list = [], []
        for ff in ['population_explore.txt', 'population_local_search.txt']:
            if os.path.exists(os.path.join(fname,ff)):
                with open(os.path.join(fname, ff), 'r') as f:
                    smi_list += f.read().split()
        for ff in ['fitness_explore.txt', 'fitness_local_search.txt']:
            if os.path.exists(os.path.join(fname,ff)):
                with open(os.path.join(fname, ff), 'r') as f:
                    fit_list += [float(i) for i in f.read().split()]

        if len(smi_list) == 0:
            continue
        count = 0
        for smi in smi_list:
            if has_stereo(smi) > 0:
                count += 1
        run_type = 'nonstereo' if 'nonstereo' in fname else 'stereo'

        gen = int(fname.split('/')[-2].split('_')[0])
        pop_results['generation'].append(gen)
        pop_results['is_stereo_percent'].append(float(count) / float(len(smi_list)))
        pop_results['avg_fitness'].append(np.mean(fit_list))
        pop_results['run_type'].append(run_type)
    
    # return generational
    pop_results = pd.DataFrame(pop_results)
    pop_results = pop_results.rename(columns={'generation': 'Generation', 'is_stereo_percent': 'Percent of sterosmiles',
        'avg_fitness': f'Average {FLAGS.target} score in population'})
    sns.lineplot(data=pop_results, x='Generation', y='Percent of sterosmiles')
    plt.savefig('generation_stereo.png', bbox_inches='tight')
    plt.close()

    sns.lineplot(data=pop_results, x='Generation', y=f'Average {FLAGS.target} score in population', hue='run_type', palette=CMAP)
    plt.savefig('trace_average.png')
    plt.close()


    




