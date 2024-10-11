import os, sys
sys.path.append('stereogeneration')

import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import rdkit.Chem as Chem
from rdkit.Chem import Draw
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

from stereogeneration.utils import remove_specified_chiral_centres

from argparse import ArgumentParser
from tqdm import tqdm
from scipy.stats import ttest_ind
from sklearn.metrics import auc
# import matplotlib as mpl

# sns.set_context('talk', font_scale=2)
# mpl.rcParams['lines.linewidth'] = 5
# mpl.rcParams['errorbar.capsize'] = 10

# define some helper functions
RUN_TYPES = ['stereo', 'nonstereo']
CMAP = {n: sns.color_palette()[i]  for i, n in enumerate(RUN_TYPES)}

def has_stereo(smi):
    mol = Chem.MolFromSmiles(smi)
    opt = StereoEnumerationOptions(unique=True, onlyUnassigned=False)
    isomers = list(EnumerateStereoisomers(mol, options=opt))
    isomers = remove_specified_chiral_centres(isomers)
    return len(isomers) > 1

def is_stereo(smi):
    """ Return boolean of whether the smiles is chiral or not.
    """
    return ('@' in smi) or ('/' in smi) or ('\\' in smi)
    
def load_janus_data(path='.', model='janus'):
    df_top1, df_explt, df_explr = [], [], []
    
    # load results
    fnames = glob.glob(os.path.join(path, f'*stereo/{model}/RESULTS_*/'))
    for fname in tqdm(fnames):
        # key = os.path.dirname(fname)
        run_type = 'stereo' if 'RESULTS_stereo' in fname else 'nonstereo'
        run = fname.split('_')[0].split('/')[-1]
        try:
            df = pd.read_csv(fname + 'generation_all_best.csv')
        except:
            continue
        df['run_type'] = run_type
        df['run'] = run
        df['generation'] += 1
        df.index += 1
        # df.iloc[0] = [0, best_in_dataset['smiles'].values[0], best_in_dataset[FLAGS.target].values[0], run_type, run]
        df_top1.append(df)

        df = pd.read_csv(fname + 'exploitation_results.csv')
        df['run_type'] = run_type
        df['run'] = run
        df['evaluation'] = list(range(1,len(df)+1))
        df_explt.append(df)

        df = pd.read_csv(fname + 'exploration_results.csv')
        df['run_type'] = run_type
        df['run'] = run
        df['evaluation'] = list(range(len(df)+1,2*len(df)+1))
        df_explr.append(df)
    
    df_top1 = pd.concat(df_top1, ignore_index=True).sort_values(['run', 'generation'])
    df_explt = pd.concat(df_explt, ignore_index=True).sort_values(['run', 'generation'])
    df_explr = pd.concat(df_explr, ignore_index=True).sort_values(['run', 'generation'])

    return df_top1, df_explt, df_explr

def load_reinvent_data(path='.'):
    results = []
    results_per_gen = []
    
    # search through result files
    fnames = glob.glob(os.path.join(path, '*stereo/reinvent/RESULTS/*/results.csv'))
    for fname in tqdm(fnames):
        key = fname
        run = key.split('_')[0].split('/')[-1]
        run_type = 'nonstereo' if 'nonstereo' in key else 'stereo'

        # this has the whole trace
        df = pd.read_csv(fname)
        # df = df[df['fitness'] > -900.0]
        df['run_type'] = [run_type]*len(df)
        df['top1'] = df['fitness'].cummax()
        df['evaluation'] = range(1,len(df)+1)
        df['run'] = int(run)
        results.append(df)

        new_df = {'generation': [], 'avg_fitness': [], 'fitness': [],  'run_type': [], 'run': []} #, 'is_stereo_percent': []}
        for gen, gdf in df.groupby('generation'):
            gdf = gdf[gdf['fitness'] > -200.0]
            new_df['run'].append(run)
            new_df['generation'].append(int(gen))
            new_df['avg_fitness'].append(gdf['fitness'].mean())
            new_df['run_type'].append(run_type)

            # best traces
            if gen == 0:
                mem = gdf['fitness'].max()
            else:
                if gdf['fitness'].max() > mem:
                    mem = gdf['fitness'].max()
            new_df['fitness'].append(mem)
        new_df = pd.DataFrame(new_df)
        new_df['generation'] += 1
        new_df.index += 1
        # new_df.iloc[0] = [0, np.nan, best_in_dataset[FLAGS.target].values[0], run_type, run]
        results_per_gen.append(new_df)

    results = pd.concat(results)
    results['generation'] = results['generation'].astype(int)
    results_per_gen = pd.concat(results_per_gen).reset_index()

    return results_per_gen, results

def plot_best_mols(*df_pops, n_plot=5):
    images = []
    for i, df in enumerate(df_pops):
        df_best = df.drop_duplicates('smiles').sort_values('fitness', ascending=False).groupby('run_type', as_index=False).head(n_plot)
        df_best['mols'] = df_best['smiles'].apply(Chem.MolFromSmiles)
        df_best['has_stereo'] = df_best['smiles'].apply(has_stereo)
        df_best = df_best.sort_values('run_type')
        labels = [
            f'{r["run_type"]} {str(r["has_stereo"])}\n{str(r["fitness"])}'
            for i, r in df_best.iterrows()
        ]
        img = Draw.MolsToGridImage(df_best['mols'].tolist(), molsPerRow=n_plot, subImgSize=(300,300), legends=labels)
        images.append(img)
    return images

def bootstrap_ci(data, num_bootstraps=100, alpha=0.05):
    bootstrapped_means = np.array([np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(num_bootstraps)])
    return np.percentile(bootstrapped_means, [alpha/2 * 100, (1 - alpha/2) * 100])

def pop_to_auroc():
    return


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--target", action="store", type=str, default="1SYH")
    parser.add_argument("--label", action="store", type=str, default="1SYH docking score", help="Y aix labels, defaults 1SYH.")
    
    FLAGS = parser.parse_args()

    df = pd.read_csv('zinc.csv')
    best_in_dataset = df.nlargest(1, FLAGS.target)


    # load data
    j_top1, j_explt, j_explr = load_janus_data()
    gj_top1, gj_explt, gj_explr = load_janus_data(model='group-janus')
    gjf_top1, gjf_explt, gjf_explr = load_janus_data(model='group-janus-fragments')
    r_top1, r_pop = load_reinvent_data()
    
    # append the populations of janus runs
    j_pop = pd.concat([j_explt, j_explr]).sort_values(['run', 'run_type', 'evaluation'])
    j_pop['top1'] = j_pop.groupby(['run', 'run_type']).cummax()['fitness']
    gj_pop = pd.concat([gj_explt, gj_explr]).sort_values(['run', 'run_type', 'evaluation'])
    gj_pop['top1'] = gj_pop.groupby(['run', 'run_type']).cummax()['fitness']
    gjf_pop = pd.concat([gjf_explt, gjf_explr]).sort_values(['run', 'run_type', 'evaluation'])
    gjf_pop['top1'] = gjf_pop.groupby(['run', 'run_type']).cummax()['fitness']


    ###### begin plotting #######

    ### Calculate AUROC print (perform t-test)
    with open('final_analysis.out', 'w') as f:
        f.write("#######\n")
        for i, df in zip(['reinvent', 'janus', 'group-janus', 'group-janus-fragments'], [r_pop, j_pop, gj_pop, gjf_pop]):
            df['evaluation'] /= max(df['evaluation'])
            df['top1'] -= best_in_dataset[FLAGS.target].values[0]
            auc_df = df.groupby(['run', 'run_type']).apply(lambda x: auc(x['evaluation'], x['top1'])).reset_index()
            auc_df = auc_df.rename(columns={0: 'auc'})
            s_df = auc_df[auc_df['run_type'] == 'stereo']
            ns_df = auc_df[auc_df['run_type'] == 'nonstereo']
            tstat, pval = ttest_ind(s_df['auc'].to_numpy(), ns_df['auc'].to_numpy())
            f.write(f"AUROC scores analysis {i}:\n")
            f.write(f"stereo {i}: {s_df['auc'].mean()} +- {s_df['auc'].std()}\n")
            f.write(f"nonstereo {i}: {ns_df['auc'].mean()} +- {ns_df['auc'].std()}\n")
            f.write(f'stereo/non-stereo t_test: {tstat:.3f}, pval: {pval:.3f}\n\n')

    # # plot the top1 per evaluation lineplot
    # fig, axes = plt.subplots(4,1, sharex=True, figsize=(5, 13))
    # axes = axes.flatten()
    # for i, (df, ax) in enumerate(zip([r_pop, j_pop, gj_pop, gjf_pop], axes)):
    #     grouped = df.groupby(['evaluation', 'run_type'])

    #     # custom confidnece interval (seaborn is too slow)
    #     results = []
    #     for (eval, run_type), group in tqdm(grouped):
    #         mean = group['top1'].mean()
    #         ci_low, ci_high = bootstrap_ci(group['top1'])
    #         results.append({'evaluations': eval, 'run_type': run_type, 'mean': mean, 'ci_low': ci_low, 'ci_high': ci_high})
    #     results_df = pd.DataFrame(results)

    #     for run_type in RUN_TYPES:
    #         data = results_df[results_df['run_type'] == run_type]
    #         ax.plot(data['evaluations'], data['mean'], label=run_type, c=CMAP[run_type])
    #         ax.fill_between(data['evaluations'], data['ci_low'], data['ci_high'], alpha=0.3, color=CMAP[run_type])
        
    #     # ax.hlines(best_in_dataset[FLAGS.target].values, min(df['evaluation']), max(df['evaluation']), color='k', linestyle='--')
    #     ax.set_xlim([min(df['evaluation']), max(df['evaluation'])])
    #     ax.set_xlabel('Evaluation')
    #     ax.set_ylabel(f'{FLAGS.label}')
    #     if i == 0:
    #         ax.legend()
    # fig.savefig('top1_traces_eval.png', bbox_inches='tight')



    # plot the top1 lineplot
    fig, axes = plt.subplots(4,1, sharex=True, figsize=(5, 13))
    axes = axes.flatten()
    with open('final_analysis.out', 'a') as f:
        f.write("#######\n")
        for i, (name, df, ax) in enumerate(zip(['reinvent', 'janus', 'group-janus', 'group-janus-fragments'], [r_top1, j_top1, gj_top1, gjf_top1], axes)):
            sns.lineplot(ax=ax, data=df, x='generation', y='fitness', hue='run_type', palette=CMAP, hue_order=RUN_TYPES)
            ax.hlines(best_in_dataset[FLAGS.target].values, min(df['generation']), max(df['generation']), color='k', linestyle='--')
            ax.set_xlim([min(df['generation']), max(df['generation'])])
            ax.set_xlabel('Generation')
            ax.set_ylabel(f'{FLAGS.label}')
            if i > 0:
                ax.get_legend().remove()
            
            ### print out the top1 max (perform t-test)
            gdf = df.groupby(['run', 'run_type'])
            max_df = gdf.max().reset_index()
            s_df = max_df[max_df['run_type'] == 'stereo']
            ns_df = max_df[max_df['run_type'] == 'nonstereo']
            tstat, pval = ttest_ind(s_df['fitness'].to_numpy(), ns_df['fitness'].to_numpy())
            f.write(f"Top1 scores analysis {name}:\n")
            f.write(f"stereo {i}: {s_df['fitness'].mean()} +- {s_df['fitness'].std()}\n")
            f.write(f"nonstereo {i}: {ns_df['fitness'].mean()} +- {ns_df['fitness'].std()}\n")
            f.write(f'stereo/non-stereo t_test: {tstat:.3f}, pval: {pval:.3f}\n\n')

    fig.savefig('top1_traces.png', bbox_inches='tight')


    # top10
    fig, axes = plt.subplots(4,1, sharex=True, figsize=(5, 13))
    axes = axes.flatten()
    with open('final_analysis.out', 'a') as f:
        f.write("#######\n")
        for i, (name, df, ax) in enumerate(zip(['reinvent', 'janus', 'group-janus', 'group-janus-fragments'], [r_pop, j_pop, gj_pop, gjf_pop], axes)):
            df = df.groupby(['run_type', 'generation', 'run']).apply(lambda x: x.nlargest(10, ['fitness']).mean())
            sns.lineplot(ax=ax, data=df, x='generation', y='fitness', hue='run_type', palette=CMAP, hue_order=RUN_TYPES)
            ax.hlines(best_in_dataset[FLAGS.target].values, min(df['generation']), max(df['generation']), color='k', linestyle='--')
            ax.set_xlim([min(df['generation']), max(df['generation'])])
            ax.set_xlabel('Generation')
            ax.set_ylabel(f'{FLAGS.label}')
            if i > 0:
                ax.get_legend().remove()

            ### print out the top10 max (perform t-test)
            df = df.reset_index(level=["run_type"]).reset_index(drop=True)
            gdf = df.groupby(['run', 'run_type'])
            stat_df = gdf.max().reset_index()
            s_df = stat_df[stat_df['run_type'] == 'stereo']
            ns_df = stat_df[stat_df['run_type'] == 'nonstereo']
            tstat, pval = ttest_ind(s_df['fitness'].to_numpy(), ns_df['fitness'].to_numpy())
            f.write(f"Top10 scores analysis {name}:\n")
            f.write(f"stereo {i}: {s_df['fitness'].mean()} +- {s_df['fitness'].std()}\n")
            f.write(f"nonstereo {i}: {ns_df['fitness'].mean()} +- {ns_df['fitness'].std()}\n")
            f.write(f'stereo/non-stereo t_test: {tstat:.3f}, pval: {pval:.3f}\n\n')

    fig.savefig('top10_traces.png', bbox_inches='tight')


    ### Plot the top 5 molecules
    r_img, j_img, gj_img, gjf_img = plot_best_mols(r_pop, j_pop, gj_pop, gjf_pop)
    r_img.save('mols_reinvent.png')
    j_img.save('mols_janus_mols.png')
    gj_img.save('mols_group-janus.png')
    gjf_img.save('mols_group-janus-fragments.png')


