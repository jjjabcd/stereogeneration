import os, csv, time, tempfile
import numpy as np
import inspect

from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions,GetStereoisomerCount
from morfeus.conformer import ConformerEnsemble
from pebble import concurrent
from concurrent.futures import TimeoutError
from .utils import assign_stereo

import subprocess

@concurrent.thread(timeout=600)
def stda(smi):
    mol = Chem.MolFromSmiles(smi)
    enantiomers = Chem.FindMolChiralCenters(mol,includeUnassigned=True, useLegacyImplementation=False)

    # do not perform CD calculation if there are no stereocentres
    if len(enantiomers) == 0:
        return 0.
    
    # start simulation
    cwd = os.getcwd()
    tmp_dir = tempfile.TemporaryDirectory(dir='/tmp')
    os.chdir(tmp_dir.name)
    
    # assign stereo
    smi = assign_stereo(smi, {})

    try:
        # random seed required, since stda algorithm is very sensitive to conformer
        ce = ConformerEnsemble.from_rdkit(smi, optimize="MMFF94", random_seed=0)
        ce.prune_rmsd()
        ce.sort()

        ce[0:1].write_xyz('conformer.xyz')
        subprocess.run(f'xtb conformer.xyz --opt > xtb_logs 2>&1', shell=True,timeout=600)
        subprocess.run(f'xtb4stda xtbopt.xyz > xtb4stda_logs 2>&1', shell=True, check=True)
        subprocess.run(f'stda_v1.6.3 -xtb wfn.xtb -e 10 > stda_logs 2>&1', shell=True)
        subprocess.run(f'g_spec < tda.dat > gspec_log 2>&1', shell=True, check=True)
        subprocess.run("sed 's/\([0-9]\)-\([0-9]\)/\\1E-\\2/g' spec.dat > spec_fix.dat", shell=True, check=True)
        spec = np.loadtxt("spec_fix.dat")
        score = extract_score(spec)
        
        # results from conformers with energies similar after rounding will be averaged
        # legacy
        if len(ce) > 1:
            if round(ce.get_energies()[0],5) == round(ce.get_energies()[1],5):
                # do the same for second conformer
                os.mkdir('conf2')
                os.chdir('conf2')
                ce[1:2].write_xyz('conformer.xyz')
                subprocess.run(f'xtb conformer.xyz --opt --squick -gfn2//gfnff > xtb_logs 2>&1', shell=True,timeout=600)
                subprocess.run(f'xtb4stda xtbopt.xyz > xtb4stda_logs 2>&1', shell=True, check=True)
                subprocess.run(f'stda_v1.6.3 -xtb wfn.xtb -e 10 > stda_logs 2>&1', shell=True)
                subprocess.run(f'g_spec < tda.dat > gspec_log 2>&1', shell=True, check=True)
                subprocess.run("sed 's/\([0-9]\)-\([0-9]\)/\\1E-\\2/g' spec.dat > spec_fix.dat", shell=True, check=True)
                spec = np.loadtxt("spec_fix.dat")
                score2 = extract_score(spec)
                
                # final averaged score
                score += score2
                score /= 2.0

    except subprocess.CalledProcessError as error:
            score = -1000.0
    except ValueError as error:
            score = -999.0
    
    os.chdir(cwd)
    tmp_dir.cleanup()

    return score

def extract_score(spec_dat, wavelength_range=[450, 550], normalize=False):
    # function will extract the score for a wavelength range in the spectral data
    # the score is normalize (if true) by the total positive area under curve
    # thus the score will range from -1 to +1
    x = spec_dat[:, 0][::-1]
    y = spec_dat[:, 1][::-1]
    indices = (x > wavelength_range[0]) & (x < wavelength_range[1])
    auc = np.trapz(y[indices], x=x[indices])

    if normalize:
        norm = np.trapz(np.abs(y), x=x)     # absolute norm 
        return auc/norm
    else:
        return auc
    
def fitness_function(smi):
    #print('starting fitness for\t',smi)
    future = stda(smi)
    try:
        results = future.result()
        return results
    except TimeoutError as error:
        #print(f'done {smi} took too long')
        return -900.0
    except Exception as error:
        #print(f'{smi} other error')
        #print(error.traceback)  # Python's traceback of remote process
        return -909.0

import multiprocessing, pickle

if __name__ =='__main__':
    # calculate the properties
    with open('../data/mols_filtered_stereo.smi','r') as f:
        smiles = f.readlines()
        smiles = [i.strip() for i in smiles]
    with multiprocessing.Pool(64) as pool:
        fitness = pool.map(fitness_function, smiles)
    with open('fitness_stereo','wb') as fp:
        pickle.dump(fitness,fp)