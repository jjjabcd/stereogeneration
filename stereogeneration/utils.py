import selfies as sf
import yaml
import random 
import numpy as np
import rdkit.Chem as Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem import MolToSmiles as mol2smi
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

def get_selfies_chars(selfies):
    """Obtain a list of all selfie characters in string selfies
    
    Parameters: 
    selfie (string) : A selfie string - representing a molecule 
    
    Example: 
    >>> get_selfies_chars('[C][=C][C][=C][C][=C][Ring1][Branch1_1]')
    ['[C]', '[=C]', '[C]', '[=C]', '[C]', '[=C]', '[Ring1]', '[Branch1_1]']
    
    Returns
    -------
    chars_selfies (list of strings) : 
        list of selfie characters present in molecule selfie
    """ 
    chars_selfies = sf.split_selfies(selfies)
    return list(chars_selfies)

def sanitize_smiles(smi):
    """
    Return a canonical smile representation of smi 

    Parameters
    ----------
    smi : str
        smile string to be canonicalized and sanitized

    Returns
    -------
    smi_canon (string)          : 
        Canonicalized smile representation of smi (None if invalid smile string smi)
    """
    if smi == '':
        return None
    try:
        mol = smi2mol(smi, sanitize=True)
        smi_canon = mol2smi(mol, isomericSmiles=True, canonical=True)
        if smi_canon == '' or mol is None:
            return None
        else:
            return smi_canon
    except:
        return None


def remove_specified_chiral_centres(iso_list, anum=[7]):
    # provided a list of isomers (Mol objs)
    # return isomers without chiral centres on atomic numbers (anum)
    new_isomers = []
    for mol in iso_list:
        chirality = Chem.FindMolChiralCenters(mol)
        for chir in chirality:
            if mol.GetAtomWithIdx(chir[0]).GetAtomicNum() in anum:
                mol.GetAtomWithIdx(chir[0]).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        new_isomers.append(mol)
    
    # remove any duplicates
    new_smiles = [Chem.MolToSmiles(m, canonical=True) for m in new_isomers]
    _, idx = np.unique(new_smiles, return_index=True)
    isomers = np.array(new_isomers)[idx].tolist()

    return isomers


def scramble_stereo(smi):
    # return a full list of stereoisomers
    mol = Chem.MolFromSmiles(smi)
    opt = StereoEnumerationOptions(unique=True, onlyUnassigned=False)
    isomers = list(EnumerateStereoisomers(mol, options=opt))
    isomers = remove_specified_chiral_centres(isomers)
    smi_list = [Chem.MolToSmiles(iso, canonical=True, isomericSmiles=True) for iso in isomers]
    return smi_list
        

def assign_stereo(smi, collector={}, random_stereo=True):
    ''' Assign stereochemistry to smiles randomly.
    Return the same smile if invalid.
    Check that smiles is not already found in collector.
    '''
    # pick an assigned stereosmiles (if more than one)
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    
    # assign isomers
    opt = StereoEnumerationOptions(unique=True, onlyUnassigned=True)
    isomers = list(EnumerateStereoisomers(mol, options=opt))
    isomers = remove_specified_chiral_centres(isomers)      # remove nitrogen chiral centres

    # other isomers 
    opt = StereoEnumerationOptions(unique=True, onlyUnassigned=False)
    other_isomers = list(EnumerateStereoisomers(mol, options=opt))
    other_isomers = remove_specified_chiral_centres(other_isomers)      # remove nitrogen chiral centres

    # return isomer that is not yet observed
    if len(isomers) >= 1:
        if random_stereo: random.shuffle(isomers)     # random selecting
        for i in isomers:
            smi = Chem.MolToSmiles(i, isomericSmiles=True, canonical=True)
            if smi not in collector:
                collector[smi] = []
                return smi #, True
        
    if len(other_isomers) > 1:
        if random_stereo: random.shuffle(other_isomers)
        for i in other_isomers:
            smi = Chem.MolToSmiles(i, isomericSmiles=True, canonical=True)
            if smi not in collector:
                collector[smi] = []
                return smi #, True
            
    # if none are found, return original smiles
    return Chem.MolToSmiles(isomers[0], isomericSmiles=True, canonical=True) #, False



def neutralize_radicals(smi):
    mol = smi2mol(smi)
    if mol is None:
        return None
    if Descriptors.NumRadicalElectrons(mol) == 0:
        return smi
    else:
        for a in mol.GetAtoms():
            num_rad = a.GetNumRadicalElectrons()
            if num_rad > 0:
                a.SetNumRadicalElectrons(0)
                a.SetNumExplicitHs(num_rad)
        try:
            smi = mol2smi(mol)
            return smi
        except:
            return None


def neutralize_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    table = Chem.GetPeriodicTable()

    #charges check, with exceptions
    functional_groups = [
        ['[N+0]=[N+]=[N-]',(1,2)], #azide
        ['[N+]#[C-]',(0,1)], #isonitrile
        ['[N+](-[O-])=[O]',(0,1)], #nitro
        # ['[C](-[N])=[N+]',(2,)], #amidinium and guanidinium
    ]
    ignore_charge = []
    for j in functional_groups:
        functional_group = Chem.MolFromSmarts(j[0])
        matches = mol.GetSubstructMatches(functional_group)
        if matches:
            for k in matches:
                for l in j[1]:
                    ignore_charge.append(k[l])

    for j in mol.GetAtoms():
        if table.GetValenceList(j.GetSymbol())[0] < j.GetTotalValence(): #cations
            if j.GetTotalNumHs() == 0:
                continue
            elif j.GetIdx() in ignore_charge: #cations that should remain cations
                continue
            else:
                j.SetFormalCharge(0) #sets formal charge to 0
                j.SetNumExplicitHs(j.GetTotalNumHs() - 1) # gets number of H, then remove 1
        elif table.GetValenceList(j.GetSymbol())[0] > j.GetTotalValence(): #anions
            if j.GetIdx() in ignore_charge: #anions that should remain anions
                continue
            else:
                j.SetFormalCharge(0) #sets formal charge to 0
                j.SetNumExplicitHs(j.GetTotalNumHs() + 1) # gets number of H, then add 1
    return Chem.CanonSmiles(Chem.MolToSmiles(mol))


def get_fp_scores(smiles_back, target_smi):
    """
    Given a list of SMILES (smiles_back), tanimoto similarities are calculated 
    (using Morgan fingerprints) to SMILES (target_smi). 
    Parameters
    ----------
    smiles_back : (list)
        List of valid SMILE strings. 
    target_smi : (str)
        Valid SMILES string. 
    Returns
    -------
    smiles_back_scores : (list of floats)
        List of fingerprint similarity scores of each smiles in input list. 
    """
    smiles_back_scores = []
    target = smi2mol(target_smi)
    fp_target = AllChem.GetMorganFingerprint(target, 2)
    for item in smiles_back:
        mol = smi2mol(item)
        fp_mol = AllChem.GetMorganFingerprint(mol, 2)
        score = TanimotoSimilarity(fp_mol, fp_target)
        smiles_back_scores.append(score)
    return smiles_back_scores

def from_yaml(work_dir, 
        fitness_function, 
        start_population,
        yaml_file, **kwargs):

    # create dictionary with parameters defined by yaml file 
    with open(yaml_file, 'r') as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
    params.update(kwargs)
    params.update({
        'work_dir': work_dir,
        'fitness_function': fitness_function,
        'start_population': start_population
    })

    return params

def scale_array(arr):
    # Get the minimum and maximum values of the array
    arr_min = np.min(arr)
    arr_max = np.max(arr)

    # If all values are the same, return an array of zeros
    if arr_min == arr_max:
        return np.zeros_like(arr)

    # Scale the array to the range [0, 1]
    scaled_arr = (arr - arr_min) / (arr_max - arr_min)
    return scaled_arr


def normalize_score(score, r=[6.0, 15.0], threshold = 0.95):
    # arbitrary range of scores given in range to [0, 1]
    # general values for dataset
    # max(1oyt) = 11.9
    # max(1syh) = 11.1
    # max(6y2f) = 9.3
    centre = r[0] + (r[1] - r[0])/2.0
    slope = (- 1.0 / (r[1] - centre))*np.log(1.0/threshold - 1.0)
    score = np.array(score)
    scaled_score = 1.0 / (1.0 + np.exp(-slope*(score - centre))) # - 1.0
    return scaled_score
