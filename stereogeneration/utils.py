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
        smile string to be canonicalized 

    Returns
    -------
    mol (rdkit.Chem.rdchem.Mol) : 
        RdKit mol object (None if invalid smile string smi)
    smi_canon (string)          : 
        Canonicalized smile representation of smi (None if invalid smile string smi)
    conversion_successful (bool): 
        True/False to indicate if conversion was  successful 
    """
    if smi == '':
        return None
    try:
        mol = smi2mol(smi, sanitize=True)
        smi_canon = mol2smi(mol, isomericSmiles=True, canonical=True)
        return smi_canon
    except:
        return None


def remove_nitrogen_chiral_centres(iso_list):
    # provided a list of isomers (Mol objs)
    # return isomers without N chiral centres
    new_isomers = []
    for mol in iso_list:
        chirality = Chem.FindMolChiralCenters(mol)
        for chir in chirality:
            if mol.GetAtomWithIdx(chir[0]).GetAtomicNum() == 7:
                mol.GetAtomWithIdx(chir[0]).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        new_isomers.append(mol)
    
    # remove any duplicates
    new_smiles = [Chem.MolToSmiles(m, canonical=True) for m in new_isomers]
    uniq_smiles, idx = np.unique(new_smiles, return_index=True)
    isomers = np.array(new_isomers)[idx].tolist()

    return isomers


def scramble_stereo(smi):
    # return a full list of stereoisomers
    mol = Chem.MolFromSmiles(smi)
    opt = StereoEnumerationOptions(unique=True, onlyUnassigned=False)
    isomers = list(EnumerateStereoisomers(mol, options=opt))
    # isomers = remove_nitrogen_chiral_centres(isomers)
    smi_list = [Chem.MolToSmiles(iso, canonical=True, isomericSmiles=True) for iso in isomers]
    return smi_list
        

def assign_stereo(smi, collector=[]):
    # pick an assigned stereosmiles (if more than one)
    mol = Chem.MolFromSmiles(smi)
    opt = StereoEnumerationOptions(unique=True, onlyUnassigned=True)
    isomers = list(EnumerateStereoisomers(mol, options=opt))

    # remove nitrogen chiral centres
    isomers = remove_nitrogen_chiral_centres(isomers)

    # return isomer that is not yet observed
    if len(isomers) > 1:
        random.shuffle(isomers)     # random selecting
        for i in range(len(isomers)):
            smi = Chem.MolToSmiles(isomers[i], isomericSmiles=True, canonical=True)
            if smi not in collector:
                return smi #, True
        return smi #, False      # if all are observed, return anyway
    else:
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
