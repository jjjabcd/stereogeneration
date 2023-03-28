"""
Filtering GDB-13
Authors: Robert Pollice, Akshat Nigam
Date: Sep. 2020
"""
import numpy as np
import pandas as pd
import rdkit as rd
from rdkit import Chem
import rdkit.Chem as rdc
import rdkit.Chem.rdMolDescriptors as rdcmd
import rdkit.Chem.Lipinski as rdcl
from rdkit.Chem import Descriptors, MolStandardize
# import argparse as ap
import pathlib as pl

cwd = pl.Path.cwd() # define current working directory

def smiles_to_mol(smiles):
    """
    Convert SMILES to mol object using RDKit
    """
    try:
        mol = rdc.MolFromSmiles(smiles)
    except:
        mol = None
    return mol

def maximum_ring_size(mol):
    """
    Calculate maximum ring size of molecule
    """
    cycles = mol.GetRingInfo().AtomRings()
    if len(cycles) == 0:
        maximum_ring_size = 0
    else:
        maximum_ring_size = max([len(ci) for ci in cycles])
    return maximum_ring_size
    
def minimum_ring_size(mol):
    """
    Calculate minimum ring size of molecule
    """
    cycles = mol.GetRingInfo().AtomRings()
    if len(cycles) == 0:
        minimum_ring_size = 0
    else:
        minimum_ring_size = min([len(ci) for ci in cycles])
    return minimum_ring_size

def substructure_match_with_exception(mol, forbidden_fragments, exceptions):
    violation = False

    try:
        bad = [Chem.MolFromSmarts(i) for i in forbidden_fragments] #specific bad structures to get rid of
    except:
        raise ValueError('Check your forbidden fragments and ensure they are valid.')
    mol_hydrogen = Chem.AddHs(mol)
    all_matches = []
    for i in bad:
        for j in mol_hydrogen.GetSubstructMatches(i):
            all_matches.append(j[0])

    ignore = []
    for j in exceptions:
        try:
            exception_group = Chem.MolFromSmarts(j[0])
        except:
            raise ValueError('Check your exceptions and ensure they are valid.')
        matches = mol.GetSubstructMatches(exception_group)
        if matches:
            for k in matches:
                for l in j[1]:
                    ignore.append(k[l])

    # check if violation is in an allowed functional group
    for j in all_matches:
        if j not in ignore:
            violation = True
            break
    return violation

def debug_substructure_violations(smiles_list):
    # frags = ['[!n;!N][ND1H2]','[!n;!N][ND1H3+]','[!n;!N][ND2H1][!n;!N]','[!n;!N][ND2H2+][!n;!N]','[!n;!N][ND3]([!n;!N])[!n;!N]','[!n;!N][ND3H1+]([!n;!N])[!n;!N]','[!n;!N][ND4+]([!n;!N])([!n;!N])[!n;!N]']
    frags = ['[!n;!N][ND1H2+0]','[!n;!N][ND1H3+]','[!n;!N][ND2H1+0][!n;!N]','[!n;!N][ND2H2+][!n;!N]','[!n;!N][ND3+0]([!n;!N])[!n;!N]','[!n;!N][ND3H1+]([!n;!N])[!n;!N]','[!n;!N][ND4+]([!n;!N])([!n;!N])[!n;!N]']
    
    counts = {k: 0 for k in frags}
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        # violation = False

        for i in frags:
            if mol.HasSubstructMatch(Chem.MolFromSmarts(i)):
                counts[i] += 1
        
    return counts
        


def substructure_violations(mol):
    """
    Check for substructure violates
    Return True: contains a substructure violation
    Return False: No substructure violation
    """
    # violation = False
    # 'a~[*;R2]~a', '[O-]', '[N-]', '[*+]', '[*-]', '[N,n,O,o,S,s]~[N,n,O,o,S,s]~[N,n,O,o,S,s]', 
    # '[N,n,O,o,S,s]~[N,n,O,o,S,s]~[C,c]=,:[O,o,S,s,N,n;!R]', '*=[S,s;!R]', '[A;R]=[*;R2]', 
    # '[S&X4]', '[S&X3]','[O,o,S,s]~[O,o,S,s]',  '*=N-[*;!R]',  '[P,p]','[B,b,N,n,O,o,S,s]~[F,Cl,Br,I]', 
    # ['[!n;!N][ND1H2]','[!n;!N][ND1H3+]','[!n;!N][ND2H1][!n;!N]','[!n;!N][ND2H2+][!n;!N]','[!n;!N][ND3]([!n;!N])[!n;!N]','[!n;!N][ND3H1+]([!n;!N])[!n;!N]','[!n;!N][ND4+]([!n;!N])([!n;!N])[!n;!N]']

    # perform filter for non-charged
    forbidden_fragments = [
        '*1=**=*1', '*=*1*=***=*1', '[PH]', '[pH]', '[N&X5]', '[S&X4]',
        '[S&X5]', '[S&X6]', '[N,n,O,o,S,s]~[F,Cl,Br,I]', '[PH2]', '[N+0]=[O+0]'
        '*=*=*', '*#*', '[O,o]~[O,o]', '[O,o,S,s]!=[O,o,S,s]!=[O,o,S,s]', 
        '[O,o,S,s]~[O,o,S,s]~[C,c]=,:[O,o,S,s;!R]', 
        '[N;R]-[N;!R]', '[N;R]-[N;R]', '[N]~[N]~[N]', '[*+]', '[*-]', '[N]-[N]'
    ]
    exceptions = [
        ['[S&X4](=[O])(=[O])', (0,1,2)],
        ['*-[N+0]=[N+]=[N-]', (1,2,3)],
        ['[C;!R]#[C;!R]', (0,1)],
        ['[NX1]#[CX2]', (0,1)],             # nitriles
        ['[!#1]-[N+0]=[N+]=[N-]',(2,3)],    # azide
        ['[!#1]-[N+]#[C-]',(1,2)],          # isonitrile
        ['[!#1]-[N+](-[O-])=[O]',(1,2)],    # nitro, also hits nitrate ester
        # ['[!n;!N][ND1H3+]', (1,)], 
        # ['[!n;!N][ND2H2+][!n;!N]', (1,)],
        # ['[!n;!N][ND3H1+]([!n;!N])[!n;!N]', (1,)]
        # ['[OX1]=[CX3]-[O-]', (2,)],         # carboxylate
    ]

    violation = substructure_match_with_exception(mol, forbidden_fragments, exceptions)

    return violation

def aromaticity_degree(mol):
    """
    Compute the percentage of non-hydrogen atoms in a molecule that are aromatic
    """
    atoms = mol.GetAtoms()
    atom_number = rdcl.HeavyAtomCount(mol)
    aromaticity_count = 0.
    
    for ai in atoms:
        if ai.GetAtomicNum() != 1:
            if ai.GetIsAromatic() == True:
                aromaticity_count += 1.
        
    degree = aromaticity_count / atom_number
    
    return degree
    
def conjugation_degree(mol):
    """
    Compute the percentage of bonds between non-hydrogen atoms in a molecule that are conjugated
    """
    bonds = mol.GetBonds()
    bond_number = 0.
    conjugation_count = 0.
    
    for bi in bonds:
        a1 = bi.GetBeginAtom()
        a2 = bi.GetEndAtom()
        if (a1.GetAtomicNum() != 1) and (a2.GetAtomicNum() != 1):
            bond_number += 1.
            if bi.GetIsConjugated() == True:
                conjugation_count += 1.
        
    degree = conjugation_count / bond_number
    
    return degree

def num_charge_species(mol):
    """
    Compute number of charged species in the molecule
    """
    num_charged = 0
    for ai in mol.GetAtoms():
        num_charged += np.abs(ai.GetFormalCharge())
    return num_charged

def passes_filter(smi): 
    if smi == '':
        return False
    mol = rdc.MolFromSmiles(smi)
    if mol is None:
        return False
    mol_hydrogen = Chem.AddHs(mol)

    # aromaticity_degree(mol) >= 0.2 and # aromaticity_degree(mol) >= 0.5 and 
    # conjugation_degree(mol) >= 0.2 and # conjugation_degree(mol) >= 0.7 and 
    # (5 <= minimum_ring_size(mol) <=7 or minimum_ring_size(mol) == 0) and 
    # 
    maxring = maximum_ring_size(mol)
    minring = minimum_ring_size(mol)
    if (
        rdcmd.CalcNumBridgeheadAtoms(mol) == 0 and 
        rdcmd.CalcNumSpiroAtoms(mol) == 0 and 
        (5 <= maxring <= 7 or maxring == 0) and
        (5 <= minring <= 7 or minring == 0) and 
        substructure_violations(mol) == False and 
        np.abs(rdc.GetFormalCharge(mol)) <= 2 and
        num_charge_species(mol) <= 5 and
        mol_hydrogen.GetNumAtoms() <= 70 and
        Descriptors.NumRadicalElectrons(mol) == 0
    ): 
        return True
    else: 
        return False 



