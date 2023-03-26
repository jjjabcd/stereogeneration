#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 13:50:21 2021

@author: akshat
"""
from selfies import encoder, decoder
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem import MolToSmiles as mol2smi
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

from .utils import assign_stereo

RDLogger.DisableLog("rdApp.*")

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)


def get_frags(smi, radius):
    ''' Create fragments from smi with some radius. Remove duplicates and any
    fragments that are blank molecules.
    '''
    mol = smi2mol(smi, sanitize=True)
    frags = []
    for ai in range(mol.GetNumAtoms()):
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, ai)
        amap = {}
        submol = Chem.PathToSubmol(mol, env, atomMap=amap)
        frag = mol2smi(submol, isomericSmiles=True, canonical=True)
        frags.append(frag)
    return list(filter(None, list(set(frags))))

def form_fragments(smi, stereo=True):
    ''' Create fragments of certain radius. Returns a list of fragments
    using SELFIES characters.
    '''
    selfies_frags = []
    unique_frags = get_frags(smi, radius=3)
    for item in unique_frags:
        # encode and decode from selfies
        try:
            sf = encoder(item)
            dec_ = decoder(sf)
        except:
            continue

        try:
            # check for chiral centers
            m = Chem.MolFromSmiles(dec_)                
            # if len(Chem.FindMolChiralCenters(m)) == 0 and stereo:
            #     continue
                
            Chem.Kekulize(m)
            dearom_smiles = Chem.MolToSmiles(
                m, canonical=True, isomericSmiles=stereo, kekuleSmiles=True
            )
            dearom_mol = Chem.MolFromSmiles(dearom_smiles)
        except:
            continue

        if dearom_mol == None:
            raise Exception("mol dearom failes")

        selfies_frags.append(encoder(dearom_smiles))

    return selfies_frags

    