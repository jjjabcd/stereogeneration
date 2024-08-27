from typing import List, Iterable

import selfies as sf
from group_selfies import GroupGrammar, Group

import pandas as pd
import numpy as np

import rdkit.Chem as Chem


def process_isoalphabet(alphabet, use_gsf=False):
    # find all the isomerisms, and ensure symmetric tokens are added
    sym_iso = []

    for a in alphabet:
        # cis-trans
        if ('/' in a):
            sym_iso += [a.replace('/', '\\')]
        elif ('\\' in a):
            sym_iso += [a.replace('\\', '/')]

        # cw and ccw chirality
        if ('@@' in a):
            sym_iso += [a.replace('@@', '@')]
        elif ('@' in a):
            sym_iso += [a.replace('@', '@@')]

    if not use_gsf:
        # chiral tokens are only added if we are using selfies
        # always include these characters
        chiral_c = [
            '[C@]', '[C@@]', '[C@H1]', '[C@@H1]',
            '[/C@]', '[/C@@]', '[/C@H1]', '[/C@@H1]',
            '[\\C@]', '[\\C@@]', '[\\C@H1]', '[\\C@@H1]',
        ]
        alphabet += chiral_c
    
    alphabet += sym_iso
    alphabet = list(sorted(set(alphabet)))

    return alphabet


def increase_token_frequency(alphabet, groups = [], n = 5):
    # increase the frequency of tokens in alphabet
    # increase sepcies found in groups by n times
    for g in groups:
        freq = n - alphabet.count(g) if g in alphabet else n
        if freq > 0:
            alphabet.extend([g] * freq)
    return sorted(alphabet)

def get_token_heavy_atom(token, use_gsf = False, grammar = None):
    # find the heaviest atom in token
    # give the decoder function required to change token to mol
    if use_gsf:
        assert grammar is not None, 'Please provide grammar if using group-selfies.'
        m = grammar.decoder(token)
    else:
        m = Chem.MolFromSmiles(sf.decoder(token))

    ele = 0
    for atom in m.GetAtoms():
        anum = atom.GetAtomicNum()
        if anum > ele:
            ele = anum
    ele = Chem.GetPeriodicTable().GetElementSymbol(ele)

    return ele

def get_proper_group_grammar(isoalphabet, abridged=False):
    # special_frags = [
    #     ('benzene', 'C1=CC=CC=C1', True),
    #     ('nitro', '[O-]-[N+]=O', True),
    #     ('carboxyl', 'O=C-[OH]', True)
    # ]
    # vocab_dict = dict([(n, Group(n, s, all_attachment=all_at, priority=priority[0] if len(priority) > 0 else 0)) for n, s, all_at, *priority in special_frags])
    # grammar = GroupGrammar(vocab_dict) | GroupGrammar().essential_set()
    grammar = GroupGrammar() | GroupGrammar().essential_set()

    atoms_found = []
    for i, a in enumerate(isoalphabet):
        # find the heaviest atom in token
        m = grammar.decoder(a)
        ele = 0
        for atom in m.GetAtoms():
            anum = atom.GetAtomicNum()
            if anum > ele:
                ele = anum
        ele = Chem.GetPeriodicTable().GetElementSymbol(ele)
        
        if ele not in atoms_found:
            atoms_found.append(ele)

    # remove 13C isotope
    grammar.delete_group('chiralc13')
    grammar.delete_group('chiral2c13')

    # remove charged nitrogen ammonium groups
    grammar.delete_group('ammonium')
    grammar.delete_group('ammonium2')

    # remove groups that are not found in the dataset
    # the isoalphabet should be generated from the dataset
    # abridged setting will remove these groups to reduce the size of alphabet
    if 'P' not in atoms_found or abridged:
        for g in ['phosphine'] + [f'phosphine{i}' for i in range(2, 8)]:
            grammar.delete_group(g)
    if 'S' not in atoms_found or abridged:
        sulf_groups = [
            'sulfoxime_cis', 'sulfoxime_trans', 
            'sulfilimine_cis', 'sulfilimine_trans', 
            'sulfan_general', 'sulfox_general', 
            'sulfonium'
        ]
        for g in sulf_groups:
            grammar.delete_group(g)
    if 'B' not in atoms_found or abridged:
        grammar.delete_group('boron')
    if abridged:
        grammar.delete_group('aziridine_n')
        grammar.delete_group('bridged_n')

    return grammar


def get_isoalphabet_weights(isoalphabet, sf2mol_func, use_gsf=False):
    # some elements are selected too often due to additional isomeric tokens
    # introduce sampling weights (based on frequency of tokens)
    weights = np.ones(len(isoalphabet))

    atoms_found = {}
    atom_map = {}
    for i, a in enumerate(isoalphabet):
        # find the heaviest atom in token
        m = sf2mol_func(a)
        ele = 0
        for atom in m.GetAtoms():
            anum = atom.GetAtomicNum()
            if anum > ele:
                ele = anum
        ele = Chem.GetPeriodicTable().GetElementSymbol(ele)

        # store the special pop operator
        if a == '[pop]':
            ele = '[pop]'

        if ele not in atoms_found.keys():
            atoms_found[ele] = 0
        atoms_found[ele] += 1
        atom_map[i] = ele

    if not use_gsf:
        # change weights based on selfies grammar
        for i, v in atom_map.items():
            if v == 'N' or v == 'O':
                weights[i] = 0.5            # N and O weights will be halved
            elif v == 'C' or v == '*':
                weights[i] = 1.0            # C and other tokens remain
            else:
                weights[i] = 1./atoms_found[v]
    else:
        for i, v in atom_map.items():
            if v == 'N':
                weights[i] = 0.25           # further reduce nitrogen weight due to nitrogen-containing groups
            elif v == 'O':
                weights[i] = 0.5            # O weight will be halved
            elif v == 'C':
                weights[i] = 1.0            # C stays at 1.0
            elif v == '*':
                weights[i] = 2.5            # special characters
            elif v == '[pop]':
                weights[i] = 5.0            # increase weight of pop character
            else:
                weights[i] = 1./atoms_found[v]
    
    return weights