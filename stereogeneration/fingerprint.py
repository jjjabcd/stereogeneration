import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
# rdkit built in isomeric fingerprints work better for the discovery tasks
# from mapchiral.mapchiral import encode_many, encode, jaccard_similarity
from .utils import assign_stereo

TAR_MAP ={
    'fp-mestranol': 'C#C[C@]1(O)CC[C@H]2[C@@H]3CCc4cc(OC)ccc4[C@H]3CC[C@@]21C',
    'fp-albuterol': 'CC(C)(C)NC[C@H](O)c1ccc(O)c(CO)c1'         # R - albuterol
}


def encode(mol, max_radius=3, mapping=False):
    return AllChem.GetMorganFingerprint(mol, radius=max_radius, useChirality=True)

def jaccard_similarity(fp1, fp2):
    return TanimotoSimilarity(fp1, fp2)


# def get_fingerprints(smi_list):
#     mols = [Chem.MolFromSmiles(s) for s in smi_list]
#     fps = encode_many(mols)
#     return fps

def get_fingerprint(smi):
    mol = Chem.MolFromSmiles(smi)
    return encode(mol, max_radius=3, mapping=False)

# def get_chiral_fp_scores(smi_list, target_smi):
#     # return a list of scores
#     target_mol = Chem.MolFromSmiles(target_smi)
#     target_fp = encode(target_mol, max_radius=3, mapping=False)
#     fps = get_fingerprints(smi_list)

#     return [jaccard_similarity(target_fp, f) for f in fps]

def get_chiral_fp_score(smi, target_smi):
    # return a list of scores
    target_mol = Chem.MolFromSmiles(target_smi)
    target_fp = encode(target_mol, max_radius=3, mapping=False)
    fps = get_fingerprint(smi)

    return jaccard_similarity(target_fp, fps)

def fitness_function(smi, target):
    target_smi = TAR_MAP[target]
    chiral_smi = assign_stereo(smi, {})        # this only returns smiles with chirality (fills in unassigned)

    try:
        return jaccard_similarity(get_fingerprint(chiral_smi), get_fingerprint(target_smi))
    except:
        return 0.0




