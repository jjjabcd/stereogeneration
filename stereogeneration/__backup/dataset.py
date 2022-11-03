from typing import Dict, List
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, Subset, random_split
import torch
from sklearn.preprocessing import MinMaxScaler

import multiprocessing
import selfies as sf
import pandas as pd
import numpy as np

from . import utils


class SELFIES_Datamodule(pl.LightningDataModule):

    def __init__(self, init_smiles: List[str], init_fitness,
            batch_size: int, len_molecule: int, num_workers: int,
            seed: int = 30624700):
        ''' Specify dataset used and parameters for the dataloaders
        data:           DataFrame with SMILES, SELFIES and TARGETS
        selfies_name:   column name of selfies string
        target_name:    column name of targets
        batch_size:     batch size of dataloaders
        len_molecule:   length of longest selfies encoding, padding for anything shorter
        num_workers:    number of cpus for datamodule
        seed:           random seed for reproducibility
        '''
        super().__init__()

        self.init_smiles = init_smiles
        self.init_fitness = np.array(init_fitness)
        if len(self.init_fitness.shape) == 1:
            self.init_fitness = self.init_fitness.reshape(-1, 1)
        self.batch_size = batch_size
        self.num_workers = num_workers
        pl.seed_everything(seed)

        with multiprocessing.Pool(self.num_workers) as pool:
            self.init_selfies = pool.map(
                sf.encoder,
                self.init_smiles
            )

        # get selfies and pad length
        max_selfies = max(sf.len_selfies(s) for s in self.init_selfies)
        print(f'Longest selfies: {max_selfies}')
        if max_selfies > len_molecule:
            print('Longest SELFIES is longer than padding length.')
            len_molecule = max_selfies
        self.len_molecule = len_molecule

        # get alphabet
        alphabet = sf.get_alphabet_from_selfies(self.init_selfies)
        alphabet.add("[nop]")
        alphabet = list(sorted(alphabet))

        self.alphabet = alphabet
        self.vocab = {s: i for i, s in enumerate(alphabet)}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.len_alphabet = len(alphabet)

    def setup(self, stage = None, training_ratio: float = 0.85):
        full_dataset = selfies_dataset(self.init_selfies, self.init_fitness, self.vocab, self.len_molecule)
        train_size = int(len(full_dataset) * training_ratio)
        valid_size = len(full_dataset) - train_size
        self.training, self.validation = random_split(full_dataset, [train_size, valid_size])

    def train_dataloader(self):
        return DataLoader(self.training, batch_size = self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.validation, batch_size = self.batch_size, num_workers=self.num_workers)

    def logits_to_smiles(self, logits):
        if type(logits) is np.array:
            logits = torch.Tensor(logits)

        labels = logits.max(dim=-1).indices
        labels = labels.detach().numpy()
        smi_list = []
        for l in labels:
            sfs = sf.encoding_to_selfies(
                l, vocab_itos = self.inv_vocab,
                enc_type='label'
            )
            smi = utils.sanitize_smiles(sf.decoder(sfs))
            smi_list.append(smi)
        
        return smi_list


class selfies_dataset(Dataset):
    def __init__(self, selfies: List[str], targets: List[float], vocab: Dict, len_molecule: int):
        ''' Create dataset from csv.
        selfies:        series of selfies strings
        target_name:    series of target values
        len_molecule:        padding length, if selfies encoding longer, use that length instead
        '''
        self.selfies = selfies
        self.targets = targets
        self.len_molecule = len_molecule
        self.vocab = vocab

    def __len__(self):
        return len(self.selfies)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        encoding = sf.selfies_to_encoding(
            self.selfies[idx], 
            vocab_stoi = self.vocab,
            pad_to_len = self.len_molecule,
            enc_type = 'one_hot'
        )
        encoding = torch.FloatTensor(encoding)
        target = self.targets[idx]

        return encoding, target

if __name__ == '__main__':
    # DEBUGGING
    pass


