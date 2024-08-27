#!/usr/bin/env python
import argparse

import torch
from torch.utils.data import DataLoader
import pickle
from rdkit import Chem
from rdkit import rdBase
from tqdm import tqdm

from .data_structs import MolData, Vocabulary
from .model import RNN, EarlyStopping
from .reinvent_utils import Variable, decrease_learning_rate
# from .custom import EarlyStopping
rdBase.DisableLog('rdApp.error')


def pretrain(
        num_epochs, 
        verbose, 
        train_ratio, 
        starting_df,
        stereo=False, 
        store_path='../data', 
        restore_from=None,
        use_gpu=True
    ):
    """Trains the Prior RNN"""

    # Initialize early stopper
    early_stop = EarlyStopping(patience=20, min_delta=1e-7, mode='minimize')

    # Read vocabulary from a file
    if stereo:
        voc = Vocabulary(init_from_file=f"{store_path}/Voc_stereo")
    else:
        voc = Vocabulary(init_from_file=f"{store_path}/Voc_nonstereo")

    moldata = MolData(starting_df, voc)

    # Create a Dataset from a SMILES file
    train_size = int(len(moldata)*train_ratio)
    train_set = torch.utils.data.Subset(moldata, range(0, train_size))
    valid_set = torch.utils.data.Subset(moldata, range(train_size, len(moldata)))

    # training and validation set
    train_data = DataLoader(train_set, batch_size=128, shuffle=True, collate_fn=MolData.collate_fn)
    valid_data = DataLoader(valid_set, batch_size=128, collate_fn=MolData.collate_fn)

    if use_gpu:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = torch.device('cpu')

    Prior = RNN(voc)
    Prior.rnn = Prior.rnn.to(device)

    # Can restore from a saved RNN
    if restore_from:
        Prior.rnn.load_state_dict(torch.load(restore_from))

    optimizer = torch.optim.Adam(Prior.rnn.parameters(), lr = 0.001)
    for epoch in range(1, num_epochs + 1):
        # When training on a few million compounds, this model converges
        # in a few of epochs or even faster. If model sized is increased
        # its probably a good idea to check loss against an external set of
        # validation SMILES to make sure we dont overfit too much.
        Prior.rnn.train()
        for step, (batch, _) in tqdm(enumerate(train_data), total=len(train_data), disable=not verbose):

            # Sample from DataLoader
            seqs = batch.long().to(device)

            # Calculate loss
            log_p, _ = Prior.likelihood(seqs)
            loss = - log_p.mean()

            # Calculate gradients and take a step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0 and step != 0:
                # decrease_learning_rate(optimizer, decrease_by=0.003)
                if verbose:
                    tqdm.write("*" * 50)
                    tqdm.write("Epoch {:3d}   step {:3d}    loss: {:5.2f}\n".format(epoch, step, loss.item()))
                else:
                    print("Epoch {:3d}   step {:3d}    loss: {:5.2f}\n".format(epoch, step, loss.item()))
                seqs, likelihood, _ = Prior.sample(128)
                valid = 0
                for i, seq in enumerate(seqs.detach().cpu().numpy()):
                    smile = voc.decode(seq)
                    if Chem.MolFromSmiles(smile):
                        valid += 1
                    if i < 5:
                        if verbose:
                            tqdm.write(smile)
                        else:
                            print(smile)
                if verbose:
                    tqdm.write("\n{:>4.1f}% valid SMILES".format(100 * valid / len(seqs)))
                    tqdm.write("*" * 50 + "\n")
                else:
                    print("\n{:>4.1f}% valid SMILES".format(100 * valid / len(seqs)))


        # validation loop
        Prior.rnn.eval()
        val_loss = 0
        for step, (batch, _) in tqdm(enumerate(valid_data), total=len(valid_data), disable=not verbose):
            with torch.no_grad():
                seqs = batch.long()
                log_p, _ = Prior.likelihood(seqs)
                val_loss += - log_p.mean()
        val_loss /= len(valid_data)

        print(f'Epoch   {epoch}: validation loss = {val_loss}')

        stop = early_stop.check_criteria(Prior.rnn, epoch, val_loss.item())
        if stop:
            Prior.rnn = early_stop.restore_best(Prior.rnn)
            break

        # # save checkpoint    
        # if stereo:
        #     torch.save(Prior.rnn.state_dict(), f"{store_path}/Prior_checkpoint_stereo.ckpt")
        # else:
        #     torch.save(Prior.rnn.state_dict(), f"{store_path}/Prior_checkpoint_nonstereo.ckpt")

        Prior.rnn.train()

    # Save the Prior
    if stereo:
        torch.save(Prior.rnn.state_dict(), f"{store_path}/Prior_stereo.ckpt")
    else:
        torch.save(Prior.rnn.state_dict(), f"{store_path}/Prior_nonstereo.ckpt")

if __name__ == "__main__":
    arg_dict = vars(parser.parse_args())
    pretrain(num_epochs=arg_dict['num_epochs'], verbose=arg_dict['verbose'], train_ratio=arg_dict['train_ratio'])
