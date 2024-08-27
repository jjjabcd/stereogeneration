#!/usr/bin/env python
import torch
import copy
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import time
import os
import multiprocessing
import itertools
from shutil import copyfile

import rdkit.Chem as Chem

from .model import RNN
from .data_structs import Vocabulary, Experience, MolData
from .reinvent_utils import Variable, seq_to_smiles, fraction_valid_smiles, unique
from ..utils import sanitize_smiles, assign_stereo
from ..filter import passes_filter


def padded_concat(arrays, pad_val = 0, max_length = 140):
    # pad_len = max(len(s) for s in arrays)
    padded = []
    for arr in arrays:
        padded.append(np.pad(arr, pad_width=(0,max_length-len(arr))))
    return np.column_stack( padded ).T

def train_agent(scoring_function,
                restore_prior_from,
                restore_agent_from,
                voc_path,
                starting_df,
                starting_size = None,
                normalize_score = None, 
                stereo=False,
                save_dir=None, learning_rate=0.0005,
                batch_size=64, n_steps=3000,
                num_workers=1, sigma=80,
                experience_replay=0,
                store_path='../data', 
                **kwargs):

    voc = Vocabulary(init_from_file=voc_path)

    start_time = time.time()

    Prior = RNN(voc)
    Agent = RNN(voc)

    # By default restore Agent to same model as Prior, but can restore from already trained Agent too.
    # Saved models are partially on the GPU, but if we dont have cuda enabled we can remap these
    # to the CPU.
    if torch.cuda.is_available():
        Prior.rnn.load_state_dict(torch.load(restore_prior_from))
        Agent.rnn.load_state_dict(torch.load(restore_agent_from))
    else:
        Prior.rnn.load_state_dict(torch.load(restore_prior_from, map_location=lambda storage, loc: storage))
        Agent.rnn.load_state_dict(torch.load(restore_agent_from, map_location=lambda storage, loc: storage))

    # We dont need gradients with respect to Prior
    for param in Prior.rnn.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(Agent.rnn.parameters(), lr=learning_rate)

    # get some samples and check validity
    seqs, likelihood, _ = Agent.sample(128)
    valid = 0
    for i, seq in enumerate(seqs.detach().cpu().numpy()):
        smile = voc.decode(seq)
        if Chem.MolFromSmiles(smile):
            valid += 1
    print("\n{:>4.1f}% valid SMILES".format(100 * valid / len(seqs)))

    # start with a first round of trianing from the initial dataset
    # if starting_size is specified, this is the number of top mols to train with
    if starting_size is not None:
        print('Running a training loop on the given starting population...')
        moldata = MolData(starting_df, voc)

        moldata.sort()      # sort based on the best fitnesses
        train_set = torch.utils.data.Subset(moldata, range(0, starting_size))
        train_data = DataLoader(train_set, batch_size=128, shuffle=True, collate_fn=MolData.collate_fn)
        
        for (batch, y) in train_data:
            seq = batch.long()
            agent_likelihood, _ = Agent.likelihood(seq)
            prior_likelihood, _ = Prior.likelihood(seq)

            # scale the score in the same way
            score = normalize_score(y.cpu().numpy()) if normalize_score is not None else y

            augmented_likelihood = prior_likelihood + sigma * Variable(score)
            loss = torch.pow((augmented_likelihood - agent_likelihood), 2).mean()

            loss_p = - (1.0 / agent_likelihood).mean()
            loss += 5 * 1e3 * loss_p

            torch.nn.utils.clip_grad_norm_(Agent.rnn.parameters(), 3.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        seqs, likelihood, _ = Agent.sample(128)
        valid = 0
        for i, seq in enumerate(seqs.detach().cpu().numpy()):
            smile = voc.decode(seq)
            if Chem.MolFromSmiles(smile):
                valid += 1
        print("\n{:>4.1f}% valid SMILES".format(100 * valid / len(seqs)))

    # For policy based RL, we normally train on-policy and correct for the fact that more likely actions
    # occur more often (which means the agent can get biased towards them). Using experience replay is
    # therefor not as theoretically sound as it is for value based RL, but it seems to work well.
    experience = Experience(voc)
    
    # collect all the results
    results = pd.DataFrame({'smiles': [], 'fitness': [], 'score': [], 'generation': []})
    best_results = pd.DataFrame({'smiles': [], 'fitness': [], 'score': [], 'generation': []})

    print("Model initialized, starting training...")

    for step in range(n_steps):

        # Current generation
        collector = {'smiles': [], 'fitness': [], 'score': [], 'generation': []}
        
        # Sampling procedure (use no_grad to save memory and speed up)
        # smiles, sequences = [], []
        temp = 1.0
        with torch.no_grad():
            seqs, _, _ = Agent.sample(batch_size, temp=temp)

            # Remove duplicates, ie only consider unique seqs
            unique_idxs = unique(seqs)
            seqs = seqs[unique_idxs]

            # apply filter to the smiles
            # assign stereochemistry to smiles
            seqs = seqs.cpu()
            sampled_smiles = seq_to_smiles(seqs, voc)
            valid_smiles, valid_sequences = [], []
            invalid_smiles, invalid_sequences = [], []
            copy_results = {k: [] for k in results['smiles']}
            for smi, seq in zip(sampled_smiles, seqs):
                cleaned_smi = sanitize_smiles(smi)
                if cleaned_smi is not None and passes_filter(cleaned_smi):
                    if stereo:
                        stereo_smi = assign_stereo(cleaned_smi, copy_results)
                        stereo_seq = voc.encode(voc.tokenize(stereo_smi))
                        if stereo_seq is not None:
                            stereo_seq = stereo_seq.astype(int)
                            valid_smiles.append(stereo_smi)
                            valid_sequences.append(stereo_seq)
                        else:
                            invalid_smiles.append(smi)
                            invalid_sequences.append(seq)
                    else:
                        valid_smiles.append(smi)
                        valid_sequences.append(seq)
                # invalid
                else:
                    invalid_smiles.append(smi)
                    invalid_sequences.append(seq)

        sequences = valid_sequences + invalid_sequences
        pad_len = max([len(s) for s in sequences])
        sequences = padded_concat(sequences, max_length=pad_len)

        # calculate likelihoods from the generated sequences
        # while smiles may be invalid, the sequences are still valid 
        # these jobs will fail, and be scored -1
        # sequences = np.concatenate(sequences, axis=0)
        num_val = len(valid_smiles)
        smiles = valid_smiles + invalid_smiles

        sequences = Variable(sequences)
        agent_likelihood, _ = Agent.likelihood(sequences)
        prior_likelihood, _ = Prior.likelihood(sequences)

        # Get prior likelihood and score
        print(f'Total of {num_val}/{batch_size} passed the filter.')
        with multiprocessing.Pool(num_workers) as pool:
            valid_fitness = pool.map(scoring_function, valid_smiles)
        invalid_fitness = [-200.]*len(invalid_smiles)
        fitness = valid_fitness + invalid_fitness

        # normalize for the scaled score
        score = normalize_score(fitness) if normalize_score is not None else np.array(fitness)

        # store the information
        collector['smiles'] = smiles
        collector['fitness'] = fitness
        collector['generation'] = [step]*len(smiles)
        collector['score'] = score.tolist()
        collector = pd.DataFrame(collector)

        results = pd.concat([results, collector])

        best = collector.nlargest(1, 'fitness')
        best_fitness = best['fitness'].values[0]
        best_smiles = best['smiles'].values[0]

        best = results.nlargest(1, 'fitness')
        best_fitness_all = best['fitness'].values[0]
        best_smiles_all = best['smiles'].values[0]
        best['generation'] = step
        best_results = pd.concat([best_results, best])

        # Calculate augmented likelihood
        augmented_likelihood = prior_likelihood + sigma * Variable(score)
        loss = torch.pow((augmented_likelihood - agent_likelihood), 2)

        # Experience Replay
        # First sample
        if experience_replay and len(experience)>10:
            exp_seqs, exp_score, exp_prior_likelihood = experience.sample(10)
            exp_agent_likelihood, _ = Agent.likelihood(exp_seqs.long())
            exp_augmented_likelihood = exp_prior_likelihood + sigma * exp_score
            exp_loss = torch.pow((Variable(exp_augmented_likelihood) - exp_agent_likelihood), 2)

            # concatenate loss and likelihood with original agent
            loss = torch.cat((loss, exp_loss), 0)
            agent_likelihood = torch.cat((agent_likelihood, exp_agent_likelihood), 0)

        # Then add new experience
        prior_likelihood = prior_likelihood.data.cpu().numpy()
        new_experience = zip(smiles, score, prior_likelihood)
        experience.add_experience(new_experience)

        # Calculate loss
        loss = loss.mean()

        # Add regularizer that penalizes high likelihood for the entire sequence
        loss_p = - (1.0 / agent_likelihood).mean()
        loss += 5 * 1e3 * loss_p

        # Calculate gradients and make an update to the network weights
        # torch.nn.utils.clip_grad_norm_(Agent.rnn.parameters(), 3.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Convert to numpy arrays so that we can print them
        augmented_likelihood = augmented_likelihood.data.cpu().numpy()
        agent_likelihood = agent_likelihood.data.cpu().numpy()

        # Print some information for this step
        time_elapsed = (time.time() - start_time) / 3600
        time_left = (time_elapsed * ((n_steps - step) / (step + 1)))
        print(f"\n       Step {step}   Time elapsed: {time_elapsed:.2f}h    Approx Time left: {time_left:.2f}h")
        print(f"Fraction of valid smiles: {fraction_valid_smiles(smiles)*100}%")
        print("Sample of results: ")
        print("  Agent    Prior   Target   Score   Fitness             SMILES")

        for i in range(min(len(agent_likelihood),5)):
            print(" {:6.2f}   {:6.2f}  {:6.2f}  {:6.2f}   {:6.2f}     {}".format(agent_likelihood[i],
                                                                       prior_likelihood[i],
                                                                       augmented_likelihood[i],
                                                                       score[i],
                                                                       fitness[i],
                                                                       smiles[i]))
        print(f"Best in generation: {best_fitness:6.2f}   {best_smiles}")
        print(f"Best in general:    {best_fitness_all:6.2f}   {best_smiles_all}")



    # If the entire training finishes, we create a new folder where we save this python file
    # as well as some sampled sequences and the contents of the experinence (which are the highest
    # scored sequences seen during training)
    if not save_dir:
        save_dir = 'RESULTS/run_' + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    os.makedirs(save_dir)
    # copyfile('train_agent.py', os.path.join(save_dir, "train_agent.py"))

    results = results.reset_index(drop=True)
    best_results = best_results.reset_index(drop=True)

    results.to_csv(os.path.join(save_dir, 'results.csv'), index=False)
    best_results.to_csv(os.path.join(save_dir, 'best_results.csv'), index=False)
    sns.lineplot(data=best_results, x='generation', y='fitness')
    plt.savefig(os.path.join(save_dir, 'trace.png'))

    fitness_dict = results[['smiles', 'fitness']].set_index('smiles').transpose().to_dict(orient='records')[0]
    experience.print_memory(os.path.join(save_dir, "memory"), fitness_dict)
    torch.save(Agent.rnn.state_dict(), os.path.join(save_dir, 'Agent.ckpt'))


if __name__ == "__main__":
    train_agent()
