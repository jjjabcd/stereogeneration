#!/usr/bin/env python
import torch
import copy
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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


def normalize_score(score, r=[-2.0, 15.0], threshold = 0.95):
    # arbitrary range of scores given in range
    centre = r[0] + (r[1] - r[0])/2.0
    slope = (- 1.0 / (r[1] - centre))*np.log(2.0 / (threshold + 1.0) - 1.0)
    score = np.array(score)
    scaled_score = 2.0 / (1.0 + np.exp(-slope*(score - centre))) - 1.0
    return scaled_score

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
                stereo=False,
                save_dir=None, learning_rate=0.0005,
                batch_size=64, n_steps=3000,
                num_workers=1, sigma=80,
                experience_replay=0,
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
        collector = {
            'smiles': [], 
            'fitness': [], 
            'score': [], 
            'generation': [],
        }
        
        # Sampling procedure (use no_grad to save memory and speed)
        smiles, sequences = [], []
        timeout_count = 0
        temp = 1.0
        with torch.no_grad():
            while len(smiles) < batch_size:
                seqs, _, _ = Agent.sample(batch_size, temp=temp)
                sampled_smiles = seq_to_smiles(seqs, voc)

                # timeout not reached yet
                if timeout_count < 5:
                    valid_smiles, valid_idx = [], []
                    for i, smi in enumerate(sampled_smiles):
                        smi = sanitize_smiles(smi)

                        # only keep if valid and passes filter
                        if smi is not None and passes_filter(smi):
                            valid_idx.append(i)
                            valid_smiles.append(smi)

                    if len(valid_smiles) == 0:
                        continue
                    seqs = seqs[valid_idx]

                else:
                    # timed out!
                    valid_smiles = sampled_smiles
                    

                # assign stereochemistry if necessary
                if stereo:
                    new_seq = []
                    for smi in valid_smiles:
                        smi = assign_stereo(smi, results['smiles'])
                        seq = voc.encode(voc.tokenize(smi))
                        if seq is not None:     # if there are invalid tokens...
                            seq = seq.astype(int)
                            smiles.append(smi)
                            new_seq.append(seq)
                    if len(new_seq) == 0:
                        continue
                    new_seq = padded_concat(new_seq)
                    sequences.append(new_seq)
                else:
                    smiles.extend(valid_smiles)
                    sequences.append(padded_concat(seqs.cpu().numpy()))

                # sampling timeout          
                if timeout_count == 5:
                    print(f'Sampling timeout... total of {len(smiles)}!')
                    break

                print(f'Successfully sampled {len(smiles)} smiles.')
                timeout_count += 1

        # calculate likelihoods from the generated sequences
        # while smiles may be invalid, the sequences are still valid 
        # these jobs will fail, and be scored -1
        sequences = np.concatenate(sequences, axis=0)
        smiles = smiles[:batch_size]
        sequences = sequences[:batch_size]

        sequences = Variable(sequences)
        agent_likelihood, _ = Agent.likelihood(sequences)
        prior_likelihood, _ = Prior.likelihood(sequences)

        # Get prior likelihood and score
        with multiprocessing.Pool(num_workers) as pool:
            fitness = pool.map(scoring_function, smiles)

        # normalize for the scaled score
        score = normalize_score(fitness)

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
        torch.nn.utils.clip_grad_norm_(Agent.rnn.parameters(), 3.0)
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
