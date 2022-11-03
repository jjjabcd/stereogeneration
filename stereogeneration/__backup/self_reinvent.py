import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl


class LanguageModel(nn.Module):
    def __init__(self, len_alphabet, len_molecule, embedding_dim, hidden_dim):
        super().__init__()
        self.len_alphabet = len_alphabet
        self.len_molecule = len_molecule
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(self.len_alphabet, self.embedding_dim)
        self.rnn = nn.GRU(
            input_size = self.embedding_dim,
            hidden_size = self.hidden_dim,
            num_layers = 3,
            batch_first = True
        )
        self.fc = nn.Linear(self.hidden_dim, self.len_alphabet)
        self.softmax = nn.LogSoftmax()
    
    def forward(self, x, h = None):
        x = self.embedding(x)
        x, h = self.rnn(x, h)
        x = self.fc(x)
        x = self.softmax(x)
        return x, h


class Prior(pl.LightningModule):
    def __init__(self, prior_network: LanguageModel):
        super().__init__()
        self.prior_network = prior_network
        self.nll = nn.NLLLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.prior_network(x)    # logsoftmax already applied
        loss = self.nll(y_hat, y)
        return loss

    def validation_step(self):
        pass

    def configure_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


class Agent(pl.LightningModule):
    def __init__(self, scoring_function, sigma, prior_network, agent_network = None):
        super().__init__()
        self.prior_network = prior_network
        for param in self.prior_network.parameters():
            param.requires_grad = False     # freeze the prior
        
        self.scoring_function = scoring_function
        self.sigma = sigma
        self.agent_network = copy.deepcopy(prior_network) if agent_network is None else agent_network
        self.nll = nn.NLLLoss()

    def sample():
        # sample the agent
        return sequences, score

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.prior_network(x)
        prior_likelihood = self.nll(y_hat, y)
        aug_likelihood = prior_likelihood + self.sigma * 
        return

    def validation_step(self, batch, batch_idx):
        return
