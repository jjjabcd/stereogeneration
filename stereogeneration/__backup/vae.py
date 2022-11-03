import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import torchmetrics
import pytorch_lightning as pl


class VAE(pl.LightningModule):
    def __init__(self, 
        latent_dim: int, len_alphabet: int, len_molecule: int,
        enc_n_layers: int, enc_hidden_dim: int, enc_dropout: float,
        dec_n_layers: int, dec_hidden_dim: int, dec_dropout: float,
        target_dim: int, 
        padding_idx: int, 
        learning_rate: float, kld_beta: float,
        verbose: bool = True,
        **kwargs):
        super().__init__()

        self.encoder = MLPEncoder(latent_dim, len_alphabet, len_molecule,
            enc_n_layers, enc_hidden_dim).to(self.device)
        self.decoder = GRUDecoder(latent_dim, len_alphabet, len_molecule,
            dec_n_layers, dec_hidden_dim).to(self.device)

        self.predictor = MLPPredictor(latent_dim, target_dim)
        self.pred_metric = torchmetrics.MeanSquaredError()

        self.learning_rate = learning_rate
        self.kld_beta = kld_beta

        self.train_vae_acc = torchmetrics.Accuracy()
        self.val_vae_acc = torchmetrics.Accuracy()
        self.kld = torchmetrics.MeanMetric()

        # padding_idx = -100 if padding_idx is None else padding_idx
        self.vae_metric = nn.CrossEntropyLoss()         # takes in logits (before softmax)

        self.verbose = verbose
        self.save_hyperparameters()

    @staticmethod
    def _vae_kld(mu, log_var):
        return -0.5 * torch.mean(1. + log_var - mu.pow(2) - log_var.exp())

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        z, mu, log_var = self.encoder(x)
        logits = self.decoder(z)            # (batch, len_molecule, len_alphabet)

        # calculate loss
        x_pred = logits.reshape(-1, logits.size(2))    # (batch*len_molecule, len_alphabet)
        target = x.reshape(-1, logits.size(2))                  # (batch*len_molecule)
        kld = self._vae_kld(mu, log_var)
        vae_loss = self.vae_metric(x_pred, target) + self.kld_beta * kld
        self.kld.update(kld)

        # calculate accuracy
        x_pred = x_pred.softmax(dim=-1)     # softmax into probabilities for last dimension
        self.train_vae_acc(x_pred, target.argmax(dim=-1))
        self.log(f'train_accuracy', self.train_vae_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        pred = self.predictor(z)
        pred_loss = self.pred_metric(pred, y)
        self.log(f'train_pred_loss', pred_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'train_vae_loss', vae_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        loss = vae_loss + pred_loss

        self.log(f'train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z, mu, log_var = self.encoder(x)
        logits = self.decoder(z)

        # calculate loss
        x_pred = logits.reshape(-1, logits.size(2))
        target = x.reshape(-1, logits.size(2))
        vae_loss = self.vae_metric(x_pred, target) + self.kld_beta * self._vae_kld(mu, log_var)

        # calculate accuracy
        x_pred = x_pred.softmax(dim=-1)
        self.val_vae_acc(x_pred, target.argmax(dim=-1))
        self.log(f'val_accuracy', self.val_vae_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # do prediction if model is specified
        pred = self.predictor(z)
        pred_loss = self.pred_metric(pred, y)
        self.log(f'val_pred_loss', pred_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'val_vae_loss', vae_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        loss = vae_loss + pred_loss

        self.log(f'val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    # print results after epoch ends
    def training_epoch_end(self, training_step_outputs):
        if not self.verbose:
            return
        avg_loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()
        self.print(f'Epoch: {self.trainer.current_epoch}        training loss: {avg_loss}    KLD: {self.kld.compute()}')
        self.kld.reset()
        # self.print(f'KLD_BETA: {self.kld_beta}')
    
    def validation_epoch_end(self, validation_step_outputs):
        if not self.verbose:
            return
        avg_loss = torch.stack(validation_step_outputs).mean()
        self.print(f'Epoch: {self.trainer.current_epoch}        validation loss: {avg_loss}')



## Models for encoder

class MLPEncoder(nn.Module):
    def __init__(self, latent_dim: int, len_alphabet: int, len_molecule: int,
        enc_n_layers: int, enc_hidden_dim: int, **kwargs):
        super(MLPEncoder, self).__init__()
        self.encode_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(len_molecule*len_alphabet, enc_hidden_dim),
            nn.ReLU(),
            nn.Linear(enc_hidden_dim, enc_hidden_dim),
        )
        self.encode_mu = nn.Linear(enc_hidden_dim, latent_dim)
        self.encode_log_var = nn.Linear(enc_hidden_dim, latent_dim)
    
    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        hidden = self.encode_mlp(x)
        mu = self.encode_mu(hidden)
        log_var = self.encode_log_var(hidden)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var


## Models for decoder

class GRUDecoder(nn.Module):

    def __init__(self, latent_dim: int, len_alphabet: int, len_molecule: int,
        dec_n_layers: int, dec_hidden_dim: int, **kwargs):
        super(GRUDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.len_molecule = len_molecule
        self.len_alphabet = len_alphabet
        self.dec_n_layers = dec_n_layers

        # Simple Decoder
        self.decode_gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=len_alphabet,
            num_layers=dec_n_layers,
            batch_first=True
        )

    def forward(self, z):
        z = z.unsqueeze(1)      # add a time dimension
        z = z.repeat([1, self.len_molecule, 1])
        decoded, hidden = self.decode_gru(z)
        return decoded


class MLPPredictor(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int, **kwargs):
        super(MLPPredictor, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Simple Decoder
        self.layers = nn.Sequential(
            nn.Linear(latent_dim, 10),
            nn.Tanh(),
            nn.Linear(10, output_dim)
        )

    def forward(self, z):
        return self.layers(z)


