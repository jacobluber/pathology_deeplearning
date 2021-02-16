from __future__ import print_function
import torch
import pytorch_lightning as pl
from torch import nn, optim
from torch.nn import functional as F
from utils import *
from torchvision.utils import save_image

class VanillaVAE(pl.LightningModule):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 kl_coeff: float = 0.1, 
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()
        self.example_input_array = torch.rand(1, 3, 512, 512)
        self.latent_dim = latent_dim
        self.kl_coeff = kl_coeff

        modules = []

        # Build Encoder
        modules.append(
            nn.Sequential(
            nn.Conv2d(3,64,6,2,2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,64,6,2,2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,64,6,2,2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,64,6,2,2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,64,6,2,2),
            nn.ReLU(),
            nn.BatchNorm2d(64)))           
        in_channels = 3

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(16384, 4096)
        self.fc_var = nn.Linear(16384, 4096)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(4096, 16384)

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(64,64,6,2,2),
                nn.BatchNorm2d(64),
                nn.ReLU(),  
                nn.ConvTranspose2d(64,64,6,2,2),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.ConvTranspose2d(64,64,6,2,2),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.ConvTranspose2d(64,64,6,2,2),
                nn.BatchNorm2d(64),
                nn.ReLU()
            ))


        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            #nn.ConvTranspose2d(64,64,6,2,2),
            #nn.BatchNorm2d(64),
            #nn.ReLU(),
            nn.ConvTranspose2d(64,3,6,2,2),
            nn.Tanh())

    def encode(self, input1: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input1)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return mu, log_var

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 64, 16, 16)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(input)
        #z = self.reparameterize(mu, log_var)
        p,q,z = self.sample(mu,log_var)
        return  self.decode(z)
    
    #def forward(self, x:
    #    x = self.encoder(x)
    #    mu = self.fc_mu(x)
    #    log_var = self.fc_var(x)
    #    p, q, z = self.sample(mu, log_var)
    #    return self.decoder(z)

    def test_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"test_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        x, y = batch
        z, x_hat, p, q = self._run_step(x)
        orig = self.reduce_image(x,2)
        recon = self.reduce_image(x_hat,2)
        out = torch.cat((orig,recon),1)
        eout = out.expand(1,3,1024,4096)
        save_image(eout, 'img1.png')

    def _run_step(self, x):
        mu, logvar = self.encode(x)
        p, q, z = self.sample(mu, logvar)
        return z, self.decode(z), p, q

    def step(self, batch, batch_idx):
        x,y = batch
        z, x_hat, p, q = self._run_step(x)

        recon_loss = F.mse_loss(x_hat, x, reduction='mean')

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = log_qz - log_pz
        kl = kl.mean()
        kl *= self.kl_coeff

        loss = kl + recon_loss

        logs = {
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": loss,
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def reduce_image(self,batch,dim):
        for i in range(0,len(batch)-1):
            if i == 0:
                res = torch.cat((batch[i],batch[i+1]),dim)
            else:
                res = torch.cat((res,batch[i+1]),dim)
        return res
    
    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

