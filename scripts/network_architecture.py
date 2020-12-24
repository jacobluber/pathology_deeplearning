from __future__ import print_function
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn, optim
from utils import *

class AutoEncoder(pl.LightningModule):
    def __init__(self,**kwargs) -> None:
        super(AutoEncoder, self).__init__()
        modules = []
        in_channels = 3
        hidden_dims = [32, 64, 128, 256, 512] #[32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=5,
                              stride=1, #3
                              padding=0), #1
                    nn.ReLU(),
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        # Build Decoder
        modules = []
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=4,#4
                                       stride=1,#2
                                       padding=0,#1
                                       output_padding=0 ), #1
                    nn.ReLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3, #3
                                               stride=1, #2
                                               padding=1, #2 
                                               output_padding=0), #1
                            nn.ReLU(),
                            nn.Conv2d(hidden_dims[-1], 
                                       out_channels=3,
                                       kernel_size=3, #3
                                       padding= 5), #1
                            nn.Sigmoid())

    def encode(self, x: Tensor) -> Tensor:
        encoded = self.encoder(x)
        #print(encoded.shape)
        return encoded

    def decode(self, x: Tensor) -> Tensor:
        decoded = self.decoder(x)
        #print(decoded.shape)
        return decoded

    def forward(self, x: Tensor) -> Tensor:
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        out = self.final_layer(decoded)
        return out

    def latent_code(self, x: Tensor) -> Tensor:
        x = self.encode(x)
        return x.view(x.size(0), -1)
    
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        criterion = nn.BCELoss()
        z = self.encoder(batch)
        x_hat = self.decode(z)
        out = self.final_layer(x_hat)
        loss = criterion(out,batch)
        self.log('loss',loss,on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #print(batch.shape)
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def generate(self, x: Tensor) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        """
        return self.forward(x)

