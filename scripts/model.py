from __future__ import print_function
import argparse
import torch
import torch.utils.data
import os
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from PIL import ImageFile
import torchvision 


from typing import TypeVar
Tensor = TypeVar('torch.tensor')
#ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser(description='Pathology Images')
parser.add_argument('--batch-size', type=int, default=5, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
])

kwargs = {'num_workers': 20, 'pin_memory': True} if args.cuda else {}
input_data = torchvision.datasets.ImageFolder(root="/data/luberjm/data/training_set",transform=transform)
t1, t2 = torch.utils.data.random_split(input_data, [35784, 1100])
train_data, test_data = torch.utils.data.random_split(t2, [1000, 100])
#test_data = torchvision.datasets.ImageFolder(root="/data/luberjm/data/training_set",transform=transform)
trainl = len(train_data)
testl = len(test_data)
train_loader = torch.utils.data.DataLoader(train_data,batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_data,batch_size=args.batch_size, shuffle=True, **kwargs)

class AutoEncoder2(nn.Module):
    def __init__(self,
                 **kwargs) -> None:
        super(AutoEncoder2, self).__init__()

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

    def loss_function(self, recon_x: Tensor, x: Tensor, **kwargs) -> dict:
        loss = F.binary_cross_entropy(recon_x, x)
        return {'loss': loss}
    
    def generate(self, x: Tensor) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        """
        return self.forward(x)



model = AutoEncoder2().to(device)

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)


optimizer = optim.Adam(model.parameters(), lr=1e-3)

criterion = nn.BCELoss()

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch = model(data)
        loss = criterion(recon_batch,data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch  = model.generate(data)
            #test_loss += loss_function(recon_batch, data, mu, logvar).item()
            test_loss += criterion(recon_batch,data)
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([x[:num_sample], recon_x[:num_sample]])    
                #comparison = torch.cat([data[:n],
                #                      recon_batch.view(args.batch_size, 1, 262144, 262144)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            #sample = model.module.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')
