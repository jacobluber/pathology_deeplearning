from __future__ import print_function
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split, distributed
from torchvision import transforms
from pytorch_lightning.loggers import TensorBoardLogger
from utils import *
from network_architecture import *
from dataloader import *
import argparse
import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl


parser = argparse.ArgumentParser(description='H&E Autoencoder')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 16)')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 5)')
#parser.add_argument('--seed', type=int, default=42, metavar='S',
#                    help='random seed (default: 42)')
parser.add_argument('--gpus',type=int,default=4,metavar='N',help='number of GPUs to utilize per node (default: 4)')
parser.add_argument('--patches',type=int,default=200,metavar='N',help='number of patches to sample per H&E image (default: 200)')
parser.add_argument('--patch-size',type=int,default=512,metavar='N',help='size of the patch X*Y where x=patch_size and y=patch_size (default: 512)')
parser.add_argument('--num-workers',type=int,default=16,metavar='N',help='number of CPUs to use in the pytorch dataloader (default: 16)')
parser.add_argument('--read-coords',dest='read_coords',action='store_true',help='add this flag to read in previously sampled patch coordinates that pass QC from the file \'patch_coords.data\'')
parser.add_argument('--write-coords', dest='write_coords', action='store_true',help='add this flag to write out sampled coordinates that pass QC to the file \'patch_coords.data\', which can be preloaded to speed up training')
#parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                    help='how many batches to wait before logging training status')
args = parser.parse_args()

kwargs = {'batch_size':args.batch_size,'pin_memory':True,'num_workers':args.num_workers}

if __name__ == '__main__':
    input_data = SvsDatasetFromFolder("/data/luberjm/data/small/svs",args.patch_size,args.patches,args.num_workers,args.write_coords,args.read_coords)
    loader = torch.utils.data.DataLoader(input_data,  **kwargs)
    tb_logger = TensorBoardLogger('../tb_logs', name='autoencoder')
    trainer = pl.Trainer(max_epochs=args.epochs, replace_sampler_ddp=True, gpus=args.gpus,logger=tb_logger,num_nodes=2,accelerator='ddp')
    autoencoder = AutoEncoder()
    trainer.fit(autoencoder, loader)
