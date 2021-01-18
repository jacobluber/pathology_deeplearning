from __future__ import print_function
from pl_bolts.models import VAE
import pytorch_lightning as pl
import PIL
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from utils import *
from datetime import datetime
import torchvision
from torchvision import datasets, transforms

class customVAE(VAE):
    def __init__(self, *args, **kwargs):
        super(customVAE, self).__init__(latent_dim=256,input_height=512,input_channels=3,*args, **kwargs)
        self.example_input_array = torch.rand(1, 3, 512, 512)
        self.test_outs = []
        self.image_count = 0
        self.time = datetime.now()        
        #self.writer = SummaryWriter('/data/luberjm/tb_logs')

    def training_epoch_end(self,output):
        now = datetime.now()
        delta = now - self.time
        self.time = now
        tensorboard_logs = {'time_secs_epoch': delta.seconds}
        self.log_dict(tensorboard_logs) 

    def reduce_image(self,batch,dim):
        for i in range(0,len(batch)-1):
            if i == 0:
                res = torch.cat((batch[i],batch[i+1]),dim)
            else:
                res = torch.cat((res,batch[i+1]),dim)
        return res

    #def test_epoch_end(self,output):
        #self.logger.experiment.add_image(str(self.image_count)+"test",self.test_outs,trainer.global_step)
        #self.image_count += 1
        #self.test_outs = []

    def test_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"test_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        #self.test_outs.append(loss)
        #if len(self.test_outs) == 0:
        #if True:
        #    x, y = batch
        #    z, x_hat, p, q = self._run_step(x)    
        #    orig = self.reduce_image(x,2)
        #    recon = self.reduce_image(x_hat,2)
        #    out = torch.cat((orig,recon),1)
        #    eout = out.expand(1,3,1024,4096)
        #    self.writer.add_images(str(datetime.now()),eout,0)
        #    #self.logger.experiment.add_image(str(self.image_count)+"test",self.global_step)
            #self.image_count += 1
            #self.test_outs = out
        #return loss
