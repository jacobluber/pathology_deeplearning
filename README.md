# Imaging Differential Expression
## Command Options
usage: experiment.py [-h] [--batch-size N] [--epochs N] [--gpus N]
                     [--patches N] [--patch-size N] [--num-workers N]
                     [--read-coords] [--write-coords]

H&E Autoencoder

optional arguments:
  -h, --help       show this help message and exit
  --batch-size N   input batch size for training (default: 16)
  --epochs N       number of epochs to train (default: 5)
  --gpus N         number of GPUs to utilize per node (default: 4)
  --patches N      number of patches to sample per H&E image (default: 200)
  --patch-size N   size of the patch X*Y where x=patch_size and y=patch_size
                   (default: 512)
  --num-workers N  number of CPUs to use in the pytorch dataloader (default:
                   16)
  --read-coords    add this flag to read in previously sampled patch
                   coordinates that pass QC from the file 'patch_coords.data'
  --write-coords   add this flag to write out sampled coordinates that pass QC
                   to the file 'patch_coords.data', which can be preloaded to
                   speed up training

## Setup
1. Install XQuartz for mac (https://www.xquartz.org/), logout and log back into NIH laptop, then use the -Y flag when connecting via ssh to Biowulf. 
2. Install [conda](https://hpc.nih.gov/apps/python.html#envs) with settings storing envs on your biowulf /data folder so that large custom packages packages can be installed without root. 
3. Run `conda env create --name distribml --file=env.yml`.
4. Run `conda activate distribml`
5. For instructions, run `python experiment.py -h`

## DDP/NCCL Parallel 
Run `sbatch ddp.sbatch`. This requires much argument tuning at the moment.
