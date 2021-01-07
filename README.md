# Imaging Differential Expression
## Usage
```
usage: argdown [-h] [--batch-size N] [--epochs N] [--gpus N] [--patches N]
               [--patch-size N] [--num-workers N] [--read-coords]
               [--write-coords]
```
## Arguments
### Quick reference table
|Short|Long            |Default|Description                                                                                                                             |
|-----|----------------|-------|----------------------------------------------------------------------------------------------------------------------------------------|
|`-h` |`--help`        |       |show this help message and exit                                                                                                         |
|     |`--batch-size`  |`16`   |input batch size for training (default: 16)                                                                                             |
|     |`--epochs`      |`5`    |number of epochs to train (default: 5)                                                                                                  |
|     |`--gpus`        |`4`    |number of GPUs to utilize per node (default: 4)                                                                                         |
|     |`--patches`     |`200`  |number of patches to sample per H&E image (default: 200)                                                                                |
|     |`--patch-size`  |`512`  |size of the patch X*Y where x=patch_size and y=patch_size (default: 512)                                                                |
|     |`--num-workers` |`16`   |number of CPUs to use in the pytorch dataloader (default: 16)                                                                           |
|     |`--read-coords` |       |add this flag to read in previously sampled patch coordinates that pass QC from the file 'patch_coords.data'                            |
|     |`--write-coords`|       |add this flag to write out sampled coordinates that pass QC to the file 'patch_coords.data', which can be preloaded to speed up training|

### `-h`, `--help`
show this help message and exit

### `--batch-size` (Default: 16)
input batch size for training (default: 16)

### `--epochs` (Default: 5)
number of epochs to train (default: 5)

### `--gpus` (Default: 4)
number of GPUs to utilize per node (default: 4)

### `--patches` (Default: 200)
number of patches to sample per H&E image (default: 200)

### `--patch-size` (Default: 512)
size of the patch X*Y where x=patch_size and y=patch_size (default: 512)

### `--num-workers` (Default: 16)
number of CPUs to use in the pytorch dataloader (default: 16)

### `--read-coords`
add this flag to read in previously sampled patch coordinates that pass QC
from the file 'patch_coords.data'

### `--write-coords`
add this flag to write out sampled coordinates that pass QC to the file
'patch_coords.data', which can be preloaded to speed up training

## Setup
1. Install XQuartz for mac (https://www.xquartz.org/), logout and log back into NIH laptop, then use the -Y flag when connecting via ssh to Biowulf. 
2. Install [conda](https://hpc.nih.gov/apps/python.html#envs) with settings storing envs on your biowulf /data folder so that large custom packages packages can be installed without root. 
3. Run `conda env create --name distribml --file=env.yml`.
4. Run `conda activate distribml`
5. For instructions, run `python experiment.py -h`

## DDP/NCCL Parallel 
Run `sbatch ddp.sbatch`. This requires much argument tuning at the moment. Installing all of the conda dependencies can be time consuming, it is recommended to use mamba to speed this up. The NCCL/DDP/pytorch lighning output is complex and difficult to determine if working correctly, example of what working output should look like is provided in the example_ddp.out file. 
