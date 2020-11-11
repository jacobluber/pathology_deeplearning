#!/bin/bash 

#SBATCH -o ~/myjob.out
#SBATCH -e ~/myjob.err
source /data/luberjm/conda/etc/profile.d/conda.sh
conda activate imaging

PREFIX=${F%????}
vips openslideload --level 2 $F ${PREFIX}.jpg
