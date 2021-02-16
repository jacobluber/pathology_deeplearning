#!/bin/bash

#SBATCH --nodes=16
#SBATCH --constraint=gpuk80
#SBATCH --gres=gpu:k80:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00               # Time limit hrs:min:sec
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --mem=30gb
#SBATCH --error=/home/luberjm/pl/code/benchmarking/best_distrib_1_k80.out
#SBATCH --output=/home/luberjm/pl/code/benchmarking/best_distrib_1_k80.out

function fail {
    echo "FAIL: $@" >&2
    exit 1  # signal failure
}

source /data/luberjm/conda/etc/profile.d/conda.sh || fail "conda load fail"
conda activate ml2 || fail "conda activate fail"
srun python /home/luberjm/pl/code/adjustments_distrib.py --batch-size 32 --epochs 20 --gpus 1 --nodes 16 --workers 8 --custom-coords-file /home/luberjm/pl/code/patch_coords.data --accelerator ddp --logging-name best_distrib_1_k80 --train-size 500000 --test-size 33500 --enc-dim 2048 --latent-dim 1024  --resnet resnet50 --read-coords || fail "python fail"

