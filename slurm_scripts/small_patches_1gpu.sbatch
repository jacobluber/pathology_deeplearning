#!/bin/bash

#SBATCH --nodes=1
#SBATCH --constraint=gpuv100x
#SBATCH --gres=gpu:v100x:1
#SBATCH --ntasks-per-node=4
#SBATCH --time=36:00:00               # Time limit hrs:min:sec
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --mem=30gb
#SBATCH --error=/home/luberjm/pl/code/benchmarking/small_patches_1gpu.out
#SBATCH --output=/home/luberjm/pl/code/benchmarking/small_patches_1gpu.out

function fail {
    echo "FAIL: $@" >&2
    exit 1  # signal failure
}

source /data/luberjm/conda/etc/profile.d/conda.sh || fail "conda load fail"
conda activate ml2 || fail "conda activate fail"
srun python /home/luberjm/pl/code/small_patches.py --batch-size 32 --epochs 30 --gpus 1 --nodes 1 --workers 8 --custom-coords-file /home/luberjm/pl/code/patch_coords.data --accelerator gpu --logging-name small_patches_1gpu --train-size 500000 --test-size 33500 --enc-dim 2048 --latent-dim 1024  --resnet resnet50 --read-coords || fail "python fail"

