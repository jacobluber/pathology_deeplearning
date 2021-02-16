#!/bin/bash

#SBATCH --nodes=1
#SBATCH --constraint=gpuv100x
#SBATCH --gres=gpu:v100x:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00               # Time limit hrs:min:sec
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --mem=30gb
#SBATCH --error=/home/luberjm/pl/code/benchmarking/bw/18_conv.out
#SBATCH --output=/home/luberjm/pl/code/benchmarking/bw/18_conv.out

function fail {
    echo "FAIL: $@" >&2
    exit 1  # signal failure
}

source /data/luberjm/conda/etc/profile.d/conda.sh || fail "conda load fail"
conda activate ml2 || fail "conda activate fail"
python /home/luberjm/pl/code/adjustments.py --batch-size 64 --epochs 10 --gpus 1 --nodes 1 --workers 8 --custom-coords-file /home/luberjm/pl/code/pc.data --accelerator gpu --logging-name 18_conv --train-size 1600 --test-size 400 --read-coords --first-conv || fail "python fail"

