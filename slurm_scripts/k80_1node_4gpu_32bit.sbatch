#!/bin/bash

#SBATCH --nodes=1
#SBATCH --constraint=gpuk80
#SBATCH --gres=gpu:k80:4
#SBATCH --ntasks-per-node=4
#SBATCH --time=01:00:00               # Time limit hrs:min:sec
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --mem=100gb
#SBATCH --error=/home/luberjm/pl/code/benchmarking/k80_1node_4gpu_32bit.out
#SBATCH --output=/home/luberjm/pl/code/benchmarking/k80_1node_4gpu_32bit.out

function fail {
    echo "FAIL: $@" >&2
    exit 1  # signal failure
}

source /data/luberjm/conda/etc/profile.d/conda.sh || fail "conda load fail"
conda activate ml2 || fail "conda activate fail"
python /home/luberjm/pl/code/experiment.py --batch-size 3 --epochs 2 --gpus 4 --nodes 1 --workers 32 --custom-coords-file /home/luberjm/pl/code/pc.data --accelerator dp --logging-name k80_1node_4gpu_32bit --train-size 1600 --test-size 400 --read-coords || fail "python fail"

