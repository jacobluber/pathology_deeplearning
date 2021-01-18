#!/bin/bash

#SBATCH --nodes=1
#SBATCH --constraint=gpuk80
#SBATCH --gres=gpu:k80:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00               # Time limit hrs:min:sec
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --mem=100gb

function fail {
    echo "FAIL: $@" >&2
    exit 1  # signal failure
}

source /data/luberjm/conda/etc/profile.d/conda.sh || fail "conda load fail"
conda activate ml2 || fail "conda activate fail"
python experiment.py --batch-size 4 --epochs 2 --gpus 1 --nodes 1 --workers 8 --custom-coords-file /home/luberjm/pl/code/pc.data --accelerator gpu --logging-name k80_1node_1gpu_32bit --train-size 1600 --test-size 400 --read-coords || fail "python fail"

