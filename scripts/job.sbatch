#!/bin/bash

#SBATCH --time=24:00:00               # Time limit hrs:min:sec
#SBATCH --cpus-per-task=50
#SBATCH --mem=100gb

function fail {
    echo "FAIL: $@" >&2
    exit 1  # signal failure
}

source /data/luberjm/conda/etc/profile.d/conda.sh || fail "conda load fail"
conda activate ml || fail "conda activate fail"
python experiment.py --write-coords --patches 500 --num-workers 49 || fail "python fail"
