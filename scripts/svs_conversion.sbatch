#!/bin/bash 

#SBATCH -o /home/luberjm/myjob.out
#SBATCH -e /home/luberjm/myjob.err
#SBATCH --time=00:45:00               # Time limit hrs:min:sec

function fail {
    echo "FAIL: $@" >&2
    exit 1  # signal failure
}

source /data/luberjm/conda/etc/profile.d/conda.sh || fail "conda load fail"
conda activate imaging || fail "imaging packages load fail"

PREFIX=${F%????} || fail "prefix fail"
vips openslideload --level 0 $F ${PREFIX}.png || fail "vips fail"
