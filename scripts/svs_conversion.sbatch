#!/bin/bash 

#SBATCH -o /home/luberjm/myjob.out
#SBATCH -e /home/luberjm/myjob.err

function fail {
    echo "FAIL: $@" >&2
    exit 1  # signal failure
}

source /data/luberjm/conda/etc/profile.d/conda.sh || fail "conda load fail"
conda activate imaging || fail "imaging packages load fail"

PREFIX=${F%????} || fail "prefix fail"
vips openslideload --level 2 $F ${PREFIX}.jpg || fail "vips fail"
