#!/bin/bash 

function fail {
    echo "FAIL: $@" >&2
    exit 1  # signal failure
}

#source /data/luberjm/conda/etc/profile.d/conda.sh || fail "conda load fail"
#conda activate imaging || fail "imaging packages load fail"

PREFIX=${1%????} || fail "prefix fail"
vips openslideload --level 0 ${1} ${PREFIX}.png || fail "vips fail"
