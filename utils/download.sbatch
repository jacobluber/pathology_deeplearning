#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00               # Time limit hrs:min:sec
#SBATCH --cpus-per-task=8
#SBATCH --partition=ccr
#SBATCH --mem=60gb
#SBATCH --error=/data/luberjm/data/gbm/sbatch.out
#SBATCH --output=/data/luberjm/data/gbm/sbatch.out

function fail {
    echo "FAIL: $@" >&2
    exit 1  # signal failure
}

module load gdc-client/1.5.0
gdc-client download -m /data/luberjm/data/gbm/gdc_manifest_20210224_205720.txt
