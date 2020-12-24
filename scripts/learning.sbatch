#!/bin/bash

#SBATCH --constraint=gpuk80
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=50g
#SBATCH --gres=gpu:k80:2
#SBATCH --time 30

function fail {
    echo "FAIL: $@" >&2
    exit 1  # signal failure
}

source /data/luberjm/conda/etc/profile.d/conda.sh || fail "conda load fail"
conda activate distrib || fail "conda activate fail"
module load nccl/2.7.8_cuda11.0
module load openmpi/4.0.1/cuda-10.1/gcc-7.4.0 openmpi/4.0.1/gcc-7.4.0
module load horovod/0.19.5

mpirun -np $SLURM_NTASKS \
-bind-to none -map-by slot -mca pml ob1 -mca btl ^openib -x NCCL_DEBUG=INFO -x CONDA_EXE='/data/luberjm/conda/bin/conda' -x CONDA_PYTHON_EXE='/data/luberjm/conda/bin/python' \
/data/luberjm/conda/envs/distrib/bin/python experiment.py --epochs 5 --gpus 2 --patches 500 --patch-size 512 --num-workers 8 --batch-size 4  --read-coords
