#!/bin/bash

#SBATCH --constraint=gpup100
#SBATCH --nodes=2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=120gb
#SBATCH --time=00:20:00

function fail {
    echo "FAIL: $@" >&2
    exit 1  # signal failure
}

source /data/luberjm/conda/etc/profile.d/conda.sh || fail "conda load fail"
conda activate distrib || fail "conda activate fail"
module load nccl/2.7.8_cuda11.0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL
# on your cluster you might need these:
# set the network interface
export NCCL_SOCKET_IFNAME=^docker0,lo

# might need the latest cuda
# module load NCCL/2.4.7-1-cuda.10.0
# -------------------------

# run script from above
#export SLURM_JOB_NODELIST=$(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ')
#export SLURM_NODELIST=$SLURM_JOB_NODELIST
#slurm_nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
#export MASTER_ADDRESS=$(echo $slurm_nodes | cut -d' ' -f1)


srun python experiment.py --epochs 5 --gpus 2 --patches 500 --patch-size 512 --num-workers 32 --batch-size 4  --read-coords
