#!/bin/bash
#SBATCH --job-name=allreduce_bench
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --constraint gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --account m4341_g
#SBATCH --time=00:30:00

#SBATCH --output=results/bench_%j.csv
#SBATCH --error=results/bench_%j.err

module purge
module load PrgEnv-gnu
module load cudatoolkit
module load cray-mpich
module load nccl

export NCCL_DEBUG=INFO
export FI_CXI_ATS=0

srun -u --cpus-per-task=32 --cpu-bind=cores ./benchmark