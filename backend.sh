#!/bin/bash
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --partition=debug
#SBATCH --time=1:00:00

source ~/.bashrc

JOB_ID=$SLURM_JOB_ID

export VLLM_CONFIG_ROOT="/iopsstor/scratch/cscs/rmachace/config/vllm"
export TRANSFORMERS_CACHE="/iopsstor/scratch/cscs/rmachace/cache/transformers"
export HF_HOME="/iopsstor/scratch/cscs/rmachace/cache/hf"
export VLLM_CACHE_ROOT="/iopsstor/scratch/cscs/rmachace/cache/vllm"

cd /iopsstor/scratch/cscs/rmachace/codegym

python src/backend/main.py