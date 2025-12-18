#!/bin/bash
#SBATCH --account=a-infra01
#SBATCH --nodes=1
#SBATCH --environment=vllm
#SBATCH --container-writable
#SBATCH --partition=debug
#SBATCH --time=1:00:00

source ~/.bashrc

USERNAME=$USER
JOB_ID=$SLURM_JOB_ID

export VLLM_CONFIG_ROOT="/iopsstor/scratch/cscs/rmachace/config/vllm"
export TRANSFORMERS_CACHE="/iopsstor/scratch/cscs/rmachace/cache/transformers"
export HF_HOME="/iopsstor/scratch/cscs/rmachace/cache/hf"
export VLLM_CACHE_ROOT="/iopsstor/scratch/cscs/rmachace/cache/vllm"

cd /iopsstor/scratch/cscs/rmachace/repos/code-reason

# Install relevant packages
pip install coverage
pip install pytest

python src/synthesizer.py augment --model Qwen/Qwen2.5-Coder-3B-Instruct --seed_path assets/example/seeds.json