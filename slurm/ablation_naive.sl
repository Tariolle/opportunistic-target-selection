#!/bin/bash
#SBATCH -J "ablation_naive"
#SBATCH -o output.%J
#SBATCH -e error.%J
#SBATCH -p ar_a100
#SBATCH --gres=gpu:a100:1
#SBATCH -n 1
#SBATCH --cpus-per-gpu 4
#SBATCH --mem 16G
#SBATCH --time=08:00:00

# ImageNet validation data must be at data/imagenet/val/ before launching.
# Clone the repo and place data, then: sbatch slurm/ablation_naive.sl
# If wall-time is hit, re-submit: the script resumes from existing CSV results.

module purge
module load aidl/pytorch/2.6.0-cuda12.6
pip install --user -r requirements-hpc.txt

python benchmark_ablation_naive.py
