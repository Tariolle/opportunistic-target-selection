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
# Install deps only if missing (git-based deps like robustbench re-clone every time otherwise).
# Run once on the login node first:
#   module load aidl/pytorch/2.6.0-cuda12.6
#   pip install --user -r requirements-hpc.txt
#   python -c "import torchvision; torchvision.models.resnet50(weights='IMAGENET1K_V1')"
python -c "import torchattacks, robustbench" 2>/dev/null || pip install --user -r requirements-hpc.txt

# Launch parallel workers on the same GPU, each handling a slice of images.
# Adjust N_WORKERS to trade GPU memory for speed (ResNet-50 is ~100MB per process).
N_WORKERS=${N_WORKERS:-10}
N_IMAGES=100
CHUNK=$(( (N_IMAGES + N_WORKERS - 1) / N_WORKERS ))

echo "Starting benchmark at $(date) with $N_WORKERS workers"
for i in $(seq 0 $((N_WORKERS - 1))); do
    START=$((i * CHUNK))
    END=$(( (i + 1) * CHUNK ))
    [ $END -gt $N_IMAGES ] && END=$N_IMAGES
    [ $START -ge $N_IMAGES ] && continue
    python -u benchmark_ablation_naive.py --image-start $START --image-end $END "$@" &
done
wait
echo "All workers finished at $(date)"
