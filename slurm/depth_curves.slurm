#!/bin/bash
#SBATCH --job-name=depth_curves
#SBATCH --partition=gpu-shared
#SBATCH --account=csd403
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/bpinto/work/adapting-fbm/logs/depth_%j.out
#SBATCH --error=/home/bpinto/work/adapting-fbm/logs/depth_%j.err

# Accuracy-vs-depth (EXPERIMENTS.md Exp 4): CBraMod native FROZEN probe on
# PhysioNet + MDD (both well above chance), all encoder blocks, 3 seeds.

REPO=/home/bpinto/work/adapting-fbm
DATA=/expanse/projects/nemar/kuntal/adapter_finetuning/data/luna_native_raw
OUT=$REPO/outputs/frozen
cd "$REPO"; mkdir -p logs "$OUT"

module purge; module load gpu; module load cuda12.2/toolkit/12.2.2; module load anaconda3/2021.05
source activate /expanse/projects/nemar/dtyoung/conda_envs/adapter-finetuning
export HF_HOME=/expanse/projects/nemar/dtyoung/huggingface_cache
export MNE_DATA=/expanse/projects/nemar/dtyoung/mne_data
export TMPDIR=/expanse/projects/nemar/eeg_finetuning/.cache/tmp
export WANDB_MODE=offline
export PYTHONPATH="$REPO:$PYTHONPATH"

echo "=== $(hostname) | $(date) ==="
for D in physionet mdd_mumtaz2016; do
  for L in 0 2 4 6 8 10 11; do
    echo "===== FROZEN $D encoder.layers.$L ====="
    python scripts/run_cbramod_experiments.py --mode native --training-mode probe \
        --probe-layer "encoder.layers.$L" --probe-aggregation mean \
        --dataset "$D" --start-seed 0 --n-seeds 3 \
        --native-dir "$DATA" --output-dir "$OUT"
  done
done
# free the intermediate checkpoints (results are in the logs); keeps Lustre quota in check
find "$OUT" -name '*.ckpt' -delete 2>/dev/null
echo "=== DONE $(date) ==="
