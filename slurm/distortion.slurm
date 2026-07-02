#!/bin/bash
#SBATCH --job-name=distortion
#SBATCH --partition=gpu-shared
#SBATCH --account=csd403
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:30:00
#SBATCH --output=/home/bpinto/work/adapting-fbm/logs/distort_%j.out
#SBATCH --error=/home/bpinto/work/adapting-fbm/logs/distort_%j.err

# Feature-distortion (EXPERIMENTS.md Exp 4 / paper V-B): on PhysioNet, SFT-train
# CBraMod native (produces a checkpoint), then probe that SFT checkpoint at each
# depth. Compare against the FROZEN PhysioNet curve from depth_curves.slurm:
# frozen_acc(L) - sft_acc(L) is the per-layer distortion.

REPO=/home/bpinto/work/adapting-fbm
DATA=/expanse/projects/nemar/kuntal/adapter_finetuning/data/luna_native_raw
OUT=$REPO/outputs/sft
cd "$REPO"; mkdir -p logs "$OUT"

module purge; module load gpu; module load cuda12.2/toolkit/12.2.2; module load anaconda3/2021.05
source activate /expanse/projects/nemar/dtyoung/conda_envs/adapter-finetuning
export HF_HOME=/expanse/projects/nemar/dtyoung/huggingface_cache
export MNE_DATA=/expanse/projects/nemar/dtyoung/mne_data
export TMPDIR=/expanse/projects/nemar/eeg_finetuning/.cache/tmp
export WANDB_MODE=offline
export PYTHONPATH="$REPO:$PYTHONPATH"

echo "=== $(hostname) | $(date) ==="
echo "=== [1/2] SFT-train CBraMod native on physionet (produces the checkpoint) ==="
python scripts/run_cbramod_experiments.py --mode native --training-mode sft \
    --dataset physionet --start-seed 0 --n-seeds 1 \
    --native-dir "$DATA" --output-dir "$OUT"

CKPT=$(ls -t "$OUT"/checkpoints/cbramod_native_sft_physionet_init0/*.ckpt 2>/dev/null | head -1)
echo "SFT checkpoint: $CKPT"
if [ -z "$CKPT" ]; then echo "ERROR: no SFT checkpoint produced"; exit 1; fi

echo "=== [2/2] probe the SFT checkpoint at each depth (physionet) ==="
for L in 0 2 4 6 8 10 11; do
  echo "===== SFT-PROBE physionet encoder.layers.$L ====="
  python scripts/run_cbramod_experiments.py --mode native --training-mode probe \
      --probe-layer "encoder.layers.$L" --probe-aggregation mean \
      --dataset physionet --start-seed 0 --n-seeds 2 \
      --init-from "$CKPT" \
      --native-dir "$DATA" --output-dir "$OUT"
done
find "$OUT" -name '*.ckpt' -delete 2>/dev/null
echo "=== DONE $(date) ==="
