#!/bin/bash
#SBATCH --job-name=probe
#SBATCH --partition=gpu-shared
#SBATCH --account=csd403
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/bpinto/work/adapting-fbm/logs/%x_%j.out
#SBATCH --error=/home/bpinto/work/adapting-fbm/logs/%x_%j.err

# Parametrized layer-probe job. Pass via sbatch --export:
#   MODEL    = cbramod | eegpt
#   KIND     = frozen  | distort         (distort = SFT-train then probe the SFT ckpt)
#   DATASETS = "bcic2a physionet tuev faced mdd_mumtaz2016"  (space-separated subset)
#   LAYERS   = "0 2 4 6 8 10 11"         (optional; default all-block sweep)
set -u
: "${MODEL:?set MODEL}"; : "${KIND:?set KIND}"; : "${DATASETS:?set DATASETS}"
LAYERS="${LAYERS:-0 2 4 6 8 10 11}"

case "$MODEL" in
  cbramod) SCRIPT=run_cbramod_experiments.py; PREFIX=encoder.layers;        NDIR=luna_native_raw ;;
  eegpt)   SCRIPT=run_eegpt_experiments.py;   PREFIX=target_encoder.blocks; NDIR=luna_native ;;
  luna)    SCRIPT=run_luna_experiments.py;    PREFIX=blocks;                NDIR=luna_native ;;
  *) echo "unknown MODEL=$MODEL"; exit 1 ;;
esac

REPO=/home/bpinto/work/adapting-fbm
DATA=/expanse/projects/nemar/kuntal/adapter_finetuning/data/$NDIR
OUT=$REPO/outputs/$KIND/$MODEL
cd "$REPO"; mkdir -p logs "$OUT"

module purge; module load gpu; module load cuda12.2/toolkit/12.2.2; module load anaconda3/2021.05
source activate /expanse/projects/nemar/dtyoung/conda_envs/adapter-finetuning
export HF_HOME=/expanse/projects/nemar/dtyoung/huggingface_cache
export MNE_DATA=/expanse/projects/nemar/dtyoung/mne_data
export TMPDIR=/expanse/projects/nemar/eeg_finetuning/.cache/tmp
export WANDB_MODE=offline
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "=== $(hostname) | MODEL=$MODEL KIND=$KIND DATASETS=[$DATASETS] | $(date) ==="
for D in $DATASETS; do
  INIT=""
  if [ "$KIND" = "distort" ]; then
    echo "===== [$MODEL/$D] SFT-train native (for the distortion checkpoint) ====="
    python scripts/$SCRIPT --mode native --training-mode sft --dataset "$D" \
        --start-seed 0 --n-seeds 1 --native-dir "$DATA" --output-dir "$OUT"
    CKPT=$(ls -t "$OUT"/checkpoints/*sft*"${D}"*/*.ckpt 2>/dev/null | head -1)
    echo "[$MODEL/$D] SFT checkpoint: $CKPT"
    [ -z "$CKPT" ] && { echo "WARN: no SFT ckpt for $MODEL/$D, skipping"; continue; }
    INIT="--init-from $CKPT"
  fi
  for L in $LAYERS; do
    echo "===== [$MODEL/$D/$KIND] probe $PREFIX.$L ====="
    python scripts/$SCRIPT --mode native --training-mode probe \
        --probe-layer "$PREFIX.$L" --probe-aggregation mean \
        --dataset "$D" --start-seed 0 --n-seeds 2 $INIT \
        --native-dir "$DATA" --output-dir "$OUT"
  done
done
find "$OUT" -name '*.ckpt' -delete 2>/dev/null
echo "=== DONE $(date) ==="
