#!/bin/bash
#SBATCH --job-name=probe_sweep
#SBATCH --partition=gpu-shared
#SBATCH --account=csd403
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/home/bpinto/work/adapting-fbm/logs/probe_%j.out
#SBATCH --error=/home/bpinto/work/adapting-fbm/logs/probe_%j.err

# Layer-wise linear-probing depth sweep (EXPERIMENTS.md Exp 4) on CBraMod native
# probe, dataset bcic2a. First de-risking run: env check + fast-dev-run smoke +
# a real accuracy-vs-depth sweep over encoder blocks.

REPO=/home/bpinto/work/adapting-fbm
DATA=/expanse/projects/nemar/kuntal/adapter_finetuning/data/luna_native_raw
cd "$REPO"
mkdir -p logs outputs

module purge
module load gpu
module load cuda12.2/toolkit/12.2.2
module load anaconda3/2021.05
source activate /expanse/projects/nemar/dtyoung/conda_envs/adapter-finetuning

export HF_HOME="/expanse/projects/nemar/dtyoung/huggingface_cache"
export MNE_DATA="/expanse/projects/nemar/dtyoung/mne_data"
export TMPDIR="/expanse/projects/nemar/eeg_finetuning/.cache/tmp"
export WANDB_MODE=offline
export PYTHONPATH="$REPO:$PYTHONPATH"

echo "=== node $(hostname) | $(date) ==="
python -c "import torch, braindecode; print('torch', torch.__version__, '| braindecode', braindecode.__version__, '| cuda', torch.cuda.is_available())"
python -c "from adapter_finetuning.probe_layer import LayerProbe; print('probe_layer import OK')"
python adapter_finetuning/probe_layer.py

echo; echo "=== SMOKE: fast-dev-run (CBraMod native probe, bcic2a, layer 6) ==="
python scripts/run_cbramod_experiments.py \
    --mode native --training-mode probe \
    --probe-layer encoder.layers.6 --probe-aggregation mean \
    --dataset bcic2a --start-seed 0 --n-seeds 1 \
    --native-dir "$DATA" --output-dir "$REPO/outputs" --fast-dev-run

echo; echo "=== DEPTH SWEEP: CBraMod native probe, bcic2a, layers {0,3,6,9,11} x 2 seeds ==="
for L in 0 3 6 9 11; do
  echo "----- probe encoder.layers.$L -----"
  python scripts/run_cbramod_experiments.py \
      --mode native --training-mode probe \
      --probe-layer "encoder.layers.$L" --probe-aggregation mean \
      --dataset bcic2a --start-seed 0 --n-seeds 2 \
      --native-dir "$DATA" --output-dir "$REPO/outputs"
done
echo; echo "=== DONE $(date) ==="
