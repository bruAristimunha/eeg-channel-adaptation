# Confirmatory Experiments

Design for experiments that **causally confirm** the thesis of *"EEG Foundation Models Need Input Adaptation, but Fixed Channel Mappings Often Suffice"*, built to run on the existing **`eeg-channel-adaptation`** harness (the `run_*_experiments.py` scripts).

**Thesis under test.** A model's best channel adapter is the one that *re-enters* its pretrained channel computation — flexible backbones win natively **because their internal channel handler learned a montage→representation map during pretraining** ("preserve, don't overwrite"), and full SFT hurts by *distorting* an already-aligned representation, not by conflicting with an external adapter.

**Harness constraint (important).** The harness is **downstream-only**: it loads pretrained weights (`braindecode` + HuggingFace) and runs probe/SFT with each adapter. It does **not** pretrain. The paper's "matched-pretraining front-end swap" therefore cannot run here (it needs a pretraining pipeline). The three experiments below instead **intervene on the loaded models** and confirm the same thesis with the machinery the harness already has.

**Instrumentation — reuse of `probe_layer` ([facebookresearch/neuroai#83](https://github.com/facebookresearch/neuroai/pull/83)).** The layer-wise linear-probing mechanism merged there — `DownstreamWrapper.probe_layer` taps any submodule's output via a forward hook and runs the probe on it (Alain & Bengio 2016), with no backbone edits and safe hook cleanup across CV folds — is the natural instrument for this thesis. It makes the "layer contribution" measurable: Experiment 4 uses it to turn *feature distortion* from an aggregate number into a per-layer **accuracy-vs-depth** curve, and to read off the channel module's contribution directly (sharpening Experiments 1–2).

---

## 0. Verified channel-handling module handles

Introspected from **braindecode 1.6.1** (`model.named_modules()`), so the ablation in Experiment 1 targets the exact submodule that carries each model's learned channel prior:

| Model | Wins natively? | Channel-prior module (attribute path) | Type | #params |
|---|---|---|---|---|
| **EEGPT** | yes (3/5) | `target_encoder.chan_embed` | `nn.Embedding` | 31,744 |
| **CBraMod** | yes (native/SSI 4/5) | `patch_embedding.positional_encoding` | `nn.Sequential` (ACPE) | 26,800 |
| **LUNA** | **no** (control) | `channel_location_embedder` (+ bare `cross_attn.query_embed`) | `_Mlp` / `nn.Parameter` | 16,576 |
| BENDR / Neuro-GPT | n/a (rigid) | channel handling *is* the first conv of `encoder` — not a separable module → **not** an ablation target | — | — |

LUNA is the **control/exception**: the paper argues its bottleneck carries *no* fixed montage prior (it re-attends), and it does not win natively — so ablating its channel module should have a *smaller* effect than for EEGPT/CBraMod. That contrast is itself a prediction.

### Shared helper (drop into each runner or a small `ablation.py`)

```python
import torch
import torch.nn as nn

@torch.no_grad()
def reinit_(module: nn.Module):
    """Re-initialize `module` from scratch, erasing its pretrained state.
    Handles standard layers (reset_parameters) and bare learned Parameters."""
    for sub in module.modules():
        if hasattr(sub, "reset_parameters"):
            sub.reset_parameters()
    # LUNA only: cross_attn holds bare Parameters with no reset_parameters
    for pname, p in module.named_parameters():
        if pname.endswith(("query_embed", "temperature")):
            nn.init.normal_(p, std=0.02)

CHANNEL_PRIOR = {          # model_name -> submodule path(s) carrying the montage prior
    "eegpt":   ["target_encoder.chan_embed"],
    "cbramod": ["patch_embedding.positional_encoding"],
    # LUNA needs BOTH: the coord MLP and the learned-query bottleneck. Resetting
    # `cross_attn` (not just `query_embed`) lets reinit_ reach its bare Parameters.
    "luna":    ["channel_location_embedder", "cross_attn"],
}

def get_submodule(model, path):
    m = model
    for p in path.split("."):
        m = getattr(m, p)
    return m
```

---

## Experiment 1 — Pretrained-prior ablation  *(PRIMARY: confirms "preserve, don't overwrite")*

**Hypothesis.** A flexible model wins natively **because** its channel handler carries a montage→representation map learned in pretraining. Reset that handler to its from-scratch init (keep the rest of the pretrained encoder), and the native advantage should collapse to external-adapter level.

**Harness hook.** In `run_eegpt` / `run_cbramod` / `run_luna`, right after the model is built (e.g. `run_eegpt_experiments.py:294  self.model = EEGPT(...)`) and *before* the probe-freeze block (`run_eegpt:316`, `run_cbramod:320`, `run_luna:391`):

```python
# inside the LightningModule __init__, after self.model = <Model>(...)
if ablate_channel_prior:
    for path in CHANNEL_PRIOR[model_name]:
        reinit_(get_submodule(self.model, path))
    log.info("ABLATION: re-initialized %s", CHANNEL_PRIOR[model_name])
```

Add the CLI flag next to the existing args (`--mode`, `--training-mode`, `--dataset`, …):

```python
parser.add_argument("--ablate-channel-prior", action="store_true",
                    help="Re-init the pretrained channel handler (confirmatory ablation).")
```
and thread it into the LightningModule (`ablate_channel_prior=args.ablate_channel_prior`).

**Run** (native probe; intact arm you already have, ablated arm is new; all datasets, 15 seeds):

```bash
for M in eegpt cbramod luna; do
  for D in bcic2a physionet tuev faced mdd_mumtaz2016; do
    python scripts/run_${M}_experiments.py --mode native --training-mode probe \
        --dataset $D --start-seed 0 --n-seeds 15 --ablate-channel-prior
  done
done
```

**Prediction (falsifiable).** Per model, at probe:

| Arm | EEGPT / CBraMod (native winners) | LUNA (control) |
|---|---|---|
| intact native | high (published) | already ≈ external methods |
| **ablated** native | **drops to ≈ best external adapter** (SSI/Conv1d) | **little change** |

Confirms the thesis if destroying the pretrained prior **erases** the native advantage for EEGPT/CBraMod while barely moving LUNA. **Refutes** it if ablated-native still ≈ intact-native (the advantage would then be architectural, not the pretrained computation).

**Read-out.** Report `Δ = intact_native − ablated_native` per (model, dataset); compare against `intact_native − best_external_adapter`. The thesis predicts `Δ ≈ (intact_native − best_external)` for EEGPT/CBraMod and `Δ ≈ 0` for LUNA.

**Compute.** 3 models × 5 datasets × 15 seeds, **frozen probe only** (cheap); the intact arm is already computed.

---

## Experiment 2 — Distortion vs. conflict  *(confirms §V-B: probe > SFT is feature distortion)*

**Hypothesis.** Full SFT degrades the **native** path via feature distortion (Kumar et al. 2022; Lee et al. 2023), not competition with an external adapter. Gentler tuning should **recover** the loss on the exact native-path cells that regressed — EEGPT native PhysioNet `53.6→34.0`, TUEV `35.2→19.5`.

**Harness hook.** Add options to `--training-mode` at the freeze branch (`run_eegpt:316`, `run_cbramod:320`, `run_luna:391`):

- **`lp-ft`** — two phases: (1) probe (freeze encoder, train the head to convergence — reuse the existing probe loop), then (2) unfreeze all at `lr=1e-5` for a few epochs.
- **`surgical`** — unfreeze only the last transformer block + head (`for p in model.parameters(): p.requires_grad=False`, then re-enable the last block).
- **`lora`** — the PEFT path the package already advertises (`LoraConfig`, `apply_peft_to_model`), once `adapter_finetuning/adapters.py` is restored; else a minimal LoRA wrap on attention/FF.

**Run** (native mode, on the models/datasets that regressed under full SFT):

```bash
for TM in lp-ft surgical lora; do
  for D in physionet tuev faced; do
    python scripts/run_eegpt_experiments.py  --mode native --training-mode $TM --dataset $D --start-seed 0 --n-seeds 15
    python scripts/run_cbramod_experiments.py --mode native --training-mode $TM --dataset $D --start-seed 0 --n-seeds 15
  done
done
```

**Prediction.** LP-FT / surgical / LoRA recover **most of the full-SFT loss on the native path** (where no external adapter exists). Confirms the loss was distortion, not conflict. **Refutes** §V-B if they do *not* help the native path.

*(This is exactly the follow-up the paper's Limitations name — "LoRA, DoRA, and LP-FT may mitigate the negative transfer we observe.")*

**Sharper version → Experiment 4.** The aggregate probe-vs-SFT gap says *that* distortion happens; the `probe_layer` accuracy-vs-depth curve (Experiment 4) says *where* — which is the mechanistic evidence for §V-B.

---

## Experiment 3 — Permutation invariance  *(optional, ~free: confirms "flexible = set/attention", §V-A)*

**Hypothesis.** Flexible backbones treat channels as an (approximately) permutation-invariant set; rigid convolutions do not.

**Harness hook.** A **test-time-only** intervention (no training): shuffle channel order before the forward pass. Add `--permute-channels-eval` and, in the test `DataLoader`/collate, apply a fixed random channel permutation (and the matching permutation to `channel_locations`/positions where the model consumes them).

**Prediction.** Native flexible models show `Δacc ≈ 0` under permutation; rigid conv models (BENDR, Neuro-GPT) show a **large** `Δacc` unless the learned adapter re-maps. A one-line demonstration of the mechanism.

---

## Experiment 4 — Accuracy-vs-depth distortion probe  *(reuses `probe_layer`, neuroai #83)*

**Why it fits.** PR #83's `DownstreamWrapper.probe_layer` taps any submodule's output through a forward hook and feeds it to the aggregation + linear-probe chain — exactly the "layer-contribution" read-out this thesis needs. It (a) *locates* the montage prior (Experiment 1) and (b) *measures* feature distortion per layer (Experiment 2), instead of inferring either from a single aggregate accuracy.

**Hypothesis.** Full SFT on a small downstream set distorts pretrained features, and the distortion is **localized** — worst at the channel front-end and early encoder blocks. A frozen model's accuracy-vs-depth curve should sit **above** the fine-tuned model's at those depths; LP-FT should preserve it.

**Protocol.** For each backbone, sweep `probe_layer` across the encoder blocks (and the channel module of §0) on three checkpoints — *frozen pretrained*, *full-SFT*, *LP-FT* — and plot balanced accuracy vs. depth.

- **Path A (no new code — if the backbone is registered in `neuralbench`; BENDR is):** run it with the merged field via a grid override.
  ```yaml
  downstream_model_wrapper.probe_layer:  [encoder.encoder.Encoder_0, ..., encoder.encoder.Encoder_N]
  downstream_model_wrapper.aggregation:  flatten
  downstream_model_wrapper.layers_to_unfreeze: [""]     # frozen backbone = pure probe
  ```
- **Path B (port the pattern into `eeg-channel-adaptation`):** add `--probe-layer PATH`; reuse the exact mechanism from #83 — `model.get_submodule(path).register_forward_hook(...)` → aggregate (flatten/mean, canonicalise `(T,B,D)`→batch-first) → linear head — **including the `weakref.finalize` hook cleanup** so a shared backbone doesn't accumulate stale hooks across seeds/folds.

**Prediction.** frozen curve **≥** SFT curve at early/mid depths, gap largest for EEGPT; LP-FT curve **≈** frozen. And for Experiment 1: the accuracy *jump across the channel module* vanishes after ablation. **Refutes** §V-B if SFT never lowers the curve at any depth (no distortion), or Experiment 1 if the channel-module jump is unchanged by ablation.

**Cost.** Linear probes on frozen activations — cheap; one accuracy-vs-depth curve per (model, checkpoint, dataset). The frozen-checkpoint curve doubles as the Experiment-1 read-out.

### Path B — drop-in wiring (validated on braindecode 1.6.1)

A self-contained `LayerProbe` (see **`probe_layer.py`**, shipped next to this file → copy to `adapter_finetuning/probe_layer.py`) implements the #83 mechanism: frozen backbone + forward-hook tap + a linear head sized eagerly (Lightning builds the optimizer before the first forward, so a `LazyLinear` head would be skipped), with `weakref.finalize` hook cleanup so a shared backbone doesn't leak hooks across the 15 seeds. It has a runnable self-check (`python probe_layer.py`), was validated live on EEGPT/CBraMod, and was hardened after a code review: it keeps the backbone in `eval()` under Lightning's per-epoch `train()`, **auto-detects** the batch axis (batch-first / sequence-first / batch-entangled), rejects batch-independent layers, and guards non-tensor captures, multi-fire hooks, and CPU/CUDA head placement.

**Validated probe targets** (accuracy-vs-depth, Exp 4):

| Backbone | depth-probe paths | aggregation | probe-head size |
|---|---|---|---|
| EEGPT | `target_encoder.blocks.0 … blocks.11` | **`mean`** | 512 |
| CBraMod | `encoder.layers.0 … layers.11` | **`mean`** | 200 |

Use `--probe-aggregation mean` for transformer blocks (EEGPT entangles `batch×tokens` in dim 0; `mean` pools to a small head — `flatten` gives a 365k-d head that overfits). The **channel modules** (`target_encoder.chan_embed`, `patch_embedding.positional_encoding`) are **ablation** targets (Exp 1), *not* probe targets — `chan_embed`'s output is batch-independent and `LayerProbe` rejects it with a clear error.

**Four edits per runner** (line anchors from `run_eegpt_experiments.py`; the others match):

1. **Import** (top of file): `from adapter_finetuning.probe_layer import LayerProbe`
2. **CLI** (`main()`, near `:630`) and thread it through the experiment fn into the module constructor (call at `:543`):
   ```python
   parser.add_argument("--probe-layer", type=str, default=None,
       help="Tap this submodule and train a linear probe on it (frozen backbone).")
   parser.add_argument("--probe-aggregation", type=str, default="flatten",
       choices=["flatten", "mean"])
   ```
3. **Module `__init__`** (signature at `:273` — add `probe_layer=None, probe_aggregation="flatten"`), just after the freeze block (`:316`):
   ```python
   self.probe = None
   if probe_layer is not None:
       ex = torch.zeros(2, n_chans, n_times)       # LUNA: also pass its position inputs
       self.probe = LayerProbe(self.model, n_outputs, probe_layer,
                               example_inputs=(ex,), aggregation=probe_aggregation)
   ```
4. **`forward`** (`:371`) — route through the probe and pass through any extra model inputs (LUNA's positions):
   ```python
   def forward(self, x, *rest):
       return self.probe(x, *rest) if self.probe is not None else self.model(x, *rest)
   ```

No optimizer change — `configure_optimizers` already trains `[p for p in self.parameters() if p.requires_grad]`, which is exactly the probe head (backbone frozen). **Ensure `training_step`/`validation_step` route through `self(...)` (this `forward`), not `self.model(...)` directly** — otherwise the probe head never gets gradients and every depth point reports chance. The batch axis is auto-detected (batch-first, sequence-first `(T,B,D)`, and batch-entangled taps all handled), so no `batch_dim` wiring is needed. Keep `--probe-aggregation mean` for transformer blocks (`flatten` builds a ~365k-d head that overfits).

**Run** (accuracy-vs-depth on the frozen pretrained model):
```bash
for L in 0 2 4 6 8 10; do
  python scripts/run_cbramod_experiments.py --mode native --training-mode probe \
      --probe-layer encoder.layers.$L --probe-aggregation mean \
      --dataset physionet --start-seed 0 --n-seeds 5
done
```

For the **before/after-SFT distortion curve** (§V-B), you need one extra hook the harness doesn't yet have: load an SFT-finetuned checkpoint into the backbone before probing (add `--init-from CKPT` in `_load_pretrained`), then run the same sweep — the frozen-pretrained curve minus the SFT curve *is* the per-layer distortion.

---

## Out of scope here: the "gold" causal test

The paper's **matched-pretraining front-end swap** (fix trunk/corpus/objective/rate/size; vary only the channel front-end; pretrain once on a single montage and once on heterogeneous montages) is the definitive de-confounder, but it needs a **pretraining pipeline** this harness does not have. **Experiment 1 is its closest downstream proxy** — it tests "the pretrained prior is what wins" without pretraining anything.

---

## Summary: what confirms / refutes the thesis

| Experiment | Claim tested | Confirms if… | Refutes if… |
|---|---|---|---|
| **1. Prior ablation** | native wins *because of* the pretrained channel prior (preserve-don't-overwrite) | EEGPT/CBraMod native → external-adapter level when ablated; LUNA barely moves | ablated native ≈ intact native |
| **2. Distortion vs. conflict** | probe > SFT is feature distortion, not adapter conflict (§V-B) | LP-FT/surgical/LoRA recover the native-path SFT loss | they don't help the native path |
| **3. Permutation** | flexible = permutation-invariant set (§V-A) | flexible-native Δacc ≈ 0; rigid large Δ | flexible-native also breaks |
| **4. Depth probe** (reuses #83) | distortion is localized; prior lives in the channel module | frozen curve ≥ SFT at early depths (LP-FT ≈ frozen); channel-module jump vanishes when ablated | SFT never lowers the curve; ablation leaves the jump |

**Recommended order:** Experiment 1 first (cheap, and it is the direct test of the headline thesis), then 2 — both **instrumented with the Experiment 4 depth-probe** (reusing `probe_layer` from #83) to *localize* the effect rather than read a single number — with 3 as a near-free sanity check.
