#!/usr/bin/env python
"""
Train LUNA on EEG datasets with multiple input modes and training regimes.

Modes:
  native:       Raw EEG with original channel counts + 3D electrode positions.
  interpolated: 19ch SSI interpolated HDF5 files from BENDR experiments.
  omneeg:       25 OmnEEG spherical harmonic coefficients (no channel positions).
  riemannian:   19ch SSI + Riemannian re-centered interpolated data.

Training:
  probe: Freeze all except final_layer (classification head only).
  sft:   Unfreeze all parameters (full supervised fine-tuning, lr=1e-5).

Usage:
    # Probe mode (frozen encoder)
    python scripts/run_luna_experiments.py --mode native --training-mode probe --dataset bcic2a

    # Full SFT
    python scripts/run_luna_experiments.py --mode native --training-mode sft --dataset bcic2a

    # With Conv1d bridge
    python scripts/run_luna_experiments.py --mode native --training-mode sft --conv1d-bridge --dataset bcic2a

    # OmnEEG
    python scripts/run_luna_experiments.py --mode omneeg --training-mode sft --dataset bcic2a physionet tuev
"""

import argparse
import logging
import os
import random
from pathlib import Path

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.serialization
from torch.utils.data import DataLoader, Dataset, Subset
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Accuracy, F1Score, MetricCollection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Paths
DEFAULT_NATIVE_DIR = Path("/expanse/projects/nemar/kuntal/adapter_finetuning/data/luna_native")
DEFAULT_INTERP_DIR = Path("/expanse/projects/nemar/kuntal/adapter_finetuning/data/interpolated")
DEFAULT_OMNEEG_DIR = Path("/expanse/projects/nemar/adapter_finetuning/data/omneeg")
DEFAULT_OUTPUT_DIR = Path("/expanse/projects/nemar/kuntal/adapter_finetuning/results/luna")

LUNA_SFREQ = 128.0
LUNA_PATCH_SIZE = 40
LUNA_HUB_REPO = "thorir/LUNA"
LUNA_HUB_FILE = "LUNA_base.safetensors"

# BENDR's 19 channels for interpolated/riemannian modes
BENDR_19_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T7", "T8", "P7", "P8",
    "Fz", "Cz", "Pz",
]

DATASET_CONFIG = {
    "bcic2a": {
        "n_classes": 4,
        "batch_size": 64,
        "train_subjects": [1, 2, 3],
        "val_subjects": [4, 5, 6],
        "test_subjects": [7, 8, 9],
        "split_key": "subject",
    },
    "physionet": {
        "n_classes": 4,
        "batch_size": 32,
        "train_subjects": list(range(1, 71)),
        "val_subjects": list(range(71, 90)),
        "test_subjects": list(range(90, 110)),
        "split_key": "subject",
    },
    "tuev": {
        "n_classes": 6,
        "batch_size": 32,
        "train_subjects": None,
        "val_subjects": None,
        "test_subjects": None,
        "split_key": "split",
    },
    "faced": {
        "n_classes": 9,
        "batch_size": 32,
        "train_subjects": list(range(0, 80)),
        "val_subjects": list(range(80, 100)),
        "test_subjects": list(range(100, 123)),
        "split_key": "subject",
    },
    "isruc-sleep": {
        "n_classes": 5,
        "batch_size": 16,
        "train_subjects": [
            "I011", "I013", "I014", "I017", "I019", "I020",
            "I021", "I023", "I025", "I027", "I028", "I029", "I030",
            "I031", "I032", "I033", "I034", "I035", "I036", "I037", "I038", "I039",
            "I041", "I042", "I043", "I044", "I045", "I046", "I047", "I048", "I049", "I050",
            "I051", "I052", "I053", "I054", "I055", "I056", "I057", "I058", "I059", "I060",
            "I061", "I062", "I063", "I064", "I065", "I066", "I067", "I068", "I069", "I070",
            "I071", "I072", "I073", "I074", "I075", "I076", "I077", "I078", "I079", "I080",
        ],
        "val_subjects": [
            "I081", "I082", "I083", "I084", "I085", "I086", "I087", "I088", "I089", "I090",
        ],
        "test_subjects": [
            "I091", "I092", "I093", "I094", "I095", "I096", "I097", "I098", "I099", "I100",
        ],
        "split_key": "subject",
    },
    "mdd_mumtaz2016": {
        "n_classes": 2,
        "batch_size": 32,
        "train_subjects": [
            "HS1", "HS10", "HS11", "HS12", "HS13", "HS14", "HS15", "HS16", "HS17", "HS18", "HS19",
            "HS2", "HS20", "HS21", "HS22", "MDDS1", "MDDS10", "MDDS11", "MDDS12", "MDDS13", "MDDS14",
            "MDDS15", "MDDS16", "MDDS17", "MDDS18", "MDDS19", "MDDS2", "MDDS20", "MDDS21",
        ],
        "val_subjects": ["HS23", "HS24", "HS25", "MDDS22", "MDDS23", "MDDS24", "MDDS25"],
        "test_subjects": [
            "HS26", "HS27", "HS28", "HS29", "HS3", "HS30", "HS4", "HS5", "HS6", "HS7", "HS8", "HS9",
            "MDDS26", "MDDS27", "MDDS28", "MDDS29", "MDDS3", "MDDS30", "MDDS31", "MDDS32", "MDDS33",
            "MDDS34", "MDDS4", "MDDS5", "MDDS6", "MDDS7", "MDDS8", "MDDS9",
        ],
        "split_key": "subject",
    },
}

TRAIN_CONFIG_PROBE = {
    "max_epochs": 50,
    "lr": 5e-4,
    "weight_decay": 0.01,
    "warmup_epochs": 5,
    "eta_min": 1e-6,
    "gradient_clip_val": 1.0,
    "patience": 10,
}

TRAIN_CONFIG_SFT = {
    "max_epochs": 50,
    "lr": 1e-5,
    "weight_decay": 0.01,
    "warmup_epochs": 5,
    "eta_min": 1e-6,
    "gradient_clip_val": 1.0,
    "patience": 10,
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LUNADataset(Dataset):
    """Dataset for LUNA experiments — supports native, interpolated, omneeg, riemannian HDF5."""

    def __init__(self, h5_path: Path, channel_locations: np.ndarray = None):
        with h5py.File(h5_path, "r") as f:
            self.signals = f["signals"][:]
            self.labels = f["labels"][:]
            self.subjects = f["subjects"][:]

            # Robust attribute reading — fall back to inferring from shape
            self.n_channels = int(f.attrs["n_channels"]) if "n_channels" in f.attrs else self.signals.shape[1]
            self.n_times = int(f.attrs["n_times"]) if "n_times" in f.attrs else self.signals.shape[2]
            self.sfreq = float(f.attrs.get("sfreq", LUNA_SFREQ))

            # Load channel locations if stored in HDF5
            if "channel_locations" in f:
                self.channel_locations = f["channel_locations"][:]
            elif channel_locations is not None:
                self.channel_locations = channel_locations
            else:
                self.channel_locations = None

            if "channel_names" in f:
                raw_names = f["channel_names"][:]
                self.channel_names = [
                    n.decode() if isinstance(n, bytes) else str(n)
                    for n in raw_names
                ]
            else:
                self.channel_names = None

        log.info(
            "Loaded %s: %d samples, shape (%d, %d, %d), sfreq=%.0f",
            h5_path.name, len(self.signals), len(self.signals),
            self.n_channels, self.n_times, self.sfreq,
        )

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        x = torch.tensor(self.signals[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def truncate_to_patch_size(dataset, patch_size=LUNA_PATCH_SIZE):
    """Truncate n_times to nearest lower multiple of patch_size."""
    n_times = dataset.n_times
    remainder = n_times % patch_size
    if remainder != 0:
        new_n_times = n_times - remainder
        log.info(
            "Truncating n_times from %d to %d (patch_size=%d)",
            n_times, new_n_times, patch_size,
        )
        dataset.signals = dataset.signals[:, :, :new_n_times]
        dataset.n_times = new_n_times
    return dataset


def split_dataset(dataset, config, seed=0):
    subjects = dataset.subjects
    if config["split_key"] == "subject":
        # Decode bytes to strings if needed
        if len(subjects) > 0 and hasattr(subjects[0], "decode"):
            subjects = np.array([s.decode() for s in subjects])
        train_mask = np.isin(subjects, config["train_subjects"])
        val_mask = np.isin(subjects, config["val_subjects"])
        test_mask = np.isin(subjects, config["test_subjects"])
    elif config["split_key"] == "split":
        if hasattr(subjects[0], "decode"):
            subjects_str = np.array([s.decode() for s in subjects])
        else:
            subjects_str = subjects.astype(str)

        train_eval_mask = np.array(["train" in s.lower() for s in subjects_str])
        eval_mask = np.array(["eval" in s.lower() for s in subjects_str])
        unmatched = ~train_eval_mask & ~eval_mask
        if unmatched.any():
            train_eval_mask = train_eval_mask | unmatched

        rng = np.random.default_rng(seed)
        train_pool_indices = np.where(train_eval_mask)[0]
        rng.shuffle(train_pool_indices)
        n_train = int(0.8 * len(train_pool_indices))

        n = len(dataset)
        train_mask = np.zeros(n, dtype=bool)
        val_mask = np.zeros(n, dtype=bool)
        test_mask = eval_mask.copy()
        train_mask[train_pool_indices[:n_train]] = True
        val_mask[train_pool_indices[n_train:]] = True
    else:
        raise ValueError(f"Unknown split_key: {config['split_key']}")

    log.info(
        "Split: train=%d, val=%d, test=%d",
        train_mask.sum(), val_mask.sum(), test_mask.sum(),
    )
    return (
        Subset(dataset, np.where(train_mask)[0]),
        Subset(dataset, np.where(val_mask)[0]),
        Subset(dataset, np.where(test_mask)[0]),
    )


# ---------------------------------------------------------------------------
# Channel location utilities
# ---------------------------------------------------------------------------

def build_chs_info(ch_names_list):
    """Build MNE-style chs_info list from channel names using standard_1020."""
    import mne
    montage = mne.channels.make_standard_montage("standard_1020")
    all_positions = montage.get_positions()["ch_pos"]

    rename_map = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8", "A1": "M1", "A2": "M2"}

    chs_info = []
    for ch in ch_names_list:
        lookup = ch
        for old, new in rename_map.items():
            if ch == old:
                lookup = new
                break
        pos = all_positions.get(lookup, np.zeros(3))
        loc = np.zeros(12)
        loc[:3] = pos
        chs_info.append({
            "ch_name": ch,
            "kind": 2,  # EEG
            "loc": loc,
        })
    return chs_info


def get_channel_locations_from_names(ch_names_list):
    """Get 3D positions array from channel names."""
    import mne
    montage = mne.channels.make_standard_montage("standard_1020")
    all_positions = montage.get_positions()["ch_pos"]
    rename_map = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8", "A1": "M1", "A2": "M2"}

    positions = []
    for ch in ch_names_list:
        lookup = ch
        for old, new in rename_map.items():
            if ch == old:
                lookup = new
                break
        positions.append(all_positions.get(lookup, np.zeros(3)))
    return np.array(positions, dtype=np.float32)


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------

class LUNAExperimentModule(pl.LightningModule):
    """LUNA model with configurable training mode (probe or SFT) and optional Conv1d bridge."""

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        n_outputs: int,
        sfreq: float,
        chs_info=None,
        channel_locations: np.ndarray = None,
        training_mode: str = "probe",
        conv1d_bridge: bool = False,
        lr: float = 5e-4,
        weight_decay: float = 0.01,
        warmup_epochs: int = 5,
        max_epochs: int = 50,
        eta_min: float = 1e-6,
        probe_layer: str | None = None,
        probe_aggregation: str = "flatten",
        init_from: str | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["chs_info", "channel_locations"])

        from braindecode.models import LUNA

        # Conv1d bridge: project n_chans -> n_chans (learnable, identity-like init)
        self.conv1d_bridge = None
        if conv1d_bridge:
            self.conv1d_bridge = nn.Conv1d(n_chans, n_chans, kernel_size=1, bias=True)
            # Initialize close to identity
            nn.init.eye_(self.conv1d_bridge.weight[:, :, 0])
            nn.init.zeros_(self.conv1d_bridge.bias)
            log.info("Conv1d bridge enabled: %d -> %d channels", n_chans, n_chans)

        self.model = LUNA(
            n_outputs=n_outputs,
            n_chans=n_chans,
            n_times=n_times,
            sfreq=sfreq,
            chs_info=chs_info,
            patch_size=LUNA_PATCH_SIZE,
            num_queries=4,
            embed_dim=64,
            depth=8,
            num_heads=2,
            mlp_ratio=4.0,
            drop_path=0.0,
            drop_prob_chan=0.0,
            attn_drop=0.0,
        )

        # Store channel locations for forward pass
        if channel_locations is not None:
            self.register_buffer(
                "default_channel_locations",
                torch.from_numpy(channel_locations).float(),
            )
        else:
            self.default_channel_locations = None

        self._load_pretrained()

        if init_from is not None:
            ckpt = torch.load(init_from, map_location="cpu", weights_only=False)
            sd = ckpt.get("state_dict", ckpt)
            model_sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
            missing, _ = self.model.load_state_dict(model_sd, strict=False)
            log.info("Loaded %d weights from checkpoint %s (%d missing)",
                     len(model_sd), init_from, len(missing))

        # Freeze based on training mode
        if training_mode == "probe":
            for name, param in self.model.named_parameters():
                if "final_layer" not in name:
                    param.requires_grad = False
        # SFT: all params remain trainable (default)

        # Optional layer-wise linear probe (EXPERIMENTS.md Path B). LUNA passes
        # channel_locations as a keyword arg, so drive the backbone via call_fn.
        self.probe = None
        if probe_layer is not None:
            from adapter_finetuning.probe_layer import LayerProbe
            ex_x = torch.randn(2, n_chans, n_times)
            if self.default_channel_locations is not None:
                ex_loc = self.default_channel_locations.unsqueeze(0).expand(2, -1, -1).contiguous()
                example_inputs = (ex_x, ex_loc)
                call_fn = lambda a, b: self.model(a, channel_locations=b)
            else:
                example_inputs = (ex_x,)
                call_fn = None
            self.probe = LayerProbe(self.model, n_outputs, probe_layer,
                                    example_inputs=example_inputs,
                                    aggregation=probe_aggregation, call_fn=call_fn)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        log.info(
            "Training mode: %s | Trainable: %d/%d (%.2f%%)",
            training_mode, trainable, total, 100 * trainable / total,
        )

        self.criterion = nn.CrossEntropyLoss()
        metrics = MetricCollection({
            "accuracy": Accuracy(task="multiclass", num_classes=n_outputs, average="macro"),
            "f1_macro": F1Score(task="multiclass", num_classes=n_outputs, average="macro"),
        })
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def _load_pretrained(self):
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        log.info("Loading pretrained LUNA weights from %s/%s", LUNA_HUB_REPO, LUNA_HUB_FILE)

        try:
            path = hf_hub_download(repo_id=LUNA_HUB_REPO, filename=LUNA_HUB_FILE)
            state_dict = load_file(path)
        except Exception as e:
            log.warning("Could not download pretrained weights: %s", e)
            return

        # Apply key mapping (handles naming differences between pretrained and braindecode)
        mapping = self.model.mapping.copy()
        mapping["cross_attn.temparature"] = "cross_attn.temperature"
        mapped_state_dict = {mapping.get(k, k): v for k, v in state_dict.items()}

        # Filter by shape compatibility
        model_state = self.model.state_dict()
        filtered = {}
        skipped = []
        for k, v in mapped_state_dict.items():
            if k in model_state and v.shape == model_state[k].shape:
                filtered[k] = v
            elif k in model_state:
                skipped.append(f"{k}: pretrained {v.shape} vs model {model_state[k].shape}")

        missing, unexpected = self.model.load_state_dict(filtered, strict=False)
        log.info(
            "Loaded %d/%d pretrained weights (%d skipped, %d missing in model)",
            len(filtered), len(mapped_state_dict), len(skipped), len(missing),
        )
        if skipped:
            log.info("Skipped (shape mismatch): %s", skipped[:5])

    def forward(self, x):
        # Apply Conv1d bridge if present
        if self.conv1d_bridge is not None:
            x = self.conv1d_bridge(x)

        # Pass channel locations if available
        if self.default_channel_locations is not None:
            batch_size = x.shape[0]
            ch_locs = self.default_channel_locations.unsqueeze(0).expand(batch_size, -1, -1)
            if self.probe is not None:
                return self.probe(x, ch_locs)
            return self.model(x, channel_locations=ch_locs)
        if self.probe is not None:
            return self.probe(x)
        return self.model(x)

    def _shared_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=-1)
        return {"loss": loss, "preds": preds, "targets": y}

    def training_step(self, batch, batch_idx):
        out = self._shared_step(batch)
        self.train_metrics.update(out["preds"], out["targets"])
        self.log("train_loss", out["loss"], on_step=True, on_epoch=True, prog_bar=True)
        return out["loss"]

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute(), prog_bar=True)
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        out = self._shared_step(batch)
        self.val_metrics.update(out["preds"], out["targets"])
        self.log("val_loss", out["loss"], on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), prog_bar=True)
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        out = self._shared_step(batch)
        self.test_metrics.update(out["preds"], out["targets"])
        self.log("test_loss", out["loss"])

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def configure_optimizers(self):
        trainable = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
        )
        from adapter_finetuning.optim import CosineAnnealingWarmupLR
        scheduler = CosineAnnealingWarmupLR(
            optimizer,
            max_epochs=self.hparams.max_epochs,
            eta_min=self.hparams.eta_min,
            warmup_start_lr=self.hparams.eta_min,
            warmup_epochs=self.hparams.warmup_epochs,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1, "monitor": "val_loss"},
        }


# ---------------------------------------------------------------------------
# Seed utilities
# ---------------------------------------------------------------------------

def seed_everything_deterministic(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    pl.seed_everything(seed, workers=True)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_experiment(
    dataset_name: str,
    seed: int,
    mode: str,
    training_mode: str,
    conv1d_bridge: bool,
    data_dir: Path,
    output_dir: Path,
    wandb_entity: str = "braindecode",
    wandb_project: str = "adapter-finetuning",
    fast_dev_run: bool = False,
    probe_layer: str | None = None,
    probe_aggregation: str = "flatten",
    init_from: str | None = None,
):
    # Fix PyTorch 2.10+ weights_only loading
    try:
        import numpy as _np
        safe_globals = [_np.dtype, _np.ndarray]
        try:
            safe_globals.append(_np._core.multiarray.scalar)
        except AttributeError:
            pass
        torch.serialization.add_safe_globals(safe_globals)
    except Exception:
        pass

    train_config = TRAIN_CONFIG_SFT if training_mode == "sft" else TRAIN_CONFIG_PROBE
    config = DATASET_CONFIG[dataset_name]

    # Determine HDF5 path based on mode
    if mode == "native":
        h5_path = data_dir / f"{dataset_name}_native_128hz.h5"
    elif mode == "interpolated":
        h5_path = data_dir / f"{dataset_name}_interpolated_spline.h5"
    elif mode == "omneeg":
        h5_path = data_dir / f"{dataset_name}_omneeg_3d.h5"
    elif mode == "riemannian":
        h5_path = data_dir / f"{dataset_name}_interpolated_spline_recenter_riemannian.h5"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if not h5_path.exists():
        raise FileNotFoundError(f"Data not found: {h5_path}")

    bridge_tag = "+conv1d" if conv1d_bridge else ""
    log.info("=" * 60)
    log.info(
        "LUNA | Dataset: %s, Seed: %d, Mode: %s%s, Training: %s",
        dataset_name, seed, mode, bridge_tag, training_mode,
    )
    log.info("=" * 60)

    seed_everything_deterministic(seed)

    # Load data
    dataset = LUNADataset(h5_path)

    # Build channel info for LUNA
    if mode == "native" and dataset.channel_names:
        ch_names = dataset.channel_names
    elif mode in ("interpolated", "riemannian"):
        ch_names = BENDR_19_CHANNELS
    elif mode == "omneeg":
        ch_names = None  # SH coefficients have no physical channel positions
    else:
        ch_names = None

    # Get channel locations
    if dataset.channel_locations is not None:
        ch_locations = dataset.channel_locations
    elif ch_names:
        ch_locations = get_channel_locations_from_names(ch_names)
    else:
        ch_locations = None

    # Build chs_info for LUNA constructor
    chs_info = build_chs_info(ch_names) if ch_names else None

    # Handle sfreq: resample non-128Hz data to 128Hz
    n_times = dataset.n_times
    actual_sfreq = dataset.sfreq
    if abs(actual_sfreq - LUNA_SFREQ) > 1.0:
        log.info("Resampling data from %.0fHz to %.0fHz", actual_sfreq, LUNA_SFREQ)
        import scipy.signal as sig
        new_n_times = int(n_times * LUNA_SFREQ / actual_sfreq)
        resampled = sig.resample(dataset.signals, new_n_times, axis=2)
        dataset.signals = resampled.astype(np.float32)
        dataset.n_times = new_n_times
        n_times = new_n_times
        actual_sfreq = LUNA_SFREQ
        log.info("Resampled to %d time points at %.0fHz", new_n_times, LUNA_SFREQ)

    # Truncate n_times to be divisible by patch_size
    truncate_to_patch_size(dataset, LUNA_PATCH_SIZE)
    n_times = dataset.n_times

    train_ds, val_ds, test_ds = split_dataset(dataset, config, seed=seed)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    # Create model
    model = LUNAExperimentModule(
        n_chans=dataset.n_channels,
        n_times=n_times,
        n_outputs=config["n_classes"],
        sfreq=actual_sfreq,
        chs_info=chs_info,
        channel_locations=ch_locations,
        training_mode=training_mode,
        probe_layer=probe_layer,
        probe_aggregation=probe_aggregation,
        init_from=init_from,
        conv1d_bridge=conv1d_bridge,
        lr=train_config["lr"],
        weight_decay=train_config["weight_decay"],
        warmup_epochs=train_config["warmup_epochs"],
        max_epochs=train_config["max_epochs"],
        eta_min=train_config["eta_min"],
    )

    # Wandb logger
    bridge_suffix = "_conv1d" if conv1d_bridge else ""
    experiment_name = f"luna_{mode}{bridge_suffix}_{training_mode}_{dataset_name}_init{seed}"
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    run_name = f"{experiment_name}_job{slurm_job_id}" if slurm_job_id else experiment_name

    tags = [
        "luna", mode, training_mode, dataset_name,
        f"init{seed}", "alignment_experiment",
    ]
    if conv1d_bridge:
        tags.append("conv1d_bridge")
    if slurm_job_id:
        tags.append(f"slurm_{slurm_job_id}")

    wandb_logger = WandbLogger(
        entity=wandb_entity, project=wandb_project, name=run_name,
        group=f"luna_{mode}{bridge_suffix}_{training_mode}_{dataset_name}",
        tags=tags, save_dir=str(output_dir),
        config={
            "model": {"name": "luna", "variant": "base"},
            "data": {"name": dataset_name, "n_classes": config["n_classes"], "batch_size": config["batch_size"]},
            "adapter": {"name": training_mode},
            "experiment": {"init_seed": seed},
            "preprocessing": f"luna_{mode}",
            "training_mode": training_mode,
            "conv1d_bridge": conv1d_bridge,
            "n_chans": dataset.n_channels,
            "n_times": n_times,
            "sfreq": actual_sfreq,
            **train_config,
        },
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=str(output_dir / "checkpoints" / experiment_name),
            monitor="val_accuracy", mode="max", save_top_k=3,
            save_last=True, save_weights_only=True,
            filename="{epoch}-{val_accuracy:.4f}",
        ),
        EarlyStopping(monitor="val_loss", patience=train_config["patience"], mode="min", min_delta=0.001),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(
        max_epochs=train_config["max_epochs"],
        accelerator="auto", devices=1, precision=32,
        gradient_clip_val=train_config["gradient_clip_val"],
        gradient_clip_algorithm="norm",
        callbacks=callbacks, logger=wandb_logger,
        deterministic="warn", fast_dev_run=fast_dev_run,
    )

    trainer.fit(model, train_loader, val_loader)

    best_score = 0.0
    if isinstance(trainer.checkpoint_callback, ModelCheckpoint) and trainer.checkpoint_callback.best_model_score is not None:
        score = trainer.checkpoint_callback.best_model_score
        best_score = score.item() if isinstance(score, torch.Tensor) else float(score)

    if test_loader is not None and len(test_ds) > 0:
        try:
            trainer.test(model, test_loader, ckpt_path="best")
        except Exception as e:
            log.warning("Test step failed: %s", e)

    log.info("Best val accuracy: %.4f", best_score)
    return best_score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train LUNA on EEG datasets")
    parser.add_argument("--dataset", type=str, nargs="+", default=["bcic2a"],
                        choices=["bcic2a", "physionet", "tuev", "faced", "isruc-sleep", "mdd_mumtaz2016"])
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--start-seed", type=int, default=0,
                        help="Starting seed index (default: 0). Runs seeds start_seed..start_seed+n_seeds-1")
    parser.add_argument("--mode", type=str, default="native",
                        choices=["native", "interpolated", "omneeg", "riemannian"])
    parser.add_argument("--training-mode", type=str, default="probe",
                        choices=["probe", "sft"])
    parser.add_argument("--conv1d-bridge", action="store_true",
                        help="Add Conv1d bridge layer before LUNA")
    parser.add_argument("--native-dir", type=Path, default=DEFAULT_NATIVE_DIR)
    parser.add_argument("--interp-dir", type=Path, default=DEFAULT_INTERP_DIR)
    parser.add_argument("--omneeg-dir", type=Path, default=DEFAULT_OMNEEG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--probe-layer", type=str, default=None,
                        help="Tap this submodule and train a linear probe on it (e.g. blocks.3).")
    parser.add_argument("--probe-aggregation", type=str, default="flatten", choices=["flatten", "mean"])
    parser.add_argument("--init-from", type=str, default=None,
                        help="Load model weights from this .ckpt into the backbone before freezing/probing.")
    parser.add_argument("--wandb-entity", type=str, default="braindecode")
    parser.add_argument("--wandb-project", type=str, default="adapter-finetuning")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TMPDIR", "/expanse/projects/nemar/eeg_finetuning/.cache/tmp")
    os.environ.setdefault("HF_HOME", "/expanse/projects/nemar/dtyoung/huggingface_cache")
    os.environ.setdefault("WANDB_MODE", os.environ.get("WANDB_MODE", "offline"))

    # Select data directory based on mode
    if args.mode == "native":
        data_dir = args.native_dir
    elif args.mode in ("interpolated", "riemannian"):
        data_dir = args.interp_dir
    elif args.mode == "omneeg":
        data_dir = args.omneeg_dir
    else:
        data_dir = args.native_dir

    train_config = TRAIN_CONFIG_SFT if args.training_mode == "sft" else TRAIN_CONFIG_PROBE

    results = {}
    for dataset_name in args.dataset:
        dataset_results = []
        for seed in range(args.start_seed, args.start_seed + args.n_seeds):
            try:
                score = run_experiment(
                    dataset_name=dataset_name, seed=seed, mode=args.mode,
                    training_mode=args.training_mode, conv1d_bridge=args.conv1d_bridge,
                    data_dir=data_dir, output_dir=args.output_dir,
                    wandb_entity=args.wandb_entity, wandb_project=args.wandb_project,
                    fast_dev_run=args.fast_dev_run,
                    probe_layer=args.probe_layer,
                    probe_aggregation=args.probe_aggregation,
                    init_from=args.init_from,
                )
                dataset_results.append(score)
            except Exception as e:
                log.error("Failed %s seed %d: %s", dataset_name, seed, e)
                dataset_results.append(0.0)

        results[dataset_name] = dataset_results
        log.info("%s: mean=%.4f, std=%.4f", dataset_name, np.mean(dataset_results), np.std(dataset_results))

    log.info("=" * 60)
    log.info("LUNA Experiment Results Summary (mode=%s, training=%s, bridge=%s)",
             args.mode, args.training_mode, args.conv1d_bridge)
    log.info("=" * 60)
    for dataset_name, scores in results.items():
        log.info("  %s: %.4f +/- %.4f (n=%d)", dataset_name, np.mean(scores), np.std(scores), len(scores))


if __name__ == "__main__":
    main()
