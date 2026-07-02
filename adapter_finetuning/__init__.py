"""Core module for adapter-based EEG foundation model fine-tuning.

Only ``optim`` (``CosineAnnealingWarmupLR``) is imported by the experiment
runners. The PEFT / callbacks / datamodule / lightning_module / config_schemas
submodules described in the original package header were part of a larger
internal codebase and are NOT included in this release; the eager imports that
referenced them broke ``import adapter_finetuning``. They are removed here so the
package imports cleanly. Import what the runners use directly:

    from adapter_finetuning.optim import CosineAnnealingWarmupLR
"""
