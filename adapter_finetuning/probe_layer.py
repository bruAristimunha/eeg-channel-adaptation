"""Layer-wise linear probing for the channel-adaptation harness (Path B).

Ports the ``probe_layer`` mechanism from facebookresearch/neuroai#83 to this
repo: tap any submodule's output with a forward hook, aggregate it, and run a
linear probe -- without modifying the backbone. The backbone is frozen, so this
measures the *pretrained* representation at depth-N (Alain & Bengio, 2016), and
lets us read off each channel module's contribution (EXPERIMENTS.md, Exp 1 & 4).

Drop this file into ``adapter_finetuning/probe_layer.py`` and wire it into a
runner in four small edits (see EXPERIMENTS.md -> "Path B -- wiring it in").

Robustness notes (hardened after a code review of the first draft):
* ``train()`` is overridden so the frozen backbone stays in ``eval()`` even when
  the parent LightningModule is put in train mode every epoch -- otherwise
  backbone dropout/BatchNorm would randomise the "frozen" representation and
  drift BN buffers across seeds.
* The batch axis of the tapped activation is *detected* (not assumed) by probing
  the layer at two batch sizes: the axis whose size scales with the batch is the
  batch axis; a layer with no such axis is batch-independent (e.g. a per-channel
  embedding) and is rejected as an ablation-only target. This also handles
  sequence-first ``(T, B, D)`` and batch-outermost-entangled ``(B*tokens, ...)``
  taps without a manual ``batch_dim``.
* Non-tensor captures, multi-fire hooks, empty captures, un-tupled
  ``example_inputs``, and CPU/CUDA head placement all raise clear errors.
* ``forward`` recomputes the frozen backbone each step; for large sweeps prefer
  caching activations once per (layer, dataset) and training the head on the
  cache (standard Alain & Bengio probing). Left out here to keep the drop-in
  minimal; noted so it is a deliberate simplification, not an oversight.
"""

from __future__ import annotations

import weakref

import torch
import torch.nn as nn


class LayerProbe(nn.Module):
    """Linear probe on the activation tapped at ``probe_layer`` of a frozen backbone.

    Parameters
    ----------
    backbone : nn.Module
        Pretrained model. All of its parameters are frozen and it is held in
        ``eval()`` for the life of the probe.
    n_outputs : int
        Number of classes.
    probe_layer : str
        Dotted submodule path resolved via ``backbone.get_submodule`` -- e.g.
        ``"target_encoder.blocks.3"`` (EEGPT) or ``"encoder.layers.6"``
        (CBraMod). A wrong path raises ``AttributeError`` naming the missing
        component. Batch-independent layers (e.g. ``target_encoder.chan_embed``)
        are rejected -- they are Experiment-1 ablation targets, not probe targets.
    example_inputs : tuple
        Args to call ``backbone(*example_inputs)`` for head sizing / batch-axis
        detection. Leading dim must be the batch (>= 2). Common case:
        ``(torch.zeros(2, n_chans, n_times),)``. Multi-input backbones (e.g. LUNA
        electrode positions) pass their extra inputs too, matching the forward
        signature used at training time.
    aggregation : {"flatten", "mean"}
        Reduce the tapped activation to ``(B, features)``: ``"flatten"`` keeps
        all information; ``"mean"`` pools tokens and keeps the last (feature)
        dim. Use ``"mean"`` for transformer blocks (small head).
    """

    def __init__(
        self,
        backbone: nn.Module,
        n_outputs: int,
        probe_layer: str,
        example_inputs: tuple,
        aggregation: str = "flatten",
    ) -> None:
        super().__init__()
        if aggregation not in ("flatten", "mean"):
            raise ValueError(
                f"aggregation must be 'flatten' or 'mean', got {aggregation!r}"
            )
        if not isinstance(example_inputs, tuple):
            raise TypeError(
                "example_inputs must be a tuple of backbone args, e.g. (x,); "
                f"got {type(example_inputs).__name__}"
            )
        if not example_inputs or not torch.is_tensor(example_inputs[0]):
            raise TypeError("example_inputs[0] must be an input tensor (the batched signal).")
        if example_inputs[0].shape[0] < 2:
            raise ValueError("example_inputs needs batch size >= 2 to detect the batch axis.")

        self.backbone = backbone
        self.probe_layer = probe_layer
        self.aggregation = aggregation

        # Resolve (and validate) the path BEFORE mutating the backbone, so a bad
        # path does not leave a shared backbone permanently frozen.
        submodule = self.backbone.get_submodule(probe_layer)  # raises on bad path

        # Freeze: probing evaluates the *pretrained* representation. Mode is held
        # in eval() by the train() override below (freezing grad != freezing mode).
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

        # Forward hook. weakref breaks the submodule -> hook -> self cycle;
        # weakref.finalize detaches the hook on GC (no stale hooks across seeds).
        self._act: list[torch.Tensor] = []
        self_ref = weakref.ref(self)

        def _capture(_module, _inputs, output):
            self_obj = self_ref()
            if self_obj is None:
                return
            if not torch.is_tensor(output):
                raise TypeError(
                    f"probe_layer={probe_layer!r} returned {type(output).__name__}, "
                    "not a Tensor. Probe a parent module that returns a tensor "
                    "(e.g. a transformer block, not the bare attention submodule)."
                )
            self_obj._act.append(output)

        self._handle = submodule.register_forward_hook(_capture)
        weakref.finalize(self, self._handle.remove)

        # Detect the batch axis + per-sample group by probing at two batch sizes,
        # then size a concrete head (Lightning builds the optimizer before the
        # first real forward, so a LazyLinear head would be skipped).
        device = next(self.backbone.parameters(), torch.empty(0)).device
        self._batch_axis, self._group = self._detect_batch(example_inputs, device)
        feats = self._aggregate(self._run(example_inputs, device), example_inputs[0].shape[0])
        self.head = nn.Linear(feats.shape[1], n_outputs).to(feats.device)

    # -- keep the frozen backbone in eval() regardless of parent .train() -------
    def train(self, mode: bool = True):  # noqa: D401
        super().train(mode)
        self.backbone.eval()
        return self

    # -- run the backbone once and return the single tapped activation ----------
    @torch.no_grad()
    def _run(self, inputs: tuple, device=None) -> torch.Tensor:
        if device is not None:
            inputs = tuple(t.to(device) if torch.is_tensor(t) else t for t in inputs)
        self._act.clear()
        self.backbone(*inputs)
        if not self._act:
            raise RuntimeError(
                f"probe_layer={self.probe_layer!r} produced no activation; the "
                "submodule was not executed on this forward pass (wrong/skipped layer)."
            )
        if len(self._act) > 1:
            raise RuntimeError(
                f"probe_layer={self.probe_layer!r} fired {len(self._act)} times in one "
                "forward (weight-shared/recurrent module); probe a block that runs once."
            )
        return self._act[0]

    @torch.no_grad()
    def _detect_batch(self, example_inputs: tuple, device) -> tuple[int, int]:
        """Return (batch_axis, group): the axis whose size scales with the batch,
        and the per-sample element count on that axis. Raise if none scales."""
        B = example_inputs[0].shape[0]

        def rebatch(inputs, n):  # resize batched tensors along dim 0 to n
            return tuple(
                t[torch.arange(n) % t.shape[0]] if (torch.is_tensor(t) and t.shape[0] == B) else t
                for t in inputs
            )

        a1 = self._run(rebatch(example_inputs, B), device)
        a2 = self._run(rebatch(example_inputs, B + 1), device)
        for d in range(a1.dim()):
            s1, s2 = a1.shape[d], a2.shape[d]
            if s1 % B == 0 and s2 % (B + 1) == 0 and s1 // B == s2 // (B + 1) and s1 >= B:
                return d, s1 // B  # outermost scaling axis
        raise ValueError(
            f"probe_layer={self.probe_layer!r} output {tuple(a1.shape)} has no batch "
            "axis (its shape does not scale with batch size): this layer is "
            "batch-independent (e.g. a per-channel embedding) -- an ABLATION target, "
            "not a probe target."
        )

    def _aggregate(self, act: torch.Tensor, batch_size: int) -> torch.Tensor:
        if self._batch_axis != 0:
            act = act.movedim(self._batch_axis, 0)
        # leading dim is batch_size * group (group=1 for batch-first); reshape
        # assumes batch is outermost on that axis (verified for EEGPT/CBraMod).
        if self.aggregation == "mean":  # pool tokens, keep the last (feature) dim
            return act.reshape(batch_size, -1, act.shape[-1]).mean(dim=1)
        return act.reshape(batch_size, -1)  # flatten -> (B, features)

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        act = self._run(inputs)  # backbone frozen + eval; only the head trains
        feats = self._aggregate(act, inputs[0].shape[0])
        return self.head(feats)


if __name__ == "__main__":
    # Self-checks (runnable without the EEG stack). Cover: correct sizing, frozen
    # backbone, train()-keeps-eval, device, batch-outermost-entangled reshape,
    # batch-independent rejection, and non-tensor rejection.
    torch.manual_seed(0)

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(nn.Flatten(), nn.Linear(8, 16), nn.Dropout(0.5), nn.Linear(16, 4))
        def forward(self, x):
            return self.enc(x)

    m = Toy()
    x = torch.randn(5, 2, 4)
    p = LayerProbe(m, n_outputs=3, probe_layer="enc.1", example_inputs=(x,))
    assert p(x).shape == (5, 3)
    assert p.head.in_features == 16
    assert not any(q.requires_grad for q in p.backbone.parameters())
    # train() must NOT re-enable backbone training mode
    p.train()
    assert not p.backbone.training, "backbone should stay in eval() after .train()"
    assert p.head.training, "head should follow train() mode"
    # gradient reaches only the head
    p(x).sum().backward()
    assert p.head.weight.grad is not None and m.enc[1].weight.grad is None

    # batch-outermost-entangled tap: a layer that outputs (B*T, D)
    class Entangled(nn.Module):
        def __init__(self):
            super().__init__()
            self.lift = nn.Linear(4, 6)   # per (b, t) -> keeps (B, T, 6)
            self.mix = nn.Linear(6, 6)    # tapped after reshape to (B*T, 6)
        def forward(self, x):             # x: (B, T=3, 4)
            h = self.lift(x)              # (B, 3, 6)
            h = h.reshape(-1, 6)          # (B*3, 6)  -- batch outermost
            return self.mix(h)
    me = Entangled()
    xe = torch.randn(5, 3, 4)
    pe = LayerProbe(me, 3, "mix", example_inputs=(xe,), aggregation="mean")
    assert pe(xe).shape == (5, 3), pe(xe).shape
    assert pe._batch_axis == 0 and pe._group == 3, (pe._batch_axis, pe._group)

    # batch-independent layer -> rejected
    class Const(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(2, 8)  # output independent of the input signal
            self.head = nn.Linear(8, 4)
        def forward(self, x):
            e = self.emb(torch.tensor([0, 1]))  # always (2, 8)
            return self.head(e.mean(0, keepdim=True).expand(x.shape[0], -1))
    try:
        LayerProbe(Const(), 3, "emb", example_inputs=(torch.randn(5, 2, 4),))
    except ValueError as e:
        assert "batch-independent" in str(e)
    else:
        raise AssertionError("expected batch-independent rejection")

    # bad path -> AttributeError
    try:
        LayerProbe(m, 3, "enc.does_not_exist", example_inputs=(x,))
    except AttributeError:
        pass
    else:
        raise AssertionError("expected AttributeError for bad path")

    print("LayerProbe self-checks passed.")
