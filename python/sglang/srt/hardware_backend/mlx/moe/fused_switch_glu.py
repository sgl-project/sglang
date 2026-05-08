"""Fused up_proj + gate_proj for SwitchGLU (mlx-lm) in SGLang's MLX backend.

Background
----------
mlx-lm's `SwitchGLU` (mlx_lm/models/switch_layers.py) implements MoE expert
forward as three sequential `mx.gather_qmm` dispatches per layer per token:

    x_up   = self.up_proj(x, idx, sorted_indices=do_sort)    # gather_qmv
    x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)  # gather_qmv
    x = self.down_proj(self.activation(x_up, x_gate), idx, ...) # gather_qmv

The first two consume the same input `x` and the same `indices`, and produce
outputs of identical shape that are immediately combined element-wise via
swiglu. They can be fused into a single `gather_qmm` against a concatenated
weight tensor without changing numerical output.

Per decode step on a 24-layer Qwen-MoE model: cuts 24 gather_qmm dispatches.
On a 48-layer Qwen3-MoE model: cuts 48. Each eliminated dispatch saves both
kernel-internal time and the runtime/scheduler boundary cost (~10us per
crossing).

This module provides:
  - FusedSwitchUpGate: holds the concatenated weights and runs the fused `gather_qmm`
  - _FusedUpProxy / _FusedGateProxy: drop-in replacements for a SwitchGLU instance's
    `up_proj` and `gate_proj` attributes. The up proxy runs the fused call and caches the gate half;
    the gate proxy returns the cached half. SwitchGLU.__call__ is left untouched.
  - patch_switch_glu_with_fused_up_gate: walks the model at load time and
    installs the proxies on each SwitchGLU block.

Activated via env var SGLANG_MLX_FUSE_SWITCHGLU=1.
"""

from __future__ import annotations

import logging

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


def _validate_compatible_projections(up_proj, gate_proj) -> None:
    """Validate that up_proj and gate_proj can be safely fused.

    Both must share weight shape, scale shape, and quantization parameters (group_size, bits, mode).
    Biases must be either both present or both absent. Any mismatch implies the upstream SwitchGLU
    layout has changed and silent miscomputation is possible, refuse to fuse.


    Raises:
        ValueError: on any structural mismatch.
    """
    if up_proj.weight.shape != gate_proj.weight.shape:
        raise ValueError(
            f"FusedSwitchUpGate: up_proj.weight.shape "
            f"{up_proj.weight.shape} != gate_proj.weight.shape "
            f"{gate_proj.weight.shape}. Cannot fuse projections with "
            f"different output dimensions."
        )
    if up_proj.scales.shape != gate_proj.scales.shape:
        raise ValueError(
            f"FusedSwitchUpGate: scales shape mismatch "
            f"({up_proj.scales.shape} vs {gate_proj.scales.shape})."
        )
    for attr in ("group_size", "bits", "mode"):
        u = getattr(up_proj, attr)
        g = getattr(gate_proj, attr)
        if u != g:
            raise ValueError(
                f"FusedSwitchUpGate: {attr} mismatch ({u!r} vs {g!r}). "
                f"Both projections must share quantization parameters."
            )
    if (up_proj.biases is None) != (gate_proj.biases is None):
        raise ValueError(
            f"FusedSwitchUpGate: biases mismatch, one projection has "
            f"biases and the other doesn't. Affine quant inconsistency."
        )


class FusedSwitchUpGate(nn.Module):
    """Single quantized gather_qmm against concat(up_proj, gate_proj) weights.

    Replaces two sequential `mx.gather_qmm` calls with one. Outputs are split
    along the last axis: the first half corresponds to up_proj, the second
    half to gate_proj.

    Numerical equivalence: bit-for-bit identical to running up_proj and
    gate_proj separately, because gather_qmm is associative across the
    output-dim concatenation (each output row is computed independently).
    """

    def __init__(self, up_proj, gate_proj):
        """Take two QuantizedSwitchLinear instances and produce a fused module.

        Args:
            up_proj: mlx_lm.models.switch_layers.QuantizedSwitchLinear for up
            gate_proj: same, for gate

        Both must share: num_experts, input_dims, output_dims, group_size,
        bits, mode. Validated at construction.
        """
        super().__init__()

        # Validate compatibility before fusing. Uses explicit raise rather
        # than assert so checks survive `python -O` (which strips asserts).
        _validate_compatible_projections(up_proj, gate_proj)

        # Concatenate along axis=1 (output dim).
        # weight: (num_experts, output_dim, packed_input) -> (num_experts, 2*output_dim, packed_input)
        # scales: (num_experts, output_dim, num_groups)   -> (num_experts, 2*output_dim, num_groups)
        # biases: same shape as scales
        # Order matters: up first, gate second. The split in __call__ assumes this.
        self.weight = mx.concatenate([up_proj.weight, gate_proj.weight], axis=1)
        self.scales = mx.concatenate([up_proj.scales, gate_proj.scales], axis=1)

        # biases is the affine quant offset, present for affine mode.
        # SwitchGLU uses bias=False so there's no additive bias to handle.
        # Compatibility (both-present or both-absent) was verified above by
        # _validate_compatible_projections, so we can branch on up_proj alone
        if up_proj.biases is not None:
            self.biases = mx.concatenate([up_proj.biases, gate_proj.biases], axis=1)
        else:
            self.biases = None

        # Quantization params (identical across both projections, validated above).
        self.group_size = up_proj.group_size
        self.bits = up_proj.bits
        self.mode = up_proj.mode

        # Single-projection output dim, used to split the fused output back into halves.
        self.hidden_dim = up_proj.weight.shape[1]

        # Freeze: these are inference-only weights.
        self.freeze()

    def __call__(self, x, indices, sorted_indices=False):
        """Fused gather_qmm followed by output split.

        Args:
            x: input activations, shape (..., 1, 1, input_dim) per SwitchGLU
                convention (expand_dims has already been applied upstream).
            indices: expert routing indices, shape (..., top_k).
            sorted_indices: passthrough to gather_qmm; True when SwitchGLU's
                _gather_sort path is active (large batch).

        Returns:
            (x_up, x_gate): tuple of two arrays with identical shape, each
            corresponding to what up_proj and gate_proj would have returned.
        """
        # Single dispatch instead of two.
        x_concat = mx.gather_qmm(
            x,
            self["weight"],
            self["scales"],
            self.get("biases"),  # None-safe via dict-style access
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )

        # Output shape: (..., 2*hidden_dim). Split into up and gate halves.
        # The order matches the concatenation order in __init__.
        x_up = x_concat[..., : self.hidden_dim]
        x_gate = x_concat[..., self.hidden_dim :]

        return x_up, x_gate


class _FusedUpProxy:
    """Drop-in replacement for `switch_mlp.up_proj`.

    Runs the fused gather_qmm, caches the gate half locally, returns just
    the up half. SwitchGLU.__call__ sees a normal callable returning a single tensor.

    Caching contract:
        After __call__, self._cached_gate holds the gate half from the most recent fusion.
        The paired _FusedGateProxy reads and clears it. Single-use: one up_proj call,
        one gate_proj call, then cleared. Concurrent invocations on the same instance are unsafe;
        SGLang's MLX backend is single-threaded.

        TODO: Add concurrent support if SGLang's MLX backend ever moves to multi-threaded execution.
    """

    def __init__(self, fused: FusedSwitchUpGate):
        self._fused = fused
        self._cached_gate = None

    def __call__(self, x, indices, sorted_indices=False):
        x_up, x_gate = self._fused(x, indices, sorted_indices=sorted_indices)
        self._cached_gate = x_gate
        return x_up

    def take_cached_gate(self):
        """Return and clear the cached gate. Single-use."""
        gate = self._cached_gate
        self._cached_gate = None
        return gate


class _FusedGateProxy:
    """Drop-in replacement for `switch_mlp.gate_proj`.

    Returns the gate half cached by the paired _FusedUpProxy. Does
    no actual computation. Relies on SwitchGLU.__call__ calling up_proj
    before gate_proj, if that order ever changes, the cache is empty
    and we raise rather than return None.
    """

    def __init__(self, up_proxy: "_FusedUpProxy"):
        self._up_proxy = up_proxy

    def __call__(self, x, indices, sorted_indices=False):
        gate = self._up_proxy.take_cached_gate()
        if gate is None:
            raise RuntimeError(
                "FusedGateProxy: gate cache is empty. up_proj must be "
                "called before gate_proj within SwitchGLU.__call__. "
                "If this fires, the upstream call order has changed."
            )
        return gate


def patch_switch_glu_with_fused_up_gate(model) -> int:
    """Walk an mlx-lm model and install fused up_proj/gate_proj proxies on
    every quantized SwitchGLU block. SwitchGLU.__call__ is left alone.

    Args:
        model: an mlx-lm model instance (e.g. from mlx_lm.load).

    Returns:
        Number of SwitchGLU instances patched.
    """
    from mlx_lm.models.switch_layers import (
        QuantizedSwitchLinear,
        SwitchGLU,
    )

    patched_count = 0

    # Walk every layer's MLP block looking for SwitchGLU
    for layer in model.model.layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue

        switch_mlp = getattr(mlp, "switch_mlp", None)
        if switch_mlp is None or not isinstance(switch_mlp, SwitchGLU):
            continue

        # Idempotent: skip if this instance is already a proxy. Without
        # this, a second call would replace a proxy's underlying fused module with
        # itself and break tests that load + patched repeatedly.

        if isinstance(switch_mlp.up_proj, _FusedUpProxy):
            continue

        # Skip non-quantized SwitchGLU. FusedSwitchUpGate assumes
        # QuantizedSwitchLinear fields (scales, group_size, bits, mode).
        # non-quantized models (SwitchLinear) would crash here.
        if not isinstance(switch_mlp.up_proj, QuantizedSwitchLinear):
            continue

        # Build the fused module and install the up-proxy.
        fused = FusedSwitchUpGate(switch_mlp.up_proj, switch_mlp.gate_proj)
        switch_mlp.up_proj = _FusedUpProxy(fused)

        # Install the gate-proxy. SwitchGLU.__call__ will invoke up_proj first
        # (running the fused gather_qmm and caching the gate half on the
        # up-proxy), then gate_proj (reading the cache from the up-proxy).
        switch_mlp.gate_proj = _FusedGateProxy(switch_mlp.up_proj)

        patched_count += 1

    if patched_count == 0:
        logger.warning(
            "patch_switch_glu_with_fused_up_gate: no SwitchGLU instances found"
        )
        return 0

    logger.info(
        f"patch_switch_glu_with_fused_up_gate: installed fused proxies on "
        f"{patched_count} SwitchGLU instances"
    )
    return patched_count
