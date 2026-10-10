"""Explicit small-T HC sum-state operations and caller-owned buffers.

GatedResidual selects the serving path and owns weight preparation. Callers
own per-in-flight-step buffers; prepared projection scratch is reused only on
the worker's serialized compute stream.
"""

import math
from typing import NamedTuple

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.kernels.ops.elementwise.hc_apply_sum import _overlap, hc_apply_sum
from sglang.kernels.ops.elementwise.hc_mix_raw import (
    hc_mix_raw,
    prepare_raw_hc_weights,
    prepare_raw_hc_weights_from_batched,
)
from sglang.kernels.ops.elementwise.hc_raw_transition import _validate_state


class HCSumState(NamedTuple):
    # Bootstrap sums observe storage values; transition sums observe the FP32
    # FMA update before the residual's BF16/FP16 cast. Epsilon is NOT in state.
    residual: torch.Tensor
    sum_sq: torch.Tensor


class HCSumBuffers(NamedTuple):
    mixed: torch.Tensor
    alpha: torch.Tensor
    output: HCSumState


@cache_once
def _bootstrap_module(c, h, dtype):
    args = make_cpp_args(c, h, dtype)
    return load_jit(
        "hc_sum_bootstrap",
        *args,
        cuda_files=["elementwise/hc_raw_transition.cuh"],
        cuda_wrappers=[("run", f"HcRawTransitionKernel<{args}>::stats_sum")],
    )


class SmallTSumHCShell:
    """Down+Inject -> Up/Mix+reset -> Apply+sum; three kernels, no inv output.

    Down and Up independently consume current sums and compute inverse RMS.
    Only Up writes next sums (reset); Apply then accumulates those next sums.

    Prepare after loading weights. Allocate one buffer set per concurrently
    live shell result; do not overwrite a still-live result. Sequential reuse
    on the same stream is supported, including Graph replay. The next shell
    uses its own epsilon. First input requires bootstrap (one extra kernel);
    a chain's final consumer may simply take state.residual.
    """

    def __init__(self, source, *, down_weights=None, norm_weight_permuted=None):
        if not source.config.hc_per_branch_norm:
            raise ValueError("sum-state requires per-branch RMSNorm")
        self.eps = source.config.rms_norm_eps
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("RMS epsilon must be positive and finite")
        if down_weights is None:
            if norm_weight_permuted is not None:
                raise ValueError("shared RMS storage requires prepared Down weights")
            self.weights = prepare_raw_hc_weights(
                source.input_mix_weight_down.weight,
                source.block_inject_weight.weight,
                source.input_mix_weight_up.weight,
                source.hc_norm.weight,
                source.hc_count,
            )
        else:
            self.weights = prepare_raw_hc_weights_from_batched(
                down_weights,
                source.input_mix_weight_up.weight,
                source.hc_norm.weight,
                norm_weight_permuted=norm_weight_permuted,
            )

    def bootstrap(self, residual):
        rows, h = _validate_state(residual, self.weights.hc_count, self.eps)
        if (
            h != self.weights.hidden_size
            or residual.dtype != self.weights.down_inject.dtype
        ):
            raise ValueError("bootstrap residual must match prepared weights")
        sums = torch.empty((rows, 4), device=residual.device, dtype=torch.float32)
        if rows:
            _bootstrap_module(4, h, residual.dtype).run(residual, sums)
        return HCSumState(residual, sums)

    def allocate_buffers(self, rows):
        if type(rows) is not int or not 0 <= rows <= 24:
            raise ValueError("sum-state requires 0<=T<=24")
        w = self.weights

        def new(shape, dtype=None):
            return torch.empty(
                shape, device=w.down_inject.device, dtype=dtype or w.down_inject.dtype
            )

        return HCSumBuffers(
            new((rows, w.hidden_size)),
            new((rows, 4), torch.float32),
            HCSumState(new((rows, 4 * w.hidden_size)), new((rows, 4), torch.float32)),
        )

    def _validate(self, state, buffers):
        if not isinstance(state, HCSumState) or not isinstance(buffers, HCSumBuffers):
            raise ValueError("explicit HCSumState / HCSumBuffers required")
        w = self.weights
        t, h = _validate_state(state.residual, w.hc_count, self.eps)
        if h != w.hidden_size:
            raise ValueError("state hidden size differs from prepared weights")
        specs = (
            (state.residual, (t, 4 * h), w.down_inject.dtype),
            (state.sum_sq, (t, 4), torch.float32),
            (buffers.mixed, (t, h), w.down_inject.dtype),
            (buffers.alpha, (t, 4), torch.float32),
            (buffers.output.residual, (t, 4 * h), w.down_inject.dtype),
            (buffers.output.sum_sq, (t, 4), torch.float32),
        )
        for tensor, shape, dtype in specs:
            if (
                tensor.shape != shape
                or tensor.dtype != dtype
                or tensor.device != w.down_inject.device
                or not tensor.is_contiguous()
                or tensor.data_ptr() % 32
            ):
                raise ValueError("sum-state buffer metadata mismatch")
        tensors = [spec[0] for spec in specs]
        for i, destination in enumerate(tensors[2:], start=2):
            if any(_overlap(destination, other) for other in tensors[:i]):
                raise ValueError("sum-state writable buffers must be disjoint")
            if any(
                _overlap(destination, weight)
                for weight in (
                    w.down_inject,
                    w.up,
                    w.norm_permuted,
                    w.scratch,
                )
            ):
                raise ValueError("sum-state outputs must not alias prepared storage")

    def mix(self, state, buffers):
        self._validate(state, buffers)
        # Both projections read CURRENT sums. Only Up clears NEXT sums.
        return hc_mix_raw(
            state.residual,
            state.sum_sq,
            self.weights,
            next_sum_sq=buffers.output.sum_sq,
            rms_eps=self.eps,
            out=buffers.mixed,
            alpha=buffers.alpha,
        )

    def combine(self, state, buffers, block_output):
        """Apply an explicit core output after ``mix`` filled these buffers.

        The caller runs attention/MoE between mix and combine. The same buffer
        set must remain live until combine finishes; there is no identity-core
        fallback and this method does not recompute mix or reset its sums.
        """
        self._validate(state, buffers)
        hc_apply_sum(
            block_output,
            state.residual,
            buffers.alpha,
            out=buffers.output.residual,
            sum_sq=buffers.output.sum_sq,
        )
        return buffers.output
