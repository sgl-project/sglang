"""Large-T HC module: branch BMM Down -> native Up -> Mix/reset -> Apply/sum.

Prepared Down, shared [H,C] RMS/source Up weights, and caller-owned buffers.
No normalized activation, inverse tensor, runtime weight pack or token chunking.
"""

import math
from typing import NamedTuple

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from sglang.kernels.ops.elementwise.hc_apply_sum import _overlap, hc_apply_sum
from sglang.kernels.ops.elementwise.hc_down_batched_sum import (
    _validate_batched_rows,
    allocate_batched_down_buffers,
    batched_down_from_sum,
    prepare_batched_down_weights,
)
from sglang.kernels.ops.elementwise.hc_sum_state import HCSumState, _bootstrap_module


class HCLargeSumBuffers(NamedTuple):
    mixed: torch.Tensor
    alpha: torch.Tensor
    output: HCSumState
    partial: torch.Tensor
    down: torch.Tensor
    logits: torch.Tensor


@gluon.jit
def _hc_mix_branch_inv(values, branch: gl.constexpr):
    # BlockedLayout([2], [32], [4], [0]): consecutive register pairs
    # belong to one token. Lanes 0..3 carry the four branch inverses.
    source = gl.full(values.shape, branch, gl.int32, values.type.layout)
    bits = gl.inline_asm_elementwise(
        asm="{ shfl.sync.idx.b32 $0, $2, $4, 31, -1; mov.b32 $1, $0; }",
        constraints="=&r,=r,r,r,r,r",
        args=[values.to(gl.int32, bitcast=True), source],
        dtype=gl.int32,
        is_pure=True,
        pack=2,
    )
    return bits.to(gl.float32, bitcast=True)


@gluon.jit
def _hc_mix_from_sum_kernel(
    logits,
    residual,
    sums,
    norm_weight,
    mixed,
    next_sums,
    rows,
    H: gl.constexpr,
    EPS: gl.constexpr,
    BLOCK: gl.constexpr,
):
    # This is the shell's existing structural H contract, not a model shape.
    # Each warp covers 64 consecutive hidden coordinates without crossing a
    # token, so four lane-carried inverses can be shared by shuffle only.
    gl.static_assert(H > 0 and H % 512 == 0)
    gl.static_assert(BLOCK == 256)
    layout: gl.constexpr = gl.BlockedLayout([2], [32], [4], [0])
    # H is divisible by BLOCK, so a CTA never straddles tokens. Compute local
    # coordinates in int32 without forming the potentially overflowing pid*BLOCK.
    pid = gl.program_id(0)
    lane_offset = gl.arange(0, BLOCK, layout=layout)
    row = pid // (H // BLOCK)
    hidden = (pid % (H // BLOCK)) * BLOCK + lane_offset
    row64 = row.to(gl.int64)
    live = row < rows
    carrier = (lane_offset // 2) % 4
    ss = gl.load(sums + row64 * 4 + carrier, live, other=0.0)
    # CURRENT and NEXT sums do not alias. Clear NEXT before the Mix math;
    # exactly one owner writes each (token, branch), before the later Apply.
    for branch in gl.static_range(4):
        gl.store(next_sums + row64 * 4 + branch, 0.0, live & (pid % (H // BLOCK) == 0))
    inv_carrier = gl.rsqrt(ss * (1.0 / H) + EPS)
    # RMS storage is [H,4], shared with small-T. Each lane owns two hidden
    # coordinates: load their eight BF16/FP16 values with one aligned LDG128,
    # then repack the two values of each branch without changing precision.
    w0, w1, w2, w3 = gl.inline_asm_elementwise(
        asm="""{
            .reg .b32 p0, p1, p2, p3;
            ld.global.v4.b32 {p0, p1, p2, p3}, [$4];
            prmt.b32 $0, p0, p2, 0x5410;
            prmt.b32 $1, p0, p2, 0x7632;
            prmt.b32 $2, p1, p3, 0x5410;
            prmt.b32 $3, p1, p3, 0x7632;
        }""",
        constraints="=r,=r,=r,=r,l,l",
        args=[norm_weight + hidden * 4],
        dtype=(
            norm_weight.dtype.element_ty,
            norm_weight.dtype.element_ty,
            norm_weight.dtype.element_ty,
            norm_weight.dtype.element_ty,
        ),
        is_pure=True,
        pack=2,
    )
    total = gl.full((BLOCK,), 0.0, gl.float32, layout)
    for branch in gl.static_range(4):
        offset = (row64 * 4 + branch) * H + hidden
        inv = _hc_mix_branch_inv(inv_carrier, branch)
        r = gl.load(residual + offset, live, other=0.0).to(gl.float32)
        if branch == 0:
            weight = w0
        elif branch == 1:
            weight = w1
        elif branch == 2:
            weight = w2
        else:
            weight = w3
        gamma = 1.0 + weight.to(gl.float32)
        norm = ((r * inv) * gamma).to(residual.dtype.element_ty).to(gl.float32)
        logit = gl.load(logits + offset, live, other=0.0).to(gl.float32)
        # Preserve the ordinary low-precision pointwise tensor boundaries.
        gate = (
            (1.0 / (1.0 + gl.exp(-logit))).to(residual.dtype.element_ty).to(gl.float32)
        )
        product = (gate * norm).to(residual.dtype.element_ty).to(gl.float32)
        total += product
    gl.store(mixed + row64 * H + hidden, total * 0.25, live)


class LargeTSumHCShell:
    """Five-kernel large-T HC operations around an explicit attention/MoE core.

    Weights are prepared after loading. All intermediates live in the supplied
    buffer set, so simultaneously live results require disjoint buffer sets.
    Each call resets NEXT sums in Mix, then accumulates in Apply. Epsilon is
    a consumer property; HCSumState can cross the small/large-T boundary.
    """

    @torch.no_grad()
    def __init__(self, source, *, down_weights=None, norm_weight_permuted=None):
        self.eps = source.config.rms_norm_eps
        if (
            not source.config.hc_per_branch_norm
            or not math.isfinite(self.eps)
            or self.eps <= 0
        ):
            raise ValueError(
                "large-T sum-state requires per-branch RMS and finite positive epsilon"
            )
        self.weights = (
            prepare_batched_down_weights(
                source.input_mix_weight_down.weight,
                source.block_inject_weight.weight,
                source.hc_norm.weight,
                source.hc_count,
            )
            if down_weights is None
            else down_weights
        )
        w = self.weights
        if w.hidden_size % 512 or w.hidden_size > 16384:
            raise ValueError("Apply/sum requires H a positive multiple of 512 <=16384")
        up, norm = source.input_mix_weight_up.weight, source.hc_norm.weight
        if (
            up.shape != (4 * w.hidden_size, w.lowrank)
            or up.dtype != w.matrix.dtype
            or up.device != w.matrix.device
        ):
            raise ValueError(
                "Up weights must match prepared Down dimensions/device/dtype"
            )
        if norm_weight_permuted is None:
            norm_weight_permuted = norm.view(4, w.hidden_size).T.contiguous().flatten()
        if (
            norm_weight_permuted.shape != (4 * w.hidden_size,)
            or norm_weight_permuted.dtype != w.matrix.dtype
            or norm_weight_permuted.device != w.matrix.device
            or norm_weight_permuted.data_ptr() % 16
        ):
            raise ValueError(
                "shared [H,4] RMS storage must match prepared Down weights"
            )
        if not up.is_contiguous() or not norm_weight_permuted.is_contiguous():
            raise ValueError("shared Up/RMS weights must be contiguous")
        # Up needs no conversion. RMS has one [H,4] CUDA layout for both paths.
        # Weight replacement requires preparation and graph recapture.
        self.up_weight = up.detach()
        self.norm_weight = norm_weight_permuted

    def _residual_shape(self, residual):
        w = self.weights
        if (
            residual.ndim != 2
            or residual.shape[1] != 4 * w.hidden_size
            or residual.dtype != w.matrix.dtype
            or residual.device != w.matrix.device
            or not residual.is_contiguous()
            or residual.data_ptr() % 32
            or residual.shape[0] >= 2**31
        ):
            raise ValueError("invalid large-T residual metadata")
        _validate_batched_rows(residual.shape[0], w)
        return residual.shape[0]

    def bootstrap(self, residual):
        t = self._residual_shape(residual)
        sums = torch.empty((t, 4), device=residual.device, dtype=torch.float32)
        if t:
            _bootstrap_module(4, self.weights.hidden_size, residual.dtype).run(
                residual, sums
            )
        return HCSumState(residual, sums)

    def allocate_buffers(self, rows):
        if type(rows) is not int or not (rows == 0 or 24 < rows < 2**31):
            raise ValueError("large-T shell requires T>24 (or empty input)")
        w = self.weights
        partial, down, alpha = allocate_batched_down_buffers(rows, w)

        def new(shape, dtype=None):
            return torch.empty(
                shape, device=w.matrix.device, dtype=dtype or w.matrix.dtype
            )

        return HCLargeSumBuffers(
            new((rows, w.hidden_size)),
            alpha,
            HCSumState(new((rows, 4 * w.hidden_size)), new((rows, 4), torch.float32)),
            partial,
            down,
            new((rows, 4 * w.hidden_size)),
        )

    def _validate(self, state, buffers):
        if not isinstance(state, HCSumState) or not isinstance(
            buffers, HCLargeSumBuffers
        ):
            raise ValueError("explicit HCSumState / HCLargeSumBuffers required")
        w = self.weights
        t = self._residual_shape(state.residual)
        if 0 < t <= 24:
            raise ValueError("use the existing small-T shell for T<=24")
        specs = (
            (state.residual, (t, 4 * w.hidden_size), w.matrix.dtype),
            (state.sum_sq, (t, 4), torch.float32),
            (buffers.mixed, (t, w.hidden_size), w.matrix.dtype),
            (buffers.alpha, (t, 4), torch.float32),
            (buffers.output.residual, (t, 4 * w.hidden_size), w.matrix.dtype),
            (buffers.output.sum_sq, (t, 4), torch.float32),
            (buffers.partial, (4, t, w.matrix.shape[-1]), torch.float32),
            (buffers.down, (t, w.lowrank), w.matrix.dtype),
            (buffers.logits, (t, 4 * w.hidden_size), w.matrix.dtype),
        )
        for tensor, shape, dtype in specs:
            if (
                tensor.shape != shape
                or tensor.dtype != dtype
                or tensor.device != w.matrix.device
                or not tensor.is_contiguous()
                or tensor.data_ptr() % 32
            ):
                raise ValueError("large-T buffer metadata mismatch")
        tensors = [x[0] for x in specs]
        for i, output in enumerate(tensors[2:], start=2):
            if any(
                _overlap(output, x)
                for x in tensors[:i] + [w.matrix, self.up_weight, self.norm_weight]
            ):
                raise ValueError(
                    "large-T writable buffers must not alias inputs/outputs/weights"
                )
        return t

    def mix(self, state, buffers):
        t = self._validate(state, buffers)
        if not t:
            return buffers.mixed, buffers.alpha
        w = self.weights
        batched_down_from_sum(
            state.residual,
            state.sum_sq,
            w,
            (buffers.partial, buffers.down, buffers.alpha),
            self.eps,
        )
        torch.mm(buffers.down, self.up_weight.T, out=buffers.logits)
        _hc_mix_from_sum_kernel[(triton.cdiv(t * w.hidden_size, 256),)](
            buffers.logits,
            state.residual,
            state.sum_sq,
            self.norm_weight,
            buffers.mixed,
            buffers.output.sum_sq,
            t,
            H=w.hidden_size,
            EPS=self.eps,
            BLOCK=256,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return buffers.mixed, buffers.alpha

    def combine(self, state, buffers, block_output):
        """Apply a core output using alpha and NEXT sums prepared by ``mix``.

        The caller owns the core and buffer lifetime. Do not invoke combine
        twice without another mix/reset; it accumulates into NEXT sums.
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
