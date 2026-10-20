"""Fused post-norm + residual-add + pre-norm operation.

Fuses the repeated pattern:
    normed_branch = post_norm(branch)
    new_stream    = residual_add(stream, normed_branch)
    output        = pre_norm(new_stream.to(branch_dtype))

This pattern appears twice per decoder layer:
1. post_attn_norm + add_attn + pre_ffn_norm   (intra-layer)
2. post_ffn_norm  + add_ffn  + pre_attn_norm  (cross-layer, deferred)
"""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl

from sglang.kernels.fused_op import BaseFusedOp, get_fused_op_backend
from sglang.kernels.ops.muse_spark import fused_op_tuning
from sglang.kernels.ops.muse_spark.all_gather_norm_residual import (
    all_gather_norm_residual,
)
from sglang.kernels.spec import CapabilityRequirement, KernelBackend
from sglang.srt.layers.muse_spark.muse_spark_layers import (
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xRMSNormWithInputScale,
)

logger = logging.getLogger(__name__)


def _fused_norm_residual_config(
    hidden_size: int,
    m_bucket: int,
    num_tokens: int,
    post_norm_is_input_scale: bool,
) -> tuple[int, int, dict]:
    """Resolve launch parameters for the exact residual-norm shape."""
    cfg = fused_op_tuning.resolve(
        "fused_norm_residual",
        {"num_warps": 8, "num_stages": 1},
        hidden_size=hidden_size,
        m_bucket=m_bucket,
        num_tokens=num_tokens,
        post_norm_is_input_scale=post_norm_is_input_scale,
    )
    extra = fused_op_tuning.split_extra(cfg, ("num_warps", "num_stages"))
    return cfg["num_warps"], cfg["num_stages"], extra


def _jit_launch_config_ok(hidden_size: int) -> bool:
    from sglang.kernels.ops.muse_spark.norm_residual import _launch_config

    try:
        _launch_config(hidden_size, 1)
    except RuntimeError:
        return False
    return True


@triton.jit
def _fused_norm_residual_triton_kernel(
    BRANCH,
    STREAM,
    OUTPUT,
    NEW_STREAM_OUT,
    ROUTER_OUT,
    POST_WEIGHT,
    PRE_WEIGHT,
    ALPHA,
    BETA,
    post_norm_eps,
    pre_norm_eps,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    POST_NORM_IS_INPUT_SCALE: tl.constexpr,
    USE_RESIDUAL_GATE: tl.constexpr,
    RESIDUAL_GATE_CHANNEL_WISE: tl.constexpr,
    POST_NORM_HAS_WEIGHT: tl.constexpr,
    PRE_NORM_HAS_WEIGHT: tl.constexpr,
    HAS_ROUTER: tl.constexpr,
    PRE_NORM_BF16: tl.constexpr,
    M_BUCKET: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    branch = tl.load(BRANCH + row * H + offs, mask=mask, other=0.0).to(tl.float32)

    if POST_NORM_IS_INPUT_SCALE:
        post_input = branch
        if POST_NORM_HAS_WEIGHT:
            post_weight = tl.load(POST_WEIGHT + offs, mask=mask, other=0.0).to(
                tl.float32
            )
            post_input = (post_input * post_weight).to(tl.bfloat16).to(tl.float32)
        post_sum_sq = tl.sum(tl.where(mask, post_input * post_input, 0.0))
        post_rstd = tl.rsqrt(post_sum_sq / H + post_norm_eps)
        normed_branch = post_input * post_rstd
    else:
        post_sum_sq = tl.sum(tl.where(mask, branch * branch, 0.0))
        post_rstd = tl.rsqrt(post_sum_sq / H + post_norm_eps)
        normed_branch = branch * post_rstd
        if POST_NORM_HAS_WEIGHT:
            post_weight = tl.load(POST_WEIGHT + offs, mask=mask, other=0.0).to(
                tl.float32
            )
            normed_branch = normed_branch * post_weight

    # MuseSpark1x norm modules return the input dtype. Branch tensors are bf16,
    # so preserve the native bf16 round-trip before residual add.
    normed_branch = normed_branch.to(tl.bfloat16).to(tl.float32)

    stream = tl.load(STREAM + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    if USE_RESIDUAL_GATE:
        if RESIDUAL_GATE_CHANNEL_WISE:
            alpha = tl.load(ALPHA + offs, mask=mask, other=0.0).to(tl.float32)
            beta = tl.load(BETA + offs, mask=mask, other=0.0).to(tl.float32)
        else:
            alpha = tl.load(ALPHA).to(tl.float32)
            beta = tl.load(BETA).to(tl.float32)
        new_stream = alpha * stream + beta * normed_branch
    else:
        new_stream = stream + normed_branch

    tl.store(NEW_STREAM_OUT + row * H + offs, new_stream, mask=mask)

    pre_input = new_stream
    if PRE_NORM_BF16:
        pre_input = pre_input.to(tl.bfloat16).to(tl.float32)

    pre_sum_sq = tl.sum(tl.where(mask, pre_input * pre_input, 0.0))
    pre_rstd = tl.rsqrt(pre_sum_sq / H + pre_norm_eps)
    normed = pre_input * pre_rstd
    if PRE_NORM_HAS_WEIGHT:
        pre_weight = tl.load(PRE_WEIGHT + offs, mask=mask, other=0.0).to(tl.float32)
        normed = normed * pre_weight

    if PRE_NORM_BF16:
        normed = normed.to(tl.bfloat16).to(tl.float32)

    tl.store(OUTPUT + row * H + offs, normed, mask=mask)
    if HAS_ROUTER:
        # The router input is the branch-dtype output upcast, not the fp32
        # intermediate, so this stays bit-exact with the unfused cast.
        tl.store(
            ROUTER_OUT + row * H + offs,
            normed.to(tl.bfloat16).to(tl.float32),
            mask=mask,
        )


# ---------------------------------------------------------------------------
# JIT compilation for the CUDA C kernel
# ---------------------------------------------------------------------------
class FusedPostNormResidualPreNorm(BaseFusedOp):
    """Fuses: post_norm(branch) + residual_add(stream, normed) + pre_norm(new_stream).

    Backends:
    - ``forward_native``: sequential PyTorch calls (reference implementation).
    - ``forward_torch_compile``: ``torch.compile``'d native.
    - ``forward_triton``: the fused Triton kernel.
    - ``forward_jit``: the JIT CUDA kernel.
    """

    priority = (KernelBackend.JIT, KernelBackend.TRITON)
    capabilities = {
        KernelBackend.JIT: {CapabilityRequirement.CUDA},
        KernelBackend.TRITON: {
            CapabilityRequirement.CUDA,
        },
    }

    def __init__(self, post_norm, residual_add, pre_norm):
        super().__init__()
        # Store module references (weights are owned by the decoder layer module tree)
        object.__setattr__(self, "_post_norm", post_norm)
        object.__setattr__(self, "_residual_add", residual_add)
        object.__setattr__(self, "_pre_norm", pre_norm)

        # Introspection flags for CUDA C kernel template specialization
        self._post_norm_is_input_scale = isinstance(
            post_norm, MuseSpark1xRMSNormWithInputScale
        )
        self._use_residual_gate = isinstance(
            residual_add, MuseSpark1xNormPreservingResidualConnection
        )
        self._pre_norm_is_input_scale = isinstance(
            pre_norm, MuseSpark1xRMSNormWithInputScale
        )
        self._hidden_size = post_norm.hidden_size

    def backend_eligible(
        self,
        backend: KernelBackend,
        stream: torch.Tensor,
        branch: torch.Tensor,
        branch_dtype: torch.dtype = torch.bfloat16,
        router_dtype: torch.dtype | None = None,
        pre_norm_dtype: torch.dtype | None = None,
    ) -> bool:
        if not super().backend_eligible(
            backend,
            stream,
            branch,
            branch_dtype,
            router_dtype,
            pre_norm_dtype,
        ):
            return False
        if backend not in (KernelBackend.TRITON, KernelBackend.JIT):
            return True
        if branch.numel() == 0 or not self._triton_supported(
            stream,
            branch,
            branch_dtype,
            router_dtype,
            pre_norm_dtype,
        ):
            return False
        return backend is not KernelBackend.JIT or _jit_launch_config_ok(
            self._hidden_size
        )

    def forward_native(
        self,
        stream: torch.Tensor,
        branch: torch.Tensor,
        branch_dtype: torch.dtype = torch.bfloat16,
        router_dtype: torch.dtype | None = None,
        pre_norm_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Reference implementation: call each module in sequence.

        Args:
            stream: residual stream ``[T, H]`` in residual dtype (e.g. fp32).
            branch: raw output from attention or MoE ``[T, H]`` in compute dtype.
            branch_dtype: dtype to cast pre-norm input to (default ``torch.bfloat16``).
            router_dtype: when set, also return the *branch_dtype* output cast
                to this dtype (e.g. ``torch.float32`` for MoE router input),
                fusing the cast into this op instead of doing it downstream.
            pre_norm_dtype: when set, cast new_stream to this dtype before pre_norm
                (e.g. ``torch.bfloat16`` for the final layer's numerical parity).

        Returns:
            output: pre-normalized result for the next compute block ``[T, H]``
                in *branch_dtype*.
            new_stream: updated residual stream ``[T, H]`` in the original
                residual dtype.
            router_input: *output* cast to *router_dtype*, or ``None`` when
                *router_dtype* is not set.
        """
        normed_branch = self._post_norm(branch)
        new_stream = self._residual_add(stream, normed_branch)
        norm_input = (
            new_stream.to(pre_norm_dtype) if pre_norm_dtype is not None else new_stream
        )
        normed = self._pre_norm(norm_input)
        output = normed.to(branch_dtype)
        router_input = output.to(router_dtype) if router_dtype is not None else None
        return output, new_stream, router_input

    def _triton_supported(
        self,
        stream: torch.Tensor,
        branch: torch.Tensor,
        branch_dtype: torch.dtype,
        router_dtype: torch.dtype | None,
        pre_norm_dtype: torch.dtype | None,
    ) -> bool:
        return (
            branch.dtype == torch.bfloat16
            and branch_dtype == torch.bfloat16
            and stream.dtype == torch.float32
            and (router_dtype is None or router_dtype == torch.float32)
            and (pre_norm_dtype is None or pre_norm_dtype == torch.bfloat16)
            and branch.dim() == 2
            and stream.dim() == 2
            and branch.shape == stream.shape
            and branch.is_contiguous()
            and stream.is_contiguous()
        )

    def _ensure_cached_weights(self, branch_dtype: torch.dtype) -> None:
        """Cache the weight tensors the Triton kernel reads (called once)."""
        if hasattr(self, "_weights_cached"):
            return

        def cache(name: str, value: torch.Tensor) -> None:
            # nn.Module.__setattr__ registers Parameter values. These are
            # inference-only kernel views owned by the source modules, so keep
            # detached ordinary tensors outside this op's state_dict.
            object.__setattr__(self, name, value.detach())

        device = None
        # Find a parameter to determine device
        for mod in (self._post_norm, self._pre_norm):
            if hasattr(mod, "weight") and mod.weight is not None:
                device = mod.weight.device
                break
        if device is None and self._use_residual_gate:
            device = self._residual_add._alpha.device

        post_weight = (
            self._post_norm._get_effective_weight().to(branch_dtype).contiguous()
            if self._post_norm.has_weight
            else torch.empty(0, dtype=torch.bfloat16, device=device)
        )
        pre_weight = (
            self._pre_norm._get_effective_weight().to(branch_dtype).contiguous()
            if self._pre_norm.has_weight
            else torch.empty(0, dtype=torch.bfloat16, device=device)
        )
        cache("_cc_post_weight", post_weight)
        cache("_cc_pre_weight", pre_weight)
        if self._use_residual_gate:
            ra = self._residual_add
            if ra._alpha is not None:
                alpha, beta = ra._alpha, ra._beta
            else:
                # precompute_coefficients() hasn't been called yet — compute
                # alpha/beta from gate directly (same math, just not cached).
                with torch.no_grad():
                    w = ra.gate.view(1, -1).float() / ra.temperature
                    alpha, beta = ra._coefficients(w)
                    if not isinstance(alpha, torch.Tensor):
                        alpha = torch.tensor(alpha, device=device)
            cc_alpha = alpha.reshape(-1).float().contiguous()
            cc_beta = beta.reshape(-1).float().contiguous()
        else:
            cc_alpha = torch.empty(0, dtype=torch.float32, device=device)
            cc_beta = torch.empty(0, dtype=torch.float32, device=device)
        cache("_cc_alpha", cc_alpha)
        cache("_cc_beta", cc_beta)
        cache("_cc_empty_fp32", torch.empty(0, dtype=torch.float32, device=device))
        object.__setattr__(self, "_weights_cached", True)

    def forward_triton(
        self,
        stream: torch.Tensor,
        branch: torch.Tensor,
        branch_dtype: torch.dtype = torch.bfloat16,
        router_dtype: torch.dtype | None = None,
        pre_norm_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Fused Triton implementation for the BF16/FP32 path."""
        if branch.numel() == 0:
            return self.forward_native(
                stream,
                branch,
                branch_dtype,
                router_dtype,
                pre_norm_dtype=pre_norm_dtype,
            )

        if not self._triton_supported(
            stream,
            branch,
            branch_dtype,
            router_dtype,
            pre_norm_dtype,
        ):
            return self.forward_native(
                stream,
                branch,
                branch_dtype,
                router_dtype,
                pre_norm_dtype=pre_norm_dtype,
            )

        M, H = branch.shape

        self._ensure_cached_weights(branch_dtype)

        output = torch.empty(M, H, dtype=branch_dtype, device=branch.device)
        new_stream_out = torch.empty(M, H, dtype=torch.float32, device=branch.device)
        router_out = (
            torch.empty(M, H, dtype=torch.float32, device=branch.device)
            if router_dtype is not None
            else self._cc_empty_fp32
        )
        block_h = triton.next_power_of_2(H)
        # decode / mid / prefill regime for _fused_norm_residual_config.
        m_bucket = 0 if M <= 256 else (1 if M <= 2048 else 2)
        residual_gate_channel_wise = (
            self._use_residual_gate
            and self._residual_add.gate.numel() == self._hidden_size
        )

        num_warps, num_stages, extra = _fused_norm_residual_config(
            H, m_bucket, M, self._post_norm_is_input_scale
        )
        _fused_norm_residual_triton_kernel[(M,)](
            branch,
            stream,
            output,
            new_stream_out,
            router_out,
            self._cc_post_weight,
            self._cc_pre_weight,
            self._cc_alpha,
            self._cc_beta,
            self._post_norm.eps,
            self._pre_norm.eps,
            H,
            block_h,
            self._post_norm_is_input_scale,
            self._use_residual_gate,
            residual_gate_channel_wise,
            self._post_norm.has_weight,
            self._pre_norm.has_weight,
            router_dtype is not None,
            pre_norm_dtype is torch.bfloat16,
            m_bucket,
            num_warps=num_warps,
            num_stages=num_stages,
            **extra,
        )

        return (
            output,
            new_stream_out,
            (router_out if router_dtype is not None else None),
        )

    def forward_jit(
        self,
        stream: torch.Tensor,
        branch: torch.Tensor,
        branch_dtype: torch.dtype = torch.bfloat16,
        router_dtype: torch.dtype | None = None,
        pre_norm_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Fused JIT CUDA implementation for the BF16/FP32 path."""
        if branch.numel() == 0 or not self._triton_supported(
            stream, branch, branch_dtype, router_dtype, pre_norm_dtype
        ):
            return self.forward_native(
                stream,
                branch,
                branch_dtype,
                router_dtype,
                pre_norm_dtype=pre_norm_dtype,
            )

        from sglang.kernels.ops.muse_spark.norm_residual import (
            fused_norm_residual as _jit_fused_norm_residual,
        )

        self._ensure_cached_weights(branch_dtype)

        return _jit_fused_norm_residual(
            branch,
            stream,
            self._post_norm.eps,
            self._pre_norm.eps,
            post_norm_is_input_scale=self._post_norm_is_input_scale,
            post_weight=self._cc_post_weight if self._post_norm.has_weight else None,
            pre_weight=self._cc_pre_weight if self._pre_norm.has_weight else None,
            alpha=self._cc_alpha if self._use_residual_gate else None,
            beta=self._cc_beta if self._use_residual_gate else None,
            want_router=router_dtype is not None,
            pre_norm_bf16=pre_norm_dtype is torch.bfloat16,
        )

    def jit_selected(self) -> bool:
        """Whether ``forward`` takes ``forward_jit`` for this model's bf16 branch.

        The stream is always fp32, contiguous, and [M, H], so only the backend
        choice matters: JIT has top priority unless forced away or compiling.
        """
        return (
            get_fused_op_backend() in (None, KernelBackend.JIT)
            and not self.is_torch_compile
            and BaseFusedOp.backend_eligible(self, KernelBackend.JIT)
            and _jit_launch_config_ok(self._hidden_size)
        )

    def forward_with_ag(
        self, stream: torch.Tensor, local: torch.Tensor, comm
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the boundary to the TP all-gather of bf16 column slices ``local``."""
        self._ensure_cached_weights(torch.bfloat16)
        return all_gather_norm_residual(
            comm,
            local,
            stream,
            self._post_norm.eps,
            self._pre_norm.eps,
            post_norm_is_input_scale=self._post_norm_is_input_scale,
            post_weight=self._cc_post_weight if self._post_norm.has_weight else None,
            pre_weight=self._cc_pre_weight if self._pre_norm.has_weight else None,
            alpha=self._cc_alpha if self._use_residual_gate else None,
            beta=self._cc_beta if self._use_residual_gate else None,
        )
