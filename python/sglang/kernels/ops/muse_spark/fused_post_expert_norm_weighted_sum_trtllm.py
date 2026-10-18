"""Fused post-expert norm + weighted sum for TRTLLM MoE finalize.

Fuses three operations per token when using TRTLLM MoE with ``do_finalize=False``:
1. Gather expert output from permuted positions (unpermute)
2. Apply weight-first RMSNorm (multiply by weight+1.0, then normalize)
3. Weighted accumulation across top-K experts

Backends:
- native: pure PyTorch (loop over experts)
- cutedsl: vectorized loads, FP32 accumulation and optional DSMEM reduction
- helion: model-specific dynamic-shape implementation
"""

from __future__ import annotations

import logging

import torch

from sglang.kernels.fused_op import BaseFusedOp
from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.kernels.ops.muse_spark import (
    fi_trtllm_moe_finalize_with_rmsnorm_cutedsl,
)
from sglang.kernels.ops.muse_spark.fi_trtllm_moe_finalize_with_rmsnorm_cutedsl import (
    FinalizePushWorkspace,
)
from sglang.kernels.spec import CapabilityRequirement, KernelBackend
from sglang.srt.runtime_context import get_forward

logger = logging.getLogger(__name__)
_missing_cutedsl_warning_logged = False

# Upper bound on the finalize grid (num_tokens * cluster_size) for launching with
# PDL. PDL blocks camp on their SM slots in griddepcontrol_wait and cannot
# retire until the producer grid completes, so a grid larger than what fits
# resident in one wave can occupy every slot and starve the producer GEMM's
# remaining thread-block clusters — a device-side deadlock that stalls this
# rank's stream and, through the MoE all-to-all combine, every EP peer.
# 1024 blocks stay well
# below one resident wave (~148 SMs x 16 blocks/SM), preserving the PDL
# launch-latency win for decode-sized batches while keeping prefill-sized
# grids on the plain stream-ordered path.
_PDL_MAX_NUM_BLOCKS = 1024


def _should_use_pdl(num_blocks: int) -> bool:
    return (
        not getattr(get_forward(), "sp_active", False)
        and is_arch_support_pdl()
        and num_blocks <= _PDL_MAX_NUM_BLOCKS
    )


def _ar_launch_config(
    num_tokens: int, controller_count: int, cluster_size=None, hidden_dim=4096
):
    if cluster_size is not None and cluster_size not in (1, 2, 4, 8, 16):
        raise ValueError("Unsupported cluster size")
    if cluster_size is None:
        cluster_size = (
            8
            if num_tokens <= 16
            else 4
            if num_tokens <= 32
            else 2
            if num_tokens <= 64
            else 1
        )
        while cluster_size > 1 and num_tokens * cluster_size > controller_count:
            cluster_size //= 2
        # Each CTA owns at least one warp of 8-element BF16 vectors.
        while cluster_size > 1 and hidden_dim % (cluster_size * 256):
            cluster_size //= 2
    if hidden_dim <= 0 or hidden_dim % (cluster_size * 256):
        raise ValueError("Finalize AR requires H divisible by cluster_size * 256")
    threads = 512 // cluster_size
    while hidden_dim % (threads * cluster_size * 8):
        threads //= 2
    return cluster_size, threads


class FusedPostExpertNormWeightedSumTRTLLM(BaseFusedOp):
    """Fused unpermute + RMSNorm + weighted sum for TRTLLM MoE finalize.

    This op handles the post-processing of TRTLLM MoE output when called with
    ``do_finalize=False``. It combines unpermuting from the TRT-LLM permuted
    layout, applying RMSNorm, and weighted accumulation across top-K experts
    into a single kernel.

    The RMSNorm form is configurable: weight-first (``rms_norm(weight * x)``)
    or standard (``x / rms(x) * weight``). Eps is also configurable so the
    kernel reproduces the same math as the Python ``MuseSpark1xRMSNorm`` modules.

    Args:
        topk: Number of experts per token.
        hidden_dim: Output hidden dimension (may differ from gemm2 padded width).
        eps: RMSNorm epsilon. Must match the ``post_expert_norm.eps`` of the
            model being served (OSS default ``1e-8``).
        weight_first: If True, implements weight-first ``rms_norm(weight*x)``;
            if False, implements standard ``rms_norm(x) * weight``.
        cluster_size: Optional explicit CTA cluster size. By default, select
            from token count for H=4096; other shapes use a single CTA.
    """

    op = "muse_spark.post_expert_finalize"
    priority = (KernelBackend.CUTE_DSL, KernelBackend.HELION)
    capabilities = {
        KernelBackend.CUTE_DSL: {CapabilityRequirement.CUDA},
        KernelBackend.HELION: {CapabilityRequirement.CUDA},
    }

    def __init__(
        self,
        topk: int,
        hidden_dim: int,
        eps: float = 1e-8,
        weight_first: bool = True,
        cluster_size: int | None = None,
    ):
        super().__init__()
        self.topk = topk
        self.hidden_dim = hidden_dim
        self.eps = float(eps)
        self.weight_first = bool(weight_first)
        self.cluster_size = cluster_size

    def backend_eligible(self, backend, *args, **kwargs) -> bool:
        if not super().backend_eligible(backend, *args, **kwargs):
            return False
        if backend is KernelBackend.HELION:
            return self._helion_eligible(*args, **kwargs)
        if backend is not KernelBackend.CUTE_DSL:
            return True
        if not fi_trtllm_moe_finalize_with_rmsnorm_cutedsl.HAS_CUTEDSL:
            global _missing_cutedsl_warning_logged
            if not _missing_cutedsl_warning_logged:
                _missing_cutedsl_warning_logged = True
                logger.warning(
                    "CuTeDSL dependencies are unavailable; falling back to the "
                    "PyTorch NVFP4 MoE finalize implementation."
                )
            return False
        gemm2_output = args[0] if args else kwargs["gemm2_output"]
        return gemm2_output.is_cuda

    def forward_native(
        self,
        gemm2_output: torch.Tensor,
        expert_weights: torch.Tensor,
        expanded_idx_to_permuted_idx: torch.Tensor,
        rms_norm_weight: torch.Tensor,
        output: torch.Tensor,
        *,
        ar_workspace: FinalizePushWorkspace | None = None,
    ) -> None:
        """Pure PyTorch fallback for fused finalize + RMSNorm.

        Args:
            gemm2_output: Permuted GEMM2 output [total_permuted_tokens, hidden_dim_padded].
            expert_weights: Router weights [num_tokens, topk] or [num_tokens * topk], BF16 or FP32.
            expanded_idx_to_permuted_idx: Index mapping [num_tokens * topk], int32.
                -1 indicates the expert should be skipped.
            rms_norm_weight: RMSNorm weight [hidden_dim] with 1.0 pre-added.
            output: Output tensor [num_tokens, hidden_dim]. Zeroed by this function.
        """
        topk = self.topk
        hidden_dim = self.hidden_dim
        num_tokens = output.shape[0]
        idx_2d = expanded_idx_to_permuted_idx.view(num_tokens, topk)
        weights_2d = expert_weights.view(num_tokens, topk).to(torch.float32)

        output.zero_()
        for k in range(topk):
            perm_idx = idx_2d[:, k]  # [num_tokens]
            # No .any() early-out: a host sync here breaks CUDA graph capture;
            # the `valid` mask below already zeroes skipped experts.
            valid = perm_idx >= 0
            # Gather expert output rows
            gathered = gemm2_output[perm_idx.clamp(min=0)][:, :hidden_dim]
            w = rms_norm_weight.float()
            if self.weight_first:
                # Weight-first: rms_norm(weight * x)
                scaled = gathered.float() * w
                inv_rms = torch.rsqrt(
                    scaled.pow(2).mean(dim=-1, keepdim=True) + self.eps
                )
                normalized = scaled * inv_rms
            else:
                # Standard: rms_norm(x) * weight
                inv_rms = torch.rsqrt(
                    gathered.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
                )
                normalized = gathered.float() * inv_rms * w
            # Weighted accumulation. Select, don't multiply-mask: invalid
            # lanes gather garbage rows (clamp(0) on a possibly fully
            # unwritten torch.empty gemm2_output — a rank whose local experts
            # got zero hits never writes any row) and 0 * NaN = NaN would
            # poison the sum. torch.where drops the unselected branch without
            # a host-sync branch, so this stays stream-capture safe.
            w = weights_2d[:, k : k + 1]
            contribution = torch.where(
                valid.unsqueeze(-1), w * normalized, torch.zeros_like(normalized)
            ).to(output.dtype)
            output += contribution
        if ar_workspace is not None:
            torch.distributed.all_reduce(output, group=ar_workspace.group)

    def forward_cute_dsl(
        self,
        gemm2_output: torch.Tensor,
        expert_weights: torch.Tensor,
        expanded_idx_to_permuted_idx: torch.Tensor,
        rms_norm_weight: torch.Tensor,
        output: torch.Tensor,
        *,
        ar_workspace: FinalizePushWorkspace | None = None,
    ) -> None:
        """CuTeDSL kernel: vectorized unpermute + RMSNorm + weighted accumulate.

        Args:
            gemm2_output: Permuted GEMM2 output [total_permuted_tokens, hidden_dim_padded].
            expert_weights: Router weights [num_tokens, topk] or [num_tokens * topk], BF16 or FP32.
            expanded_idx_to_permuted_idx: Index mapping [num_tokens * topk], int32.
                -1 indicates the expert should be skipped.
            rms_norm_weight: RMSNorm weight [hidden_dim] with 1.0 pre-added.
            output: Output tensor [num_tokens, hidden_dim].
            ar_workspace: Optional workspace for all-reduced output.
        """
        if ar_workspace is not None and gemm2_output.shape[1] % 8:
            if not ar_workspace.fallback_logged:
                logger.warning(
                    "Finalize AR fusion requires GEMM2 padding divisible by 8; "
                    "using separate finalize and AR."
                )
                ar_workspace.fallback_logged = True
            # The model delegated AR to this op; preserve that ownership on fallback.
            self.forward_cute_dsl(
                gemm2_output,
                expert_weights,
                expanded_idx_to_permuted_idx,
                rms_norm_weight,
                output,
            )
            torch.distributed.all_reduce(output, group=ar_workspace.group)
            return

        import cuda.bindings.driver as cuda
        from cutlass import Int32
        from cutlass.cute.runtime import make_ptr

        kernel_module = fi_trtllm_moe_finalize_with_rmsnorm_cutedsl

        assert gemm2_output.is_cuda
        cutlass_dtype = kernel_module._torch_to_cutlass_dtype[gemm2_output.dtype]

        num_tokens = output.shape[0]
        hidden_dim_padded = gemm2_output.shape[1]
        cluster_size = self.cluster_size
        if cluster_size is None:
            cluster_size = 1
            if (
                self.hidden_dim == 4096
                and kernel_module._get_sm_version(output.device) >= 90
            ):
                cluster_size = 8 if num_tokens <= 32 else 4 if num_tokens <= 128 else 1
        if num_tokens == 0:
            return
        threads_per_row = None
        if ar_workspace is not None:
            cluster_size, threads_per_row = _ar_launch_config(
                num_tokens,
                ar_workspace.controller_count,
                self.cluster_size,
                self.hidden_dim,
            )
            ar_workspace.validate(output, cluster_size)
            if not ar_workspace.fusion_logged:
                logger.info(
                    "Using fused MoE finalize + all-reduce: tokens=%s, capacity=%s, "
                    "hidden_dim=%s, cluster_size=%s",
                    num_tokens,
                    ar_workspace.max_tokens,
                    self.hidden_dim,
                    cluster_size,
                )
                ar_workspace.fusion_logged = True

        assert expert_weights.dtype in (torch.bfloat16, torch.float32), (
            f"expert_weights must be BF16 or FP32, got {expert_weights.dtype}"
        )
        assert expert_weights.is_contiguous(), "expert_weights must be contiguous"
        expert_weights_flat = expert_weights.view(-1)
        expert_weights_dtype = kernel_module._torch_to_cutlass_dtype[
            expert_weights.dtype
        ]

        assert expanded_idx_to_permuted_idx.dtype == torch.int32, (
            f"expanded_idx_to_permuted_idx must be int32, got "
            f"{expanded_idx_to_permuted_idx.dtype}"
        )
        assert expanded_idx_to_permuted_idx.is_contiguous(), (
            "expanded_idx_to_permuted_idx must be contiguous"
        )
        idx_flat = expanded_idx_to_permuted_idx.view(-1)

        torch_stream = torch.cuda.current_stream()
        stream = cuda.CUstream(torch_stream.cuda_stream)

        compiled_kernel = kernel_module._get_compiled_kernel(
            cutlass_dtype,
            num_tokens,
            self.hidden_dim,
            self.topk,
            hidden_dim_padded,
            stream,
            threads_per_row=threads_per_row,
            use_pdl=_should_use_pdl(num_tokens * cluster_size),
            eps=self.eps,
            weight_first=self.weight_first,
            expert_weights_dtype=expert_weights_dtype,
            cluster_size=cluster_size,
            ar_world_size=0 if ar_workspace is None else ar_workspace.world_size,
            ar_multicast=ar_workspace is not None and ar_workspace.multicast,
            ar_controller_count=ar_workspace.controller_count
            if ar_workspace is not None
            else 0,
        )

        gemm2_ptr = make_ptr(cutlass_dtype, gemm2_output.data_ptr())
        expw_ptr = make_ptr(expert_weights_dtype, expert_weights_flat.data_ptr())
        idx_ptr = make_ptr(Int32, idx_flat.data_ptr())
        rmsw_ptr = make_ptr(cutlass_dtype, rms_norm_weight.data_ptr())
        out_ptr = make_ptr(cutlass_dtype, output.data_ptr())

        compiled_kernel(
            gemm2_ptr,
            expw_ptr,
            idx_ptr,
            rmsw_ptr,
            out_ptr,
            Int32(num_tokens),
            stream,
            None if ar_workspace is None else ar_workspace.arguments(),
        )

    def _helion_eligible(
        self,
        gemm2_output: torch.Tensor,
        expert_weights: torch.Tensor,
        expanded_idx_to_permuted_idx: torch.Tensor,
        rms_norm_weight: torch.Tensor,
        output: torch.Tensor,
        *,
        ar_workspace: FinalizePushWorkspace | None = None,
    ) -> bool:
        from sglang.kernels.ops.muse_spark.helion_runtime import helion_supported

        return (
            helion_supported("post_expert_finalize", output)
            and output.shape[0] > 0
            and self.topk == 8
            and self.hidden_dim == 4096
            and gemm2_output.dtype == torch.bfloat16
            and expert_weights.dtype == torch.bfloat16
            and expanded_idx_to_permuted_idx.dtype == torch.int32
            and rms_norm_weight.dtype == torch.bfloat16
            and output.dtype == torch.bfloat16
            and gemm2_output.is_contiguous()
            and expert_weights.is_contiguous()
            and expanded_idx_to_permuted_idx.is_contiguous()
            and rms_norm_weight.is_contiguous()
            and output.is_contiguous()
        )

    def forward_helion(
        self,
        gemm2_output: torch.Tensor,
        expert_weights: torch.Tensor,
        expanded_idx_to_permuted_idx: torch.Tensor,
        rms_norm_weight: torch.Tensor,
        output: torch.Tensor,
        *,
        ar_workspace: FinalizePushWorkspace | None = None,
    ) -> None:
        """Run Helion finalize, then a separate AR when delegated by the model."""

        from sglang.kernels.ops.muse_spark.helion_post_expert import (
            POST_EXPERT_FINALIZE_CONFIGS,
            _post_expert_finalize_helion_kernel,
        )
        from sglang.kernels.ops.muse_spark.helion_runtime import (
            run_pinned_helion,
        )

        run_pinned_helion(
            _post_expert_finalize_helion_kernel,
            POST_EXPERT_FINALIZE_CONFIGS,
            (
                gemm2_output,
                expert_weights,
                expanded_idx_to_permuted_idx,
                rms_norm_weight,
                output,
                self.topk,
                self.hidden_dim,
                self.eps,
                self.weight_first,
            ),
            num_tokens=output.shape[0],
            launch_pdl=_should_use_pdl(output.shape[0]),
        )
        if ar_workspace is not None:
            # The model skips its outer AR when it supplies a workspace.
            torch.distributed.all_reduce(output, group=ar_workspace.group)
