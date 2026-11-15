"""Fused post-expert norm + weighted sum for TRTLLM MoE finalize.

Fuses three operations per token when using TRTLLM MoE with ``do_finalize=False``:
1. Gather expert output from permuted positions (unpermute)
2. Apply weight-first RMSNorm (multiply by weight+1.0, then normalize)
3. Weighted accumulation across top-K experts

Backends:
- native: pure PyTorch (loop over experts)
- cutedsl: vectorized 128-bit tiled copies with TensorSSA reductions
"""

from __future__ import annotations

import logging

import torch

from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.srt.rocm_muse_spark import fi_trtllm_moe_finalize_with_rmsnorm_cutedsl
from sglang.srt.rocm_muse_spark.base_fused_op import BaseFusedOp, FusedOpBackend

logger = logging.getLogger(__name__)
_missing_cutedsl_warning_logged = False

# Upper bound on the finalize grid (one block per token) for launching with
# PDL. PDL blocks camp on their SM slots in griddepcontrol_wait and cannot
# retire until the producer grid completes, so a grid larger than what fits
# resident in one wave can occupy every slot and starve the producer GEMM's
# remaining thread-block clusters — a device-side deadlock that stalls this
# rank's stream and, through the MoE all-to-all combine, every EP peer.
# 1024 blocks stay well
# below one resident wave (~148 SMs x 16 blocks/SM), preserving the PDL
# launch-latency win for decode-sized batches while keeping prefill-sized
# grids on the plain stream-ordered path.
_PDL_MAX_NUM_TOKENS = 1024


def _should_use_pdl(num_tokens: int) -> bool:
    return is_arch_support_pdl() and num_tokens <= _PDL_MAX_NUM_TOKENS


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
    """

    def __init__(
        self,
        topk: int,
        hidden_dim: int,
        eps: float = 1e-8,
        weight_first: bool = True,
    ):
        self.topk = topk
        self.hidden_dim = hidden_dim
        self.eps = float(eps)
        self.weight_first = bool(weight_first)
        super().__init__()

    def _backend_runtime_eligible(self, backend, *args, **kwargs) -> bool:
        if backend is not FusedOpBackend.CUTEDSL:
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
    ) -> None:
        """Pure PyTorch fallback for fused finalize + RMSNorm.

        Args:
            gemm2_output: Permuted GEMM2 output [total_permuted_tokens, hidden_dim_padded].
            expert_weights: Router weights [num_tokens, topk] or [num_tokens * topk], bfloat16.
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

    def forward_cutedsl(
        self,
        gemm2_output: torch.Tensor,
        expert_weights: torch.Tensor,
        expanded_idx_to_permuted_idx: torch.Tensor,
        rms_norm_weight: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """CuTeDSL kernel: vectorized unpermute + RMSNorm + weighted accumulate.

        Args:
            gemm2_output: Permuted GEMM2 output [total_permuted_tokens, hidden_dim_padded].
            expert_weights: Router weights [num_tokens, topk] or [num_tokens * topk], bfloat16.
            expanded_idx_to_permuted_idx: Index mapping [num_tokens * topk], int32.
                -1 indicates the expert should be skipped.
            rms_norm_weight: RMSNorm weight [hidden_dim] with 1.0 pre-added.
            output: Output tensor [num_tokens, hidden_dim].
        """
        import cuda.bindings.driver as cuda
        from cutlass import BFloat16, Int32
        from cutlass.cute.runtime import make_ptr

        kernel_module = fi_trtllm_moe_finalize_with_rmsnorm_cutedsl

        assert gemm2_output.is_cuda
        cutlass_dtype = kernel_module._torch_to_cutlass_dtype[gemm2_output.dtype]

        num_tokens = output.shape[0]
        hidden_dim_padded = gemm2_output.shape[1]

        assert expert_weights.dtype == torch.bfloat16, (
            f"expert_weights must be bfloat16, got {expert_weights.dtype}"
        )
        assert expert_weights.is_contiguous(), "expert_weights must be contiguous"
        expert_weights_flat = expert_weights.view(-1)

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
            self.hidden_dim,
            self.topk,
            hidden_dim_padded,
            stream,
            use_pdl=_should_use_pdl(num_tokens),
            eps=self.eps,
            weight_first=self.weight_first,
        )

        gemm2_ptr = make_ptr(cutlass_dtype, gemm2_output.data_ptr())
        expw_ptr = make_ptr(BFloat16, expert_weights_flat.data_ptr())
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
        )
