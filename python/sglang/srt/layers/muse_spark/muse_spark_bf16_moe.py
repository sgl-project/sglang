"""BF16 routed experts for Muse Spark, on FlashInfer's TRT-LLM MoE kernel.

The NVFP4 checkpoint's ``exclude_modules`` keeps layers 0 and 61 of the 820B
export out of quantization, so their experts stay BF16. Muse's RMSNorm on each
expert output lands before the top-k sum, which the fused MoE finalize cannot
express, so the kernel runs with ``do_finalize=False`` and the FP4 path's
finalize combines the un-permuted output.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn

from sglang.kernels.ops.communication.muse_spark_moe_finalize_all_reduce import (
    FusedPostExpertNormWeightedSumTRTLLM,
)
from sglang.kernels.ops.moe.muse_spark_post_expert_finalize_triton import (
    fused_post_expert_norm_weighted_sum,
)
from sglang.srt.layers.moe import MoeRunnerBackend
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.muse_spark.muse_spark_fp4_moe import (
    finalize_expert_tp,
    finalize_with_post_expert_norm,
)
from sglang.srt.layers.utils.common import set_weight_attrs
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
    is_in_breakable_cuda_graph,
)
from sglang.srt.utils.triton_utils import next_power_of_2

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.layers.communication.muse_spark_finalize_workspace import (
        FinalizePushWorkspace,
    )
    from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
        FlashinferCombineInput,
        FlashinferDispatchOutput,
    )

# Cap on the batched-expert intermediate, in elements (64 MiB at bf16).
_EXPERT_GROUP_ELEMS = 32 << 20

# BlockMajorK blocks along K in 128-*byte* units, and the row shuffle is defined
# on the weight's byte view; both helpers therefore run over a uint8 view.
_EPILOGUE_TILE_M = 128
_BLOCK_K_BYTES = 128

_SWIGLU = 3
# Precomputed routing skips the routing kernel; the launcher branches on this
# only for Llama4, so any other value is inert.
_ROUTING_METHOD_UNUSED = 1

_TRITON_ROUTE_ELEMS = 256 << 20
_GROUPED_MIN_TOKENS = 4096
_GROUPED_WS_TOKENS = 17408
_GROUPED_DG = dict(use_psum_layout=True, expected_m_for_psum_layout=512)


@triton.jit
def _rows_kernel(
    src_ptr,
    rows_ptr,
    out_ptr,
    stride,
    n_ptr,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
    SILU_MUL: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    for r in range(tl.program_id(0), tl.load(n_ptr), tl.num_programs(0)):
        t = tl.load(rows_ptr + r)
        if t >= 0:
            if SILU_MUL:
                gu = tl.load(
                    src_ptr
                    + r.to(tl.int64) * 2 * H
                    + 2 * offs[:, None]
                    + tl.arange(0, 2)[None, :],
                    mask=(offs < H)[:, None],
                )
                g, u = tl.split(gu.to(tl.float32))
                v = (g / (1.0 + tl.exp(-g)) * u).to(out_ptr.dtype.element_ty)
            else:
                v = tl.load(src_ptr + t.to(tl.int64) * stride + offs, mask=offs < H)
            tl.store(out_ptr + r.to(tl.int64) * H + offs, v, mask=offs < H)


@triton.jit(do_not_specialize=["n"])
def _route_kernel(
    ids,
    local,
    ws,
    rows,
    eids,
    ntp,
    n,
    lo,
    E,
    EB: tl.constexpr,
    BM: tl.constexpr,
    CH: tl.constexpr,
):
    # One CTA: local expert per route (-1: other rank), expert-sorted BM-padded route order (any order within an expert)
    cnt = ws + n + tl.arange(0, EB) * 32  # a 128-byte line per expert
    tl.store(cnt, 0)
    tl.debug_barrier()
    for s in range(0, n, CH):
        o = s + tl.arange(0, CH)
        l = tl.load(ids + o, mask=o < n, other=-1) - lo
        l = tl.where((l >= 0) & (l < E) & (o < n), l, -1)
        tl.store(local + o, l, mask=o < n)
        tl.store(
            ws + o,
            tl.atomic_add(ws + n + l * 32, 1, mask=l >= 0, sem="relaxed"),
            mask=l >= 0,
        )
    tl.debug_barrier()
    pad = (tl.load(cnt, cache_modifier=".cg") + BM - 1) // BM * BM
    end, total = tl.cumsum(pad, 0), tl.sum(pad, 0)
    tl.store(ntp, total)
    for s in range(0, total, 128 * BM):
        p = s + tl.arange(0, 128) * BM
        tl.store(
            eids + p // BM,
            tl.sum((end[None, :] <= p[:, None]).to(tl.int32), 1),
            mask=p < total,
        )
    for s in range(0, total, CH):
        tl.store(rows + s + tl.arange(0, CH), n, mask=s + tl.arange(0, CH) < total)
    tl.store(cnt, end - pad)
    tl.debug_barrier()
    for s in range(0, n, CH):
        o = s + tl.arange(0, CH)
        l = tl.load(local + o, mask=o < n, other=-1)
        tl.store(
            rows + tl.load(ws + n + l * 32, mask=l >= 0) + tl.load(ws + o, mask=l >= 0),
            o,
            mask=l >= 0,
        )


def _trtllm_bf16_moe_supported(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    # The TRT-LLM Gen BF16 MoE cubins are SM10x-only.
    if torch.cuda.get_device_capability(device)[0] != 10:
        return False
    try:
        from flashinfer.fused_moe import trtllm_bf16_routed_moe  # noqa: F401
        from flashinfer.fused_moe.core import convert_to_block_layout  # noqa: F401
    except ImportError:
        logger.warning(
            "FlashInfer's TRT-LLM BF16 MoE is unavailable; Muse Spark's BF16 "
            "experts fall back to the PyTorch expert loop."
        )
        return False
    return True


def _triton_bf16_moe_supported(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    try:
        from sglang.srt.layers.moe.moe_runner.triton_utils import (  # noqa: F401
            fused_experts,
        )
    except ImportError:
        logger.warning(
            "SGLang's Triton fused MoE is unavailable; Muse Spark's BF16 "
            "experts fall back to the PyTorch expert loop."
        )
        return False
    return True


class MuseSpark1xBf16Experts(nn.Module):
    """Local BF16 expert weights, run by TRT-LLM or by the PyTorch loop."""

    def __init__(
        self,
        *,
        num_experts: int,
        num_local_experts: int,
        local_expert_offset: int,
        expert_hidden_size: int,
        intermediate_size: int,
        top_k: int,
        expert_tp_size: int = 1,
        expert_tp_rank: int = 0,
    ):
        super().__init__()
        self.expert_tp_size = expert_tp_size
        self.expert_tp_rank = expert_tp_rank
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.local_expert_offset = local_expert_offset
        self.expert_hidden_size = expert_hidden_size
        self.intermediate_size = intermediate_size
        self.top_k = top_k
        self.moe_runner_config = MoeRunnerConfig(
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            runner_backend=MoeRunnerBackend.FLASHINFER_TRTLLM,
            hidden_size=expert_hidden_size,
            intermediate_size_per_partition=intermediate_size,
            top_k=top_k,
            num_fused_shared_experts=0,
            params_dtype=torch.bfloat16,
            routing_method_type=_ROUTING_METHOD_UNUSED,
            activation="silu",
            is_gated=True,
            routed_scaling_factor=1.0,
            inplace=False,
        )
        self.use_trtllm = False
        self.use_triton = False
        self._triton_runner_config = None
        self._finalize_op: Optional[FusedPostExpertNormWeightedSumTRTLLM] = None
        self.register_buffer("_post_expert_norm_weight", None, persistent=False)
        self.gate_up_proj = nn.Parameter(
            torch.empty(
                num_local_experts,
                2 * intermediate_size,
                expert_hidden_size,
            ),
            requires_grad=False,
        )
        self.down_proj = nn.Parameter(
            torch.empty(
                num_local_experts,
                expert_hidden_size,
                intermediate_size,
            ),
            requires_grad=False,
        )

        for param in (self.gate_up_proj, self.down_proj):
            set_weight_attrs(param, {"weight_loader": self.weight_loader})

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
    ) -> None:
        if self.use_trtllm:
            raise RuntimeError(
                f"Cannot load {weight_name}: BF16 expert weights are already "
                "shuffled for TRT-LLM, and per-expert loading needs the "
                "checkpoint layout"
            )
        local_id = expert_id - self.local_expert_offset
        if not 0 <= local_id < self.num_local_experts:
            return
        target = param.data[local_id]
        if self.expert_tp_size > 1:
            dim = 0 if shard_id in ("w1", "w3") else -1
            loaded_weight = loaded_weight.narrow(
                dim,
                self.expert_tp_rank * self.intermediate_size,
                self.intermediate_size,
            )
        if shard_id in ("w1", "w3"):
            offset = 0 if shard_id == "w1" else self.intermediate_size
            target = target.narrow(0, offset, self.intermediate_size)
        elif shard_id != "w2":
            raise ValueError(f"Unsupported expert shard: {shard_id}")
        if target.shape != loaded_weight.shape:
            raise RuntimeError(
                f"Expert weight shape mismatch for {weight_name}: "
                f"parameter={tuple(target.shape)}, checkpoint={tuple(loaded_weight.shape)}"
            )
        target.copy_(loaded_weight)

    @torch.no_grad()
    def prepare_trtllm_weights(self, post_expert_norm: Optional[nn.Module]) -> None:
        """Shuffle the weights into BlockMajorK and bind the finalize op.

        Must run after the norms' ``precompute_effective_weight``; the norm
        weight is bound here rather than on the first forward because that copy
        would otherwise be captured into a CUDA graph.
        """
        if self.use_trtllm or self.use_triton:
            return
        if not _trtllm_bf16_moe_supported(self.gate_up_proj.device):
            if _triton_bf16_moe_supported(self.gate_up_proj.device):
                from sglang.srt.layers.muse_spark.fused import glue

                self._triton_runner_config = self._build_triton_runner_config()
                self.use_triton = True
                self._triton_interleaved = glue.hopper_kernels_enabled(
                    self.gate_up_proj.device
                )
                if not self._triton_interleaved:
                    return
                # gate / up rows interleaved: the up-GEMM epilogue applies the SiLU, stores half width
                for w in self.gate_up_proj.data:
                    w.copy_(w.view(2, -1, w.shape[1]).transpose(0, 1).reshape(w.shape))
                if torch.cuda.get_device_capability(self.gate_up_proj.device) == (9, 0):
                    self._compile_grouped(post_expert_norm)
            return
        if getattr(self, "use_megamoe", False):
            from sglang.srt.layers.muse_spark.muse_spark_megamoe import (
                build_bf16_megamoe_weights,
            )

            build_bf16_megamoe_weights(self)
        self._shuffle_weights()
        if getattr(self, "use_megamoe", False):
            from sglang.srt.layers.muse_spark.muse_spark_megamoe import (
                bind_megamoe_shared_weights,
            )

            bind_megamoe_shared_weights(self, "bf16xbf16")
        self._build_finalize_op(post_expert_norm)
        self.use_trtllm = True

    def _build_triton_runner_config(self):
        from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig

        return MoeRunnerConfig(
            num_experts=self.num_experts,
            num_local_experts=self.num_local_experts,
            hidden_size=self.expert_hidden_size,
            intermediate_size_per_partition=self.intermediate_size,
            top_k=self.top_k,
            activation="silu",
            is_gated=True,
            apply_router_weight_on_input=False,
            inplace=False,
            no_combine=True,
            gate_up_interleaved=False,
        )

    def _shuffle_weights(self) -> None:
        from flashinfer.fused_moe.core import (
            _maybe_get_cached_w3_w1_permute_indices,
            convert_to_block_layout,
            get_w2_permute_indices_with_cache,
        )

        cache: dict = {}
        half = self.intermediate_size
        old_shape_w13 = self.gate_up_proj.data[0].shape
        old_shape_w2 = self.down_proj.data[0].shape
        new_shape_w13 = new_shape_w2 = None
        for i in range(self.num_local_experts):
            # Muse's checkpoint is [w1, w3]; the kernel's gated row interleave
            # assumes [w3, w1] (see muse_spark_nvfp4_moe for the FP4 twin).
            gate, up = self.gate_up_proj.data[i].split(half, dim=0)
            w13 = torch.cat([up, gate], dim=0).view(torch.uint8)
            indices = _maybe_get_cached_w3_w1_permute_indices(
                cache, w13, _EPILOGUE_TILE_M
            )
            w13 = convert_to_block_layout(
                w13[indices.to(w13.device)].contiguous(), _BLOCK_K_BYTES
            )

            w2 = self.down_proj.data[i].view(torch.uint8)
            indices = get_w2_permute_indices_with_cache(cache, w2, _EPILOGUE_TILE_M)
            w2 = convert_to_block_layout(
                w2[indices.to(w2.device)].contiguous(), _BLOCK_K_BYTES
            )

            new_shape_w13 = w13.view(torch.bfloat16).shape
            new_shape_w2 = w2.view(torch.bfloat16).shape
            self.gate_up_proj.data[i] = w13.view(torch.bfloat16).reshape(old_shape_w13)
            self.down_proj.data[i] = w2.view(torch.bfloat16).reshape(old_shape_w2)
        self.gate_up_proj.data = self.gate_up_proj.data.reshape(
            self.num_local_experts, *new_shape_w13
        )
        self.down_proj.data = self.down_proj.data.reshape(
            self.num_local_experts, *new_shape_w2
        )

    def _build_finalize_op(self, post_expert_norm: Optional[nn.Module]) -> None:
        """Which RMSNorm form applies is selected by ``use_ffn_learnable_scales``
        in the model config; see ``MuseSpark1xFp4Experts.set_post_expert_norm``.
        """
        if post_expert_norm is None:
            return
        from sglang.srt.layers.muse_spark.muse_spark_layers import (
            MuseSpark1xRMSNormWithInputScale,
        )

        weight = post_expert_norm._get_effective_weight()
        if weight is None:
            weight = torch.ones(
                self.expert_hidden_size,
                dtype=torch.bfloat16,
                device=self.gate_up_proj.device,
            )
        self._post_expert_norm_weight = weight.detach().to(torch.bfloat16)
        self._finalize_op = FusedPostExpertNormWeightedSumTRTLLM(
            topk=self.top_k,
            hidden_dim=self.expert_hidden_size,
            eps=float(post_expert_norm.eps),
            weight_first=isinstance(post_expert_norm, MuseSpark1xRMSNormWithInputScale),
        )

    def forward_dispatched(
        self,
        dispatch_output: FlashinferDispatchOutput,
        post_expert_norm: Optional[nn.Module],
    ) -> FlashinferCombineInput:
        """Run BF16 expert-owned SP rows and build the A2A combine payload."""
        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
            FlashInferTrtllmBf16MoeQuantInfo,
            FlashInferTrtllmDeferredFinalizeOutput,
            flashinfer_trtllm_deferred_finalize_context,
            fused_experts_none_to_flashinfer_trtllm_bf16,
        )
        from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
            FlashinferCombineInput,
        )

        if not self.use_trtllm:
            raise RuntimeError(
                "Muse Spark SP prefill requires prepared TRTLLM BF16 expert weights."
            )
        quant_info = FlashInferTrtllmBf16MoeQuantInfo(
            gemm1_weights=self.gate_up_proj,
            gemm2_weights=self.down_proj,
            global_num_experts=self.num_experts,
            local_expert_offset=self.local_expert_offset,
        )
        with flashinfer_trtllm_deferred_finalize_context():
            runner_output = fused_experts_none_to_flashinfer_trtllm_bf16(
                dispatch_output,
                quant_info,
                self.moe_runner_config,
                use_routed_topk=True,
            )
        deferred_output = runner_output.hidden_states
        if not isinstance(deferred_output, FlashInferTrtllmDeferredFinalizeOutput):
            raise TypeError(
                "Muse Spark SP prefill requires deferred TRTLLM BF16 output, got "
                f"{type(deferred_output).__name__}."
            )
        combine_payload = self._finalize_deferred_output(
            deferred_output,
            num_tokens=dispatch_output.hidden_states.shape[0],
            post_expert_norm=post_expert_norm,
            out_dtype=torch.bfloat16,
        )
        return FlashinferCombineInput(combine_payload)

    def _finalize_deferred_output(
        self,
        deferred_output,
        *,
        num_tokens: int,
        post_expert_norm: Optional[nn.Module],
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        if self._finalize_op is None:
            return finalize_with_post_expert_norm(
                gemm2_out=deferred_output.gemm2_out,
                expert_weights=deferred_output.expert_weights,
                expanded_idx_to_permuted_idx=(
                    deferred_output.expanded_idx_to_permuted_idx
                ),
                num_tokens=num_tokens,
                hidden_size=self.expert_hidden_size,
                top_k=deferred_output.top_k,
                post_expert_norm=post_expert_norm,
                out_dtype=out_dtype,
            )

        output = torch.empty(
            num_tokens,
            self.expert_hidden_size,
            dtype=torch.bfloat16,
            device=deferred_output.gemm2_out.device,
        )
        self._finalize_op(
            gemm2_output=deferred_output.gemm2_out,
            expert_weights=deferred_output.expert_weights,
            expanded_idx_to_permuted_idx=(deferred_output.expanded_idx_to_permuted_idx),
            rms_norm_weight=self._post_expert_norm_weight,
            output=output,
        )
        return output.to(out_dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_scores: torch.Tensor,
        topk_ids: torch.Tensor,
        post_expert_norm: Optional[nn.Module],
        packed_topk_ids: Optional[torch.Tensor] = None,
        *,
        ar_workspace: FinalizePushWorkspace | None = None,
    ) -> torch.Tensor:
        if self.use_trtllm:
            return self._forward_trtllm(
                hidden_states,
                topk_scores,
                topk_ids,
                post_expert_norm,
                packed_topk_ids,
                ar_workspace=ar_workspace,
            )
        if self.expert_tp_size > 1:
            raise RuntimeError("Muse BF16 expert TP needs the TRT-LLM MoE backend")
        if self.use_triton:
            return self._forward_triton(
                hidden_states, topk_scores, topk_ids, post_expert_norm
            )
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            output = self._forward_capturable(
                hidden_states, topk_scores, topk_ids, post_expert_norm
            )
            if ar_workspace is not None:
                torch.distributed.all_reduce(output, group=ar_workspace.group)
            return output
        output = torch.zeros_like(hidden_states)
        for local_id in range(self.num_local_experts):
            global_id = self.local_expert_offset + local_id
            token_slot = (topk_ids == global_id).nonzero(as_tuple=False)
            if token_slot.numel() == 0:
                continue
            token_ids = token_slot[:, 0]
            slots = token_slot[:, 1]
            selected = hidden_states.index_select(0, token_ids)
            gate_up = F.linear(selected, self.gate_up_proj[local_id])
            gate, up = gate_up.chunk(2, dim=-1)
            expert_output = F.linear(F.silu(gate) * up, self.down_proj[local_id])
            if post_expert_norm is not None:
                expert_output = post_expert_norm(expert_output)
            expert_output = expert_output * topk_scores[token_ids, slots].unsqueeze(-1)
            output.index_add_(0, token_ids, expert_output.to(output.dtype))
        if ar_workspace is not None:
            torch.distributed.all_reduce(output, group=ar_workspace.group)
        return output

    def _forward_trtllm(
        self,
        hidden_states: torch.Tensor,
        topk_scores: torch.Tensor,
        topk_ids: torch.Tensor,
        post_expert_norm: Optional[nn.Module],
        packed_topk_ids: Optional[torch.Tensor] = None,
        *,
        ar_workspace: FinalizePushWorkspace | None = None,
    ) -> torch.Tensor:
        from flashinfer.fused_moe import trtllm_bf16_routed_moe

        num_tokens = hidden_states.shape[0]
        top_k = topk_ids.shape[1]
        # Preserve Muse's packed fast path; the upstream FlashInfer API also
        # accepts materialized IDs and FP32 routing weights directly.
        routing = packed_topk_ids
        if routing is None:
            routing = (
                topk_ids.to(torch.int32).contiguous(),
                topk_scores.to(torch.float32).contiguous(),
            )

        gemm2_out, expert_weights, expanded_idx_to_permuted_idx = (
            trtllm_bf16_routed_moe(
                topk_ids=routing,
                hidden_states=hidden_states,
                gemm1_weights=self.gate_up_proj,
                gemm2_weights=self.down_proj,
                num_experts=self.num_experts,
                top_k=top_k,
                n_group=None,
                topk_group=None,
                intermediate_size=self.intermediate_size,
                local_expert_offset=self.local_expert_offset,
                local_num_experts=self.num_local_experts,
                # Muse folds its route_scale into the top-k weights, so the
                # kernel applies none of its own -- but it wants 1.0, not None.
                routed_scaling_factor=1.0,
                routing_method_type=_ROUTING_METHOD_UNUSED,
                do_finalize=False,  # we need the per-expert outputs
                tune_max_num_tokens=next_power_of_2(num_tokens),
                activation_type=_SWIGLU,
            )[:3]
        )

        if self.expert_tp_size > 1:
            weight, eps, weight_first = self._finalize_norm_params(post_expert_norm)
            return finalize_expert_tp(
                gemm2_out,
                expanded_idx_to_permuted_idx,
                topk_scores,
                topk_ids,
                hidden_size=self.expert_hidden_size,
                norm_weight=weight,
                eps=eps,
                weight_first=weight_first,
                apply_norm=post_expert_norm is not None,
            ).to(hidden_states.dtype)

        if self._finalize_op is None:
            output = finalize_with_post_expert_norm(
                gemm2_out=gemm2_out,
                expert_weights=expert_weights,
                expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                num_tokens=num_tokens,
                hidden_size=self.expert_hidden_size,
                top_k=top_k,
                post_expert_norm=post_expert_norm,
                out_dtype=hidden_states.dtype,
            )
            if ar_workspace is not None:
                torch.distributed.all_reduce(output, group=ar_workspace.group)
            return output

        output = torch.empty(
            num_tokens,
            self.expert_hidden_size,
            dtype=torch.bfloat16,
            device=gemm2_out.device,
        )
        self._finalize_op(
            gemm2_output=gemm2_out,
            expert_weights=expert_weights,
            expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
            rms_norm_weight=self._post_expert_norm_weight,
            output=output,
            ar_workspace=ar_workspace,
        )
        return output.to(hidden_states.dtype)

    def _forward_triton(
        self,
        hidden_states: torch.Tensor,
        topk_scores: torch.Tensor,
        topk_ids: torch.Tensor,
        post_expert_norm: Optional[nn.Module],
    ) -> torch.Tensor:
        if self._triton_interleaved:
            return self._forward_triton_interleaved(
                hidden_states, topk_scores, topk_ids, post_expert_norm
            )
        from sglang.srt.layers.moe.moe_runner.triton_utils import fused_experts
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        offset = self.local_expert_offset
        local_topk_ids = torch.where(
            (topk_ids >= offset) & (topk_ids < offset + self.num_local_experts),
            topk_ids - offset,
            -1,
        )
        local_topk_ids = local_topk_ids.to(torch.int32).contiguous()

        expert_output = fused_experts(
            hidden_states=hidden_states.contiguous(),
            w1=self.gate_up_proj,
            w2=self.down_proj,
            topk_output=StandardTopKOutput(
                topk_weights=topk_scores.to(torch.float32).contiguous(),
                topk_ids=local_topk_ids,
                router_logits=None,
            ),
            moe_runner_config=self._triton_runner_config,
        )

        weight, eps, weight_first = self._finalize_norm_params(post_expert_norm)
        return fused_post_expert_norm_weighted_sum(
            expert_output,
            topk_scores,
            norm_weight=weight,
            eps=eps,
            weight_first=weight_first,
            apply_norm=post_expert_norm is not None,
        ).to(hidden_states.dtype)

    def _forward_triton_interleaved(
        self, hidden_states, topk_scores, topk_ids, post_expert_norm, grouped=True
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        if (
            grouped
            and hidden_states.dtype == torch.bfloat16
            and num_tokens >= _GROUPED_MIN_TOKENS
            and torch.cuda.get_device_capability(hidden_states.device) == (9, 0)
            and (
                is_in_breakable_cuda_graph()
                or not torch.cuda.is_current_stream_capturing()
            )
        ):
            out = hidden_states.new_empty((num_tokens, self.down_proj.shape[1]))
            # captured: the workspace is graph-pool memory
            ws = (
                self._grouped_ws(hidden_states)
                if torch.cuda.is_current_stream_capturing()
                else None
            )
            self._forward_grouped(
                hidden_states, topk_scores, topk_ids, post_expert_norm, out, ws
            )
            return out
        width = max(2 * self.intermediate_size, self.down_proj.shape[1])
        chunk = max(1, _TRITON_ROUTE_ELEMS // (topk_ids.shape[1] * width))
        if num_tokens > chunk:
            output = hidden_states.new_empty((num_tokens, self.down_proj.shape[1]))
            for start in range(0, num_tokens, chunk):
                rows = slice(start, start + chunk)
                output[rows] = self._forward_triton_interleaved(
                    hidden_states[rows],
                    topk_scores[rows],
                    topk_ids[rows],
                    post_expert_norm,
                    grouped=False,
                )
            return output
        from sglang.srt.layers.moe.moe_runner.triton import (
            TritonMoeQuantInfo,
            TritonRunnerCore,
            TritonRunnerInput,
        )
        from sglang.srt.layers.moe.moe_runner.triton_utils import (
            try_get_optimal_moe_config,
        )

        w1, w2, n, e = (
            self.gate_up_proj,
            self.down_proj,
            topk_ids.numel(),
            self.num_local_experts,
        )
        config, (down_config, _) = try_get_optimal_moe_config(
            w1.shape,
            w2.shape,
            self.top_k,
            None,
            topk_ids.shape[0],
            return_down_config=True,
        )
        if topk_ids.shape[0] <= 128:  # faster than the table rows (H200 sweep)
            config = down_config = dict(
                config,
                BLOCK_SIZE_M=16,
                BLOCK_SIZE_N=64,
                BLOCK_SIZE_K=128,
                GROUP_SIZE_M=8,
                num_warps=4,
                num_stages=5,
            )
        bm, eb = config["BLOCK_SIZE_M"], next_power_of_2(e)
        cap = n + min(n, e) * (bm - 1)
        local_topk_ids, rows, expert_ids, ntp, ws = (
            topk_ids.new_empty(k)
            for k in (topk_ids.shape, cap, -(-cap // bm), 1, n + 32 * eb)
        )
        _route_kernel[(1,)](
            topk_ids,
            local_topk_ids,
            ws,
            rows,
            expert_ids,
            ntp,
            n,
            self.local_expert_offset,
            e,
            EB=eb,
            BM=bm,
            CH=1024,
        )
        expert_output = (
            TritonRunnerCore(self._triton_runner_config)
            .run(
                TritonRunnerInput(
                    hidden_states.contiguous(),
                    topk_scores.to(torch.float32).contiguous(),
                    topk_ids,
                    rows,
                    expert_ids,
                    ntp,
                ),
                TritonMoeQuantInfo(
                    w13_weight=w1, w2_weight=w2, fuse_swiglu_interleaved=True
                ),
                {"config": config, "down_config": down_config},
            )
            .hidden_states
        )

        weight, eps, weight_first = self._finalize_norm_params(post_expert_norm)
        return fused_post_expert_norm_weighted_sum(
            expert_output,
            topk_scores,
            norm_weight=weight,
            eps=eps,
            weight_first=weight_first,
            apply_norm=post_expert_norm is not None,
            slot_expert_ids=local_topk_ids,
        ).to(hidden_states.dtype)

    def _grouped_ws(self, x) -> torch.Tensor:
        """Per row an input / activation and a gate-up / output row."""
        import deep_gemm

        align = deep_gemm.get_mk_alignment_for_contiguous_layout()
        rows = (
            max(x.shape[0], _GROUPED_WS_TOKENS) * 3 // 2 // align
            + self.num_local_experts
        ) * align
        h, inter = x.shape[1], self.intermediate_size
        return x.new_empty(rows * (max(h, inter) + max(2 * inter, h)))

    def _compile_grouped(self, post_expert_norm) -> None:
        x = self.gate_up_proj.new_zeros(
            (_GROUPED_MIN_TOKENS, self.gate_up_proj.shape[2])
        )
        ids = x.new_full(
            (x.shape[0], self.top_k), self.local_expert_offset, dtype=torch.int32
        )
        self._forward_grouped.run_eager(
            x, ids.float(), ids, post_expert_norm, torch.empty_like(x), None
        )
        torch.cuda.empty_cache()

    @eager_on_graph
    def _forward_grouped(
        self, x, topk_scores, topk_ids, post_expert_norm, out, ws
    ) -> None:
        """The Triton path's arithmetic reading each local expert's weights once."""
        import deep_gemm

        if ws is None and is_in_breakable_cuda_graph():
            out.copy_(
                self._forward_triton_interleaved(
                    x, topk_scores, topk_ids, post_expert_norm, grouped=False
                )
            )
            torch.cuda.empty_cache()
            return
        n, k = topk_ids.shape
        e, h, inter = self.num_local_experts, x.shape[1], self.intermediate_size
        align = deep_gemm.get_mk_alignment_for_contiguous_layout()
        ws = self._grouped_ws(x) if ws is None else ws
        w1, w2 = max(h, inter), max(2 * inter, h)
        cap = ws.numel() // (w1 + w2)
        budget = cap - e * (align - 1) - (k - 1)
        parts = (n * k - 1) // budget + 1
        ids = topk_ids.reshape(-1).long() - self.local_expert_offset
        here = (ids >= 0) & (ids < e)
        per_tok = here.view(n, k).sum(1)
        part = (per_tok.cumsum(0) - per_tok) // budget
        key = (part * (e + 1)).unsqueeze(1) + torch.where(here, ids, e).view(n, k)
        order = torch.argsort(key.view(-1).short(), stable=True)
        key = key.view(-1)[order]
        first = torch.searchsorted(
            key, torch.arange(parts * (e + 1) + 1, device=x.device)
        )
        padded = (first.diff() + align - 1) // align * align
        padded.view(parts, e + 1)[:, e] = 0
        end = padded.view(parts, e + 1).cumsum(1)
        psum = end[:, :e].int()
        dst = (
            torch.arange(n * k, device=x.device)
            + (end.view(-1) - padded - first[:-1])[key]
        )
        local = key % (e + 1) < e
        token = x.new_full((parts * cap + 1,), -1, dtype=torch.int32)
        token[torch.where(local, key // (e + 1) * cap + dst, parts * cap)] = (
            order // k
        ).int()
        src = x.new_empty(n * k, dtype=torch.int64)
        src[order] = torch.where(local, dst, -1)
        gate_up = ws[cap * w1 : cap * (w1 + 2 * inter)].view(cap, 2 * inter)
        xs, act = ws[: cap * h].view(cap, h), ws[: cap * inter].view(cap, inter)
        y = ws[-cap * h :].view(cap, h)
        weight, eps, weight_first = self._finalize_norm_params(post_expert_norm)
        bounds = torch.searchsorted(
            part, torch.arange(parts + 1, device=x.device), out_int32=True
        )
        for p in range(parts):
            rows = token[p * cap : (p + 1) * cap]
            used = psum[p, e - 1 :]
            _rows_kernel[(min(cap, 4096),)](
                x,
                rows,
                xs,
                x.stride(0),
                used,
                H=h,
                BLOCK=next_power_of_2(h),
                SILU_MUL=False,
            )
            deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
                xs, self.gate_up_proj, gate_up, psum[p], **_GROUPED_DG
            )
            _rows_kernel[(min(cap, 4096),)](
                gate_up,
                rows,
                act,
                0,
                used,
                H=inter,
                BLOCK=next_power_of_2(inter),
                SILU_MUL=True,
            )
            deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
                act, self.down_proj, y, psum[p], **_GROUPED_DG
            )
            fused_post_expert_norm_weighted_sum(
                y,
                topk_scores,
                weight,
                eps,
                weight_first,
                post_expert_norm is not None,
                rows=src.view(n, k),
                out=out,
                part=(bounds, p),
            )

    @torch.no_grad()
    def _finalize_norm_params(self, post_expert_norm):
        if post_expert_norm is None:
            return None, 1e-8, True
        if getattr(self, "_finalize_norm_src", None) is post_expert_norm:
            return self._finalize_norm_cache
        from sglang.srt.layers.muse_spark.muse_spark_layers import (
            MuseSpark1xRMSNormWithInputScale,
        )

        getter = getattr(post_expert_norm, "_get_effective_weight", None)
        weight = (
            getter()
            if getter is not None
            else getattr(post_expert_norm, "weight", None)
        )
        if weight is not None:
            weight = weight.detach().to(torch.bfloat16).contiguous()
        eps = float(getattr(post_expert_norm, "eps", 1e-8))
        weight_first = isinstance(post_expert_norm, MuseSpark1xRMSNormWithInputScale)
        object.__setattr__(self, "_finalize_norm_src", post_expert_norm)
        object.__setattr__(self, "_finalize_norm_cache", (weight, eps, weight_first))
        return self._finalize_norm_cache

    def _forward_capturable(
        self,
        hidden_states: torch.Tensor,
        topk_scores: torch.Tensor,
        topk_ids: torch.Tensor,
        post_expert_norm: Optional[nn.Module],
    ) -> torch.Tensor:
        """Fixed-shape equivalent of the expert loop for CUDA graph capture.

        The loop above reads device data on the host (``nonzero``, the empty
        skip), which stream capture rejects. Here every local expert runs over
        all tokens and is scaled by its routing weight, which is zero for a
        token that did not select it.
        """
        # A token's top-k ids are distinct, so at most one slot matches each
        # expert and this sum is exactly that slot's score (zero when unrouted).
        local_ids = torch.arange(
            self.local_expert_offset,
            self.local_expert_offset + self.num_local_experts,
            device=topk_ids.device,
        ).view(-1, 1, 1)
        weights = (topk_scores * (topk_ids == local_ids)).sum(dim=-1, keepdim=True)

        # One group unless the [G, tokens, 2I] intermediate exceeds the budget.
        group_elems = hidden_states.shape[0] * 2 * self.intermediate_size
        group_size = max(1, _EXPERT_GROUP_ELEMS // group_elems)
        output = None
        for start in range(0, self.num_local_experts, group_size):
            stop = min(start + group_size, self.num_local_experts)
            w13 = self.gate_up_proj[start:stop].transpose(1, 2)
            gate, up = torch.matmul(hidden_states, w13).chunk(2, dim=-1)
            group = torch.bmm(
                F.silu(gate) * up, self.down_proj[start:stop].transpose(1, 2)
            )
            if post_expert_norm is not None:
                group = post_expert_norm(group)
            group = (group * weights[start:stop]).sum(dim=0)
            output = group if output is None else output + group
        return output.to(hidden_states.dtype)


__all__ = ["MuseSpark1xBf16Experts"]
