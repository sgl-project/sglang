# Copyright 2023 Antgroup and The HuggingFace Inc. team. All rights reserved.
from __future__ import annotations

import copy
import logging
from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.kernels.ops.moe.router import (
    ROUTER_GATE_MATVEC_MAX_M,
    router_gate_matvec,
)
from sglang.kernels.ops.quantization.fp8_kernel import (
    block_quant_fp8,
    is_fp8_fnuz,
)
from sglang.srt.configs import KimiLinearConfig
from sglang.srt.configs.bailing_hybrid import is_bailing_multi_gate_enabled
from sglang.srt.distributed.utils import get_group_rank_size
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.aux_hidden_states import AuxHiddenStateList
from sglang.srt.layers.chunked_gemm_ar import chunked_gemm_ar_enabled
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.layer_boundary import (
    append_stages,
    declare_attn,
    declare_ffn,
    is_dense_ffn_fully_dp,
)
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    LinearParallelGroup,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import (
    get_moe_a2a_backend,
    get_moe_runner_backend,
)
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.moe.utils import is_shared_experts_fusion_disabled
from sglang.srt.layers.multi_gate import (
    create_multi_gate_mm_indices,
    multi_gate_triton_kernel,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8 import (
    Fp8Config,
    Fp8LinearMethod,
    Fp8MoEMethod,
)
from sglang.srt.layers.quantization.fp8_utils import (
    block_quant_dequant,
    block_quant_to_tensor_quant,
    channel_quant_to_tensor_quant,
    deepgemm_w8a8_block_fp8_linear_with_fallback,
    dispatch_w8a8_block_fp8_linear,
    normalize_e4m3fn_to_e4m3fnuz,
    triton_w8a8_block_fp8_linear,
)
from sglang.srt.layers.quantization.int8_utils import (
    block_dequant as int8_block_dequant,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
)
from sglang.srt.models.deepseek_common.utils import (
    _is_cpu,
    _is_cpu_amx_available,
    _is_cuda,
    _is_hip,
)
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
from sglang.srt.models.kimi_linear import KimiDeltaAttention
from sglang.srt.runtime_context import (
    get_exec,
    get_parallel,
    get_platform,
    get_stream,
)
from sglang.srt.utils import (
    BumpAllocator,
    add_prefix,
    bind_or_assign,
    is_cuda,
    is_flashinfer_available,
    log_info_on_rank0,
    make_pp_layers,
)

_is_fp8_fnuz = is_fp8_fnuz()


def _get_awq_dequantize():
    """Load the platform AWQ kernel only when an AWQ tensor needs it."""
    if _is_cuda:
        from sglang.kernels.ops.quantization.awq_dequantize import awq_dequantize

        return awq_dequantize
    if _is_hip:
        from sglang.kernels.ops.quantization.awq_triton import awq_dequantize_triton

        return awq_dequantize_triton
    if not (_is_cpu and _is_cpu_amx_available):
        from vllm._custom_ops import awq_dequantize

        return awq_dequantize
    return None


_is_flashinfer_available = is_flashinfer_available()
_is_sm100_supported = is_cuda() and get_platform().is_sm100


class DsV3MLA(DeepseekV2AttentionMLA):
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        layer_id: int = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        skip_rope: bool = False,
    ) -> None:
        super().__init__(
            config,
            hidden_size,
            num_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            q_lora_rank,
            kv_lora_rank,
            rope_theta,
            rope_scaling,
            max_position_embeddings,
            quant_config,
            reduce_results,
            layer_id,
            prefix,
            alt_stream,
            skip_rope,
        )
        self.gated_attention_proj_granularity_type = getattr(
            config, "gated_attention_proj_granularity_type", None
        )
        # gated_attn now not support NPU
        if self.gated_attention_proj_granularity_type == "head_wise":
            self.g_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads,
                bias=False,
                prefix=f"{prefix}.output_gate",
                quant_config=None,
                parallel_group="attn_tp",
            )
        elif self.gated_attention_proj_granularity_type == "element_wise":
            self.g_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.v_head_dim,
                bias=False,
                prefix=f"{prefix}.output_gate",
                parallel_group="attn_tp",
            )
        else:
            self.g_proj = None

        if "mrope_section" in rope_scaling and rope_scaling.get("video_rope", False):
            rope_scaling["rope_type"] = "default"
            self.rotary_emb = get_rope(
                qk_rope_head_dim,
                rotary_dim=qk_rope_head_dim,
                max_position=max_position_embeddings,
                base=rope_theta,
                rope_scaling=rope_scaling,
                is_neox_style=not getattr(config, "rope_interleave", True),
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
        llama_4_scaling: Optional[torch.Tensor] = None,
        inline_tp_reduce: bool = False,
    ):
        # Same per-forward bookkeeping as DeepseekV2AttentionMLA.forward.
        self._inline_tp_reduce = inline_tp_reduce
        self._inline_tp_reduced = False
        try:
            s = self.forward_prepare(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                zero_allocator=zero_allocator,
                llama_4_scaling=llama_4_scaling,
            )
            gate = self._forward_gated(hidden_states)
            s = (s[0], s[1], s[2], s[3] + (gate,))
            return self.forward_core(s)
        finally:
            self._inline_tp_reduce = False

    def _forward_gated(self, hidden_states: torch.Tensor):
        if self.g_proj:
            gate, _ = self.g_proj(hidden_states)
            gate = F.sigmoid(gate.float()).type_as(hidden_states)
            return gate
        else:
            return None

    def _apply_gated(self, attn_output: torch.Tensor, gate: torch.Tensor):
        if self.gated_attention_proj_granularity_type == "head_wise":
            attn_output = (
                attn_output.view(-1, self.num_local_heads, self.v_head_dim)
                * gate[:, :, None]
            )
            attn_output = attn_output.view(-1, self.num_local_heads * self.v_head_dim)
        else:
            attn_output = attn_output * gate
        return attn_output


logger = logging.getLogger(__name__)


def is_linear_layer(layer_idx, layer_group_size):
    if layer_idx is None:
        return False
    if isinstance(layer_group_size, list):
        return layer_group_size[layer_idx] == 1
    if layer_group_size > 0:
        return (layer_idx + 1) % layer_group_size != 0
    else:
        return False


_NEXTN_SPEC_WEIGHT_NAMES = (
    "final_layernorm",
    "eh_proj",
    "enorm",
    "hnorm",
)


def resolve_nextn_layer_id(config: PretrainedConfig) -> int:
    """Locate the nextn predict layer index in the HF checkpoint name space."""
    if not hasattr(config, "num_nextn_predict_layers"):
        raise ValueError("num nextn_predict_layers is not in the config")
    assert config.num_nextn_predict_layers == 1, "Only 1 nextn layer is supported"
    return 0 if config.num_hidden_layers == 1 else config.num_hidden_layers


def rewrite_nextn_weight_name(name: str, nextn_layer_prefix: str) -> Optional[str]:
    """Map a HF nextn-layer weight name onto the local NextN module namespace.

    The caller must already have ensured ``name.startswith(nextn_layer_prefix)``.
    Returns the rewritten name, or None to signal the weight should be skipped
    (e.g. shared head / embed tokens which are reused from the target model).
    """
    if "shared_head.head" in name or "embed_tokens" in name:
        return None
    for spec in _NEXTN_SPEC_WEIGHT_NAMES:
        if spec in name:
            return name.replace(nextn_layer_prefix, "model")
    return name.replace(nextn_layer_prefix, "model.decoder")


def is_pp_missing_parameter(
    name: str,
    model: torch.nn.Module,
) -> bool:
    if isinstance(model, PPMissingLayer):
        return True
    return False


class BailingMLP(nn.Module):
    @classmethod
    def requires_standalone_fp8_tail(
        cls,
        *,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig],
        tp_size: int,
    ) -> bool:
        """Whether TP cuts a standalone block-FP8 MLP through a block."""
        if (
            quant_config is None
            or quant_config.get_name() != "fp8"
            or getattr(quant_config, "use_mxfp8", False)
            or intermediate_size % tp_size != 0
        ):
            return False
        block_size = getattr(quant_config, "weight_block_size", None)
        if block_size is None or len(block_size) != 2:
            return False
        local_intermediate_size = intermediate_size // tp_size
        return any(local_intermediate_size % block != 0 for block in block_size)

    @classmethod
    def standalone_fp8_tail_route(
        cls,
        *,
        hidden_size: int,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig],
        tp_size: int,
    ) -> Optional[str]:
        """GEMM route for a standalone shared MLP that keeps its logical TP K tail."""
        if not cls.requires_standalone_fp8_tail(
            intermediate_size=intermediate_size,
            quant_config=quant_config,
            tp_size=tp_size,
        ) or getattr(quant_config, "weight_block_size", None) != [128, 128]:
            return None
        if not getattr(quant_config, "is_checkpoint_fp8_serialized", False):
            return None
        # gate_up uses hidden_size as K, quantized per 128-wide token group.
        if not (_is_cuda and not _is_hip and hidden_size % 128 == 0):
            return None

        parallel = get_parallel()
        runtime = get_exec()
        if not (
            tp_size == parallel.tp_size
            and parallel.moe_ep_size == 1
            and parallel.moe_dp_size == 1
            and parallel.moe_tp_size == parallel.tp_size
            and parallel.moe_tp_rank == parallel.tp_rank
            and get_moe_a2a_backend().is_none()
            and not runtime.moe.enable_eplb
            and runtime.moe.ep_num_redundant_experts == 0
            and runtime.moe.init_expert_location == "trivial"
            and runtime.offload.cpu_offload_gb == 0
            and runtime.offload.offload_group_size <= 0
        ):
            return None

        local_intermediate_size = intermediate_size // tp_size
        if (
            deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
            and not deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
            # down_proj uses the merged gate_up width as N (DeepGEMM: N % 64).
            and (2 * local_intermediate_size) % 64 == 0
        ):
            try:
                if cls._is_deepgemm_block_linear(
                    dispatch_w8a8_block_fp8_linear(
                        weight_block_size=quant_config.weight_block_size,
                        act_scale_ue8m0=getattr(quant_config, "scale_fmt", None)
                        == "ue8m0",
                    )
                ):
                    return "deepgemm"
            except RuntimeError:
                pass
        # Triton block-FP8 masks the partial K block and reads the requantized
        # fp32 local-grid scales; UE8M0 DeepGEMM (SM100+) cannot.
        return "triton"

    @classmethod
    def supports_standalone_fp8_tail(
        cls,
        *,
        hidden_size: int,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig],
        tp_size: int,
    ) -> bool:
        """Whether a standalone shared MLP can keep its logical TP K tail."""
        return (
            cls.standalone_fp8_tail_route(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                quant_config=quant_config,
                tp_size=tp_size,
            )
            is not None
        )

    @staticmethod
    def _is_deepgemm_block_linear(linear) -> bool:
        # Dispatch binds checkpoint activation policies (e.g. ue8m0) via partial.
        return (
            getattr(linear, "func", linear)
            is deepgemm_w8a8_block_fp8_linear_with_fallback
        )

    @classmethod
    def validate_standalone_fp8_tail(
        cls,
        *,
        hidden_size: int,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig],
        tp_size: int,
    ) -> bool:
        requires_tail = cls.requires_standalone_fp8_tail(
            intermediate_size=intermediate_size,
            quant_config=quant_config,
            tp_size=tp_size,
        )
        if not requires_tail:
            return False
        supports_tail = cls.supports_standalone_fp8_tail(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            quant_config=quant_config,
            tp_size=tp_size,
        )
        if not supports_tail:
            raise NotImplementedError(
                "Bailing standalone shared experts with blockwise-FP8 TP tails "
                "require CUDA, a serialized [128, 128] block-FP8 checkpoint and "
                "pure-TP MoE; model-level intermediate-size padding is not allowed."
            )
        return supports_tail

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        config: PretrainedConfig,
        reduce_results=True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        swiglu_limit: Optional[float] = None,
        padded_intermediate_size: Optional[int] = None,
        *,
        parallel_group: LinearParallelGroup = "tp",
        allow_fp8_blockwise_tail: bool = False,
    ) -> None:
        super().__init__()

        self.config = config
        self.swiglu_limit = swiglu_limit
        self.quant_config = quant_config
        self.allow_fp8_blockwise_tail = allow_fp8_blockwise_tail

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.padded_intermediate_size = padded_intermediate_size or intermediate_size

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [self.padded_intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
            parallel_group=parallel_group,
            skip_block_quant_check=allow_fp8_blockwise_tail,
        )
        self.down_proj = RowParallelLinear(
            self.padded_intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
            parallel_group=parallel_group,
            skip_block_quant_check=allow_fp8_blockwise_tail,
        )

        self.tp_rank, self.tp_size = get_group_rank_size(self.gate_up_proj.tp_group)
        tp_size = self.tp_size

        if self.padded_intermediate_size > self.intermediate_size:
            self.padded_size_per_partition = self.padded_intermediate_size // tp_size
            self.effective_size_per_partition = self.intermediate_size // tp_size
            self.pad_size_per_partition = (
                self.padded_size_per_partition - self.effective_size_per_partition
            )
        else:
            self.padded_size_per_partition = None
            self.effective_size_per_partition = None
            self.pad_size_per_partition = None

        down_quant_method = self.down_proj.quant_method
        local_intermediate_size = self.intermediate_size // tp_size
        if allow_fp8_blockwise_tail and self.down_proj.orig_dtype != torch.bfloat16:
            raise NotImplementedError(
                "Bailing standalone blockwise-FP8 TP tails currently require "
                f"BF16 activations, but got {self.down_proj.orig_dtype}."
            )
        self.fp8_tail_route = (
            self.standalone_fp8_tail_route(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                quant_config=quant_config,
                tp_size=tp_size,
            )
            if allow_fp8_blockwise_tail
            else None
        )
        if self.fp8_tail_route == "triton":
            # Keeps the fp32 local-grid scales: post-load UE8M0 requant only
            # runs for DeepGEMM-dispatched layers.
            triton_linear = partial(
                triton_w8a8_block_fp8_linear,
                act_scale_ue8m0=getattr(quant_config, "scale_fmt", None) == "ue8m0",
            )
            for proj in (self.gate_up_proj, self.down_proj):
                proj.quant_method.w8a8_block_fp8_linear = triton_linear
            log_info_on_rank0(
                logger,
                f"{prefix}: blockwise-FP8 TP K-tail runs on Triton block-FP8.",
            )
        self.use_prequantized_fp8_tail = (
            allow_fp8_blockwise_tail
            and isinstance(down_quant_method, Fp8LinearMethod)
            and (
                self.fp8_tail_route == "triton"
                or self._is_deepgemm_block_linear(
                    down_quant_method.w8a8_block_fp8_linear
                )
            )
            and local_intermediate_size
            % down_quant_method.quant_config.weight_block_size[1]
            != 0
        )

        self.act_fn = SiluAndMul()

    def _requires_serialized_fp8_tp_requant(self) -> bool:
        if not self.allow_fp8_blockwise_tail or not isinstance(
            self.quant_config, Fp8Config
        ):
            return False
        if (
            not self.quant_config.is_checkpoint_fp8_serialized
            or getattr(self.quant_config, "use_mxfp8", False)
            or self.quant_config.weight_block_size is None
            or self.tp_size <= 1
        ):
            return False
        local_intermediate_size = self.intermediate_size // self.tp_size
        block_n, block_k = self.quant_config.weight_block_size
        return (
            local_intermediate_size % block_n != 0
            or local_intermediate_size % block_k != 0
        )

    def cache_serialized_fp8_scale_for_tp_requant(
        self, loaded_scale: torch.Tensor, projection: str
    ) -> bool:
        if not self._requires_serialized_fp8_tp_requant():
            return False
        if projection not in {"gate_proj", "up_proj", "down_proj"}:
            return False
        if loaded_scale.ndim != 2:
            raise ValueError(
                "Bailing serialized blockwise FP8 shared-expert scales must be "
                f"2D, but got {tuple(loaded_scale.shape)} for {projection}."
            )
        if not hasattr(self, "_serialized_fp8_tp_scale_cache"):
            self._serialized_fp8_tp_scale_cache: Dict[str, torch.Tensor] = {}
        self._serialized_fp8_tp_scale_cache[projection] = loaded_scale.to(
            device=self.gate_up_proj.weight.device, dtype=loaded_scale.dtype
        ).clone()
        return True

    @staticmethod
    def _dequantize_serialized_fp8_tp_shard(
        quantized_weight: torch.Tensor,
        checkpoint_scale: torch.Tensor,
        block_size: Tuple[int, int],
        *,
        row_offset: int = 0,
        column_offset: int = 0,
    ) -> torch.Tensor:
        block_n, block_k = block_size
        rows, columns = quantized_weight.shape
        if rows == 0 or columns == 0:
            raise ValueError(
                "Bailing shared-expert TP requantization does not support empty "
                f"weight shards, but got {tuple(quantized_weight.shape)}."
            )
        required_scale_rows = (row_offset + rows - 1) // block_n + 1
        required_scale_columns = (column_offset + columns - 1) // block_k + 1
        if (
            required_scale_rows > checkpoint_scale.shape[0]
            or required_scale_columns > checkpoint_scale.shape[1]
        ):
            raise ValueError(
                "Bailing shared-expert TP shard exceeds its serialized FP8 scale: "
                f"weight={tuple(quantized_weight.shape)}, "
                f"scale={tuple(checkpoint_scale.shape)}, offsets="
                f"({row_offset}, {column_offset}), block_size={block_size}."
            )
        scale_rows = (
            torch.arange(rows, device=quantized_weight.device) + row_offset
        ) // block_n
        scale_columns = (
            torch.arange(columns, device=quantized_weight.device) + column_offset
        ) // block_k
        element_scale = checkpoint_scale.index_select(0, scale_rows).index_select(
            1, scale_columns
        )
        return (quantized_weight.float() * element_scale.float()).to(torch.bfloat16)

    def requantize_serialized_fp8_tp_weights(self) -> bool:
        if not self._requires_serialized_fp8_tp_requant() or not hasattr(
            self, "_serialized_fp8_tp_scale_cache"
        ):
            return False

        cache = self._serialized_fp8_tp_scale_cache
        missing = {"gate_proj", "up_proj", "down_proj"} - cache.keys()
        if missing:
            raise ValueError(
                "Missing serialized blockwise FP8 scales required for Bailing "
                f"shared-expert TP requantization: {sorted(missing)}."
            )

        block_n, block_k = self.quant_config.weight_block_size
        if block_n != block_k:
            raise ValueError(
                "Bailing shared-expert TP requantization requires square FP8 "
                f"blocks, but got {self.quant_config.weight_block_size}."
            )
        block_size = (block_n, block_k)
        local_intermediate_size = self.intermediate_size // self.tp_size
        tp_offset = self.tp_rank * local_intermediate_size

        gate_up_weight = self.gate_up_proj.weight.data
        down_weight = self.down_proj.weight.data
        expected_gate_up_shape = (2 * local_intermediate_size, self.hidden_size)
        expected_down_shape = (self.hidden_size, local_intermediate_size)
        if (
            tuple(gate_up_weight.shape) != expected_gate_up_shape
            or tuple(down_weight.shape) != expected_down_shape
        ):
            raise ValueError(
                "Unexpected Bailing shared-expert TP weight layout: "
                f"gate_up={tuple(gate_up_weight.shape)}, "
                f"down={tuple(down_weight.shape)}, expected "
                f"gate_up={expected_gate_up_shape}, down={expected_down_shape}."
            )

        gate = self._dequantize_serialized_fp8_tp_shard(
            gate_up_weight[:local_intermediate_size],
            cache["gate_proj"],
            block_size,
            row_offset=tp_offset,
        )
        up = self._dequantize_serialized_fp8_tp_shard(
            gate_up_weight[local_intermediate_size:],
            cache["up_proj"],
            block_size,
            row_offset=tp_offset,
        )
        requant_gate_up, requant_gate_up_scale = block_quant_fp8(
            torch.cat((gate, up), dim=0), block_size
        )

        down = self._dequantize_serialized_fp8_tp_shard(
            down_weight,
            cache["down_proj"],
            block_size,
            column_offset=tp_offset,
        )
        requant_down, requant_down_scale = block_quant_fp8(down, block_size)

        gate_up_weight.copy_(requant_gate_up)
        down_weight.copy_(requant_down)
        self.gate_up_proj.weight_scale_inv.data = requant_gate_up_scale
        self.down_proj.weight_scale_inv.data = requant_down_scale
        self.gate_up_proj.weight_scale_inv.format_ue8m0 = False
        self.down_proj.weight_scale_inv.format_ue8m0 = False
        del self._serialized_fp8_tp_scale_cache
        return True

    def forward(
        self,
        x,
        forward_batch: Optional[ForwardBatch] = None,
    ):
        x, _ = self.gate_up_proj(x)

        if self.padded_size_per_partition is not None:
            gate_padded = x[..., : self.padded_size_per_partition]
            up_padded = x[..., self.padded_size_per_partition :]

            gate_effective = gate_padded[..., : self.effective_size_per_partition]
            up_effective = up_padded[..., : self.effective_size_per_partition]

            if self.swiglu_limit is not None:
                x = F.silu(gate_effective).clamp(
                    max=self.swiglu_limit
                ) * up_effective.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
            else:
                x = F.silu(gate_effective) * up_effective

            x = F.pad(x, (0, self.pad_size_per_partition))
        else:
            if self.swiglu_limit is not None:
                d = x.shape[-1] // 2
                gate = F.silu(x[..., :d]).clamp(max=self.swiglu_limit)
                up = x[..., d:].clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
                x = gate * up
            else:
                x = self.act_fn(x)

        if self.use_prequantized_fp8_tail:
            if x.numel() == 0:
                return x.new_empty(*x.shape[:-1], self.hidden_size)
            block_k = self.quant_config.weight_block_size[1]
            logical_shape = x.shape
            x, x_scale = block_quant_fp8(x.reshape(-1, logical_shape[-1]), (1, block_k))
            if self.fp8_tail_route != "triton":
                x_scale = deep_gemm_wrapper.get_mn_major_tma_aligned_tensor(x_scale)
            x = (x.view(logical_shape), x_scale)

        x, _ = self.down_proj(x)
        return x


class BailingMoEGate(nn.Module):
    def __init__(
        self,
        config,
        params_dtype: Optional[torch.dtype] = None,
        prefix: str = "",
    ):
        super().__init__()

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        self.weight = nn.Parameter(
            torch.empty(
                (config.num_experts, config.hidden_size),
                dtype=self.params_dtype,
            ),
        )

        if getattr(config, "moe_router_enable_expert_bias", False):
            self.expert_bias = nn.Parameter(
                torch.empty((config.num_experts,), dtype=torch.float32),
            )
        else:
            self.expert_bias = None

    def forward(self, hidden_states):
        if (
            hidden_states.is_cuda
            and 0 < hidden_states.shape[0] <= ROUTER_GATE_MATVEC_MAX_M
            and self.weight.dtype in (torch.float32, torch.bfloat16)
        ):
            # Decode-sized M: one fp32-accumulating triton matvec for either
            # gate dtype. Cold-cache (rotating weights) it beats the library
            # path up to M=8 — by ~2.6-3x at the bs=1 verify shape (M=4) —
            # and ties it at M=1; see router_gate_matvec for the numbers.
            return router_gate_matvec(hidden_states, self.weight)
        logits = F.linear(hidden_states.to(self.weight.dtype), self.weight, None)
        return logits


def _get_bailing_num_shared_experts(config: PretrainedConfig) -> int:
    shared_intermediate_size = getattr(
        config, "moe_shared_expert_intermediate_size", None
    )
    num_shared_experts = getattr(config, "num_shared_experts", 0) or 0
    if shared_intermediate_size is None:
        return num_shared_experts
    return max(1, num_shared_experts) if shared_intermediate_size > 0 else 0


class BailingMoE(nn.Module):
    @staticmethod
    def _get_swiglu_limit(limit_list, layer_num):
        if limit_list is None or not 0 <= layer_num < len(limit_list):
            return None
        return limit_list[layer_num] or None

    @classmethod
    def _resolve_num_fused_shared_experts(
        cls,
        config: PretrainedConfig,
        layer_id: int,
        requested_num_fused_shared_experts: int,
    ) -> int:
        if requested_num_fused_shared_experts <= 0:
            return 0
        # The fused MoE applies one GEMM1 clamp to routed and shared experts alike,
        # so a layer whose two limits differ keeps its shared expert standalone.
        expert_limit = cls._get_swiglu_limit(
            getattr(config, "expert_swiglu_limit_list", None), layer_id
        )
        shared_limit = cls._get_swiglu_limit(
            getattr(config, "share_expert_swiglu_limit_list", None), layer_id
        )
        if expert_limit != shared_limit:
            return 0
        return requested_num_fused_shared_experts

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        layer_id: int = 0,
        prefix: str = "moe",
        num_fused_shared_experts: int = 0,
        alt_stream: Optional[torch.cuda.Stream] = None,
        swiglu_limit_layer_id: Optional[int] = None,
    ):
        super().__init__()

        self.layer_id = layer_id
        if swiglu_limit_layer_id is None:
            swiglu_limit_layer_id = layer_id
        self.alt_stream = alt_stream

        self.quant_config = quant_config

        self.tp_size = get_parallel().tp_size
        self.tp_rank = get_parallel().tp_rank
        self.moe_ep_size = get_parallel().moe_ep_size
        self.moe_tp_size = get_parallel().moe_tp_size
        self.moe_tp_rank = get_parallel().moe_tp_rank

        self.top_k = config.num_experts_per_tok
        self.norm_expert_prob = getattr(config, "norm_topk_prob", False)
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.num_shared_experts = _get_bailing_num_shared_experts(config)
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.score_function = getattr(config, "score_function", None)
        self.multi_gate = is_bailing_multi_gate_enabled(config)

        expert_swiglu_limit_list = getattr(config, "expert_swiglu_limit_list", None)
        share_expert_swiglu_limit_list = getattr(
            config, "share_expert_swiglu_limit_list", None
        )
        self.expert_swiglu_limit = self._get_swiglu_limit(
            expert_swiglu_limit_list, swiglu_limit_layer_id
        )
        self.share_expert_swiglu_limit = self._get_swiglu_limit(
            share_expert_swiglu_limit_list, swiglu_limit_layer_id
        )
        self.num_fused_shared_experts = self._resolve_num_fused_shared_experts(
            config, swiglu_limit_layer_id, num_fused_shared_experts
        )
        if num_fused_shared_experts > 0 and self.num_fused_shared_experts == 0:
            log_info_on_rank0(
                logger,
                f"Bailing shared-expert fusion is disabled for layer {layer_id}: "
                f"routed/shared SwiGLU limits differ ({self.expert_swiglu_limit} "
                f"vs {self.share_expert_swiglu_limit}).",
            )

        # bf16 here perturbs routing enough to cause output variance, so the
        # gate dtype follows the checkpoint config; absent means fp32.
        router_dtype = getattr(config, "router_dtype", None)
        if router_dtype is None or router_dtype == "fp32":
            self.router_dtype = torch.float32
        else:
            self.router_dtype = torch.bfloat16

        self.num_expert_group = getattr(config, "n_group", 0)
        self.topk_group = getattr(config, "topk_group", 0)
        if self.num_expert_group > 0 or self.topk_group > 0:
            assert (
                self.num_expert_group > 0
                and 0 < self.topk_group <= self.num_expert_group
            )
            self.use_grouped_topk = True
        else:
            self.num_expert_group = self.topk_group = None
            self.use_grouped_topk = False

        self.num_experts = config.num_experts

        self.gate = BailingMoEGate(
            config=config,
            params_dtype=self.router_dtype,
            prefix=add_prefix("gate", prefix),
        )
        self.correction_bias = (
            self.gate.expert_bias.data if self.gate.expert_bias is not None else None
        )

        self.image_gate = None
        self.audio_gate = None
        self.image_correction_bias = None
        self.audio_correction_bias = None
        if self.multi_gate:
            self.image_gate = BailingMoEGate(
                config=config,
                params_dtype=self.router_dtype,
                prefix=add_prefix("image_gate", prefix),
            )
            self.audio_gate = BailingMoEGate(
                config=config,
                params_dtype=self.router_dtype,
                prefix=add_prefix("audio_gate", prefix),
            )
            self.image_correction_bias = (
                self.image_gate.expert_bias.data
                if self.image_gate.expert_bias is not None
                else None
            )
            self.audio_correction_bias = (
                self.audio_gate.expert_bias.data
                if self.audio_gate.expert_bias is not None
                else None
            )
            if any(
                bias is None
                for bias in (
                    self.correction_bias,
                    self.image_correction_bias,
                    self.audio_correction_bias,
                )
            ):
                raise ValueError(
                    "Bailing MultiRouter requires expert bias for text, image, and audio gates"
                )

        if self.score_function is not None:
            assert (
                self.score_function == "softmax" and self.correction_bias is None
            ) or (
                self.score_function == "sigmoid" and self.correction_bias is not None
            ), (
                "score_function and correction_bias should be in 2 combination (softmax, None) or (sigmoid, not None)"
            )

        self._enable_a2a_moe = not get_moe_a2a_backend().is_none()

        # Scaling factor for fused shared experts in EP mode.
        # Non-A2A EP (standard): each GPU computes shared expert, outputs are summed
        # via all_reduce → scale down by 1/ep_size to avoid double counting.
        # Note: A2A EP (DeepEP) + fused shared experts is impossible (expert routing
        # breaks for the extra shared expert ID), so it's auto-disabled in
        # determine_num_fused_shared_experts().
        fused_shared_experts_scaling_factor = None
        if self.moe_ep_size > 1 and self.num_fused_shared_experts > 0:
            fused_shared_experts_scaling_factor = 1.0 / float(self.moe_ep_size)

        self.experts = get_moe_impl_class(quant_config)(
            num_experts=self.num_experts + self.num_fused_shared_experts,
            top_k=self.top_k + self.num_fused_shared_experts,
            num_fused_shared_experts=self.num_fused_shared_experts,
            layer_id=self.layer_id,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            quant_config=quant_config,
            routed_scaling_factor=self.routed_scaling_factor,
            prefix=f"{prefix}.experts",
            gemm1_clamp_limit=self.expert_swiglu_limit,
            allow_fp8_blockwise_tail=True,
        )
        self.topk = TopK(
            top_k=self.top_k + self.num_fused_shared_experts,
            layer_id=self.layer_id,
            use_grouped_topk=self.use_grouped_topk,
            renormalize=self.norm_expert_prob,
            num_expert_group=self.num_expert_group,
            topk_group=self.topk_group,
            correction_bias=self.correction_bias,
            scoring_func=self.score_function
            or ("sigmoid" if self.correction_bias is not None else "softmax"),
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scaling_factor_on_output=(
                self.experts.should_fuse_routed_scaling_factor_in_topk
            ),
            num_fused_shared_experts=self.num_fused_shared_experts,
            fused_shared_experts_scaling_factor=fused_shared_experts_scaling_factor,
            is_fp4_experts=(
                quant_config is not None
                and getattr(quant_config, "is_fp4_experts", False)
            ),
        )

        # Whether to apply routed_scaling_factor at model layer.
        # For A2A MoE paths (e.g., DeepEP), the runner/post_permute does not apply it,
        # so apply it here unless the selected runner fuses it into TopK weights.
        # This scales only routed expert output; shared experts are added unscaled.
        self._apply_routed_scaling_factor_on_output = (
            self._enable_a2a_moe
            and not self.experts.should_fuse_routed_scaling_factor_in_topk
            and self.routed_scaling_factor is not None
            and self.routed_scaling_factor != 1.0
        )

        if self.num_shared_experts > 0 and self.num_fused_shared_experts == 0:
            intermediate_size = getattr(
                config,
                "moe_shared_expert_intermediate_size",
                self.intermediate_size * self.num_shared_experts,
            )
            # When DeepEP is enabled, shared experts should not be TP-sharded
            # because MoE output is already complete after EP combine.
            # Using tp_size=1 ensures shared output is also complete,
            # so no all-reduce is needed at the MoE level.
            shared_tp_size = self.tp_size
            if self._enable_a2a_moe:
                shared_tp_size = 1
            allow_fp8_blockwise_tail = self._allow_shared_fp8_blockwise_tail(
                moe_ep_size=self.moe_ep_size,
                enable_a2a_moe=self._enable_a2a_moe,
                hidden_size=self.hidden_size,
                intermediate_size=intermediate_size,
                quant_config=quant_config,
                tp_size=shared_tp_size,
            )
            padded_intermediate_size = (
                None
                if allow_fp8_blockwise_tail
                else self._compute_padded_intermediate_size(
                    intermediate_size, quant_config, shared_tp_size
                )
            )
            self.shared_experts = BailingMLP(
                hidden_size=self.hidden_size,
                intermediate_size=intermediate_size,
                config=config,
                reduce_results=False,
                quant_config=quant_config,
                prefix=f"{prefix}.shared_experts",
                swiglu_limit=self.share_expert_swiglu_limit,
                padded_intermediate_size=padded_intermediate_size,
                parallel_group="replicated" if self._enable_a2a_moe else "tp",
                allow_fp8_blockwise_tail=allow_fp8_blockwise_tail,
            )
        else:
            self.shared_experts = None

    @staticmethod
    def _allow_shared_fp8_blockwise_tail(
        *,
        moe_ep_size: int,
        enable_a2a_moe: bool,
        hidden_size: int,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig],
        tp_size: int,
    ) -> bool:
        # EP deployments keep the padded shared MLP; only TP-only MoE
        # requantizes the shared shard onto its local block grid.
        if moe_ep_size != 1 or enable_a2a_moe:
            return False
        return BailingMLP.validate_standalone_fp8_tail(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            quant_config=quant_config,
            tp_size=tp_size,
        )

    def _compute_padded_intermediate_size(
        self,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig],
        tp_size: int,
    ) -> Optional[int]:
        """Compute padded intermediate size to satisfy FP8 blockwise quantization alignment.

        FP8 blockwise quantization requires:
        - output_partition_size % block_n == 0 (for column parallel)
        - input_size_per_partition % block_k == 0 (for row parallel)

        When TP size is large, intermediate_size / tp_size may not satisfy these constraints.
        We pad the intermediate_size to make it divisible by block_size * tp_size.

        Args:
            intermediate_size: Original intermediate size
            quant_config: Quantization configuration
            tp_size: TP degree used by the MLP being padded

        Returns:
            Padded intermediate size if padding is needed, None otherwise
        """
        if quant_config is None:
            return None

        if quant_config.get_name() != "fp8":
            return None

        weight_block_size = getattr(quant_config, "weight_block_size", None)
        if weight_block_size is None:
            return None

        block_n = weight_block_size[0]
        block_k = weight_block_size[1]
        block_size = max(block_n, block_k)

        intermediate_size_per_partition = intermediate_size // tp_size

        if (
            intermediate_size_per_partition % block_n == 0
            and intermediate_size_per_partition % block_k == 0
        ):
            return None

        alignment = block_size * tp_size
        padded_intermediate_size = (
            (intermediate_size + alignment - 1) // alignment
        ) * alignment

        log_info_on_rank0(
            logger,
            f"Padding shared_experts intermediate_size from {intermediate_size} to {padded_intermediate_size} "
            f"to satisfy FP8 blockwise quantization alignment (block_size={block_size}, tp_size={tp_size}).",
        )

        return padded_intermediate_size

    def _requires_serialized_fp8_tp_requant(self) -> bool:
        """Whether this Bailing MoE needs model-specific TP requantization."""
        if not isinstance(self.quant_config, Fp8Config):
            return False
        if getattr(self.quant_config, "is_fp4_experts", False):
            # Routed experts ship as MXFP4 (packed e2m1 + per-row group-32
            # e8m0 scales); the blockwise-FP8 [128, 128] scale grid only
            # describes the dense/shared projections, so a non-block-aligned
            # per-rank intermediate size needs no requantization here.
            return False
        if not self.quant_config.is_checkpoint_fp8_serialized:
            return False
        if getattr(self.quant_config, "use_mxfp8", False):
            return False

        block_size = self.quant_config.weight_block_size
        if block_size is None or len(block_size) != 2:
            return False

        # Serialized scales use checkpoint-global tensor coordinates. If a TP
        # boundary cuts through a block, slicing those scales cannot describe
        # the local shard.
        local_intermediate_size = self.intermediate_size // self.moe_tp_size
        requires_requant = self.moe_tp_size > 1 and (
            local_intermediate_size % block_size[0] != 0
            or local_intermediate_size % block_size[1] != 0
        )
        if not requires_requant:
            return False

        self._validate_serialized_fp8_tp_requant_support()
        return True

    def _validate_serialized_fp8_tp_requant_support(self) -> None:
        """Reject configurations for which TP-local requantization is unsafe."""
        parallel = get_parallel()
        runtime = get_exec()
        unsupported = []

        if not (
            self.moe_ep_size == 1
            and parallel.moe_dp_size == 1
            and self.moe_tp_size == self.tp_size
            and self.moe_tp_rank == self.tp_rank
            and not self._enable_a2a_moe
        ):
            unsupported.append("non-pure-TP MoE topology")

        if (
            runtime.moe.enable_eplb
            or runtime.moe.ep_num_redundant_experts != 0
            or runtime.moe.init_expert_location != "trivial"
        ):
            unsupported.append("EPLB or non-trivial expert placement")

        if runtime.offload.cpu_offload_gb > 0 or runtime.offload.offload_group_size > 0:
            unsupported.append("weight offloading")

        quant_method = getattr(self.experts, "quant_method", None)
        if not isinstance(
            quant_method, Fp8MoEMethod
        ) or not quant_method._supports_blockwise_fp8_tail(self.experts):
            unsupported.append("non-Triton MoE runner")

        scales_are_ue8m0 = any(
            getattr(getattr(self.experts, name, None), "format_ue8m0", False)
            for name in ("w13_weight_scale_inv", "w2_weight_scale_inv")
        )
        will_use_ue8m0 = (
            deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
            and Fp8MoEMethod.is_deepgemm_moe_runner_backend_enabled()
        )
        if scales_are_ue8m0 or will_use_ue8m0:
            unsupported.append("UE8M0 scales")

        if unsupported:
            raise NotImplementedError(
                "Bailing serialized blockwise FP8 TP requantization does not "
                f"support layer {self.layer_id} with: {', '.join(unsupported)}."
            )

    def cache_serialized_fp8_scale_for_tp_requant(
        self,
        loaded_scale: torch.Tensor,
        shard_id: str,
        global_expert_id: int,
    ) -> bool:
        """Cache a global checkpoint scale instead of TP-slicing it directly."""
        if not self._requires_serialized_fp8_tp_requant():
            return False
        if shard_id not in {"w1", "w2", "w3"}:
            return False
        if loaded_scale.ndim != 2:
            raise ValueError(
                "Bailing serialized blockwise FP8 expert scales must be 2D, "
                f"but got shape {tuple(loaded_scale.shape)} for {shard_id}."
            )

        local_expert_id = self.experts._map_global_expert_id_to_local_expert_id(
            global_expert_id
        )
        if local_expert_id == -1:
            return True

        if not hasattr(self, "_serialized_fp8_tp_scale_cache"):
            self._serialized_fp8_tp_scale_cache: Dict[str, torch.Tensor] = {}
            self._serialized_fp8_tp_scale_loaded: Dict[str, Set[int]] = {
                "w1": set(),
                "w2": set(),
                "w3": set(),
            }

        scale_cache = self._serialized_fp8_tp_scale_cache.get(shard_id)
        expected_shape = (self.experts.num_local_experts, *loaded_scale.shape)
        if scale_cache is None:
            scale_cache = torch.empty(
                expected_shape,
                dtype=loaded_scale.dtype,
                device=self.experts.w13_weight.device,
            )
            self._serialized_fp8_tp_scale_cache[shard_id] = scale_cache
        elif tuple(scale_cache.shape) != expected_shape:
            raise ValueError(
                "Inconsistent Bailing serialized blockwise FP8 scale shapes for "
                f"{shard_id}: expected {tuple(scale_cache.shape[1:])}, got "
                f"{tuple(loaded_scale.shape)}."
            )

        scale_cache[local_expert_id].copy_(
            loaded_scale.to(device=scale_cache.device, dtype=scale_cache.dtype)
        )
        self._serialized_fp8_tp_scale_loaded[shard_id].add(local_expert_id)
        return True

    @staticmethod
    def _dequantize_serialized_fp8_tp_shard(
        quantized_weight: torch.Tensor,
        checkpoint_scale: torch.Tensor,
        block_size: Tuple[int, int],
        row_offset: int = 0,
        column_offset: int = 0,
    ) -> torch.Tensor:
        """Dequantize a local TP shard using checkpoint-global block indices."""
        block_n, block_k = block_size
        rows, columns = quantized_weight.shape
        last_scale_row = (row_offset + rows - 1) // block_n
        last_scale_column = (column_offset + columns - 1) // block_k
        if (
            last_scale_row >= checkpoint_scale.shape[0]
            or last_scale_column >= checkpoint_scale.shape[1]
        ):
            raise ValueError(
                "Bailing serialized blockwise FP8 TP shard exceeds its checkpoint "
                f"scale: weight={tuple(quantized_weight.shape)}, "
                f"scale={tuple(checkpoint_scale.shape)}, offsets="
                f"({row_offset}, {column_offset}), block_size={block_size}."
            )

        scale_rows = (
            torch.arange(rows, device=quantized_weight.device) + row_offset
        ) // block_n
        scale_columns = (
            torch.arange(columns, device=quantized_weight.device) + column_offset
        ) // block_k
        element_scale = checkpoint_scale.index_select(0, scale_rows).index_select(
            1, scale_columns
        )
        return (quantized_weight.float() * element_scale.float()).to(torch.bfloat16)

    def requantize_serialized_fp8_tp_weights(self) -> bool:
        """Convert serialized global blocks into blocks local to this TP rank."""
        if not self._requires_serialized_fp8_tp_requant() or not hasattr(
            self, "_serialized_fp8_tp_scale_cache"
        ):
            return False

        expected_experts = set(range(self.experts.num_local_experts))
        missing = {
            shard_id: sorted(
                expected_experts
                - self._serialized_fp8_tp_scale_loaded.get(shard_id, set())
            )
            for shard_id in ("w1", "w2", "w3")
        }
        missing = {key: value for key, value in missing.items() if value}
        if missing:
            raise ValueError(
                "Missing serialized blockwise FP8 scales required for Bailing TP "
                f"requantization in layer {self.layer_id}: {missing}."
            )

        block_n, block_k = self.quant_config.weight_block_size
        if block_n != block_k:
            raise ValueError(
                "Bailing serialized FP8 TP requantization currently requires square "
                f"blocks, but got {self.quant_config.weight_block_size}."
            )

        local_intermediate_size = self.intermediate_size // self.moe_tp_size
        if self.experts.intermediate_size_per_partition != local_intermediate_size:
            raise ValueError(
                "Bailing serialized FP8 TP requantization does not support a padded "
                "MoE intermediate dimension: "
                f"expected {local_intermediate_size}, got "
                f"{self.experts.intermediate_size_per_partition}."
            )

        w13_weight = self.experts.w13_weight.data
        w2_weight = self.experts.w2_weight.data
        expected_w13_shape = (
            self.experts.num_local_experts,
            2 * local_intermediate_size,
            self.hidden_size,
        )
        expected_w2_shape = (
            self.experts.num_local_experts,
            self.hidden_size,
            local_intermediate_size,
        )
        if (
            tuple(w13_weight.shape) != expected_w13_shape
            or tuple(w2_weight.shape) != expected_w2_shape
        ):
            raise ValueError(
                "Unexpected Bailing MoE weight layout for serialized FP8 TP "
                f"requantization: w13={tuple(w13_weight.shape)}, "
                f"w2={tuple(w2_weight.shape)}, expected w13={expected_w13_shape}, "
                f"w2={expected_w2_shape}."
            )

        new_w13_scale = torch.empty(
            self.experts.num_local_experts,
            (2 * local_intermediate_size + block_n - 1) // block_n,
            (self.hidden_size + block_k - 1) // block_k,
            dtype=torch.float32,
            device=w13_weight.device,
        )
        new_w2_scale = torch.empty(
            self.experts.num_local_experts,
            (self.hidden_size + block_n - 1) // block_n,
            (local_intermediate_size + block_k - 1) // block_k,
            dtype=torch.float32,
            device=w2_weight.device,
        )

        tp_offset = self.moe_tp_rank * local_intermediate_size
        switch_w13 = getattr(
            self.experts.quant_method, "load_up_proj_weight_first", False
        )
        first_shard, second_shard = ("w3", "w1") if switch_w13 else ("w1", "w3")
        scale_cache = self._serialized_fp8_tp_scale_cache

        for expert_id in range(self.experts.num_local_experts):
            first_weight = self._dequantize_serialized_fp8_tp_shard(
                w13_weight[expert_id, :local_intermediate_size],
                scale_cache[first_shard][expert_id],
                (block_n, block_k),
                row_offset=tp_offset,
            )
            second_weight = self._dequantize_serialized_fp8_tp_shard(
                w13_weight[expert_id, local_intermediate_size:],
                scale_cache[second_shard][expert_id],
                (block_n, block_k),
                row_offset=tp_offset,
            )
            requant_w13, requant_w13_scale = block_quant_fp8(
                torch.cat((first_weight, second_weight), dim=0),
                (block_n, block_k),
            )

            local_w2 = self._dequantize_serialized_fp8_tp_shard(
                w2_weight[expert_id],
                scale_cache["w2"][expert_id],
                (block_n, block_k),
                column_offset=tp_offset,
            )
            requant_w2, requant_w2_scale = block_quant_fp8(local_w2, (block_n, block_k))

            w13_weight[expert_id].copy_(requant_w13)
            w2_weight[expert_id].copy_(requant_w2)
            new_w13_scale[expert_id].copy_(requant_w13_scale)
            new_w2_scale[expert_id].copy_(requant_w2_scale)

        # Preserve the original Parameter objects and their weight-loader attrs.
        self.experts.w13_weight_scale_inv.data = new_w13_scale
        self.experts.w2_weight_scale_inv.data = new_w2_scale
        self.experts.w13_weight_scale_inv.format_ue8m0 = False
        self.experts.w2_weight_scale_inv.format_ue8m0 = False
        del self._serialized_fp8_tp_scale_cache
        del self._serialized_fp8_tp_scale_loaded
        return True

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
    ) -> torch.Tensor:
        if self._enable_a2a_moe:
            return self.forward_deepep(hidden_states, forward_batch)
        return self.forward_normal(hidden_states, forward_batch)

    def forward_normal(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
    ) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)

        if num_tokens == 0:
            shared_output = None
            topk_output = self.topk.empty_topk_output(hidden_states.device)
            final_hidden_states = self.experts(hidden_states, topk_output)
        elif (
            self.num_fused_shared_experts == 0
            and self.shared_experts is not None
            and self.alt_stream is not None
            and get_is_capture_mode()
        ):
            final_hidden_states, shared_output = self.forward_normal_dual_stream(
                hidden_states, forward_batch
            )
        else:
            # Fold the shared output into MoE finalize when supported.
            output_add = self._forward_shared_experts(hidden_states)
            shared_output = None
            final_hidden_states = self._forward_router_experts(
                hidden_states, forward_batch, output_add=output_add
            )

        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output
        return final_hidden_states

    def _forward_shared_experts(
        self, hidden_states: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if self.shared_experts is None:
            return None
        return self.shared_experts(hidden_states)

    def _forward_gate(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if (
            self.multi_gate
            and forward_batch is not None
            and forward_batch.mm_token_modalities is not None
        ):
            if forward_batch.mm_token_modalities.shape[0] != hidden_states.shape[0]:
                raise ValueError(
                    "Bailing modality metadata must align with MoE tokens: "
                    f"modalities={forward_batch.mm_token_modalities.shape[0]}, "
                    f"hidden_states={hidden_states.shape[0]}"
                )
            if forward_batch.multi_gate_indices is None:
                forward_batch.multi_gate_indices = create_multi_gate_mm_indices(
                    forward_batch.mm_token_modalities
                )
            return multi_gate_triton_kernel(
                hidden_states,
                forward_batch.multi_gate_indices,
                self.gate.weight,
                self.image_gate.weight,
                self.audio_gate.weight,
                self.correction_bias,
                self.image_correction_bias,
                self.audio_correction_bias,
            )
        return self.gate(hidden_states), None

    def _forward_router_experts(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch],
        output_add: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        router_logits, dynamic_expert_bias = self._forward_gate(
            hidden_states, forward_batch
        )
        topk_output = self.topk(
            hidden_states,
            router_logits,
            dynamic_expert_bias=dynamic_expert_bias,
        )
        return self.experts(hidden_states, topk_output, output_add=output_add)

    def forward_normal_dual_stream(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        current_stream = torch.cuda.current_stream()
        self.alt_stream.wait_stream(current_stream)

        shared_output = self._forward_shared_experts(hidden_states.clone())

        with torch.cuda.stream(self.alt_stream):
            final_hidden_states = self._forward_router_experts(
                hidden_states, forward_batch
            )

        current_stream.wait_stream(self.alt_stream)
        return final_hidden_states, shared_output

    def forward_deepep(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        assert forward_batch is not None, "forward_batch is required for DeepEP MoE"
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)

        if num_tokens == 0:
            shared_output = None
            topk_output = self.topk.empty_topk_output(hidden_states.device)
        else:
            shared_output = self._forward_shared_experts(hidden_states)
            router_logits, dynamic_expert_bias = self._forward_gate(
                hidden_states, forward_batch
            )
            topk_output = self.topk(
                hidden_states,
                router_logits,
                dynamic_expert_bias=dynamic_expert_bias,
                num_token_non_padded=forward_batch.moe_num_token_non_padded(),
                expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                    layer_id=self.layer_id,
                ),
            )
        final_hidden_states = self.experts(hidden_states, topk_output)

        # In DeepEP mode, MoE output is already complete after EP combine.
        # Apply routed_scaling_factor here since the runner does not apply it
        # for DeepEP paths (post_permute_deep_gemm_to_deepep_normal/ll).
        if shared_output is not None:
            if self._apply_routed_scaling_factor_on_output:
                shared_output.add_(
                    final_hidden_states, alpha=self.routed_scaling_factor
                )
            else:
                shared_output.add_(final_hidden_states)
            final_hidden_states = shared_output
        elif self._apply_routed_scaling_factor_on_output:
            final_hidden_states = final_hidden_states * self.routed_scaling_factor

        # No all-reduce needed: both MoE output (complete after EP combine)
        # and shared output (tp_size=1, complete) are already full results.
        return final_hidden_states


class BailingKDA(KimiDeltaAttention):
    def __init__(
        self,
        layer_id: int,
        hidden_size: int,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        reduce_results: bool = True,
        **kwargs,
    ) -> None:
        kimi_linear_config = KimiLinearConfig(
            linear_attn_config={
                "head_dim": config.head_dim,
                "num_heads": config.num_attention_heads,
                "short_conv_kernel_size": config.short_conv_kernel_size,
                "kda_layers": [],
                "full_attn_layers": [],
            },
            v_head_dim=config.v_head_dim,
        )
        super().__init__(
            layer_id,
            hidden_size,
            kimi_linear_config,
            quant_config,
            prefix=prefix,
            rms_norm_eps=1e-6,
            no_kda_lora=config.no_kda_lora,
            safe_gate=config.kda_safe_gate,
            lower_bound=config.kda_lower_bound,
            reduce_results=reduce_results,
            # Ling-V3 shards KDA on the attention-TP group (DP attention) and
            # carries its own value head dim; upstream Kimi-Linear does neither,
            # so both are opt-in on KimiDeltaAttention.
            shard_on_attn_tp=True,
            v_head_dim=config.v_head_dim,
        )


class BailingMoEAttention(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        layer_id: int = None,
        prefix: str = "mha",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        self.hidden_size = config.hidden_size
        tp_size = get_parallel().attn_tp_size
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = getattr(config, "head_dim", None)
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.total_num_heads

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.split_qkv = getattr(config, "using_split_qkv_in_self_attention", False)
        assert not self.split_qkv, "split_qkv is not supported for now"
        self.use_qk_norm = getattr(config, "use_qk_norm", False)

        self.query_key_value = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=(config.use_bias or config.use_qkv_bias),
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        if self.use_qk_norm:
            self.query_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.key_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.dense = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=config.use_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
            reduce_results=False,
        )
        if hasattr(config, "rotary_dim"):
            self.rotary_dim = config.rotary_dim
        elif hasattr(config, "partial_rotary_factor"):
            self.rotary_dim = int(self.head_dim * config.partial_rotary_factor)
        else:
            self.rotary_dim = self.head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.rotary_dim,
            max_position=self.max_position_embeddings,
            base=config.rope_parameters.get("rope_theta", 600000),
            rope_scaling=config.rope_parameters,
            dtype=torch.float32,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def _apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_by_head = q.reshape(-1, self.head_dim)
        q_by_head = self.query_layernorm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.reshape(-1, self.head_dim)
        k_by_head = self.key_layernorm(k_by_head)
        k = k_by_head.view(k.shape)
        return q, k

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> torch.Tensor:
        qkv, _ = self.query_key_value(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.use_qk_norm:
            q, k = self._apply_qk_norm(q, k)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.dense(attn_output)
        return output


class BailingMoELinearDecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        layer_id: int = 0,
        prefix: str = "layer",
        num_fused_shared_experts: int = 0,
        is_nextn: bool = False,
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.use_mla = getattr(config, "full_attention_type", "mla") == "mla"
        self.attention_type = config.attention_type
        self.config = config

        if config.attention_type == 0:  # Linear layer
            self.attention = BailingKDA(
                layer_id=self.layer_id,
                hidden_size=config.hidden_size,
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                reduce_results=False,
            )
        elif config.attention_type == 1:  # softmax layer
            if self.use_mla:
                self.attention = DsV3MLA(
                    config=config,
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    qk_nope_head_dim=config.qk_nope_head_dim,
                    qk_rope_head_dim=config.qk_rope_head_dim,
                    v_head_dim=config.v_head_dim,
                    q_lora_rank=(
                        config.q_lora_rank if hasattr(config, "q_lora_rank") else None
                    ),
                    kv_lora_rank=config.kv_lora_rank,
                    rope_theta=config.rope_parameters.get("rope_theta", 600000),
                    rope_scaling=config.rope_parameters,
                    max_position_embeddings=config.max_position_embeddings,
                    quant_config=quant_config,
                    layer_id=layer_id,
                    reduce_results=False,
                    prefix=add_prefix("attention", prefix),
                    alt_stream=alt_stream,
                    skip_rope=(
                        getattr(config, "use_mla_nope", False)
                        or config.qk_rope_head_dim == 0
                    ),
                )
            else:
                self.attention = BailingMoEAttention(
                    config,
                    quant_config=quant_config,
                    layer_id=self.layer_id,
                    prefix=prefix + ".attention",
                )
        else:
            raise ValueError(f"Unsupported attention type: {config.attention_type}")

        self.expert_num = config.num_experts
        self.hidden_size = config.hidden_size
        attn, ffn = self.stage_facts(config, self.layer_id, is_nextn=is_nextn)
        self.is_layer_sparse = ffn.sparse
        mlp_parallel_group = "replicated" if is_dense_ffn_fully_dp() else "tp"

        if self.expert_num == 1:
            self.mlp = BailingMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
                parallel_group=mlp_parallel_group,
                reduce_results=False,
            )
        else:
            if is_nextn or self.layer_id >= config.first_k_dense_replace:
                self.mlp = BailingMoE(
                    config,
                    quant_config=quant_config,
                    layer_id=self.layer_id,
                    prefix=prefix,
                    num_fused_shared_experts=num_fused_shared_experts,
                    alt_stream=alt_stream,
                    # The MTP layer is built with layer_id 0; its limits live at
                    # its checkpoint index, after the target decoder layers.
                    swiglu_limit_layer_id=(
                        config.num_hidden_layers if is_nextn else self.layer_id
                    ),
                )
            else:
                self.mlp = BailingMLP(
                    hidden_size=self.hidden_size,
                    intermediate_size=config.intermediate_size,
                    config=config,
                    quant_config=quant_config,
                    prefix=prefix,
                    parallel_group=mlp_parallel_group,
                    reduce_results=False,
                )
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-5))
        self.input_layernorm = RMSNorm(self.hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=rms_norm_eps)

        self.attn_boundary, self.ffn_boundary = append_stages(
            (
                attn,
                self.input_layernorm,
                {
                    "qkv_latent_func": self.attention.prepare_qkv_latent
                    if self.attention_type == 1 and self.use_mla
                    else None
                },
            ),
            (ffn, self.post_attention_layernorm),
        )

    @staticmethod
    def stage_facts(config: PretrainedConfig, layer_id: int, is_nextn: bool = False):
        """The stages the layer at ``layer_id`` declares, from the config: the
        model's shared declaration function (see make_layers)."""
        is_moe_layer = is_nextn or (
            not (config.num_experts == 1) and (layer_id >= config.first_k_dense_replace)
        )
        is_next_layer_sparse = not (config.num_experts == 1) and (
            layer_id + 1 >= config.first_k_dense_replace
        )
        return (
            declare_attn(),
            declare_ffn(
                sparse=is_moe_layer,
                next_layer_sparse=is_next_layer_sparse,
            ),
        )

    @torch.inference_mode()
    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
        capture_output=None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = self.attn_boundary.prepare(
            hidden_states, forward_batch, capture=capture_output
        )

        attn_output_reduced = False
        if not forward_batch.forward_mode.is_idle():
            # An o_proj that reduced inline must not leave an owed sum behind.
            inline_tp_reduce = (
                chunked_gemm_ar_enabled()
                and self.attn_boundary.can_inline_attn_tp_reduce(forward_batch)
            )
            if self.attention_type == 0:
                hidden_states = self.attention(
                    hidden_states=hidden_states,
                    positions=positions,
                    forward_batch=forward_batch,
                    zero_allocator=zero_allocator,
                    inline_tp_reduce=inline_tp_reduce,
                )
                attn_output_reduced = self.attention._inline_tp_reduced
            elif self.use_mla:
                hidden_states = self.attention(
                    positions=positions,
                    hidden_states=hidden_states,
                    forward_batch=forward_batch,
                    zero_allocator=zero_allocator,
                    inline_tp_reduce=inline_tp_reduce,
                )
                attn_output_reduced = self.attention._inline_tp_reduced
            else:
                hidden_states = self.attention(
                    hidden_states=hidden_states,
                    positions=positions,
                    forward_batch=forward_batch,
                )

        hidden_states = self.attn_boundary.finish(
            hidden_states, forward_batch, already_reduced=attn_output_reduced
        )
        hidden_states = self.ffn_boundary.prepare(hidden_states, forward_batch)

        if not (
            is_dense_ffn_fully_dp()
            and (not self.is_layer_sparse)
            and hidden_states.shape[0] == 0
        ):
            hidden_states = self.mlp(
                hidden_states,
                forward_batch=forward_batch,
            )
        hidden_states = self.ffn_boundary.finish(hidden_states, forward_batch)

        return hidden_states

    @staticmethod
    def shared_moe_coefficient_loader(
        param: torch.Tensor, loaded_weight: torch.Tensor
    ) -> None:
        assert param.size() == loaded_weight.size()

        param.data.copy_(loaded_weight.to(torch.float32))
        return


class BailingMoELinearModel(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        num_fused_shared_experts: int = 0,
    ) -> None:
        super().__init__()
        self.pp_group = get_parallel().pp_group
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_dim = config.hidden_size
        self.num_layers = config.num_hidden_layers

        self.layer_group_size = getattr(config, "layer_group_size", 1)
        self.decoder_attention_types = [
            0 if is_linear_layer(i, self.layer_group_size) else 1
            for i in range(self.num_layers)
        ]
        logger.info(
            f"attention type of layers:{self.decoder_attention_types}, 0 is linear layer and 1 is softmax layer!"
        )

        assert (
            isinstance(self.layer_group_size, list)
            or self.num_layers % self.layer_group_size == 0
        ), (
            f"num_layers={self.num_layers} must be divided by layer_group_size={self.layer_group_size}"
        )

        keep_word_embeddings = self.pp_group.is_first_rank or (
            config.tie_word_embeddings and self.pp_group.is_last_rank
        )
        if keep_word_embeddings:
            self.word_embeddings = VocabParallelEmbedding(
                self.vocab_size,
                self.embed_dim,
                enable_tp=not is_dp_attention_enabled(),
                org_num_embeddings=self.vocab_size,
            )
        else:
            self.word_embeddings = PPMissingLayer()

        self.alt_stream = get_stream("alt") if _is_cuda else None

        def layer_fn(idx, prefix):
            layer_idx = idx
            layer_config = copy.deepcopy(config)
            layer_config.attention_type = self.decoder_attention_types[layer_idx]

            decoder_kwargs = {
                "quant_config": quant_config,
                "layer_id": layer_idx,
                "num_fused_shared_experts": num_fused_shared_experts,
            }
            return BailingMoELinearDecoderLayer(
                layer_config,
                **decoder_kwargs,
                prefix=prefix,
                alt_stream=self.alt_stream,
            )

        self.layers, self.start_layer, self.end_layer = make_pp_layers(
            self.num_layers,
            layer_fn,
            prefix=f"{prefix}.layers",
            stage_facts=lambda idx: BailingMoELinearDecoderLayer.stage_facts(
                config, idx
            ),
        )

        linear_layer_nums = sum(
            1 for i in range(self.num_layers) if self.decoder_attention_types[i] == 0
        )
        logger.info(f"linear_layer_nums={linear_layer_nums}")

        norm_kwargs = {}
        if hasattr(config, "rms_norm_eps"):
            norm_kwargs["eps"] = config.rms_norm_eps
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, **norm_kwargs)
        else:
            self.norm = PPMissingLayer()
        self.embed_scale = 1.0

        # Hidden-state capture for speculative decoding (DSpark / DFlash / EAGLE3).
        # When set, ``forward`` appends each captured layer's final
        # ``hidden_states + residual`` (shape ``[N, hidden_size]``) to an
        # aux list and returns ``(hidden_states, aux_hidden_states)``.
        self.layers_to_capture: Optional[List[int]] = None
        self.capture_aux_hidden_states = False
        return

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.word_embeddings(input_ids)
            else:
                hidden_states = input_embeds
            residual_batch.start(forward_batch)
        else:
            assert pp_proxy_tensors is not None
            hidden_states = self.layers[self.start_layer].attn_boundary.from_pp(
                pp_proxy_tensors, forward_batch
            )

        total_num_layers = self.end_layer - self.start_layer
        device = hidden_states.device
        zero_allocator = BumpAllocator(
            buffer_size=total_num_layers * 2 * (2 if forward_batch.can_run_tbo else 1),
            dtype=torch.float32,
            device=device,
        )

        # DSpark / DFlash capture relies on the per-layer eager loop exposing the
        # completed hidden state of each layer, so it is only supported when this
        # rank owns the final pipeline stage (the aux list is consumed there).
        # Gate on capture_hidden_mode so plain (non-capturing) prefills / decodes
        # keep the zero-overhead path and only the dspark target-prefill asks for
        # aux hidden states.
        capture_mode = forward_batch.capture_hidden_mode
        capture_aux = (
            self.capture_aux_hidden_states
            and self.pp_group.is_last_rank
            and self.layers_to_capture is not None
            and capture_mode is not None
            and capture_mode.need_capture()
        )
        if capture_aux:
            dspark_aux_hidden_states = AuxHiddenStateList()

            def capture_output(value, *, owned=False):
                if value.shape[0] != 0:
                    dspark_aux_hidden_states.capture(value, owned=owned)

        for i in range(self.start_layer, self.end_layer):
            with get_global_expert_distribution_recorder().with_current_layer(i):
                layer = self.layers[i]
                hidden_states = layer(
                    hidden_states=hidden_states,
                    positions=positions,
                    forward_batch=forward_batch,
                    zero_allocator=zero_allocator,
                    capture_output=capture_output
                    if capture_aux
                    and i > self.start_layer
                    and (i - 1 in self.layers_to_capture)
                    else None,
                )

        if not self.pp_group.is_last_rank:
            return residual_batch.to_pp(hidden_states, forward_batch)
        else:
            hidden_states = residual_batch.complete_output(hidden_states, forward_batch)
            if not forward_batch.forward_mode.is_idle():
                hidden_states = residual_batch.final_norm(
                    hidden_states,
                    forward_batch,
                    self.norm,
                    capture=capture_output
                    if capture_aux and self.end_layer - 1 in self.layers_to_capture
                    else None,
                )
            if capture_aux and len(dspark_aux_hidden_states) > 0:
                return hidden_states, dspark_aux_hidden_states
            return hidden_states


class BailingMoeV3ForCausalLM(nn.Module):
    def __init__(
        self,
        *,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.pp_group = get_parallel().pp_group
        self.config = config
        self.quant_config = quant_config

        # Determine num_fused_shared_experts
        self.determine_num_fused_shared_experts()

        self.model = BailingMoELinearModel(
            self.config,
            quant_config,
            prefix=add_prefix("model", prefix),
            num_fused_shared_experts=self.num_fused_shared_experts,
        )

        if self.pp_group.is_last_rank:
            self.lm_head = (
                self.model.word_embeddings
                if config.tie_word_embeddings
                else ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    # bf16 lm_head (was fp32): fp32 vocab GEMM uses a slow sm80
                    # path on Hopper; logits are still cast to fp32 for sampling
                    # in the logits processor. Accuracy-neutral on ling-v3.
                    params_dtype=torch.bfloat16,
                    quant_config=quant_config,
                    use_attn_tp_group=get_parallel().enable_dp_lm_head,
                )
            )
            self.logits_processor = LogitsProcessor(config)
        else:
            self.lm_head = PPMissingLayer()

        self.capture_aux_hidden_states = False

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    @classmethod
    def _supports_fused_fp8_blockwise_tail(cls, quant_config) -> bool:
        """Strict Bailing-only admission for native Triton MoE K tails."""
        if (
            quant_config is None
            or quant_config.get_name() != "fp8"
            or getattr(quant_config, "use_mxfp8", False)
            or tuple(getattr(quant_config, "weight_block_size", ()) or ()) != (128, 128)
            or not getattr(quant_config, "is_checkpoint_fp8_serialized", False)
            or not _is_cuda
            or _is_hip
        ):
            return False

        parallel = get_parallel()
        runtime = get_exec()
        if not (
            parallel.moe_ep_size == 1
            and parallel.moe_dp_size == 1
            and parallel.moe_tp_size == parallel.tp_size
            and parallel.moe_tp_rank == parallel.tp_rank
            and get_moe_a2a_backend().is_none()
            and not runtime.moe.enable_eplb
            and runtime.moe.ep_num_redundant_experts == 0
            and runtime.moe.init_expert_location == "trivial"
            and runtime.offload.cpu_offload_gb == 0
            and runtime.offload.offload_group_size <= 0
        ):
            return False

        moe_runner_backend = get_moe_runner_backend()
        return moe_runner_backend.is_triton() or (
            moe_runner_backend.is_auto()
            and not Fp8MoEMethod.is_deepgemm_moe_runner_backend_enabled()
        )

    @classmethod
    def shared_experts_fusion_disable_reason(
        cls,
        hf_config,
        quant_config,
        expected_architecture="BailingMoeV3ForCausalLM",
    ):
        num_shared_experts = _get_bailing_num_shared_experts(hf_config)
        if num_shared_experts == 0:
            return None
        if not get_moe_a2a_backend().is_none():
            return (
                "A2A MoE backend (e.g., DeepEP) is enabled. Fused shared experts is "
                "incompatible with A2A backends because the extra shared expert ID cannot "
                "be correctly routed."
            )
        if hf_config.architectures[0] != expected_architecture:
            return "Config does not support fused shared expert(s)."
        if (not _is_cuda or torch.cuda.get_device_capability("cuda") < (8, 0)) and (
            not _is_hip or torch.cuda.get_device_capability("cuda") < (9, 4)
        ):
            return (
                "Only Bailing MoE V3 on NV-platform with capability >= 80 "
                "or AMD-platform with capability >= gfx942(MI30x) can use shared experts fusion optimization."
            )
        if quant_config and (
            quant_config.get_name() == "w4afp8"
            or getattr(quant_config, "is_fp4_experts", False)
        ):
            return (
                "Bailing MoE V3 uses different quant methods for routed experts "
                "and shared experts."
            )
        if quant_config and quant_config.get_name() == "compressed_tensors":
            from sglang.srt.layers.quantization.compressed_tensors.utils import (
                should_ignore_layer,
            )

            ignore = getattr(quant_config, "ignore", ())
            fused_mapping = getattr(quant_config, "packed_modules_mapping", {})
            shared_ignored = should_ignore_layer(
                "model.layers.0.mlp.shared_experts.gate_proj",
                ignore=ignore,
                fused_mapping=fused_mapping,
            )
            routed_ignored = should_ignore_layer(
                "model.layers.0.mlp.experts.0.gate_proj",
                ignore=ignore,
                fused_mapping=fused_mapping,
            )
            if shared_ignored != routed_ignored:
                return (
                    "Bailing MoE V3 uses different quant methods for routed experts "
                    "and shared experts."
                )
        shared_expert_intermediate_size = getattr(
            hf_config,
            "moe_shared_expert_intermediate_size",
            hf_config.moe_intermediate_size * num_shared_experts,
        )
        if shared_expert_intermediate_size != hf_config.moe_intermediate_size:
            return (
                f"Shared experts have different intermediate_size ({shared_expert_intermediate_size}) "
                f"from routed experts ({hf_config.moe_intermediate_size}). Fusion requires them to be equal."
            )
        if quant_config and quant_config.get_name() == "fp8":
            weight_block_size = getattr(quant_config, "weight_block_size", None)
            if weight_block_size is not None:
                block_n, block_k = weight_block_size
                tp_size = get_parallel().tp_size
                moe_ep_size = get_parallel().moe_ep_size
                moe_tp_size = tp_size // moe_ep_size if moe_ep_size > 1 else tp_size
                intermediate_size_per_partition = (
                    hf_config.moe_intermediate_size // moe_tp_size
                )
                if (
                    intermediate_size_per_partition % block_n != 0
                    or intermediate_size_per_partition % block_k != 0
                ) and not cls._supports_fused_fp8_blockwise_tail(quant_config):
                    return (
                        "FP8 blockwise quantization requires "
                        f"intermediate_size_per_partition ({intermediate_size_per_partition}) "
                        f"to be divisible by block_n ({block_n}) and block_k ({block_k}). "
                        f"Current config: moe_intermediate_size={hf_config.moe_intermediate_size}, "
                        f"tp_size={tp_size}, moe_tp_size={moe_tp_size}. Consider using "
                        "--disable-shared-experts-fusion to use padding solution instead."
                    )
        return None

    def determine_num_fused_shared_experts(self):
        self.num_fused_shared_experts = (
            0
            if is_shared_experts_fusion_disabled()
            else _get_bailing_num_shared_experts(self.config)
        )
        if self.num_fused_shared_experts == 0:
            return

        # Safety check: current CUDA implementation only supports num_fused_shared_experts == 1.
        # The grouped_topk_gpu and _post_process_topk_ids functions only handle the last column,
        # which is incorrect when num_fused_shared_experts > 1.
        # AMD platform with aiter handles this correctly via fused_append_shared_experts kernel.
        if self.num_fused_shared_experts > 1 and not _is_hip:
            raise ValueError(
                f"num_fused_shared_experts > 1 ({self.num_fused_shared_experts}) is not "
                f"supported on CUDA platform. The current TopK implementation only handles "
                f"one fused shared expert. AMD platform with aiter supports multiple shared experts."
            )

        moe_ep_size = get_parallel().moe_ep_size
        if moe_ep_size > 1:
            log_info_on_rank0(
                logger,
                f"Shared experts fusion optimization is enabled with {self.num_fused_shared_experts} fused shared expert(s) under EP mode (ep_size={moe_ep_size}). "
                f"Shared experts will be distributed across GPUs along with routed experts.",
            )
        else:
            log_info_on_rank0(
                logger,
                f"Shared experts fusion optimization is enabled with {self.num_fused_shared_experts} fused shared expert(s).",
            )

    def post_load_weights(self, is_nextn=False, weight_names=None):
        if is_nextn:
            layer_ids = [self.config.num_hidden_layers]
        else:
            if weight_names is None:
                layer_ids = range(self.model.start_layer, self.model.end_layer)
            else:
                layer_ids = set()
                for name in weight_names:
                    if "kv_b_proj" in name:
                        layer_id = int(name.split(".")[2])
                        if (
                            layer_id < self.model.end_layer
                            and layer_id >= self.model.start_layer
                        ):
                            layer_ids.add(layer_id)
        for layer_id in layer_ids:
            self_attn = (
                self.model.layers[layer_id].attention
                if not is_nextn
                else self.model.decoder.attention
            )
            if not hasattr(self_attn, "kv_b_proj"):
                continue
            if hasattr(self_attn.kv_b_proj, "qweight"):
                awq_dequantize = _get_awq_dequantize()
                if awq_dequantize is None:
                    raise RuntimeError(
                        "AWQ dequantization is unavailable on this CPU platform"
                    )
                if _is_cuda or _is_hip:
                    w = awq_dequantize(
                        self_attn.kv_b_proj.qweight,
                        self_attn.kv_b_proj.scales,
                        self_attn.kv_b_proj.qzeros,
                    ).T
                else:
                    w = awq_dequantize(
                        self_attn.kv_b_proj.qweight,
                        self_attn.kv_b_proj.scales,
                        self_attn.kv_b_proj.qzeros,
                        0,
                        0,
                        0,
                    ).T
            else:
                w = self_attn.kv_b_proj.weight
            # NOTE(HandH1998): Since `bmm_fp8` only supports per-tensor scale, we have to requantize `self_attn.kv_b_proj`.
            # This may affect the accuracy of fp8 model.
            # Fix deepseek v3 blockwise bmm by using deep_gemm
            use_deep_gemm_bmm = False

            if w.dtype in (
                torch.float8_e4m3fn,
                torch.float8_e4m3fnuz,
            ):
                if (
                    hasattr(self.quant_config, "weight_block_size")
                    and self.quant_config.weight_block_size is not None
                ):
                    weight_block_size = self.quant_config.weight_block_size
                    assert hasattr(self_attn.kv_b_proj, "weight_scale_inv")
                    if _is_fp8_fnuz:
                        weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                            weight=w,
                            weight_scale=self_attn.kv_b_proj.weight_scale_inv,
                            input_scale=None,
                        )
                    else:
                        weight = w
                        weight_scale = self_attn.kv_b_proj.weight_scale_inv

                    if (
                        _is_cuda
                        and weight_block_size[0] == 128
                        and weight_block_size[1] == 128
                    ):
                        if (
                            deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
                            and not deep_gemm_wrapper.DEEPGEMM_BLACKWELL
                            and envs.SGLANG_USE_DEEPGEMM_BMM.get()
                        ):
                            block_scale = weight_scale
                            use_deep_gemm_bmm = True
                        else:
                            w = block_quant_dequant(
                                weight,
                                weight_scale,
                                weight_block_size,
                                torch.bfloat16,
                            )
                    else:
                        w, scale = block_quant_to_tensor_quant(
                            weight, weight_scale, weight_block_size
                        )
                        self_attn.w_scale = scale
                else:
                    if _is_fp8_fnuz:
                        weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                            weight=w,
                            weight_scale=self_attn.kv_b_proj.weight_scale,
                            input_scale=None,
                        )
                    else:
                        weight = w
                        weight_scale = self_attn.kv_b_proj.weight_scale

                    w, scale = channel_quant_to_tensor_quant(weight, weight_scale)
                    self_attn.w_scale = scale

            if w.dtype == torch.int8:
                if hasattr(self.quant_config, "weight_block_size"):
                    weight_block_size = self.quant_config.weight_block_size
                    if weight_block_size is not None:
                        assert hasattr(self_attn.kv_b_proj, "weight_scale_inv")
                        weight = w
                        weight_scale = self_attn.kv_b_proj.weight_scale_inv
                        w = int8_block_dequant(
                            weight, weight_scale, weight_block_size
                        ).to(torch.bfloat16)
                else:
                    w = w.to(torch.bfloat16) * self_attn.kv_b_proj.weight_scale.to(
                        torch.bfloat16
                    )

            w_kc, w_vc = w.unflatten(
                0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
            ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
            if not use_deep_gemm_bmm:
                self_attn.w_kc = bind_or_assign(
                    self_attn.w_kc, w_kc.transpose(1, 2).contiguous().transpose(1, 2)
                )
                self_attn.w_vc = bind_or_assign(
                    self_attn.w_vc, w_vc.contiguous().transpose(1, 2)
                )
                if (
                    hasattr(self_attn.kv_b_proj, "weight_scale")
                    and self_attn.w_scale is None
                ):
                    self_attn.w_scale = bind_or_assign(
                        self_attn.w_scale, self_attn.kv_b_proj.weight_scale
                    )
                    if _is_hip:
                        self_attn.w_scale *= 2.0
                # TODO: remove this after adding FP8 support in bmm cpu kernel
                if _is_cpu and _is_cpu_amx_available and w.dtype == torch.float8_e4m3fn:
                    self_attn.w_kc = (
                        self_attn.w_kc.to(torch.bfloat16) * self_attn.w_scale
                    )
                    self_attn.w_vc = (
                        self_attn.w_vc.to(torch.bfloat16) * self_attn.w_scale
                    )
            else:
                num_tiles_k = self_attn.qk_nope_head_dim // weight_block_size[1]
                num_tiles_n = self_attn.v_head_dim // weight_block_size[0]
                ws_kc, ws_vc = block_scale.unflatten(
                    0, (-1, (num_tiles_k + num_tiles_n))
                ).split([num_tiles_k, num_tiles_n], dim=1)
                self_attn.w_scale_k = bind_or_assign(
                    self_attn.w_scale_k, ws_kc.transpose(1, 2).contiguous()
                )
                self_attn.w_scale_v = bind_or_assign(
                    self_attn.w_scale_v, ws_vc.contiguous()
                )
                self_attn.w_kc = bind_or_assign(
                    self_attn.w_kc, w_kc.transpose(1, 2).contiguous()
                )
                self_attn.w_vc = bind_or_assign(self_attn.w_vc, w_vc.contiguous())
                self_attn.use_deep_gemm_bmm = True

        # NOTE: no model-level ue8m0 requant here. Fp8LinearMethod /
        # Fp8MoEMethod .process_weights_after_loading requant every
        # block-fp8 weight for DeepGEMM (with the format_ue8m0 reentry
        # guard); a second model-level pass double-packs the scales and
        # crashes, which is why deepseek_v2 dropped its copy too.

    def get_decoder_attention_types(self):
        return self.model.decoder_attention_types

    def get_input_embeddings(self) -> nn.Module:
        return self.model.word_embeddings

    def set_dspark_layers_to_capture(self, layer_ids: List[int]) -> None:
        """Configure per-layer hidden-state capture for DSpark.

        The target model exposes the final ``hidden_states + residual`` of each
        requested layer so the DSpark draft can project them into its context KV
        cache. ``layer_ids`` are raw target decoder layer indices (e.g.
        ``[1, 11, 23, 29, 35]``); capture happens after layer ``i`` runs, so no
        ``+1`` offset is applied (in contrast to the DFlash convention used by
        models that capture before the layer runs).
        """
        if not self.pp_group.is_last_rank:
            return
        if layer_ids is None:
            raise ValueError(
                "DSPARK requires explicit layer_ids for aux hidden capture."
            )
        self.capture_aux_hidden_states = True
        self.model.capture_aux_hidden_states = True
        self.model.layers_to_capture = list(layer_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            input_embeds=input_embeds,
            forward_batch=forward_batch,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        if self.pp_group.is_last_rank:
            aux_hidden_states = None
            if self.capture_aux_hidden_states and isinstance(hidden_states, tuple):
                hidden_states, aux_hidden_states = hidden_states
            return self.logits_processor(
                input_ids,
                # keep hidden_states in bf16 so the lm_head matmul runs in bf16
                # (logits are cast to fp32 inside the logits processor)
                hidden_states,
                self.lm_head,
                forward_batch,
                aux_hidden_states=aux_hidden_states,
            )
        else:
            return hidden_states

    @staticmethod
    def weight_direct_load(param: torch.Tensor, loaded_weight: torch.Tensor):
        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight)

    def _resolve_shared_expert_weight_name(self, weight_name: str) -> str:
        layer = self.get_submodule(weight_name.split(".mlp.", 1)[0])
        if isinstance(layer, PPMissingLayer) or layer.mlp.num_fused_shared_experts == 0:
            return weight_name
        return weight_name.replace(
            "mlp.shared_experts", f"mlp.experts.{self.config.num_experts}"
        )

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        num_groups = getattr(config, "n_group", 0)
        from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation

        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_experts,
            num_groups=None if num_groups == 0 else num_groups,
        )

    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False
    ) -> Set[str]:
        def load_linear_attn_weight(
            name: str, loaded_weight: torch.Tensor, self
        ) -> None:
            if is_pp_missing_parameter(name, self):
                return
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", self.weight_direct_load)
            if "A_log" in name:
                # A_log param shape differs from Kimi's
                loaded_weight = loaded_weight[None, None, :, None]
            weight_loader(param, loaded_weight)
            return

        stacked_params_mapping = [
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
            (".fused_qkvbfg_proj", ".q_proj", 0),
            (".fused_qkvbfg_proj", ".k_proj", 1),
            (".fused_qkvbfg_proj", ".v_proj", 2),
            (".fused_qkvbfg_proj", ".b_proj", 3),
            (".fused_qkvbfg_proj", ".f_proj", 3),
            (".fused_qkvbfg_proj", ".g_proj", 4),
            (".fused_qkvbfg_a_proj", ".q_proj", 0),
            (".fused_qkvbfg_a_proj", ".k_proj", 1),
            (".fused_qkvbfg_a_proj", ".v_proj", 2),
            (".fused_qkvbfg_a_proj", ".b_proj", 3),
            (".fused_qkvbfg_a_proj", ".f_a_proj", 4),
            (".fused_qkvbfg_a_proj", ".g_a_proj", 5),
            (".fused_fg_b_proj", ".f_b_proj", 0),
            (".fused_fg_b_proj", ".g_b_proj", 1),
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".qkv_conv1d", ".q_conv1d", 0),
            (".qkv_conv1d", ".k_conv1d", 1),
            (".qkv_conv1d", ".v_conv1d", 2),
        ]
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts + self.num_fused_shared_experts,
        )

        if is_nextn:
            nextn_layer_id = resolve_nextn_layer_id(self.config)
            nextn_layer_prefix = f"model.layers.{nextn_layer_id}"

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        weight_names = []
        fuse_qkv_a_proj = hasattr(self.config, "q_lora_rank") and (
            self.config.q_lora_rank is not None
        )
        cached_a_proj = {} if fuse_qkv_a_proj else None

        if self.num_fused_shared_experts > 0:
            log_info_on_rank0(logger, "Shared experts fusion optimization enabled.")

        for name, loaded_weight in weights:
            if name.startswith("model.mtp"):
                continue
            layer_idx = None
            if "model.layers." in name:
                layer_idx = int(name.split(".")[2])
                if not is_nextn and layer_idx >= self.config.num_hidden_layers:
                    continue
            if (
                ("v_head" in name)
                or ("inv_freq" in name)
                or (self.config.tie_word_embeddings and "lm_head" in name)
            ):
                continue

            if is_nextn:
                if not name.startswith(nextn_layer_prefix):
                    continue
                rewritten = rewrite_nextn_weight_name(name, nextn_layer_prefix)
                if rewritten is None:
                    continue
                name = rewritten
                layer_idx = 0

            if self.num_fused_shared_experts > 0 and "mlp.shared_experts" in name:
                name = self._resolve_shared_expert_weight_name(name)

            weight_names.append(name)

            if "mlp.shared_experts" in name and name.endswith("weight_scale_inv"):
                projection = next(
                    (
                        projection
                        for projection in ("gate_proj", "up_proj", "down_proj")
                        if f".{projection}." in name
                    ),
                    None,
                )
                shared_layer = self.get_submodule(name.split(".mlp.", 1)[0])
                shared_mlp = getattr(
                    getattr(shared_layer, "mlp", None), "shared_experts", None
                )
                if (
                    projection is not None
                    and isinstance(shared_mlp, BailingMLP)
                    and shared_mlp.cache_serialized_fp8_scale_for_tp_requant(
                        loaded_weight, projection
                    )
                ):
                    loaded_params.add(name)
                    continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                if param_name in {
                    ".fused_qkvbfg_a_proj",
                    ".fused_fg_b_proj",
                    ".fused_qkvbfg_proj",
                }:
                    layer = (
                        self.model.decoder
                        if is_nextn
                        else self.model.layers[int(name.split(".")[2])]
                    )
                    if is_pp_missing_parameter(name, layer):
                        continue
                    layer_attn = layer.attention
                    if not getattr(layer_attn, "do_fuse_qkvbfg", False):
                        continue
                    if param_name == ".fused_qkvbfg_proj":
                        if not getattr(layer_attn, "no_kda_lora", False):
                            continue
                        if weight_name == ".b_proj" and not getattr(
                            layer_attn, "fuse_no_lora_beta", False
                        ):
                            continue
                        if weight_name in {".f_proj", ".g_proj"} and getattr(
                            layer_attn, "fuse_no_lora_beta", False
                        ):
                            shard_id += 1
                    elif getattr(layer_attn, "no_kda_lora", False):
                        continue

                new_name = name.replace(weight_name, param_name)
                if new_name not in params_dict:
                    continue

                param = params_dict[new_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)

                    if name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name.endswith("weight_scale_inv"):
                        moe_module = self.get_submodule(name.rsplit(".experts.", 1)[0])
                        if isinstance(
                            moe_module, BailingMoE
                        ) and moe_module.cache_serialized_fp8_scale_for_tp_requant(
                            loaded_weight,
                            shard_id,
                            expert_id,
                        ):
                            break
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if "slope" in name:
                        continue

                    if fuse_qkv_a_proj and (
                        "q_a_proj" in name or "kv_a_proj_with_mqa" in name
                    ):
                        cached_a_proj[name] = loaded_weight
                        q_a_proj_name = (
                            name
                            if "q_a_proj" in name
                            else name.replace("kv_a_proj_with_mqa", "q_a_proj")
                        )
                        kv_a_proj_name = (
                            name
                            if "kv_a_proj_with_mqa" in name
                            else name.replace("q_a_proj", "kv_a_proj_with_mqa")
                        )

                        if (
                            q_a_proj_name in cached_a_proj
                            and kv_a_proj_name in cached_a_proj
                        ):
                            q_a_proj_weight = cached_a_proj[q_a_proj_name]
                            kv_a_proj_weight = cached_a_proj[kv_a_proj_name]
                            cat_dim = 0
                            if self.quant_config is not None and (
                                self.quant_config.get_name() == "awq"
                                or self.quant_config.get_name() == "awq_marlin"
                                or self.quant_config.get_name() == "moe_wna16"
                            ):
                                cat_dim = 1
                            fused_weight = torch.cat(
                                [q_a_proj_weight, kv_a_proj_weight], dim=cat_dim
                            )
                            param_name = (
                                name.replace("q_a_proj", "fused_qkv_a_proj_with_mqa")
                                if "q_a_proj" in name
                                else name.replace(
                                    "kv_a_proj_with_mqa",
                                    "fused_qkv_a_proj_with_mqa",
                                )
                            )
                            if param_name not in params_dict:
                                continue
                            param = params_dict[param_name]
                            weight_loader = getattr(
                                param, "weight_loader", default_weight_loader
                            )

                            weight_loader(param, fused_weight)
                            cached_a_proj.pop(q_a_proj_name)
                            cached_a_proj.pop(kv_a_proj_name)
                    else:
                        if name not in params_dict:
                            name = name.replace(".dense.", ".o_proj.")
                            if name not in params_dict:
                                continue
                        if is_pp_missing_parameter(name, self):
                            continue
                        if (
                            "attention" in name
                            and "slope" not in name
                            and is_linear_layer(layer_idx, self.model.layer_group_size)
                        ):
                            load_linear_attn_weight(name, loaded_weight, self)
                            loaded_params.add(name)
                            continue

                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
            loaded_params.add(name)

        # DefaultModelLoader invokes each quant method immediately after this
        # method returns, so rebuild TP-local blocks before generic post-load
        # quantization can transform the scale layout.
        requantized_moe_layers = sum(
            module.requantize_serialized_fp8_tp_weights()
            for module in self.modules()
            if isinstance(module, BailingMoE)
        )
        requantized_shared_mlps = sum(
            module.requantize_serialized_fp8_tp_weights()
            for module in self.modules()
            if isinstance(module, BailingMLP)
        )
        if requantized_moe_layers or requantized_shared_mlps:
            log_info_on_rank0(
                logger,
                "Requantized serialized blockwise FP8 TP weights in "
                f"{requantized_moe_layers} Bailing MoE layer(s) and "
                f"{requantized_shared_mlps} standalone shared MLP(s).",
            )

        self.post_load_weights(is_nextn=is_nextn, weight_names=weight_names)

        return loaded_params

    def post_process_weights_if_quant(self):
        for name, module in self.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None:
                post_process = getattr(
                    quant_method, "process_weights_after_loading", None
                )
                if post_process is not None:
                    post_process(module)

    def get_embed_and_head(self):
        return self.model.word_embeddings.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.word_embeddings.weight
        del self.lm_head.weight
        self.model.word_embeddings.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()


EntryClass = [
    BailingMoeV3ForCausalLM,
]
