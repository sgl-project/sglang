# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 BharatGen AI team. All rights reserved.
#
# SGLang inference implementation for Param2MoE.
# Architecture: Hybrid MoE with GQA attention, sigmoid-scored grouped top-k routing,
# per-expert bias correction, shared (always-active) experts, and SwiGLU MLP.
#
"""SGLang Param2MoE model."""

import logging
from typing import Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    enable_moe_dense_fully_dp,
)
from sglang.srt.layers.dp_attention import (
    get_attention_dp_size,
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import (
    get_moe_a2a_backend,
    should_use_flashinfer_cutlass_moe_fp4_allgather,
)
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.utils import (
    apply_qk_norm,
    create_fused_set_kv_buffer_arg,
    enable_fused_set_kv_buffer,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, is_cuda, is_non_idle_and_non_empty, make_layers

LoraConfig = None
logger = logging.getLogger(__name__)
_is_cuda = is_cuda()


class Param2MoEMLP(nn.Module):
    """SwiGLU feed-forward block (SiLU + Mul gating).

    Used both for dense transformer layers and as the shared-expert sub-network
    inside Param2MoESparseMoeBlock.
    """

    def __init__(
        self,
        intermediate_size: int,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: Optional[bool] = True,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.tp_size = tp_size
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate_size] * 2,
            bias=config.use_bias,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=config.use_bias,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act!r}. "
                "Only 'silu' is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        # Early-exit for empty batches (avoids NCCL hangs in TP > 1 when
        # the rank receives zero tokens).
        if self.tp_size == 1 and hidden_states.shape[0] == 0:
            return hidden_states
        gate_up, _ = self.gate_up_proj(hidden_states)
        hidden_states = self.act_fn(gate_up)
        hidden_states, _ = self.down_proj(
            hidden_states,
            skip_all_reduce=should_allreduce_fusion or use_reduce_scatter,
        )
        return hidden_states


class Param2MoEGate(nn.Module):
    """Linear router for the Mixture-of-Experts block.

    Computes per-expert logits from the hidden state.  An optional per-expert
    bias (``expert_bias``) can be added to the logits for load balancing; it is
    stored in float32 regardless of the model precision so that bias corrections
    remain numerically stable.

    Matching vLLM: ``moe_router_enable_expert_bias`` defaults to ``True``
    because the Param2MoE architecture always ships with a bias tensor.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        params_dtype: Optional[torch.dtype] = None,
        prefix: str = "",
    ) -> None:
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
        # Default True aligns with the vLLM implementation — the checkpoint
        # ships with an expert_bias tensor for all released Param2MoE variants.
        if getattr(config, "moe_router_enable_expert_bias", True):
            self.expert_bias = nn.Parameter(
                torch.empty((config.num_experts,), dtype=torch.float32),
            )
        else:
            self.expert_bias = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Cast to weight dtype for the matmul, then cast back so downstream
        # ops keep the model's native precision.
        return F.linear(hidden_states.to(self.weight.dtype), self.weight, None).to(
            hidden_states.dtype
        )


class Param2MoESparseMoeBlock(nn.Module):
    """Mixture-of-Experts feed-forward block for Param2MoE.

    Routing details
    ---------------
    * Sigmoid scoring  (``config.score_function = "sigmoid"``)
    * Grouped top-k   (``n_group``, ``topk_group``)
    * Per-expert bias  (``gate.expert_bias`` → passed to TopK as
      ``correction_bias``)
    * ``routed_scaling_factor`` is applied *inside* the expert kernel
      (``self.experts``), **not** by TopK — avoid double-scaling.

    Two code paths
    --------------
    * Normal (default): ``forward_normal`` — optionally overlaps shared-expert
      compute on an alternate CUDA stream when CUDA graphs are being captured.
    * DeepEP all-to-all: ``forward_deepep`` — used when
      ``get_moe_a2a_backend().is_deepep()`` is True.
    """

    def __init__(
        self,
        layer_id: int,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.alt_stream = alt_stream
        self.tp_size = get_tensor_model_parallel_world_size()

        # ---- Basic MoE hyper-parameters ---------------------------------- #
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_size = config.hidden_size
        # num_shared_experts may be an int or None in the config.
        self.num_shared_experts: Optional[int] = config.num_shared_experts
        self.routed_scaling_factor: float = getattr(
            config, "routed_scaling_factor", 1.0
        )
        self.score_function: Optional[str] = getattr(
            config, "score_function", "sigmoid"
        )

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act!r}. "
                "Only 'silu' is supported for now."
            )

        # ---- Router dtype ------------------------------------------------ #
        router_dtype = getattr(config, "router_dtype", None)
        if router_dtype is None:
            self.router_dtype: Optional[torch.dtype] = None
        elif router_dtype == "fp32":
            self.router_dtype = torch.float32
        else:
            self.router_dtype = torch.bfloat16

        # ---- Grouped top-k ----------------------------------------------- #
        self.num_expert_group: Optional[int] = getattr(config, "n_group", 0)
        self.topk_group: Optional[int] = getattr(config, "topk_group", 0)
        if self.num_expert_group > 0 or self.topk_group > 0:
            assert (
                self.num_expert_group > 0
                and 0 < self.topk_group <= self.num_expert_group
            ), (
                f"n_group ({self.num_expert_group}) and topk_group "
                f"({self.topk_group}) must satisfy: "
                "n_group > 0 and 0 < topk_group <= n_group."
            )
            self.use_grouped_topk = True
        else:
            # Sentinel values expected by TopK when grouping is disabled
            self.num_expert_group = None
            self.topk_group = None
            self.use_grouped_topk = False

        # Total experts including any redundant EP replicas
        self.num_experts: int = (
            config.num_experts + get_global_server_args().ep_num_redundant_experts
        )

        # ---- Gate / router ----------------------------------------------- #
        self.gate = Param2MoEGate(
            config=config,
            params_dtype=self.router_dtype,
            prefix=add_prefix("gate", prefix),
        )
        # Extract the raw tensor so TopK can use it as a correction bias.
        self.correction_bias: Optional[torch.Tensor] = (
            self.gate.expert_bias.data if self.gate.expert_bias is not None else None
        )

        # Validate score_function <-> correction_bias consistency
        if self.score_function is not None:
            assert (
                self.score_function == "softmax" and self.correction_bias is None
            ) or (
                self.score_function == "sigmoid" and self.correction_bias is not None
            ), (
                "score_function and correction_bias must be one of: "
                "(softmax, None) or (sigmoid, not None). "
                f"Got score_function={self.score_function!r}, "
                f"correction_bias="
                f"{'set' if self.correction_bias is not None else 'None'}."
            )

        # ---- TopK router ------------------------------------------------- #
        # IMPORTANT: routed_scaling_factor must NOT be passed here — it is
        # applied inside self.experts (get_moe_impl_class).  Passing it to both
        # TopK and the expert kernel would double-scale the output.
        self.topk = TopK(
            top_k=self.top_k,
            renormalize=self.norm_topk_prob,
            use_grouped_topk=self.use_grouped_topk,
            num_expert_group=self.num_expert_group,
            topk_group=self.topk_group,
            correction_bias=self.correction_bias,
            routed_scaling_factor=None,
            apply_routed_scaling_factor_on_output=False,
            scoring_func=self.score_function,  # ← must match gate score fn
            quant_config=quant_config,
            layer_id=layer_id,
        )

        # ---- Routed experts ---------------------------------------------- #
        self.experts = get_moe_impl_class(quant_config)(
            num_experts=self.num_experts,
            top_k=self.top_k,
            layer_id=self.layer_id,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            routed_scaling_factor=self.routed_scaling_factor,
            prefix=add_prefix("experts", prefix),
        )

        # ---- Shared (always-active) experts ------------------------------ #
        if config.num_shared_experts is not None:
            # If the config carries `moe_shared_expert_intermediate_size` it
            # already encodes the TOTAL intermediate size across all shared
            # experts (i.e. moe_intermediate_size * num_shared_experts).
            # Do NOT multiply again — only compute the product when the
            # dedicated field is absent.
            if (
                hasattr(config, "moe_shared_expert_intermediate_size")
                and config.moe_shared_expert_intermediate_size is not None
            ):
                shared_intermediate_size = config.moe_shared_expert_intermediate_size
            else:
                shared_intermediate_size = (
                    config.moe_intermediate_size * self.num_shared_experts
                )
            self.shared_experts = Param2MoEMLP(
                intermediate_size=shared_intermediate_size,
                config=config,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_experts", prefix),
                # Shared experts run on a single rank when using DeepEP so that
                # they don't cross EP-group boundaries.
                **(
                    dict(tp_rank=0, tp_size=1)
                    if get_moe_a2a_backend().is_deepep()
                    else {}
                ),
            )
        else:
            self.shared_experts = None

        # NOTE: No standalone DeepEPDispatcher is created here.
        # The DeepEPMoE expert layer (self.experts) creates its own internal
        # dispatcher. A second dispatcher would waste GPU memory due to the
        # large communication buffers allocated by DeepEP and is never used
        # by forward_deepep (which delegates directly to self.experts).

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        if not get_moe_a2a_backend().is_deepep():
            return self.forward_normal(
                hidden_states, should_allreduce_fusion, use_reduce_scatter
            )
        return self.forward_deepep(hidden_states, forward_batch)

    def get_moe_weights(self) -> List[torch.Tensor]:
        """Return expert parameter tensors (used by EPLB)."""
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ["correction_bias"]
        ]

    def _forward_shared_experts(
        self, hidden_states: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Return shared-expert output, or None when there are no shared experts."""
        if self.num_shared_experts:
            return self.shared_experts(hidden_states)
        return None

    def _forward_router_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        return self.experts(hidden_states, topk_output)

    def forward_normal_dual_stream(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Overlap shared-expert compute on the alternate CUDA stream.

        Returns ``(router_output, shared_output)``; shared_output may be None.
        The hidden tensor is *cloned* before the shared-expert branch so that
        both branches see an independent copy of the input.
        """
        current_stream = torch.cuda.current_stream()
        self.alt_stream.wait_stream(current_stream)
        # Clone guarantees shared and routed branches read independent memory.
        shared_output = self._forward_shared_experts(hidden_states.clone())

        with torch.cuda.stream(self.alt_stream):
            router_output = self._forward_router_experts(hidden_states)
        current_stream.wait_stream(self.alt_stream)

        return router_output, shared_output

    def forward_normal(
        self,
        hidden_states: torch.Tensor,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)

        if (
            self.alt_stream is not None
            and hidden_states.shape[0] > 0
            and get_is_capture_mode()
        ):
            final_hidden_states, shared_output = self.forward_normal_dual_stream(
                hidden_states
            )
        else:
            shared_output = self._forward_shared_experts(hidden_states)
            final_hidden_states = self._forward_router_experts(hidden_states)

        if self.num_shared_experts:
            final_hidden_states = final_hidden_states + shared_output

        if (
            self.tp_size > 1
            and not should_allreduce_fusion
            and not use_reduce_scatter
            and not should_use_flashinfer_cutlass_moe_fp4_allgather()
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_size)

    def forward_deepep(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        shared_output = None
        forward_mode = forward_batch.forward_mode
        if is_non_idle_and_non_empty(forward_mode, hidden_states):
            router_logits = self.gate(hidden_states)
            if self.num_shared_experts:
                shared_output = self.shared_experts(hidden_states)

            topk_output = self.topk(
                hidden_states,
                router_logits,
                num_token_non_padded=forward_batch.num_token_non_padded,
                expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                    layer_id=self.layer_id,
                ),
            )
        else:
            topk_output = self.topk.empty_topk_output(hidden_states.device)

        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )

        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output
        return final_hidden_states


class Param2MoEAttention(nn.Module):
    """Grouped-Query Attention (GQA) for Param2MoE.

    Weight names follow the HuggingFace checkpoint layout:
      - ``query_key_value``  (fused QKV)
      - ``dense``            (output projection)
      - ``query_layernorm`` / ``key_layernorm``  (optional, when use_qk_norm=True)
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_kv_heads = config.num_key_value_heads
        self.dp_size = get_attention_dp_size()
        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        assert self.total_num_heads % attn_tp_size == 0
        if self.total_kv_heads >= attn_tp_size:
            assert self.total_kv_heads % attn_tp_size == 0
        else:
            assert attn_tp_size % self.total_kv_heads == 0
        assert self.total_num_heads >= self.total_kv_heads

        self.num_heads = self.total_num_heads // attn_tp_size
        self.head_dim = config.head_dim or (self.hidden_size // self.total_num_heads)
        self.q_size = self.head_dim * self.num_heads

        self.num_kv_heads = max(1, self.total_kv_heads // attn_tp_size)
        self.kv_size = max(1, self.num_kv_heads * self.head_dim)

        self.scale = self.head_dim**-0.5
        self.use_qk_norm: bool = getattr(config, "use_qk_norm", False)

        # Fused QKV projection (HF name: query_key_value)
        self.query_key_value = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_kv_heads,
            bias=(config.use_bias or config.use_qkv_bias),
            quant_config=quant_config,
            prefix=add_prefix("query_key_value", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )

        if self.use_qk_norm:
            self.query_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.key_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Output projection (HF name: dense)
        self.dense = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=config.use_bias,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("dense", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )

        # Rotary embedding — respect partial_rotary_factor / rotary_dim if set.
        if hasattr(config, "partial_rotary_factor"):
            self.rotary_dim = int(self.head_dim * config.partial_rotary_factor)
        elif hasattr(config, "rotary_dim"):
            self.rotary_dim = config.rotary_dim
        else:
            self.rotary_dim = self.head_dim

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.rotary_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scale,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
        )

        self.alt_stream = alt_stream

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states

        qkv, _ = self.query_key_value(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        if self.use_qk_norm:
            q, k = apply_qk_norm(
                q=q,
                k=k,
                q_norm=self.query_layernorm,
                k_norm=self.key_layernorm,
                head_dim=self.head_dim,
                alt_stream=self.alt_stream,
            )

        q, k = self.rotary_emb(
            positions,
            q,
            k,
            fused_set_kv_buffer_arg=(
                create_fused_set_kv_buffer_arg(
                    value=v,
                    layer=self.attn,
                    forward_batch=forward_batch,
                )
                if enable_fused_set_kv_buffer(forward_batch)
                else None
            ),
        )

        context_layer = self.attn(
            q,
            k,
            v,
            forward_batch,
            save_kv_cache=not enable_fused_set_kv_buffer(forward_batch),
        )
        attn_output, _ = self.dense(context_layer)
        return attn_output


class Param2MoEBlock(nn.Module):
    """Single Param2MoE transformer decoder layer.

    The first ``config.first_k_dense_replace`` layers use a dense MLP;
    all subsequent layers use a Mixture-of-Experts MLP.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size

        self.input_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.dp_size = get_attention_dp_size()
        self.attention = Param2MoEAttention(
            config,
            layer_id,
            quant_config,
            reduce_results=False,
            prefix=add_prefix("attention", prefix),
            alt_stream=alt_stream,
        )
        self.layer_id = layer_id
        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()

        self.is_layer_sparse = self._is_layer_sparse(config, layer_id=layer_id)
        is_previous_layer_sparse = self._is_layer_sparse(config, layer_id=layer_id - 1)
        is_next_layer_sparse = self._is_layer_sparse(config, layer_id=layer_id + 1)

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )

        self.is_last_layer = self.layer_id == config.num_hidden_layers - 1

        if self.is_layer_sparse:
            self.mlp = Param2MoESparseMoeBlock(
                layer_id=layer_id,
                config=config,
                quant_config=quant_config,
                alt_stream=alt_stream,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            if enable_moe_dense_fully_dp():
                mlp_tp_rank, mlp_tp_size = 0, 1
            else:
                mlp_tp_rank, mlp_tp_size = None, None
            self.mlp = Param2MoEMLP(
                intermediate_size=config.intermediate_size,
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                tp_rank=mlp_tp_rank,
                tp_size=mlp_tp_size,
            )

        self.post_attention_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
            is_last_layer=(self.layer_id == self.config.num_hidden_layers - 1),
        )

    # layer_id=-1 (prev of layer 0) and layer_id=num_hidden_layers (next of
    # last) will both return False because neither satisfies >= first_k_dense_replace
    # for sensible config values — no special casing needed.
    def _is_layer_sparse(self, config: PretrainedConfig, layer_id: int) -> bool:
        return (
            config.num_experts is not None and layer_id >= config.first_k_dense_replace
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        captured_last_layer_outputs: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden_states, residual = (
            self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
                hidden_states,
                residual,
                forward_batch,
                captured_last_layer_outputs=captured_last_layer_outputs,
            )
        )

        if hidden_states.shape[0] != 0:
            hidden_states = self.attention(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=forward_batch,
        )

        should_allreduce_fusion = (
            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
                forward_batch
            )
        )
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )

        hidden_states = self.mlp(
            hidden_states, forward_batch, should_allreduce_fusion, use_reduce_scatter
        )

        if should_allreduce_fusion:
            hidden_states._sglang_needs_allreduce_fusion = True
        else:
            hidden_states, residual = self.layer_communicator.postprocess_layer(
                hidden_states, residual, forward_batch
            )

        return hidden_states, residual


class Param2MoEModel(nn.Module):
    """Stack of Param2MoEBlock layers with embedding + final norm."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_dim = config.hidden_size

        if self.pp_group.is_first_rank:
            self.word_embeddings = VocabParallelEmbedding(
                self.vocab_size,
                self.embed_dim,
                quant_config=quant_config,
                prefix=add_prefix("word_embeddings", prefix),
                enable_tp=not is_dp_attention_enabled(),
            )
        else:
            self.word_embeddings = PPMissingLayer()

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: Param2MoEBlock(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )

        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(self.embed_dim, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.pp_group.is_first_rank:
            hidden_states = (
                input_embeds
                if input_embeds is not None
                else self.word_embeddings(input_ids)
            )
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        aux_hidden_states: List[torch.Tensor] = []
        for i in range(self.start_layer, self.end_layer):
            with get_global_expert_distribution_recorder().with_current_layer(i):
                layer = self.layers[i]
                hidden_states, residual = layer(
                    positions,
                    hidden_states,
                    forward_batch,
                    residual,
                    captured_last_layer_outputs=(
                        aux_hidden_states
                        if getattr(layer, "_is_layer_to_capture", False)
                        else None
                    ),
                )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )

        if not forward_batch.forward_mode.is_idle():
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Param2MoEForCausalLM(nn.Module):
    """Param2MoE causal language model for SGLang inference."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        alt_stream = torch.cuda.Stream() if _is_cuda else None

        self.model = Param2MoEModel(
            config,
            quant_config,
            alt_stream=alt_stream,
            prefix=add_prefix("model", ""),
        )

        if config.tie_word_embeddings:
            self.lm_head = self.model.word_embeddings
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
                use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
            )
        self.logits_processor = LogitsProcessor(config)

    @property
    def start_layer(self) -> int:
        return self.model.start_layer

    @property
    def end_layer(self) -> int:
        return self.model.end_layer

    def get_embed_and_head(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (embedding weight, lm_head weight). Used by eagle_worker."""
        return self.model.word_embeddings.weight, self.lm_head.weight

    def set_embed_and_head(self, embed: torch.Tensor, head: torch.Tensor) -> None:
        """Replace embedding and head weights in-place. Used by eagle_worker."""
        del self.model.word_embeddings.weight
        del self.lm_head.weight
        self.model.word_embeddings.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch
            )
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        # (param_name_suffix, ckpt_weight_name_suffix, shard_id)
        # Leading dot prevents substring collisions, e.g. "output_gate_proj".
        stacked_params_mapping = [
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            if (
                "v_head" in name
                or "inv_freq" in name
                or (self.config.tie_word_embeddings and "lm_head" in name)
            ):
                continue

            if (
                hasattr(self.config, "norm_head")
                and self.config.norm_head
                and "lm_head.weight" in name
            ):
                loaded_weight = F.normalize(loaded_weight, dim=0, p=2, eps=1e-7)

            if name.endswith(".mlp.gate.expert_bias"):
                loaded_weight = loaded_weight - loaded_weight.mean()

            matched_stacked = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # Routed-expert weights are handled in step 5 below.
                if "mlp.experts" in name:
                    continue
                new_name = name.replace(weight_name, param_name)
                if new_name.endswith(".bias") and new_name not in params_dict:
                    continue
                if new_name not in params_dict:
                    continue

                param = params_dict[new_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                matched_stacked = True
                break

            if matched_stacked:
                continue

            matched_expert = False
            for param_name, weight_name, expert_id, shard_id in expert_params_mapping:
                if weight_name not in name:
                    continue
                new_name = name.replace(weight_name, param_name)
                if new_name not in params_dict:
                    continue

                param = params_dict[new_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(
                    param,
                    loaded_weight,
                    name,  # original ckpt name — used by FusedMoE to
                    shard_id=shard_id,  # identify gate/up/down shard
                    expert_id=expert_id,
                )
                matched_expert = True
                break

            if matched_expert:
                continue

            if name.endswith(".bias") and name not in params_dict:
                continue
            if name not in params_dict:
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)

        # Build per-layer MoE weight cache consumed by the EPLB rebalancer.
        self.routed_experts_weights_of_layer = {
            layer_id: layer.mlp.get_moe_weights()
            for layer_id, layer in enumerate(self.model.layers)
            if not isinstance(layer, PPMissingLayer)
            and isinstance(layer.mlp, Param2MoESparseMoeBlock)
        }

    @classmethod
    def get_model_config_for_expert_location(
        cls, config: PretrainedConfig
    ) -> ModelConfigForExpertLocation:
        num_groups = getattr(config, "n_group", 0)
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_experts,
            num_groups=None if num_groups == 0 else num_groups,
        )


EntryClass = [Param2MoEForCausalLM]
