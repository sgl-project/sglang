# Copyright 2026 Baidu ERNIE Team and The SGLang Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""ERNIE 5.5 Flash (``Ernie5_5_MoeForCausalLM``).

The checkpoint's ``compress_ratios`` (``config.checkpoint_compress_ratios``)
picks each layer's attention: ``128`` is an HCA layer (:class:`MQALayer`), ``-2``
a full-context MLA layer (:class:`DeepseekV2AttentionMLA`, dims under
``hybrid_mla_*``) with a DSA indexer. Both add a sink, a sigmoid output gate and a
VHA postmix. Layers are joined by mHC; the FFN is a SiTU dense MLP for the first
``first_k_dense_replace`` layers and a latent multi-router MoE afterwards.

:mod:`sglang.srt.configs.ernie55` glosses the checkpoint's field names (HCA,
MLA, DSA, VHA, mHC, SiTU, ``wo_a``/``wo_b``).
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple, Union

import torch
from torch import nn

from sglang.kernels.ops.activation.multimax import multimax
from sglang.kernels.ops.attention.dsv4 import fused_q_norm_rope, fused_rope_inplace
from sglang.kernels.ops.attention.postmix_fp8_wo_a import postmix_fp8_wo_a_quant
from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul
from sglang.kernels.ops.gemm.bf16_fp32 import linear_bf16_fp32
from sglang.kernels.ops.layernorm.mhc import mhc_post, mhc_pre
from sglang.kernels.ops.layernorm.mhc_head import fused_hc_head
from sglang.kernels.ops.moe.multi_router import multi_router_topk
from sglang.srt.configs.ernie55 import Ernie5_5_MoeConfig
from sglang.srt.distributed import tensor_model_parallel_all_reduce
from sglang.srt.environ import envs
from sglang.srt.layers.activation import SituAndMul
from sglang.srt.layers.cp.utils import is_cp_active
from sglang.srt.layers.deep_gemm_wrapper.configurer import DEEPGEMM_SCALE_UE8M0
from sglang.srt.layers.dp_attention import (
    attn_tp_all_gather_into_tensor,
    attn_tp_all_reduce,
    dp_gather_replicate,
    dp_scatter,
    get_global_dp_buffer,
    get_is_extend_in_batch,
    get_local_dp_buffer,
    get_moe_cp_size,
    is_enable_moe_cp_allgather,
)
from sglang.srt.layers.layer_boundary import AttentionInputs, get_attn_tp_context
from sglang.srt.layers.layer_boundary.layout import moe_cp_gathered_rows
from sglang.srt.layers.layer_boundary.ops import moe_cp_gather, moe_cp_take_back
from sglang.srt.layers.layernorm import RMSNorm, exact_rms_norm
from sglang.srt.layers.layernorm_sp import runs_sp
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import (
    LogitsProcessor,
    extend_reads_input_logprobs,
)
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.utils.common import add_prefix, set_weight_attrs
from sglang.srt.layers.utils.layer_list import make_pp_layers
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
    get_embedding_tp_kwargs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    DeepseekV2WeightLoaderMixin,
    _clone_if_runai_streamed_tensor,
    _load_fused_indexer_wk,
)
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
from sglang.srt.models.deepseek_v4 import (
    MQALayer,
    _fp8_wo_a_einsum,
    make_hc_head_params,
    make_hc_mixing_params,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import BumpAllocator, is_cuda


def check_supported_deployment(config: Ernie5_5_MoeConfig) -> None:
    if not is_cuda():
        # The HIP / NPU prepare paths call wq_b directly and skip the gate split.
        raise ValueError("ERNIE 5.5 Flash is supported on CUDA only")
    attn_tp = get_parallel().attn_tp_size
    # HCA heads split by output group and both halves of the merged [q_up | gate]
    # GEMM carry fp8 block scales, so each rank's share must be whole 128-row blocks.
    heads = (
        ("num_attention_heads", config.num_attention_heads),
        ("hybrid_mla_num_attention_heads", config.hybrid_mla_num_attention_heads),
        ("o_groups", config.o_groups),
        (
            "num_attention_heads * head_dim / 128",
            config.num_attention_heads * config.head_dim // 128,
        ),
        (
            "o_groups * o_lora_rank / 128",
            config.o_groups * config.o_lora_rank // 128,
        ),
    )
    for name, value in heads:
        if value % attn_tp:
            raise ValueError(
                f"ERNIE 5.5 Flash splits attention by output group, so attention "
                f"TP ({attn_tp}) must divide {name} ({value})"
            )


def _heads_are_split() -> bool:
    return get_parallel().attn_tp_size > 1


def logits_rows(forward_batch: ForwardBatch) -> Optional[torch.Tensor]:
    """Each extend segment's last token (the LM head's rows), or None."""
    if (
        forward_batch.forward_mode != ForwardMode.EXTEND
        or forward_batch._original_forward_mode == ForwardMode.IDLE
        or forward_batch.is_prefill_only
        or runs_sp(forward_batch.forward_mode)
        or _heads_are_split()
        or (forward_batch.attn_cp_metadata is not None and is_cp_active(forward_batch))
        or torch.cuda.is_current_stream_capturing()
        or extend_reads_input_logprobs(forward_batch)
    ):
        return None
    return torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1


def scatter_rows(x: torch.Tensor, rows: torch.Tensor, num_rows: int) -> torch.Tensor:
    out = x.new_zeros((num_rows, *x.shape[1:]))
    out[rows] = x
    return out


def _situ_betas(config: Ernie5_5_MoeConfig) -> Tuple[float, float]:
    # hidden_act == "situ" is validated once, by the config class.
    return float(config.activation_situ_beta), float(config.activation_situ_linear_beta)


class VhaPostMix(nn.Module):
    """Cross-head low-rank mix ``o <- (I + V @ U.T) @ o`` over the head axis."""

    def __init__(self, num_heads: int, total_heads: int):
        super().__init__()
        # Rank num_heads / 4 of the whole head axis, as every checkpoint ships it.
        rank = total_heads // 4
        self.num_heads = num_heads
        self.sharded = num_heads != total_heads
        self.vha_postmix_U = nn.Parameter(torch.zeros(num_heads, rank))
        self.vha_postmix_V = nn.Parameter(torch.zeros(num_heads, rank))
        if self.sharded:
            for param in (self.vha_postmix_U, self.vha_postmix_V):
                set_weight_attrs(
                    param,
                    {
                        "weight_loader": sharded_weight_loader(
                            0, parallel_group="attn_tp"
                        )
                    },
                )
        else:
            self.register_buffer("mix", torch.eye(num_heads), persistent=False)

    def fold(self) -> None:
        if self.sharded:
            # Nothing to fold: no rank holds the whole [H, H] product.
            return
        u, v = self.vha_postmix_U.data, self.vha_postmix_V.data
        eye = torch.eye(self.num_heads, dtype=u.dtype, device=u.device)
        # In place: a captured CUDA graph reads this buffer's address.
        self.mix.copy_(torch.addmm(eye, v, u.t()))

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        # o: [T, ...] with trailing dims flattening to num_heads * d.
        heads = o.reshape(o.shape[0], self.num_heads, -1)
        if not self.sharded:
            return torch.matmul(self.mix, heads).reshape(o.shape)
        # fp32 throughout: a bf16 latent or residual add loses accuracy vs the fold.
        heads32 = heads.float()
        latent = torch.matmul(self.vha_postmix_U.t().float(), heads32)
        latent = attn_tp_all_reduce(latent)
        mixed = heads32 + torch.matmul(self.vha_postmix_V.float(), latent)
        return mixed.to(heads.dtype).reshape(o.shape)


def _load_transposed(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
    param.data.copy_(loaded_weight.t())


class Ernie5_5HcHead(nn.Module):
    """Collapses the mHC streams (the checkpoint's ``mhc_contract``)."""

    def __init__(self, config: Ernie5_5_MoeConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.norm_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        self.fn, self.base, self.scale = make_hc_head_params(
            config.hc_mult, config.hidden_size
        )
        # Stored [hc_mult * hidden, hc_mult]; the parameter is the transpose.
        set_weight_attrs(self.fn, {"weight_loader": _load_transposed})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[T, hc_mult, hidden]`` -> ``[T, hidden]``."""
        if x.shape[0] == 0:
            return x.new_empty((0, self.hidden_size))
        return fused_hc_head(
            x,
            self.fn,
            self.scale,
            self.base,
            norm_eps=self.norm_eps,
            hc_eps=self.hc_eps,
            eps_outside_sqrt=True,
        )


class Ernie5_5GatedQMixin:
    """[q_up | gate] GEMM over the q-LoRA latent; the gate waits in ``self._gate``
    until wo_b reads it."""

    def _init_gated_wq_b(
        self, quant_config: Optional[QuantizationConfig], prefix: str
    ) -> None:
        # Merged so q_up shards by head and gate by output group, as wo_a / wo_b.
        self.q_up_local_size = self.n_local_heads * self.head_dim
        self.wq_b = MergedColumnParallelLinear(
            self.q_lora_rank,
            [self.n_heads * self.head_dim, self.n_groups * self.o_lora_rank],
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_b", prefix),
            parallel_group="attn_tp",
        )
        self._gate: Optional[torch.Tensor] = None

    def _gated_q_norm_rope(self, q_lora, positions, q_out=None):
        qg, _ = self.wq_b(q_lora)
        if self._gate is not None:
            raise RuntimeError("previous output gate never reached wo_b")
        # Strided views; the norm+rope kernel and the gate-mul read the row stride.
        self._gate = qg[:, self.q_up_local_size :]
        q = qg[:, : self.q_up_local_size].unflatten(
            1, (self.n_local_heads, self.head_dim)
        )
        if q_out is None:
            q_out = q.new_empty(q.shape)
        fused_q_norm_rope(q, q_out, self.eps, self.freqs_cis, positions)
        return q_out

    def _take_gate(self) -> torch.Tensor:
        gate, self._gate = self._gate, None
        if gate is None:
            raise RuntimeError("output gate was not produced by the q projection")
        return gate


class Ernie5_5HcaAttention(Ernie5_5GatedQMixin, MQALayer):
    def __init__(
        self,
        config: Ernie5_5_MoeConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
        alt_streams: Optional[list] = None,
    ):
        super().__init__(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=prefix,
            alt_streams=alt_streams,
        )
        self.postmix = VhaPostMix(self.n_local_heads, self.n_heads)
        self._init_gated_wq_b(quant_config, prefix)
        # The mix fuses into the UE8M0 fp8 wo_a quant, but only from the folded
        # [H, H] matrix, which a split head axis has no rank-local form of.
        self._postmix_in_quant = (
            envs.SGLANG_OPT_ERNIE5_5_FUSE_INEXACT.get()
            and self.wo_a_fp8
            and DEEPGEMM_SCALE_UE8M0
            and not self.postmix.sharded
        )

    def _compute_q_b(self, q, positions, q_out=None):
        return self._gated_q_norm_rope(q, positions, q_out)

    def _project_wo_a(self, o, *, positions, **kwargs):
        if not self._postmix_in_quant:
            # The mix is over heads and the inverse RoPE rotates every head the
            # same way, so mixing before the parent's inverse RoPE is equivalent.
            return super()._project_wo_a(self.postmix(o), positions=positions, **kwargs)
        fused_rope_inplace(
            o[..., -self.qk_rope_head_dim :],
            None,
            self.freqs_cis,
            positions=positions,
            inverse=True,
        )
        x = o.view(o.shape[0], self.n_local_groups, -1)
        x_fp8, x_scale = postmix_fp8_wo_a_quant(x, self.postmix.mix)
        _, groups, hidden = x.shape
        return _fp8_wo_a_einsum(
            x_fp8,
            x_scale,
            self.wo_a.weight.view(groups, self.o_lora_rank, hidden),
            self.wo_a.weight_scale_inv.data,
            recipe=(1, 1, 128),
        )

    def _project_wo_b(self, o, defer_all_reduce: bool = False):
        o = fused_sigmoid_mul(o.flatten(1), self._take_gate(), round_sigmoid=True)
        return super()._project_wo_b(o, defer_all_reduce)


class Ernie5_5MlaAttention(DeepseekV2AttentionMLA):
    """MLA with DSA sparse attention; sink and gate go through the MLA hooks."""

    def __init__(
        self,
        config: Ernie5_5_MoeConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ):
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.hybrid_mla_num_attention_heads,
            qk_nope_head_dim=config.hybrid_mla_qk_nope_head_dim,
            qk_rope_head_dim=config.hybrid_mla_qk_rope_head_dim,
            v_head_dim=config.hybrid_mla_v_head_dim,
            q_lora_rank=config.hybrid_mla_q_lora_rank,
            kv_lora_rank=config.hybrid_mla_kv_lora_rank,
            rope_theta=config.rope_parameters["rope_theta"],
            # The MLA layers use plain RoPE; YaRN belongs to the HCA layers.
            rope_scaling=None,
            max_position_embeddings=config.max_position_embeddings,
            quant_config=quant_config,
            # Reduced in forward() instead: with attention DP, o_proj's own reduce
            # would span the whole TP group rather than the attention-TP group.
            reduce_results=False,
            layer_id=layer_id,
            prefix=prefix,
        )
        if not self.use_dsa or self.skip_topk:
            raise ValueError(
                "ERNIE 5.5 Flash's MLA layers need the DSA indexer with top-k "
                "selection enabled (check index_topk in the checkpoint config)"
            )
        self.learnable_sink_param = nn.Parameter(
            torch.zeros(self.num_local_heads, dtype=torch.float32)
        )
        if _heads_are_split():
            set_weight_attrs(
                self.learnable_sink_param,
                {"weight_loader": sharded_weight_loader(0, parallel_group="attn_tp")},
            )
        self.gate_proj = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * self.v_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_proj", prefix),
            parallel_group="attn_tp",
        )
        self.postmix = VhaPostMix(self.num_local_heads, self.num_heads)
        self._gate: Optional[torch.Tensor] = None

    def q_b_proj_forward(self, q_lora: torch.Tensor) -> torch.Tensor:
        # The only hook that receives the normalized q-LoRA latent the gate reads.
        if self._gate is not None:
            raise RuntimeError(
                "previous output gate never reached the attention output"
            )
        self._gate, _ = self.gate_proj(q_lora)
        return super().q_b_proj_forward(q_lora)

    def prepare_attention_output_gate(self, hidden_states):
        # Placeholder for the inner-state slot; forward_absorb_prepare swaps in
        # the gate projected in q_b_proj_forward.
        return hidden_states

    def forward_absorb_prepare(self, *args, **kwargs):
        inner_state = super().forward_absorb_prepare(*args, **kwargs)
        gate, self._gate = self._gate, None
        if gate is None:
            raise RuntimeError(
                "the q-LoRA latent for the output gate was not projected"
            )
        return (*inner_state[:-1], gate)

    def apply_attention_output_gate(self, attn_out, gate):
        return fused_sigmoid_mul(self.postmix(attn_out), gate, round_sigmoid=True)

    def forward(self, positions, hidden_states, forward_batch, zero_allocator):
        get_attn_tp_context().set_attn_inputs(
            AttentionInputs(hidden_states, forward_batch, self.prepare_qkv_latent)
        )
        try:
            with get_attn_backend().mla_layer_context():
                out = super().forward(
                    positions, hidden_states, forward_batch, zero_allocator
                )
        finally:
            get_attn_tp_context().clear_attn_inputs()
        # The DSA layer also returns top-k indices; no layer reuses them here.
        out = out[0] if isinstance(out, tuple) else out
        # The decoder layer has no boundary reduce to defer the head sum to.
        return attn_tp_all_reduce(out) if _heads_are_split() else out


class Ernie5_5MLP(nn.Module):
    def __init__(
        self,
        config: Ernie5_5_MoeConfig,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
        parallel_group: str,
        reduce_results: bool,
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            parallel_group=parallel_group,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("down_proj", prefix),
            parallel_group=parallel_group,
        )
        beta, linear_beta = _situ_betas(config)
        self.act_fn = SituAndMul(beta=beta, linear_beta=linear_beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x, _ = self.down_proj(self.act_fn(gate_up))
        return x


class Ernie5_5Moe(nn.Module):
    """Experts run in a ``moe_latent_size`` space; the shared expert does not."""

    def __init__(
        self,
        config: Ernie5_5_MoeConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        self.alt_stream = alt_stream
        self.layer_id = layer_id
        self.num_experts = config.n_routed_experts
        latent_size = config.moe_latent_size
        self.latent_size = latent_size
        # gate and gate_1 fused into one bf16 GEMM with fp32 logits; an fp32
        # checkpoint rounds on load.
        self.gate_weight = nn.Parameter(
            torch.empty(
                2 * self.num_experts,
                config.hidden_size,
                dtype=torch.bfloat16,
            )
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.empty(self.num_experts, dtype=torch.float32)
        )
        self.per_expert_scale = nn.Parameter(
            torch.ones(self.num_experts, dtype=torch.float32)
        )
        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            layer_id=layer_id,
            renormalize=config.norm_topk_prob,
            custom_routing_function=self._route,
            output_format=TopKOutputFormat.STANDARD,
        )
        beta, linear_beta = _situ_betas(config)
        self.experts = get_moe_impl_class(quant_config)(
            num_experts=self.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=latent_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
            activation="situ",
            gemm1_alpha=beta,
            gemm1_clamp_limit=linear_beta,
            reduce_results=False,
        )
        self.shared_experts = Ernie5_5MLP(
            config,
            config.moe_intermediate_size * config.n_shared_experts,
            quant_config,
            add_prefix("shared_experts", prefix),
            parallel_group="replicated",
            reduce_results=False,
        )
        self.fc1_latent_proj = ReplicatedLinear(
            config.hidden_size,
            latent_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("fc1_latent_proj", prefix),
        )
        self.fc2_latent_proj = ReplicatedLinear(
            latent_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("fc2_latent_proj", prefix),
        )
        self.latent_norm = RMSNorm(latent_size, eps=config.rms_norm_eps)

    def _route(self, hidden_states, gating_output, topk, renormalize):
        # gating_output is the raw [T, 2E] gate|gate_1 GEMM output.
        return multi_router_topk(
            gating_output,
            self.e_score_correction_bias,
            self.per_expert_scale,
            topk,
            renormalize,
            num_routers=2,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            if get_moe_a2a_backend().is_none():
                # Every TP rank sees the same (DP-gathered) tokens: all skip together.
                return hidden_states
            # An all-to-all dispatch still needs this rank, tokens or not.
            self.experts(
                hidden_states.new_empty((0, self.latent_size)),
                self.topk.empty_topk_output(
                    hidden_states.device, layer_id=self.layer_id
                ),
            )
            return hidden_states
        shared = None
        in_dispatch = None
        # A breakable graph's eager break inside the routed experts would leave one
        # fork/join spanning two captures, so overlap only in a whole-graph capture.
        if (
            self.alt_stream is not None
            and get_is_capture_mode()
            and not is_in_breakable_cuda_graph()
        ):
            self.alt_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.alt_stream):
                shared = self.shared_experts(hidden_states)
        elif (
            not get_is_capture_mode()
            and not is_in_breakable_cuda_graph()
            and get_moe_a2a_backend().is_deepep()
            and get_is_extend_in_batch()
        ):
            in_dispatch = self._shared_experts_in_dispatch(hidden_states)
        topk_output = self.topk(
            hidden_states, linear_bf16_fp32(hidden_states, self.gate_weight)
        )
        latent, _ = self.fc1_latent_proj(hidden_states)
        routed = self.experts(latent, topk_output)
        # An all-to-all combine already returns the complete sum, so all-reducing
        # it again would multiply by tp_size.
        if get_parallel().tp_size > 1 and get_moe_a2a_backend().is_none():
            routed = tensor_model_parallel_all_reduce(routed)
        routed, _ = self.fc2_latent_proj(self.latent_norm(routed))
        if in_dispatch:
            return routed + in_dispatch[0]
        if shared is None:
            shared = self.shared_experts(hidden_states)
        else:
            torch.cuda.current_stream().wait_stream(self.alt_stream)
            # Allocated on alt_stream, consumed on the main one.
            shared.record_stream(torch.cuda.current_stream())
        return routed + shared

    def _shared_experts_in_dispatch(self, hidden_states: torch.Tensor) -> list:
        """Run the shared expert in the eager DeepEP dispatch hook (after quantize,
        before the all-to-all); the returned list holds the output once it ran."""
        result = []

        def hook(dispatcher):
            for handle in handles:
                handle.remove()
            result.append(self.shared_experts(hidden_states))

        handles = self.experts.dispatcher.register_deepep_dispatch_hook(hook)
        return result


class Ernie5_5DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Ernie5_5_MoeConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
        alt_streams: Optional[list] = None,
    ):
        super().__init__()
        self.config = config
        self.self_attn = self._build_self_attn(
            config, layer_id, quant_config, add_prefix("self_attn", prefix), alt_streams
        )
        # Attention DP with whole heads per rank: replicate the dense MLP, with no
        # DP gather, all-reduce or scatter.
        self._dense_replicated = False
        if layer_id >= config.first_k_dense_replace:
            self.mlp = Ernie5_5Moe(
                config,
                layer_id,
                quant_config,
                add_prefix("mlp", prefix),
                # Attention joins its side streams before the FFN starts.
                alt_stream=alt_streams[0] if alt_streams else None,
            )
        else:
            self._dense_replicated = (
                get_parallel().attn_dp_size > 1 and not _heads_are_split()
            )
            self.mlp = Ernie5_5MLP(
                config,
                config.intermediate_size,
                quant_config,
                add_prefix("mlp", prefix),
                parallel_group="replicated" if self._dense_replicated else "tp",
                reduce_results=not self._dense_replicated,
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # Fuse the input RMSNorm into mhc_pre; not bit-exact with the eager norm.
        self.fuse_input_norm = envs.SGLANG_OPT_ERNIE5_5_FUSE_INEXACT.get()
        self.hc_mult = config.hc_mult
        (
            self.hc_attn_fn,
            self.hc_ffn_fn,
            self.hc_attn_base,
            self.hc_ffn_base,
            self.hc_attn_scale,
            self.hc_ffn_scale,
        ) = make_hc_mixing_params(self.hc_mult, config.hidden_size)

    @staticmethod
    def _build_self_attn(config, layer_id, quant_config, prefix, alt_streams):
        if layer_id in config.mla_layer_ids:
            return Ernie5_5MlaAttention(config, layer_id, quant_config, prefix)
        return Ernie5_5HcaAttention(config, layer_id, quant_config, prefix, alt_streams)

    def _hc_pre(self, residual, fn, scale, base, norm):
        if residual.shape[0] == 0:
            hc = self.hc_mult
            return (
                residual.new_empty((0, residual.shape[-1])),
                residual.new_empty((0, hc, 1), dtype=torch.float32),
                residual.new_empty((0, hc, hc), dtype=torch.float32),
            )
        post, comb, layer_input = mhc_pre(
            residual=residual,
            fn=fn,
            hc_scale=scale,
            hc_base=base,
            rms_eps=self.config.rms_norm_eps,
            hc_pre_eps=self.config.hc_eps,
            # The checkpoint uses one epsilon for both.
            hc_sinkhorn_eps=self.config.hc_eps,
            hc_post_mult_value=self.config.hc_post_mult_value,
            sinkhorn_repeat=self.config.hc_sinkhorn_iters,
            eps_outside_sqrt=True,
            norm_weight=norm.weight if self.fuse_input_norm else None,
            norm_eps=norm.variance_epsilon if self.fuse_input_norm else None,
        )
        if self.fuse_input_norm:
            return layer_input, post, comb
        return exact_rms_norm(layer_input, norm), post, comb

    @staticmethod
    def _hc_post(x, residual, post, comb):
        if x.shape[0] == 0:
            return torch.empty_like(residual)
        return mhc_post(x, residual, post, comb)

    def ffn_reads_own_rows(self) -> bool:
        """True if each row's FFN output depends on that row alone."""
        return (
            self._dense_replicated
            or get_parallel().attn_dp_size == 1
            or (
                isinstance(self.mlp, Ernie5_5Moe)
                and not get_moe_a2a_backend().is_none()
            )
        )

    def _forward_ffn(self, x: torch.Tensor, forward_batch: ForwardBatch):
        if isinstance(self.mlp, Ernie5_5Moe) and not get_moe_a2a_backend().is_none():
            if not _heads_are_split():
                return self.mlp(x)
            if forward_batch.attn_tp_sequence_sharded:
                return self._forward_moe_attn_tp_shard(x)
        if self._dense_replicated:
            return self.mlp(x) if x.shape[0] else x
        if get_parallel().attn_dp_size == 1:
            return self._forward_ffn_no_dp(x, forward_batch)
        tp_group = get_parallel().tp_group
        # Under prefill CP each CP rank gathers and takes back only its own shard
        # of the DP group's rows; mHC needs the FFN output row count to match its input.
        cp_shard_counts = None
        if forward_batch.attn_cp_metadata is not None and is_cp_active(forward_batch):
            metadata = forward_batch.attn_cp_metadata
            cp_shard_counts = (
                metadata.per_rank_logical_token or metadata.per_rank_actual_token
            )
        global_x = get_global_dp_buffer(tp_group)
        # Replicate, not partial: attention already reduced its output over
        # attention TP, so every attention-TP rank holds the same complete rows.
        dp_gather_replicate(global_x, x, forward_batch, cp_shard_counts=cp_shard_counts)
        global_out = self.mlp(global_x)
        out = get_local_dp_buffer(tp_group)
        dp_scatter(out, global_out, forward_batch, cp_shard_counts=cp_shard_counts)
        if cp_shard_counts is not None:
            # dp_scatter fills the whole DP-local buffer, zeroing rows this CP
            # rank does not own; keep only this rank's shard.
            out = out[: x.shape[0]]
        return out

    def _forward_moe_attn_tp_shard(self, x: torch.Tensor) -> torch.Tensor:
        """An all-to-all MoE under split heads: the dispatcher bounds each
        attention-TP rank to its own ``rows / attn_tp_size`` slice
        (``num_token_non_padded``); route that slice and all-gather."""
        attn_tp = get_parallel().attn_tp_size
        if x.shape[0] % attn_tp:
            raise RuntimeError(
                f"{x.shape[0]} rows do not split evenly over attention TP {attn_tp}"
            )
        chunk = x.shape[0] // attn_tp
        start = get_parallel().attn_tp_rank * chunk
        local = self.mlp(x[start : start + chunk].contiguous())
        out = torch.empty_like(x)
        attn_tp_all_gather_into_tensor(out, local.contiguous())
        return out

    def _forward_ffn_no_dp(self, x: torch.Tensor, forward_batch: ForwardBatch):
        """Gather prefill-CP shards over the MoE-CP group (an alias of attention CP
        here) so the TP-group reduce sums one token set."""
        if is_enable_moe_cp_allgather():
            rows = moe_cp_gathered_rows(forward_batch)
            if rows is not None:
                gathered = moe_cp_gather(x, rows, get_moe_cp_size())
                return moe_cp_take_back(self.mlp(gathered), rows)
        return self.mlp(x)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
        out_rows: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Only ``out_rows`` continue past attention (KV is still written for all);
        needs ffn_reads_own_rows()."""
        x, post, comb = self._hc_pre(
            hidden_states,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            self.input_layernorm,
        )
        if isinstance(self.self_attn, Ernie5_5MlaAttention):
            x = self.self_attn(positions, x, forward_batch, zero_allocator)
        else:
            x = self.self_attn(x, positions, forward_batch)
        if out_rows is not None:
            x, hidden_states = x[out_rows], hidden_states[out_rows]
            post, comb = post[out_rows], comb[out_rows]
        hidden_states = self._hc_post(x, hidden_states, post, comb)

        x, post, comb = self._hc_pre(
            hidden_states,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.post_attention_layernorm,
        )
        x = self._forward_ffn(x, forward_batch)
        return self._hc_post(x, hidden_states, post, comb)


class Ernie5_5Model(nn.Module):
    def __init__(
        self,
        config: Ernie5_5_MoeConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ):
        super().__init__()
        self.config = config
        self.pp_group = get_parallel().pp_group
        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=add_prefix("embed_tokens", prefix),
                **get_embedding_tp_kwargs(),
            )
        else:
            self.embed_tokens = PPMissingLayer()
        # Shared by every layer, sized as MQALayer's: the HCA prepare uses [:3],
        # the shared-expert overlap [0].
        self.alt_streams = (
            [torch.cuda.Stream() for _ in range(5)]
            if envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get()
            else None
        )
        # Out-of-stage slots are PPMissingLayer, so global layer ids stay indices.
        self.layers, self.start_layer, self.end_layer = make_pp_layers(
            config.num_hidden_layers,
            lambda idx, prefix: Ernie5_5DecoderLayer(
                config, idx, quant_config, prefix, self.alt_streams
            ),
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.hc_head = Ernie5_5HcHead(config)
        self.dspark_layers_to_capture: Optional[list[int]] = None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[PPProxyTensors, Tuple[torch.Tensor, torch.Tensor, list]]:
        """The last stage returns the normed hidden states, the mHC streams
        before the head (what an MTP draft reads) and the DSpark captures."""
        hc_mult, hidden_size = self.config.hc_mult, self.config.hidden_size
        if self.pp_group.is_first_rank:
            hidden_states = (
                self.embed_tokens(input_ids) if input_embeds is None else input_embeds
            )
            hidden_states = (
                hidden_states.unsqueeze(1).expand(-1, hc_mult, -1).contiguous()
            )
        else:
            # The mHC stream crosses a stage flattened to [T, hc_mult * hidden],
            # the width the runner sizes its PP buffers with (hc_hidden_size).
            hidden_states = pp_proxy_tensors["hidden_states"].view(
                -1, hc_mult, hidden_size
            )
        zero_allocator = BumpAllocator(
            buffer_size=2 * (self.end_layer - self.start_layer),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        captures = self.dspark_layers_to_capture or ()
        aux_hidden_states = []
        for layer_id in range(self.start_layer, self.end_layer):
            hidden_states = self.layers[layer_id](
                positions, hidden_states, forward_batch, zero_allocator
            )
            if layer_id in captures:
                # The draft reads layer i's output, not its input, as the mean
                # of the mHC streams, as it was trained.
                aux_hidden_states.append(hidden_states.mean(dim=1))
        if not self.pp_group.is_last_rank:
            return PPProxyTensors({"hidden_states": hidden_states.flatten(1)})
        pre_hc_head = hidden_states.flatten(1)
        # A draft reads every row's streams, the LM head only logits_rows():
        # collapse and norm just those.
        rows = logits_rows(forward_batch) if hidden_states.shape[0] > 0 else None
        hidden_states = self.hc_head(
            hidden_states if rows is None else hidden_states[rows]
        )
        hidden_states = exact_rms_norm(hidden_states, self.norm)
        if rows is not None:
            hidden_states = scatter_rows(hidden_states, rows, pre_hc_head.shape[0])
        return hidden_states, pre_hc_head, aux_hidden_states


class Ernie5_5LogitsProcessor(LogitsProcessor):
    def __init__(self, config: Ernie5_5_MoeConfig):
        super().__init__(config)
        self.multimax_ranges = nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))
        self.multimax_ts = nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))

    def _compute_lm_head(
        self, hidden_states, lm_head, embedding_bias=None, *, out=None
    ):
        x = super()._compute_lm_head(hidden_states, lm_head, embedding_bias, out=out)
        return multimax(x, self.multimax_ranges, self.multimax_ts)


class Ernie5_5_MoeForCausalLM(nn.Module):
    def __init__(
        self,
        config: Ernie5_5_MoeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        if not isinstance(config, Ernie5_5_MoeConfig):
            raise ValueError(
                "Ernie5_5_MoeForCausalLM serves ERNIE 5.5 Flash checkpoints only, "
                f"got a {type(config).__name__}"
            )
        config.validate_checkpoint_fields()
        check_supported_deployment(config)
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_parallel().pp_group
        self.model = Ernie5_5Model(config, quant_config, add_prefix("model", prefix))
        if self.pp_group.is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
                use_attn_tp_group=get_parallel().enable_dp_lm_head,
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = Ernie5_5LogitsProcessor(config)
        self.capture_aux_hidden_states = False
        get_attn_tp_context().init_context(config.hybrid_mla_q_lora_rank, is_dsa=True)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        hidden_states = self.model(
            input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
        )
        if not self.pp_group.is_last_rank:
            return hidden_states
        hidden_states, pre_hc_head, aux_hidden_states = hidden_states
        if not self.capture_aux_hidden_states:
            aux_hidden_states = None
        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
            aux_hidden_states=aux_hidden_states,
            # An EAGLE (MTP) draft reads the mHC streams, not the collapsed state.
            hidden_states_before_norm=(
                None if aux_hidden_states is not None else pre_hc_head
            ),
        )

    def get_embed_and_head(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_dspark_layers_to_capture(self, layer_ids: list[int]) -> None:
        if not layer_ids:
            raise ValueError(
                "DSPARK requires explicit layer_ids for aux hidden capture"
            )
        if not self.pp_group.is_last_rank:
            return
        outside = [i for i in layer_ids if not self.start_layer <= i < self.end_layer]
        if outside:
            raise ValueError(
                f"DSpark target layers {outside} are outside the last pipeline "
                f"stage [{self.start_layer}, {self.end_layer})"
            )
        self.capture_aux_hidden_states = True
        self.model.dspark_layers_to_capture = [int(i) for i in layer_ids]

    def get_input_embeddings(self) -> nn.Module:
        # The prefill-CP runner embeds before sharding tokens across CP ranks;
        # Ernie5_5Model.forward expands the mHC streams on the rank-local slice.
        return self.model.embed_tokens

    @property
    def start_layer(self) -> int:
        return self.model.start_layer

    @property
    def end_layer(self) -> int:
        return self.model.end_layer

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        Ernie5_5WeightLoader(self).load_all(weights)
        self.post_load_weights()

    def post_load_weights(self) -> None:
        # Splits the MLA layers' kv_b_proj into w_kc / w_vc (HCA layers have none).
        DeepseekV2WeightLoaderMixin.post_load_weights(self)
        for layer in self.model.layers[self.start_layer : self.end_layer]:
            layer.self_attn.postmix.fold()


# wo_a's fp8 block edge, as fp8_einsum's recipe=(1, 128, 128) reads it.
_WO_A_BLOCK = 128


def _quantize_wo_a(attn: Ernie5_5HcaAttention, weight: torch.Tensor) -> None:
    from sglang.srt.layers.quantization.fp8_utils import per_block_cast_to_fp8

    groups, rank = attn.n_local_groups, attn.o_lora_rank
    # The checkpoint holds every group; attention TP gives each rank a run of them.
    rows = groups * rank
    weight = weight[attn.attn_tp_rank * rows : (attn.attn_tp_rank + 1) * rows]
    grouped = weight.to(device=attn.wo_a.weight.device, dtype=torch.bfloat16)
    grouped = grouped.view(groups, rank, weight.shape[1])
    # One group at a time: a 128x128 block must not straddle two groups.
    # The scale is rounded to UE8M0 before dividing, so DeepGEMM packing stays exact.
    quantized = [per_block_cast_to_fp8(grouped[g]) for g in range(groups)]
    _store_fp8_wo_a(
        attn,
        torch.stack([q for q, _ in quantized]).view(weight.shape),
        torch.stack([s for _, s in quantized]),
    )


def _load_fp8_wo_a(
    attn: Ernie5_5HcaAttention, weight: torch.Tensor, scale: torch.Tensor
) -> None:
    groups, rank = attn.n_local_groups, attn.o_lora_rank
    total_rows, in_dim = weight.shape
    expected = [total_rows // _WO_A_BLOCK, -(-in_dim // _WO_A_BLOCK)]
    # Whole-block groups make the full-matrix blocks the per-group ones.
    if rank % _WO_A_BLOCK or list(scale.shape) != expected:
        raise ValueError(
            f"wo_a: expected a {expected} block scale and o_lora_rank ({rank}) a "
            f"multiple of {_WO_A_BLOCK}, got a {list(scale.shape)} scale"
        )
    if DEEPGEMM_SCALE_UE8M0 and not (torch.frexp(scale.float())[0] == 0.5).all():
        raise ValueError("wo_a: UE8M0 DeepGEMM needs power-of-two block scales")
    rows = groups * rank
    start = attn.attn_tp_rank * rows
    local_scale = scale[start // _WO_A_BLOCK : (start + rows) // _WO_A_BLOCK]
    _store_fp8_wo_a(
        attn,
        weight[start : start + rows],
        local_scale.reshape(groups, rank // _WO_A_BLOCK, -1),
    )


def _store_fp8_wo_a(
    attn: Ernie5_5HcaAttention, weight: torch.Tensor, scale: torch.Tensor
) -> None:
    """Write a rank's fp8 ``wo_a`` and its scale in the layout ``fp8_einsum`` reads."""
    wo_a = attn.wo_a
    # In place: a captured CUDA graph holds the weight's address.
    wo_a.weight.data.copy_(weight)
    scale = scale.to(device=wo_a.weight.device, dtype=torch.float32)
    if DEEPGEMM_SCALE_UE8M0:
        from deep_gemm import transform_sf_into_required_layout

        scale = transform_sf_into_required_layout(
            scale,
            mn=attn.o_lora_rank,
            k=weight.shape[1],
            recipe=(1, 128, 128),
            num_groups=attn.n_local_groups,
            is_sfa=False,
        )
    # create_weights sized the scale for the flat 2D layout, so the first write rebinds.
    if wo_a.weight_scale_inv.data.shape == scale.shape:
        wo_a.weight_scale_inv.data.copy_(scale)
    else:
        wo_a.weight_scale_inv.data = scale
    # Keeps the linear method's own UE8M0 requant off this scale.
    wo_a.weight_scale_inv.format_ue8m0 = DEEPGEMM_SCALE_UE8M0


class Ernie5_5WeightLoader:
    """Maps checkpoint names onto the module tree; fails on unloaded parameters."""

    # Checkpoint tensors stacked into one parameter: suffix -> (param, stack index).
    _HCA_STACKED = {
        "self_attn.linear_q_down_proj.weight": ("self_attn.wqkv_a.weight", 0),
        "self_attn.linear_kv_proj.weight": ("self_attn.wqkv_a.weight", 1),
        "self_attn.core_attention.compressor.linear_wkv.weight": (
            "self_attn.compressor.wkv_gate.weight",
            0,
        ),
        "self_attn.core_attention.compressor.linear_wgate.weight": (
            "self_attn.compressor.wkv_gate.weight",
            1,
        ),
    }
    # With SGLANG_OPT_FUSE_WQA_WKV off, the HCA attention keeps them apart.
    _HCA_UNFUSED_QKV_A = {
        "self_attn.linear_q_down_proj.weight": "self_attn.wq_a.weight",
        "self_attn.linear_kv_proj.weight": "self_attn.wkv.weight",
    }
    _MLA_STACKED = {
        "self_attn.q_a_proj.weight": ("self_attn.fused_qkv_a_proj_with_mqa.weight", 0),
        "self_attn.kv_a_proj_with_mqa.weight": (
            "self_attn.fused_qkv_a_proj_with_mqa.weight",
            1,
        ),
    }
    _MOE_STACKED = {
        "block_sparse_moe.gate.weight": ("mlp.gate_weight", 0),
        "block_sparse_moe.gate.weight_1": ("mlp.gate_weight", 1),
    }
    _HCA_RENAMES = {
        "self_attn.q_layernorm.weight": "self_attn.q_norm.weight",
        "self_attn.kv_layernorm.weight": "self_attn.kv_norm.weight",
        "self_attn.o_proj.weight": "self_attn.wo_b.weight",
        "self_attn.core_attention.attn_sink": "self_attn.attn_sink",
        "self_attn.core_attention.compressor.norm.weight": (
            "self_attn.compressor.norm.weight"
        ),
        "self_attn.core_attention.compressor.ape": "self_attn.compressor.ape",
    }
    _COMMON_RENAMES = {
        "self_attn.vha_postmix_U": "self_attn.postmix.vha_postmix_U",
        "self_attn.vha_postmix_V": "self_attn.postmix.vha_postmix_V",
        "self_attn.core_attention.softmax_offset": "self_attn.learnable_sink_param",
        "block_sparse_moe.e_score_correction_bias": "mlp.e_score_correction_bias",
        "block_sparse_moe.gate.routed_scaling_factor_param": "mlp.per_expert_scale",
        "block_sparse_moe.fc1_latent_proj.weight": "mlp.fc1_latent_proj.weight",
        "block_sparse_moe.fc2_latent_proj.weight": "mlp.fc2_latent_proj.weight",
        "block_sparse_moe.latent_norm.weight": "mlp.latent_norm.weight",
        "block_sparse_moe.shared_experts.w2.weight": (
            "mlp.shared_experts.down_proj.weight"
        ),
    }
    # suffix -> (param, shard id) for MergedColumnParallelLinear.
    _HCA_MERGED_SHARDS = {
        "self_attn.linear_q_up_proj.weight": ("self_attn.wq_b.weight", 0),
        "self_attn.gate_proj.weight": ("self_attn.wq_b.weight", 1),
    }
    _MERGED_SHARDS = {
        "mlp.gate_proj.weight": ("mlp.gate_up_proj.weight", 0),
        "mlp.up_proj.weight": ("mlp.gate_up_proj.weight", 1),
        "block_sparse_moe.shared_experts.w1.weight": (
            "mlp.shared_experts.gate_up_proj.weight",
            0,
        ),
        "block_sparse_moe.shared_experts.w3.weight": (
            "mlp.shared_experts.gate_up_proj.weight",
            1,
        ),
    }
    _MULTIMAX_RENAMES = {
        "model.lm_head.multimax_ranges": "logits_processor.multimax_ranges",
        "model.lm_head.multimax_ts": "logits_processor.multimax_ts",
    }
    _ROOT_RENAMES = {
        **_MULTIMAX_RENAMES,
        "model.mhc_contract.hc_head_fn": "model.hc_head.fn",
        "model.mhc_contract.hc_head_base": "model.hc_head.base",
        "model.mhc_contract.hc_head_scale": "model.hc_head.scale",
    }

    def __init__(self, model: Ernie5_5_MoeForCausalLM):
        self._init_tables(
            model, layers=model.model.layers, mla_layers=model.config.mla_layer_ids
        )
        self.stage = range(model.start_layer, model.end_layer)
        pp_group = model.pp_group
        # A bundled DSpark draft lives under model.mtp.*; the draft model loads it.
        skipped = ["model.mtp."]
        # Root tensors only one end of the pipeline builds a module for.
        if not pp_group.is_first_rank:
            skipped.append("model.embed_tokens.")
        if not pp_group.is_last_rank:
            skipped += ["model.norm.", "model.mhc_contract.", "lm_head."]
        self.skipped_roots = tuple(skipped)

    def _init_tables(self, model, *, layers, mla_layers, preloaded=()) -> None:
        """The state every loader shares; ``preloaded`` names params bound later."""
        self.config = model.config
        self.quant_config = model.quant_config
        self.params = dict(model.named_parameters())
        self.layers = layers
        self.loaded: set[str] = set(preloaded)
        # Stacked expert params are "loaded" after one shard, so track the shards
        # themselves: {param name: {(expert id, shard id)}}.
        self.expert_shards: dict[str, set] = {}
        self.mla_layers = set(mla_layers)
        # An fp8 wo_a's weight or block scale, held until the other one arrives.
        self.pending_wo_a: dict[str, dict[str, torch.Tensor]] = {}
        # The same for the fused indexer wk_weights_proj's fp8 wk.
        self.pending_indexer_wk: dict[str, dict[str, torch.Tensor]] = {}
        # {wk_weights_proj name: halves written}; it is loaded once both are.
        self.fused_indexer_halves: dict[str, set[str]] = {}

    def load_all(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        for name, tensor in weights:
            self.load(name, tensor)
        self.finish()

    def _load(self, name: str, tensor: torch.Tensor, *args) -> None:
        param = self.params[name]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, tensor, *args)
        self.loaded.add(name)

    def _load_stacked(self, name: str, index: int, tensor: torch.Tensor) -> None:
        param = self.params[name]
        rows = tensor.shape[0]
        # Every stacked pair here is [first | second]; the second ends at the last row.
        start = 0 if index == 0 else param.shape[0] - rows
        param.data[start : start + rows].copy_(tensor)
        self.loaded.add(name)

    def load(self, name: str, tensor: torch.Tensor) -> None:
        if not name.startswith("model.layers."):
            if not name.startswith(self.skipped_roots):
                self._load(self._ROOT_RENAMES.get(name, name), tensor)
            return
        layer_id_text, _, suffix = name[len("model.layers.") :].partition(".")
        layer_id = int(layer_id_text)
        # Ids past the main stack are the MTP (NextN) draft layers; the others
        # outside this pipeline stage belong to another rank.
        if layer_id not in self.stage:
            return
        self._load_layer_tensor(f"model.layers.{layer_id}.", layer_id, suffix, tensor)

    def _load_layer_tensor(
        self, prefix: str, layer_id: int, suffix: str, tensor: torch.Tensor
    ) -> None:
        if self._load_hc(prefix, suffix, tensor):
            return
        if suffix.startswith("block_sparse_moe.experts."):
            self._load_expert(prefix, suffix, tensor)
            return
        # An offline-quantized weight ships a block scale beside it; strip the
        # suffix and re-append it to whatever target the weight maps to.
        scale_suffix = "_scale_inv" if suffix.endswith("_scale_inv") else ""
        suffix = suffix.removesuffix(scale_suffix)
        if suffix == "self_attn.linear_o_group_proj":
            self._load_wo_a(prefix, layer_id, tensor, is_scale=bool(scale_suffix))
            return
        if suffix.startswith("self_attn.core_attention.indexer."):
            field = suffix[len("self_attn.core_attention.indexer.") :]
            name = f"{prefix}self_attn.indexer.{field}{scale_suffix}"
            # With fusion on, wk and weights_proj fill one bf16 wk_weights_proj.
            if field.startswith(("wk.", "weights_proj.")) and _load_fused_indexer_wk(
                name, tensor, self.params, self.pending_indexer_wk, self.quant_config
            ):
                fused = f"{prefix}self_attn.indexer.wk_weights_proj.weight"
                halves = self.fused_indexer_halves.setdefault(fused, set())
                halves.add(field.partition(".")[0])
                if halves == {"wk", "weights_proj"}:
                    self.loaded.add(fused)
                return
            self._load(name, tensor)
            return
        stacked = {
            **self._MOE_STACKED,
            **(self._MLA_STACKED if layer_id in self.mla_layers else self._HCA_STACKED),
        }
        if (
            suffix in self._HCA_UNFUSED_QKV_A
            and layer_id not in self.mla_layers
            and not self.layers[layer_id].self_attn.fuse_wqa_wkv
        ):
            target = self._HCA_UNFUSED_QKV_A[suffix]
            self._load(prefix + target + scale_suffix, tensor)
            return
        if suffix in stacked:
            target, index = stacked[suffix]
            self._load_stacked(prefix + target + scale_suffix, index, tensor)
            return
        merged = self._MERGED_SHARDS
        if layer_id not in self.mla_layers:
            merged = {**merged, **self._HCA_MERGED_SHARDS}
        if suffix in merged:
            target, shard_id = merged[suffix]
            self._load(prefix + target + scale_suffix, tensor, shard_id)
            return
        renames = self._COMMON_RENAMES
        if layer_id not in self.mla_layers:
            renames = {**renames, **self._HCA_RENAMES}
        self._load(prefix + renames.get(suffix, suffix) + scale_suffix, tensor)

    def _load_wo_a(
        self, prefix: str, layer_id: int, tensor: torch.Tensor, *, is_scale: bool
    ) -> None:
        attn = self.layers[layer_id].self_attn
        name = f"{prefix}self_attn.wo_a.weight"
        if not is_scale and tensor.dtype != torch.float8_e4m3fn:
            if attn.wo_a_fp8:
                _quantize_wo_a(attn, tensor)
                self.loaded.add(name)
            else:
                self._load(name, tensor)
            return
        # An fp8 checkpoint ships the weight and its block scale apart, in any order.
        pair = self.pending_wo_a.setdefault(name, {})
        pair["scale" if is_scale else "weight"] = _clone_if_runai_streamed_tensor(
            tensor
        )
        if len(pair) < 2:
            return
        del self.pending_wo_a[name]
        if attn.wo_a_fp8:
            _load_fp8_wo_a(attn, pair["weight"], pair["scale"])
            self.loaded.add(name)
            return
        # Layers without the fp8 wo_a GEMM (the DSpark draft, or
        # SGLANG_OPT_FP8_WO_A_GEMM=0) run it in bf16.
        from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant

        device = self.params[name].device
        self._load(
            name,
            block_quant_dequant(
                pair["weight"].to(device),
                pair["scale"].to(device),
                [_WO_A_BLOCK, _WO_A_BLOCK],
                torch.bfloat16,
            ),
        )

    def _load_hc(self, prefix: str, suffix: str, tensor: torch.Tensor) -> bool:
        path, _, field = suffix.partition(".")
        kind = {
            "self_attention_hyper_connection": "attn",
            "mlp_hyper_connection": "ffn",
        }.get(path)
        if kind is None:
            return False
        if field == "mapping_proj.weight":
            name = f"{prefix}hc_{kind}_fn"
            self.params[name].data.copy_(tensor.t())
        elif field == "bias":
            name = f"{prefix}hc_{kind}_base"
            self.params[name].data.copy_(tensor)
        else:
            name = f"{prefix}hc_{kind}_scale"
            # The [pre, post, res] order mhc_pre reads hc_scale in.
            index = ("alpha_pre", "alpha_post", "alpha_res").index(field)
            self.params[name].data[index] = tensor.reshape(())
        self.loaded.add(name)
        return True

    def _load_expert(self, prefix: str, suffix: str, tensor: torch.Tensor) -> None:
        # block_sparse_moe.experts.{E}.{w1|w2|w3}.{weight|weight_scale_inv};
        # w1 = gate, w3 = up.
        _, _, expert_id, shard_id, field = suffix.split(".")
        fused = "w2" if shard_id == "w2" else "w13"
        target = f"{prefix}mlp.experts.{fused}_{field}"
        param = self.params[target]
        param.weight_loader(
            param, tensor, target, shard_id=shard_id, expert_id=int(expert_id)
        )
        self.loaded.add(target)
        self.expert_shards.setdefault(target, set()).add((int(expert_id), shard_id))

    def finish(self) -> None:
        # w1 and w3 share one w13 param, so a missing shard passes the leftover check;
        # EP makes the count rank-dependent, so check both halves hold the same experts.
        for target, seen in self.expert_shards.items():
            if not target.endswith("w13_weight"):
                continue
            gate = {expert for expert, shard in seen if shard == "w1"}
            up = {expert for expert, shard in seen if shard == "w3"}
            if gate != up:
                raise RuntimeError(
                    f"{target}: experts {sorted(gate ^ up)[:16]} loaded only one of "
                    "w1 (gate) / w3 (up)"
                )
        pending = {**self.pending_wo_a, **self.pending_indexer_wk}
        if pending:
            raise RuntimeError(
                "fp8 weights or block scales without the other half: "
                f"{sorted(pending)[:8]}"
            )
        # Not loaded by name: wo_a's block scale (written with its weight) and the
        # KV cache scales.
        derived = ("self_attn.wo_a.weight_scale_inv", ".k_scale", ".v_scale")
        missing = sorted(
            name
            for name in set(self.params) - self.loaded
            if not name.endswith(derived)
        )
        if missing:
            raise RuntimeError(
                f"{len(missing)} ERNIE 5.5 Flash parameters were never loaded; "
                f"first: {missing[:16]}"
            )


EntryClass = [Ernie5_5_MoeForCausalLM]
