# Copyright 2025-2026 SGLang Team
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
"""Inference-only BerryLM-OS model compatible with HuggingFace weights.

BerryLM is a text-only hybrid MoE decoder: three gated delta-net linear-attention
layers per full-attention layer, sparse MoE + shared expert on every layer, and two
additions on top of the usual hybrid:

* a per-channel **KDA forget gate** on the linear layers
  (``g = -exp(A_log)[h] * softplus(f_up(f_down(x))[h, k] + dt_bias[h, k])``) on the
  gated-delta-net projection layout (fused q|k|v conv, ``in_proj_z`` output gate) — served
  by the KDA linear-attention backend (registered in ``configs/berrylm.py``);
* a **Gated Block AttnRes** mixer on the residual stream: before layer *i* the
  stream ``x`` becomes ``x + tanh(gate_i) * (mix_i - x)`` where ``mix_i`` is a
  softmax mixture over depth of the streams committed at block boundaries and
  ``x`` itself. Token-local, so it does not touch any cache.
"""

import logging
from typing import Iterable, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.kernels.ops.attention.fla.layernorm_gated import RMSNorm as RMSNormGated
from sglang.kernels.ops.elementwise.elementwise import fused_gate_sigmoid_mul_add
from sglang.srt.configs.berrylm import BerryLMConfig
from sglang.srt.distributed import get_pp_group, tensor_model_parallel_all_reduce
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.attention.mamba.mamba import mamba_v2_sharded_weight_loader
from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import (
    get_moe_a2a_backend,
    should_skip_post_experts_all_reduce,
)
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.moe.utils import (
    RoutingMethodType,
    filter_moe_weight_param_global_expert,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from sglang.srt.runtime_context import get_exec, get_forward, get_parallel, get_stream
from sglang.srt.utils import add_prefix, is_cuda, make_layers, set_weight_attrs
from sglang.srt.utils.custom_op import register_custom_op

logger = logging.getLogger(__name__)
_is_cuda = is_cuda()


# ---------------------------------------------------------------------------
# Gated Block AttnRes
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Fused Gated Block AttnRes mixer (Triton, forward only): one program per token,
# one pass over the committed streams for the norms / logits and one for the
# mixture; numerics follow the torch chain (fp32 accumulation, the blend rounded
# in the stream dtype op by op). The torch chain stays as the off-GPU fallback.
# ---------------------------------------------------------------------------

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    HAS_TRITON = True
except ImportError:  # pragma: no cover - CPU-only environments
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _gated_attnres_kernel(
        stream_ptr,
        blocks_ptr,
        query_ptr,
        gate_ptr,
        out_ptr,
        n_blocks,
        D,
        stride_st,  # stream / out: row stride
        stride_bn,  # block_streams: source stride
        stride_bt,  # block_streams: row stride
        eps,
        HAS_GATE: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        t = tl.program_id(0)
        offs = tl.arange(0, BLOCK_D)
        mask = offs < D
        q = tl.load(query_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        inv_d = 1.0 / D

        # pass 1: logits of the committed sources and of the current stream, running max for a stable softmax
        x_cur = tl.load(stream_ptr + t * stride_st + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        rstd = tl.rsqrt(tl.sum(x_cur * x_cur, axis=0) * inv_d + eps)
        logit_cur = tl.sum(x_cur * q, axis=0) * rstd
        m = logit_cur
        for i in range(n_blocks):
            x = tl.load(
                blocks_ptr + i * stride_bn + t * stride_bt + offs, mask=mask, other=0.0
            ).to(tl.float32)
            rstd = tl.rsqrt(tl.sum(x * x, axis=0) * inv_d + eps)
            logit = tl.sum(x * q, axis=0) * rstd
            m = tl.maximum(m, logit)

        # pass 2: the softmax denominator (recomputing a logit is cheaper than keeping n rows live)
        w_cur = tl.exp(logit_cur - m)
        denom = w_cur
        for i in range(n_blocks):
            x = tl.load(
                blocks_ptr + i * stride_bn + t * stride_bt + offs, mask=mask, other=0.0
            ).to(tl.float32)
            rstd = tl.rsqrt(tl.sum(x * x, axis=0) * inv_d + eps)
            denom += tl.exp(tl.sum(x * q, axis=0) * rstd - m)
        # pass 3: the mixture with normalized weights, committed sources first, the current stream last (the
        # reference's order of summation)
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        for i in range(n_blocks):
            x = tl.load(
                blocks_ptr + i * stride_bn + t * stride_bt + offs, mask=mask, other=0.0
            ).to(tl.float32)
            rstd = tl.rsqrt(tl.sum(x * x, axis=0) * inv_d + eps)
            p = tl.exp(tl.sum(x * q, axis=0) * rstd - m) / denom
            acc += p * x
        mixed = acc + (w_cur / denom) * x_cur

        dt: tl.constexpr = out_ptr.dtype.element_ty
        if HAS_GATE:
            # The reference blends in the stream dtype op by op (`stream + tanh(gate) * (mixed - stream)` on bf16
            # tensors rounds after every op); mirror that rounding sequence so the results match ulp for ulp.
            g = tl.load(gate_ptr).to(tl.float32)
            scale = libdevice.tanh(g).to(dt).to(tl.float32)
            mixed_r = mixed.to(dt).to(tl.float32)
            diff = (mixed_r - x_cur).to(dt).to(tl.float32)
            prod = (scale * diff).to(dt).to(tl.float32)
            out = x_cur + prod
        else:
            out = mixed
        tl.store(out_ptr + t * stride_st + offs, out.to(dt), mask=mask)


def gated_attnres(
    stream: torch.Tensor,
    block_streams: torch.Tensor,
    pseudo_query: torch.Tensor,
    gate: Optional[torch.Tensor],
    eps: float,
) -> torch.Tensor:
    """Fused mixer. ``stream`` [T, D], ``block_streams`` [n, T, D] (n >= 0), ``pseudo_query`` [D], ``gate`` 0-d or
    None. Falls back to the torch chain off-GPU or for rows wider than 8192."""
    T, D = stream.shape
    if (
        not HAS_TRITON
        or stream.device.type not in ("cuda", "xpu")
        or D > 8192
        or T == 0
    ):
        return _gated_attnres_torch(stream, block_streams, pseudo_query, gate, eps)
    if stream.stride(-1) != 1 or block_streams.stride(-1) != 1:
        stream, block_streams = stream.contiguous(), block_streams.contiguous()
    n = block_streams.shape[0]
    out = torch.empty_like(stream)
    pq = pseudo_query if pseudo_query.dtype == torch.float32 else pseudo_query.float()
    has_gate = gate is not None
    _gated_attnres_kernel[(T,)](
        stream,
        block_streams,
        pq,
        gate if has_gate else stream,  # any pointer when unused
        out,
        n,
        D,
        stream.stride(0),
        block_streams.stride(0) if n else 0,
        block_streams.stride(1) if n else 0,
        eps,
        HAS_GATE=has_gate,
        BLOCK_D=triton.next_power_of_2(D),
        num_warps=8 if D > 1024 else 4,
    )
    return out


def _gated_attnres_torch(
    stream: torch.Tensor,
    block_streams: torch.Tensor,
    pseudo_query: torch.Tensor,
    gate: Optional[torch.Tensor],
    eps: float,
) -> torch.Tensor:
    """The torch chain of the mixer (fallback off-GPU): ``stream`` [T, D], ``block_streams`` [n, T, D]."""
    stacked = torch.cat([block_streams, stream.unsqueeze(0)], dim=0).float()
    keys = F.rms_norm(stacked, (stream.shape[-1],), None, eps)
    logits = torch.einsum("ntd,d->nt", keys, pseudo_query.float())
    mixed = (logits.softmax(dim=0).unsqueeze(-1) * stacked).sum(dim=0).to(stream.dtype)
    if gate is None:
        return mixed
    scale = torch.tanh(gate.float()).to(stream.dtype)
    return stream + scale * (mixed - stream)


def _gated_block_attn_res(
    pseudo_query: torch.Tensor,
    stream: torch.Tensor,
    completed: List[torch.Tensor],
    gate: Optional[torch.Tensor],
    eps: float,
) -> torch.Tensor:
    hidden = stream.shape[-1]
    stream2d = stream.reshape(-1, hidden)
    if completed:
        block_streams = torch.stack([s.reshape(-1, hidden) for s in completed], dim=0)
    else:
        block_streams = stream2d.new_empty((0, *stream2d.shape))
    return gated_attnres(stream2d, block_streams, pseudo_query, gate, eps).reshape(
        stream.shape
    )


@register_custom_op(mutates_args=["output"])
def berrylm_gated_attnres_with_output(
    pseudo_query: torch.Tensor,
    stream: torch.Tensor,
    completed: List[torch.Tensor],
    gate: Optional[torch.Tensor],
    eps: float,
    output: torch.Tensor,
) -> None:
    # Opaque op: the multi-source fp32 stack/rms_norm/softmax chain must not be
    # fused by a graph compiler (it stays a plain kernel sequence inside graphs).
    output.copy_(_gated_block_attn_res(pseudo_query, stream, completed, gate, eps))


class BerryLMAttnRes(nn.Module):
    """Per-layer mixer: ``pseudo_query`` [hidden] (zero == uniform mean over the
    sources) and a scalar ``gate`` (zero == exact identity). Replicated on every
    TP rank. Checkpoint keys: ``model.layers.{i}.attn_res.{pseudo_query,gate}``."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, gated: bool = True):
        super().__init__()
        self.eps = eps
        self.pseudo_query = nn.Parameter(torch.zeros(hidden_size))
        self.gate = nn.Parameter(torch.zeros(())) if gated else None

    def forward(
        self, stream: torch.Tensor, completed: List[torch.Tensor]
    ) -> torch.Tensor:
        output = torch.empty_like(stream)
        berrylm_gated_attnres_with_output(
            self.pseudo_query, stream, list(completed), self.gate, self.eps, output
        )
        return output


# ---------------------------------------------------------------------------
# Linear attention with the KDA forget gate
# ---------------------------------------------------------------------------


class BerryLMKDA(nn.Module):
    """Gated delta net (stock GDN projection layout) with a per-channel KDA gate.

    Projections: ``in_proj_qkv`` [2K+V, D] (fused in the checkpoint), ``in_proj_z`` and
    ``in_proj_b`` (beta logits) — one merged GEMM at run time —, ``f_down_proj`` [bottleneck, D] / ``f_up_proj``
    [HV*K, bottleneck] (raw forget gate), depthwise ``conv1d`` over q|k|v,
    ``A_log`` [HV], ``dt_bias`` [HV*K], SiLU-gated output RMSNorm on ``z``,
    ``out_proj``. The KDA backend applies the gate activation (prefill in
    ``chunk_kda``, decode in the fused recurrent kernel), like Kimi Linear.
    """

    def __init__(
        self,
        config: BerryLMConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.attn_tp_rank = get_parallel().attn_tp_rank
        self.attn_tp_size = get_parallel().attn_tp_size
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.gate_bottleneck = config.kda_gate_bottleneck
        self.layer_id = layer_id
        self.activation = config.hidden_act
        self.layer_norm_epsilon = config.rms_norm_eps
        self.lower_bound = (
            float(config.kda_gate_lower_bound)
            if getattr(config, "kda_safe_gate", False)
            else None
        )
        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError(
                "linear_num_value_heads must be a multiple of linear_num_key_heads"
            )
        if self.num_k_heads % self.attn_tp_size or self.num_v_heads % self.attn_tp_size:
            raise ValueError(
                "BerryLM linear attention needs the key and value head counts "
                f"divisible by the attention TP size ({self.attn_tp_size})"
            )
        self.local_num_k_heads = self.num_k_heads // self.attn_tp_size
        self.local_num_v_heads = self.num_v_heads // self.attn_tp_size

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=self.conv_dim,
            bias=False,
            quant_config=None,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            prefix=add_prefix("conv1d", prefix),
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        # Checkpoint ships in_proj_qkv / in_proj_z / in_proj_b separately; q|k|v|z run as one GEMM
        # (every partition is sharded by head, so the merged layer's TP split is the natural one).
        # The beta projection (one row per value head) stays a separate bf16 layer: smaller than a
        # quantization block, and a merged layer takes a single quantization scheme.
        self.in_proj_qkvz = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_sizes=[self.key_dim, self.key_dim, self.value_dim, self.value_dim],
            bias=False,
            quant_config=quant_config,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            prefix=add_prefix("in_proj_qkvz", prefix),
        )
        self.in_proj_b = ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.num_v_heads,
            bias=False,
            quant_config=None,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            prefix=add_prefix("in_proj_b", prefix),
        )
        # Low-rank forget gate: D -> bottleneck (replicated) -> HV*K (sharded by head). Never
        # quantized: the decay gate is the precision-critical part of the recurrence.
        self.f_down_proj = ReplicatedLinear(
            self.hidden_size,
            self.gate_bottleneck,
            bias=False,
            quant_config=None,
            prefix=add_prefix("f_down_proj", prefix),
        )
        self.f_up_proj = ColumnParallelLinear(
            input_size=self.gate_bottleneck,
            output_size=self.num_v_heads * self.head_k_dim,
            bias=False,
            quant_config=None,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            prefix=add_prefix("f_up_proj", prefix),
        )

        query_key_settings = (self.key_dim, 0, False)
        value_settings = (self.value_dim, 0, False)
        delattr(self.conv1d.weight, "weight_loader")
        set_weight_attrs(
            self.conv1d.weight,
            {
                "weight_loader": mamba_v2_sharded_weight_loader(
                    [query_key_settings, query_key_settings, value_settings],
                    self.attn_tp_size,
                    self.attn_tp_rank,
                )
            },
        )
        self.A_log = nn.Parameter(
            torch.empty(self.local_num_v_heads, dtype=torch.float32)
        )
        self.dt_bias = nn.Parameter(
            torch.empty(self.local_num_v_heads * self.head_k_dim, dtype=torch.float32)
        )
        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(0)})
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        self.attn = RadixLinearAttention(
            layer_id=layer_id,
            num_q_heads=self.local_num_k_heads,
            num_k_heads=self.local_num_k_heads,
            num_v_heads=self.local_num_v_heads,
            head_q_dim=self.head_k_dim,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            conv_weights=conv_weights,
            bias=self.conv1d.bias,
            activation=self.activation,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
        )
        self.attn.lower_bound = self.lower_bound
        # BerryLM's per-channel log-decays reach -300 and beyond; the fused "small-grid" intra-chunk
        # kernel factorizes exp2(g_i - g_j) with factors clamped to +-126 and collapses such blocks.
        # Route prefill through the difference-form (token-parallel) intra-chunk kernels instead.
        self.attn.kda_fused_intra = False
        act = getattr(config, "kda_output_gate_act", "silu")
        act = "swish" if act in ("silu", "swish") else act
        self.norm = RMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            group_size=None,
            norm_before_gate=True,
            device=torch.get_device_module().current_device(),
            dtype=torch.get_default_dtype(),
            activation=act,
        )
        self.out_proj = RowParallelLinear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            reduce_results=False,
            quant_config=quant_config,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            prefix=add_prefix("out_proj", prefix),
        )

    def forward(self, hidden_states: torch.Tensor, forward_batch: ForwardBatch):
        num_tokens = hidden_states.shape[0]
        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        qkv_size = (2 * self.key_dim + self.value_dim) // self.attn_tp_size
        z_size = self.value_dim // self.attn_tp_size
        mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
        z = z.reshape(num_tokens, -1, self.head_v_dim)
        beta, _ = self.in_proj_b(hidden_states)
        beta = beta.float()
        raw_g = self.f_up_proj(self.f_down_proj(hidden_states)[0])[0]
        # Prefill passes raw gates to chunk KDA; decode and target-verify kernels
        # apply the activation internally (same contract as Kimi Linear).
        if (
            not forward_batch.forward_mode.is_decode()
            and not forward_batch.forward_mode.is_target_verify()
        ):
            raw_g = raw_g.unflatten(-1, (-1, self.head_k_dim)).unsqueeze(0)
            beta = beta.sigmoid()
        beta = beta.unsqueeze(0)
        core_attn_out = self.attn(forward_batch, mixed_qkv=mixed_qkv, a=raw_g, b=beta)
        core_attn_out = self.norm(
            core_attn_out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim)
        )
        core_attn_out = core_attn_out.reshape(num_tokens, -1)
        output, _ = self.out_proj(core_attn_out)
        return output


# ---------------------------------------------------------------------------
# Full attention (gated, partial RoPE, QK Gemma-RMSNorm)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Sparse MoE block (routed experts + a sigmoid-gated shared expert). The standard
# TP / EP / DeepEP paths only; kept in this file so the model has no dependency on
# another model file.
# ---------------------------------------------------------------------------
class BerryLMMLP(nn.Module):
    """Dense SwiGLU MLP (the shared expert)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("down_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class BerryLMSparseMoeBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        config: BerryLMConfig,
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_parallel().tp_size
        self.layer_id = layer_id
        self.alt_stream = alt_stream
        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )
        self.num_experts = config.num_experts
        a2a = get_moe_a2a_backend()
        self._a2a_is_ep = a2a.is_deepep() or a2a.is_deepep_v2() or a2a.is_mori()

        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            renormalize=config.norm_topk_prob,
            layer_id=layer_id,
        )
        # The shared-expert add reads hidden_states after the experts -> no inplace MoE.
        self.experts = get_moe_impl_class(quant_config)(
            layer_id=self.layer_id,
            top_k=config.num_experts_per_tok,
            num_experts=config.num_experts + get_exec().moe.ep_num_redundant_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
            routing_method_type=RoutingMethodType.RenormalizeNaive,
            inplace=config.shared_expert_intermediate_size <= 0,
        )
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )
        if config.shared_expert_intermediate_size > 0:
            self.shared_expert = BerryLMMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_expert", prefix),
                **(
                    dict(tp_rank=0, tp_size=1)
                    if (
                        self._a2a_is_ep
                        or a2a.is_flashinfer()
                        or a2a.is_flashinfer_megamoe()
                    )
                    else {}
                ),
            )
        else:
            self.shared_expert = None
        self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)

        if self._a2a_is_ep:
            # TODO: tp < ep is not supported yet
            self.ep_size = get_parallel().moe_ep_size
            self.num_experts = (
                config.num_experts + get_exec().moe.ep_num_redundant_experts
            )
            self.top_k = config.num_experts_per_tok

    def get_moe_weights(self):
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ["correction_bias"]
            and filter_moe_weight_param_global_expert(
                name, x, self.experts.num_local_experts
            )
        ]

    def _forward_shared_experts(
        self, hidden_states: torch.Tensor, apply_gate: bool = True
    ):
        if self.shared_expert is None:
            return None
        shared_output = self.shared_expert(hidden_states)
        if apply_gate:
            shared_output = (
                F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_output
            )
        return shared_output

    def _forward_router_experts(self, hidden_states: torch.Tensor):
        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        return self.experts(hidden_states, topk_output)

    def _forward_deepep(self, hidden_states: torch.Tensor, forward_batch: ForwardBatch):
        shared_output = None
        if hidden_states.shape[0] > 0:
            router_logits, _ = self.gate(hidden_states)
            shared_output = self._forward_shared_experts(hidden_states)
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
            final_hidden_states.add_(shared_output)
        return final_hidden_states

    def forward_normal_dual_stream(
        self, hidden_states: torch.Tensor, use_fused_gate: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Shared expert on the alternate stream, routed experts on the current one
        # (CUDA-graph capture path).
        current_stream = torch.cuda.current_stream()
        self.alt_stream.wait_stream(current_stream)
        shared_output = (
            self._forward_shared_experts(
                hidden_states.clone(), apply_gate=not use_fused_gate
            )
            if self.shared_expert is not None
            else None
        )
        with torch.cuda.stream(self.alt_stream):
            router_output = self._forward_router_experts(hidden_states)
        current_stream.wait_stream(self.alt_stream)
        return router_output, shared_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self._a2a_is_ep:
            return self._forward_deepep(hidden_states, forward_batch)

        # sigmoid(gate) * shared + routed in one fused kernel
        use_fused_gate = self.shared_expert_gate is not None

        if hidden_states.shape[0] == 0:
            # M=0 guard for idle DP ranks: still call self.experts() so the rank takes
            # part in the all-to-all collective.
            shared_output = None
            topk_output = self.topk.empty_topk_output(hidden_states.device)
            final_hidden_states = self.experts(hidden_states, topk_output)
        elif (
            self.alt_stream is not None
            and get_is_capture_mode()
            and not torch.compiler.is_compiling()
        ):
            final_hidden_states, shared_output = self.forward_normal_dual_stream(
                hidden_states, use_fused_gate=use_fused_gate
            )
        else:
            shared_output = self._forward_shared_experts(
                hidden_states, apply_gate=not use_fused_gate
            )
            final_hidden_states = self._forward_router_experts(hidden_states)

        if shared_output is not None:
            if use_fused_gate:
                fused_gate_sigmoid_mul_add(
                    hidden_states,
                    self.shared_expert_gate.weight.squeeze(),
                    shared_output,
                    final_hidden_states,
                )
            else:
                final_hidden_states += shared_output
        if (
            self.tp_size > 1
            and not should_skip_post_experts_all_reduce(is_tp_path=True)
            and not get_moe_a2a_backend().is_flashinfer()
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_dim)


class BerryLMAttention(nn.Module):
    def __init__(
        self,
        config: BerryLMConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attn_tp_rank = get_parallel().attn_tp_rank
        self.attn_tp_size = get_parallel().attn_tp_size
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % self.attn_tp_size == 0
        self.num_heads = self.total_num_heads // self.attn_tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= self.attn_tp_size:
            assert self.total_num_kv_heads % self.attn_tp_size == 0
        else:
            assert self.attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.attn_tp_size)
        self.head_dim = config.head_dim or (self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.layer_id = layer_id
        self.alt_stream = alt_stream
        self.attn_output_gate = getattr(config, "attn_output_gate", True)

        rope = (
            getattr(config, "rope_parameters", None)
            or getattr(config, "rope_scaling", None)
            or {}
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=getattr(config, "max_position_embeddings", 8192),
            rope_scaling=rope,
            base=rope.get("rope_theta", 10000),
            partial_rotary_factor=rope.get("partial_rotary_factor", 1.0),
            is_neox_style=True,
            dtype=torch.get_default_dtype(),
        )
        attn_quant_config = (
            None
            if quant_config and quant_config.get_name() == "modelopt_fp4"
            else quant_config
        )
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=False,
            quant_config=attn_quant_config,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=attn_quant_config,
            reduce_results=False,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            prefix=add_prefix("o_proj", prefix),
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def _apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.alt_stream is not None and get_is_capture_mode():
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            q_by_head = self.q_norm(q.reshape(-1, self.head_dim))
            with torch.cuda.stream(self.alt_stream):
                k_by_head = self.k_norm(k.reshape(-1, self.head_dim))
            current_stream.wait_stream(self.alt_stream)
        else:
            q_by_head = self.q_norm(q.reshape(-1, self.head_dim))
            k_by_head = self.k_norm(k.reshape(-1, self.head_dim))
        return q_by_head.view(q.shape), k_by_head.view(k.shape)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        if self.attn_output_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )
            orig_shape = q_gate.shape[:-1]
            q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(*orig_shape, -1)
            gate = gate.reshape(*orig_shape, -1)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            gate = None
        q, k = self._apply_qk_norm(q, k)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        if gate is not None:
            attn_output = attn_output * torch.sigmoid(gate)
        output, _ = self.o_proj(attn_output)
        return output


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------


class BerryLMDecoderLayer(nn.Module):
    """One decoder layer: KDA linear attention or full attention, then sparse MoE.
    The AttnRes mixer is owned by the layer (``attn_res``) but applied in the model
    loop, on the residual stream, before ``input_layernorm``."""

    def __init__(
        self,
        config: BerryLMConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.layer_type = config.layers_block_type[layer_id]
        if self.layer_type == "linear_attention":
            if not getattr(config, "kda_insert", True):
                raise NotImplementedError(
                    "BerryLM without the KDA gate is not supported"
                )
            linear_attn_quant_config = (
                None
                if quant_config and quant_config.get_name() == "modelopt_fp4"
                else quant_config
            )
            self.linear_attn = BerryLMKDA(
                config,
                layer_id,
                linear_attn_quant_config,
                prefix=add_prefix("linear_attn", prefix),
            )
        else:
            self.self_attn = BerryLMAttention(
                config,
                layer_id,
                quant_config,
                prefix=add_prefix("self_attn", prefix),
                alt_stream=alt_stream,
            )
        self.mlp = BerryLMSparseMoeBlock(
            layer_id=layer_id,
            config=config,
            quant_config=quant_config,
            alt_stream=alt_stream,
            prefix=add_prefix("mlp", prefix),
        )
        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=True,
            is_previous_layer_sparse=True,
            is_next_layer_sparse=True,
        )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
            is_last_layer=(layer_id == config.num_hidden_layers - 1),
        )
        if config.attn_res_block_size > 0:
            self.attn_res = BerryLMAttnRes(
                config.hidden_size,
                eps=getattr(config, "attn_res_eps", 1e-6),
                gated=getattr(config, "attn_res_gated", True),
            )
        else:
            self.attn_res = None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        **kwargs,
    ):
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
        )
        if not forward_batch.forward_mode.is_idle():
            if self.layer_type == "linear_attention":
                hidden_states = self.linear_attn(hidden_states, forward_batch)
            else:
                hidden_states = self.self_attn(
                    positions=positions,
                    hidden_states=hidden_states,
                    forward_batch=forward_batch,
                )
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )
        mlp_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )
        fuse_mlp_allreduce = (
            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
                forward_batch
            )
        )
        with get_forward().scoped(
            fuse_mlp_allreduce=fuse_mlp_allreduce,
            mlp_reduce_scatter=mlp_reduce_scatter,
        ):
            hidden_states = self.mlp(hidden_states, forward_batch)
        if fuse_mlp_allreduce:
            hidden_states._sglang_needs_allreduce_fusion = True
        else:
            hidden_states, residual = self.layer_communicator.postprocess_layer(
                hidden_states, residual, forward_batch
            )
        return hidden_states, residual


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class BerryLMModel(nn.Module):
    def __init__(
        self,
        config: BerryLMConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.attn_res_block_size = int(config.attn_res_block_size)
        self.pp_group = get_pp_group()
        if self.pp_group.world_size > 1:
            raise NotImplementedError(
                "BerryLM does not support pipeline parallelism yet"
            )
        if self.attn_res_block_size > 0 and is_dp_attention_enabled():
            # The mixer needs every rank to hold the full residual stream of every
            # token; DP-attention scatter modes break that invariant.
            raise NotImplementedError(
                "BerryLM AttnRes does not support DP attention yet"
            )
        alt_stream = get_stream("alt") if _is_cuda else None
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            enable_tp=not is_dp_attention_enabled(),
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.layers, self._start_layer, self._end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: BerryLMDecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds
        residual = None
        block = self.attn_res_block_size
        # Residual streams committed at block boundaries (embeddings first).
        completed: List[torch.Tensor] = []
        for layer_idx in range(self._start_layer, self._end_layer):
            layer = self.layers[layer_idx]
            with get_global_expert_distribution_recorder().with_current_layer(
                layer_idx
            ):
                if block:
                    stream = (
                        hidden_states if residual is None else hidden_states + residual
                    )
                    if layer_idx == 0:
                        completed.append(stream)
                    hidden_states = layer.attn_res(stream, completed)
                    residual = None
                hidden_states, residual = layer(
                    positions=positions,
                    hidden_states=hidden_states,
                    residual=residual,
                    forward_batch=forward_batch,
                )
                if block and (layer_idx + 1) % block == 0:
                    completed.append(hidden_states + residual)
        if hidden_states.shape[0] != 0:
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class BerryLMForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        config: BerryLMConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = BerryLMModel(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
        self.logits_processor = LogitsProcessor(config)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )
        # transformers-v5 fused expert tensors: experts.gate_up_proj [E, 2I, D],
        # experts.down_proj [E, D, I].
        fused_expert_params_mapping = [
            ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
            ("experts.w2_weight", "experts.down_proj", 0, "w2"),
        ]
        ignore_suffixes = (
            ".bias",
            "_bias",
            ".k_scale",
            "_k_scale",
            ".v_scale",
            "_v_scale",
            ".weight_scale",
            "_weight_scale",
            ".input_scale",
            "_input_scale",
        )
        num_experts = self.config.num_experts
        is_fused_expert = False

        def load_fused_expert_weights(name, params_dict, loaded_weight, shard_id):
            param = params_dict[name]
            weight_loader = param.weight_loader
            for expert_id in range(num_experts):
                weight_loader(
                    param, loaded_weight[expert_id], name, shard_id, expert_id
                )

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name or "mtp" in name or "visual" in name:
                continue
            if "language_model" in name:
                name = name.replace("model.language_model.", "model.")
            if "experts.gate_up_proj" in name or "experts.down_proj" in name:
                is_fused_expert = True
                expert_params_mapping = fused_expert_params_mapping
            if ".linear_attn.in_proj_qkv." in name or ".linear_attn.in_proj_z." in name:
                # in_proj_qkv [2K+V, D] -> partitions 0..2 (split here: the v1 loader takes one index),
                # in_proj_z -> 3 of the merged GEMM; in_proj_b is its own layer (default loader)
                key_dim = (
                    self.config.linear_num_key_heads * self.config.linear_key_head_dim
                )
                value_dim = (
                    self.config.linear_num_value_heads
                    * self.config.linear_value_head_dim
                )
                if ".in_proj_qkv." in name:
                    name = name.replace(".in_proj_qkv.", ".in_proj_qkvz.")
                    param = params_dict[name]
                    # block-quantized checkpoints ship `weight_scale_inv` with one row per
                    # 128-row block: split it in the same proportions as the weight
                    scale = loaded_weight.shape[0] / (2 * key_dim + value_dim)
                    sizes = [
                        int(key_dim * scale),
                        int(key_dim * scale),
                        int(value_dim * scale),
                    ]
                    for sid, part in enumerate(loaded_weight.split(sizes, dim=0)):
                        param.weight_loader(param, part, sid)
                else:
                    name = name.replace(".in_proj_z.", ".in_proj_qkvz.")
                    param = params_dict[name]
                    param.weight_loader(param, loaded_weight, 3)
                loaded_params.add(name)
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # ".linear_attn." tensors are never stacked here (f_up_proj contains "up_proj").
                if (
                    weight_name not in name
                    or "mlp.experts" in name
                    or ".linear_attn." in name
                ):
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(ignore_suffixes) and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                for (
                    param_name,
                    weight_name,
                    expert_id,
                    shard_id,
                ) in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    if is_fused_expert:
                        if "experts.gate_up_proj" in name:
                            w1, w3 = loaded_weight.chunk(2, dim=-2)
                            load_fused_expert_weights(
                                name_mapped, params_dict, w1, "w1"
                            )
                            load_fused_expert_weights(
                                name_mapped, params_dict, w3, "w3"
                            )
                        else:
                            load_fused_expert_weights(
                                name_mapped, params_dict, loaded_weight, shard_id
                            )
                    else:
                        if (
                            name_mapped.endswith(ignore_suffixes)
                            and name_mapped not in params_dict
                        ):
                            continue
                        param = params_dict[name_mapped]
                        param.weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                    name = name_mapped
                    break
                else:
                    if is_expert_weight:
                        continue
                    if name.endswith(ignore_suffixes) and name not in params_dict:
                        continue
                    if name not in params_dict:
                        logger.warning(f"Parameter {name} not found in params_dict")
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_experts,
            num_groups=None,
        )


EntryClass = BerryLMForCausalLM
