# Adapted from the DFlash reference implementation (HF) but implemented with
# SGLang primitives (RadixAttention + SGLang KV cache). Most drafts borrow the
# target embedding and LM head; Nemotron 3.5 drafts carry their own embedding.

from __future__ import annotations

import itertools
import logging
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.kernels.ops.speculative.dflash import selector_walk_triton
from sglang.srt.configs.laguna import normalize_gating
from sglang.srt.distributed.communication_op import tensor_model_parallel_all_gather
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import (
    LogitsProcessorOutput,
    should_apply_lm_head_quant_method,
)
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from sglang.srt.models.utils import apply_qk_norm
from sglang.srt.runtime_context import get_parallel, get_spec
from sglang.srt.speculative.dflash_utils import (
    DFlashKDAConfig,
    can_dflash_slice_qkv_weight,
    get_dflash_attention_modes,
    get_dflash_attention_sliding_window_size,
    get_dflash_layer_types,
    is_dense_head_weight,
    is_nemotron_35_draft_config,
    parse_dflash_draft_config,
    parse_dflash_kda_config,
)
from sglang.srt.utils import is_npu, set_weight_attrs
from sglang.srt.utils.common import get_compiler_backend
from sglang.srt.utils.hf_transformers_utils import get_rope_config

_is_npu = is_npu()
if _is_npu:
    from sgl_kernel_npu.norm.split_qkv_rmsnorm_rope import split_qkv_rmsnorm_rope
logger = logging.getLogger(__name__)

try:
    from flashinfer import top_k as _flashinfer_top_k
except ImportError:
    _flashinfer_top_k = None


def _radix_topk(scores: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # The selector's largest single cost: it reads the whole logits tensor.
    if _flashinfer_top_k is not None:
        return _flashinfer_top_k(scores, k, sorted=True, deterministic=True)
    return torch.topk(scores, k, dim=-1)


def _logical_linear_weight_shape(
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
    *,
    output_features: int,
) -> Tuple[int, ...]:
    """Return a checkpoint linear weight shape in logical elements."""
    loaded_shape = tuple(loaded_weight.shape)
    pack_factor = getattr(param, "pack_factor", None)
    if pack_factor is None or loaded_shape != tuple(param.shape):
        return loaded_shape

    logical_numel = int(loaded_weight.numel() * pack_factor)
    if logical_numel % output_features == 0:
        return (output_features, logical_numel // output_features)
    return (logical_numel,)


def _project_candidate_logits(
    hidden: torch.Tensor, lm_head: nn.Module, *, num_org: int, use_quant_head: bool
) -> torch.Tensor:
    """Project draft hiddens through the target head, restricted to the org vocab."""
    if not use_quant_head:
        weight = lm_head.weight
        return torch.matmul(hidden.to(weight.dtype), weight[:num_org].T)
    # A packed weight can't be row-sliced to the org vocab like the dense path,
    # and flashinfer's radix top-k rejects the crop view (non-contiguous), so
    # mask the padded tail out of the top-k instead.
    logits = lm_head.quant_method.apply(lm_head, hidden, None).contiguous()
    if logits.shape[-1] > num_org:
        logits[:, num_org:] = float("-inf")
    return logits


def _get_dflash_attention_type(config, *, default: AttentionType) -> AttentionType:
    """Honor explicit causality while preserving legacy layer defaults."""
    text_config = config.get_text_config()
    is_causal = getattr(text_config, "is_causal", None)
    if is_causal is None:
        return default
    return AttentionType.DECODER if is_causal else AttentionType.ENCODER_ONLY


def _get_dflash_layer_attention_params(
    config, layer_id: int
) -> Tuple[int, AttentionType]:
    layer_types = get_dflash_layer_types(config)
    if layer_types is None:
        return -1, AttentionType.ENCODER_ONLY
    if layer_id >= len(layer_types):
        raise ValueError(
            "DFLASH config.layer_types must contain one entry per draft layer. "
            f"Got {len(layer_types)} entries, layer_id={layer_id}."
        )

    layer_type = layer_types[layer_id]
    if layer_type == "full_attention":
        return -1, _get_dflash_attention_type(
            config, default=AttentionType.ENCODER_ONLY
        )
    if layer_type == "sliding_attention":
        sliding_window_size = get_dflash_attention_sliding_window_size(config)
        assert sliding_window_size is not None
        return sliding_window_size, _get_dflash_attention_type(
            config, default=AttentionType.DECODER
        )
    raise ValueError(
        f"Unsupported DFLASH draft layer type. layer_types[{layer_id}]={layer_type!r}."
    )


class DFlashAttention(nn.Module):
    def __init__(
        self, config, layer_id: int, quant_config=None, prefix: str = ""
    ) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        tp_size = int(get_parallel().tp_size)
        total_num_heads = int(config.num_attention_heads)
        total_num_kv_heads = int(
            getattr(config, "num_key_value_heads", total_num_heads)
        )
        head_dim = int(getattr(config, "head_dim", hidden_size // total_num_heads))

        self.hidden_size = hidden_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        assert self.total_num_heads % tp_size == 0, (
            f"DFlashAttention requires total_num_heads divisible by tp_size. "
            f"total_num_heads={self.total_num_heads}, tp_size={tp_size}."
        )
        self.num_heads = self.total_num_heads // tp_size
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0, (
                f"DFlashAttention requires total_num_kv_heads divisible by tp_size when >= tp_size. "
                f"total_num_kv_heads={self.total_num_kv_heads}, tp_size={tp_size}."
            )
        else:
            assert tp_size % self.total_num_kv_heads == 0, (
                f"DFlashAttention requires tp_size divisible by total_num_kv_heads when total_num_kv_heads < tp_size. "
                f"total_num_kv_heads={self.total_num_kv_heads}, tp_size={tp_size}."
            )
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * head_dim
        self.kv_size = self.num_kv_heads * head_dim

        attention_bias = bool(getattr(config, "attention_bias", False))
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj" if prefix else "qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj" if prefix else "o_proj",
        )

        # Per-head Q/K RMSNorm, matching HF Qwen3.
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

        rope_theta, rope_scaling = get_rope_config(config)
        rope_is_neox_style = bool(
            getattr(
                config, "rope_is_neox_style", getattr(config, "is_neox_style", True)
            )
        )
        max_position_embeddings = int(getattr(config, "max_position_embeddings", 32768))
        self.rotary_emb = get_rope(
            head_dim,
            rotary_dim=head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=rope_is_neox_style,
        )

        self.scaling = head_dim**-0.5
        rotary = self.rotary_emb
        self.use_table_qk_norm_rope = (
            not _is_npu
            and hasattr(rotary, "cos_sin_cache")
            and getattr(rotary, "rotary_dim", None) == head_dim
            and getattr(rotary, "is_neox_style", False)
        )
        self.sliding_window_size, self.attn_type = _get_dflash_layer_attention_params(
            config, layer_id
        )
        self.attention_sink_bias = None
        if is_nemotron_35_draft_config(config) and bool(
            getattr(config, "attention_sink_bias", False)
        ):
            draft_attention_backend = get_spec().speculative_draft_attention_backend
            if draft_attention_backend != "trtllm_mha":
                raise ValueError(
                    "Nemotron 3.5 DSpark attention sinks require "
                    "--speculative-draft-attention-backend trtllm_mha, "
                    f"got {draft_attention_backend!r}."
                )
            self.attention_sink_bias = nn.Parameter(
                torch.empty(self.num_heads, dtype=torch.float32), requires_grad=False
            )
            set_weight_attrs(
                self.attention_sink_bias,
                {"weight_loader": sharded_weight_loader(0)},
            )
        self.attn = RadixAttention(
            num_heads=self.num_heads,
            head_dim=head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=self.sliding_window_size,
            attn_type=self.attn_type,
        )

    def forward_prepare_npu(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)

        if self.attn.layer_id == 0:
            self.rotary_emb.get_cos_sin_with_position(positions)
        q, k, v = split_qkv_rmsnorm_rope(
            qkv,
            self.rotary_emb.position_sin,
            self.rotary_emb.position_cos,
            self.q_size,
            self.kv_size,
            self.head_dim,
            eps=self.q_norm.variance_epsilon,
            q_weight=self.q_norm.weight,
            k_weight=self.k_norm.weight,
            q_bias=getattr(self.q_norm, "bias", None),
            k_bias=getattr(self.k_norm, "bias", None),
        )
        return q, k, v

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        if _is_npu:
            q, k, v = self.forward_prepare_npu(positions, hidden_states)
        elif self.use_table_qk_norm_rope and qkv.dtype == torch.bfloat16:
            from sglang.srt.speculative.dflash_utils import table_qk_norm_rope_

            table_qk_norm_rope_(
                qkv,
                positions,
                self.q_norm.weight,
                self.k_norm.weight,
                self.rotary_emb.cos_sin_cache,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.q_norm.variance_epsilon,
            )
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q, k = apply_qk_norm(q, k, self.q_norm, self.k_norm, self.head_dim)
            q, k = self.rotary_emb(positions, q, k)
        if self.attention_sink_bias is None:
            attn_output = self.attn(q, k, v, forward_batch)
        else:
            attn_output = self.attn(
                q, k, v, forward_batch, sinks=self.attention_sink_bias
            )
        attn_output = self.apply_attention_output(attn_output, hidden_states)
        output, _ = self.o_proj(attn_output)
        return output

    def apply_attention_output(
        self, attn_output: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        return attn_output

    def kv_proj_only(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project hidden_states to K/V only (skip Q).

        This is used by DFlash to materialize ctx tokens into the draft KV cache:
        we only need K/V for the cached tokens; Q is never consumed.
        """
        # Fast path for unquantized weights: slice the fused QKV weight and run one GEMM.
        can_slice_qkv_weight, _ = can_dflash_slice_qkv_weight(self.qkv_proj)
        if can_slice_qkv_weight:
            kv_slice = slice(self.q_size, self.q_size + 2 * self.kv_size)
            weight = self.qkv_proj.weight[kv_slice]
            bias = (
                self.qkv_proj.bias[kv_slice] if self.qkv_proj.bias is not None else None
            )
            kv = F.linear(hidden_states, weight, bias)
            k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
            return k, v

        # Fallback: compute full QKV and discard Q (keeps compatibility with quantized weights).
        qkv, _ = self.qkv_proj(hidden_states)
        _, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        return k, v

    def apply_k_norm(self, k: torch.Tensor) -> torch.Tensor:
        k_by_head = k.reshape(-1, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        return k_by_head.view_as(k)

    def apply_k_rope(self, positions: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        # Match K shape so RoPE kernel head-count check passes on all backends.
        dummy_q = k.new_empty(k.shape)
        _, k = self.rotary_emb(positions, dummy_q, k)
        return k


class DFlashKDAShortConvolution(nn.Module):
    """Checkpoint-compatible causal depthwise convolution for KDA."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.weight = nn.Parameter(torch.empty(channels, self.kernel_size))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        channels_first = inputs.transpose(1, 2)
        channels_first = F.pad(channels_first, (self.kernel_size - 1, 0))
        convolved = F.conv1d(
            channels_first,
            self.weight.unsqueeze(1),
            bias=None,
            groups=self.weight.shape[0],
        )
        return F.silu(convolved.transpose(1, 2))


class DFlashKDAGatedRMSNorm(nn.Module):
    """RMSNorm followed by KDA's sigmoid output gate."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = float(eps)

    def forward(self, inputs: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        variance = inputs.float().square().mean(dim=-1, keepdim=True)
        normalized = inputs * torch.rsqrt(variance + self.eps).to(inputs.dtype)
        return normalized * self.weight.to(inputs.dtype) * torch.sigmoid(gate)


def _kda_kernel_beta(raw_beta: torch.Tensor) -> torch.Tensor:
    """Beta as the Triton KDA kernels expect it: post-sigmoid, fp32.

    ``chunk_kda`` activates the decay gate in-kernel from the raw gate, A_log,
    dt_bias and lower_bound, but it does NOT apply sigmoid to beta (the
    ``beta_is_raw`` keyword is swallowed by ``**kwargs``). Feeding the raw
    ``b_proj`` output silently changes the delta rule; SpecForge's FLA call
    uses ``use_beta_sigmoid_in_kernel=True``, so the sigmoid must happen here.
    """
    return torch.sigmoid(raw_beta.float())


def reference_dflash_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_gate: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: Optional[float],
    *,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
):
    """Small-sequence KDA oracle and non-CUDA fallback.

    ``initial_state`` (``[B, H, K, V]`` fp32) seeds the recurrence and
    ``output_final_state`` also returns the state after the last step, the
    same contract as SpecForge's ``reference_kda``. Note the FLA/Triton pool
    layout is ``[.., H, V, K]``: transpose the last two dims when crossing.
    """
    q = F.normalize(q.float(), dim=-1).to(q.dtype)
    k = F.normalize(k.float(), dim=-1).to(k.dtype)
    beta = torch.sigmoid(beta.float()).to(q.dtype)

    gate_input = raw_gate.float() + dt_bias.view(1, 1, *raw_gate.shape[-2:])
    decay_scale = A_log.float().exp().view(1, 1, -1, 1)
    if lower_bound is None:
        log_decay = -decay_scale * F.softplus(gate_input)
    else:
        log_decay = float(lower_bound) * torch.sigmoid(decay_scale * gate_input)

    if initial_state is None:
        state = torch.zeros(
            q.shape[0],
            q.shape[2],
            q.shape[3],
            v.shape[3],
            dtype=torch.float32,
            device=q.device,
        )
    else:
        state = initial_state.to(device=q.device, dtype=torch.float32)
    outputs = []
    score_scale = q.shape[-1] ** -0.5
    for step in range(q.shape[1]):
        state = state * log_decay[:, step].exp().unsqueeze(-1)
        step_key = k[:, step].float()
        step_value = v[:, step].float()
        prediction = torch.einsum("bhd,bhdv->bhv", step_key, state)
        delta = (step_value - prediction) * beta[:, step].float().unsqueeze(-1)
        state = state + torch.einsum("bhd,bhv->bhdv", step_key, delta)
        output = torch.einsum("bhd,bhdv->bhv", q[:, step].float(), state)
        outputs.append((output * score_scale).to(q.dtype))
    if outputs:
        output = torch.stack(outputs, dim=1)
    else:
        output = q.new_zeros((q.shape[0], 0, q.shape[2], v.shape[3]))
    if output_final_state:
        return output, state
    return output


class DFlashKDAAttention(nn.Module):
    """KDA recurrent attention over DFlash proposal blocks.

    ``linear_attn_config.context_state`` selects where a block's recurrent
    state starts (SpecForge PR #836 semantics):

    * ``reset``: every block is an independent sequence starting from a zero
      state; target context reaches the layer only through the hybrid stack's
      KV-cache attention layers.
    * ``scan``: the recurrence first consumes the target context. A per-request
      running state (and the last ``kernel_size - 1`` context rows for the
      short convolution) lives in a slot pool indexed by ``req_pool_indices``;
      the worker advances it with every newly verified context slice
      (:meth:`advance_context_state`) and each block forward starts from it.
      The context scan runs through the same raw-gate ``chunk_kda`` kernel as
      the block, which writes the final state back into the pool slot.
    """

    is_dflash_kda = True

    def __init__(
        self, config, layer_id: int, quant_config=None, prefix: str = ""
    ) -> None:
        super().__init__()
        del layer_id, prefix
        tp_size = int(get_parallel().tp_size)
        if tp_size != 1:
            raise ValueError(
                "DFLASH KDA draft attention currently requires tp_size=1, "
                f"got tp_size={tp_size}. Run one independent draft server per GPU."
            )
        if quant_config is not None:
            raise ValueError(
                "DFLASH KDA draft attention currently requires unquantized "
                "draft weights."
            )

        spec = parse_dflash_kda_config(config)
        if spec is None:
            raise ValueError("DFlashKDAAttention requires a KDA draft config.")
        self.kda_config: DFlashKDAConfig = spec
        self.hidden_size = int(config.hidden_size)
        self.head_dim = spec.head_dim
        self.num_heads = spec.num_heads
        self.block_size = parse_dflash_draft_config(
            draft_hf_config=config
        ).resolve_block_size(default=16)
        self.lower_bound = spec.gate_lower_bound
        projection_size = spec.projection_size

        self.q_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        self.q_conv1d = DFlashKDAShortConvolution(
            projection_size, spec.short_conv_kernel_size
        )
        self.k_conv1d = DFlashKDAShortConvolution(
            projection_size, spec.short_conv_kernel_size
        )
        self.v_conv1d = DFlashKDAShortConvolution(
            projection_size, spec.short_conv_kernel_size
        )

        self.A_log = nn.Parameter(
            torch.log(torch.empty(spec.num_heads, dtype=torch.float32).uniform_(1, 16))
        )
        self.f_a_proj = nn.Linear(self.hidden_size, spec.head_dim, bias=False)
        self.f_b_proj = nn.Linear(spec.head_dim, projection_size, bias=False)
        self.dt_bias = nn.Parameter(torch.zeros(projection_size, dtype=torch.float32))
        self.b_proj = nn.Linear(self.hidden_size, spec.num_heads, bias=False)
        if spec.use_full_rank_gate:
            self.g_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        else:
            self.g_a_proj = nn.Linear(self.hidden_size, spec.head_dim, bias=False)
            self.g_b_proj = nn.Linear(spec.head_dim, projection_size, bias=False)
        self.o_norm = DFlashKDAGatedRMSNorm(
            spec.head_dim, eps=float(config.rms_norm_eps)
        )
        self.o_proj = nn.Linear(projection_size, self.hidden_size, bias=False)

        # Context-scanning policy: per-request running state, allocated by
        # init_context_state() once the worker knows the request-slot count.
        self.scans_context = bool(spec.scans_context)
        self.is_dflash_kda_scan = self.scans_context
        self.conv_window = int(spec.short_conv_kernel_size) - 1
        self._ctx_state: Optional[torch.Tensor] = None  # [slots, H, V, K] fp32
        self._ctx_tail: Optional[torch.Tensor] = None  # [slots, window, hidden]

    def set_block_size(self, block_size: int) -> None:
        self.block_size = int(block_size)

    def _output_gate(self, blocks: torch.Tensor) -> torch.Tensor:
        if self.kda_config.use_full_rank_gate:
            return self.g_proj(blocks)
        return self.g_b_proj(self.g_a_proj(blocks))

    def _gates(self, rows: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Raw decay gate ``[..., H, D]`` and raw beta ``[..., H]`` for rows."""
        leading = tuple(rows.shape[:-1])
        raw_gate = self.f_b_proj(self.f_a_proj(rows)).reshape(
            *leading, self.num_heads, self.head_dim
        )
        beta = self.b_proj(rows).reshape(*leading, self.num_heads)
        return raw_gate, beta

    @staticmethod
    def _conv_with_left_rows(
        conv: DFlashKDAShortConvolution,
        proj: nn.Linear,
        left_rows: torch.Tensor,
        rows: torch.Tensor,
    ) -> torch.Tensor:
        """Convolve ``rows`` as if ``left_rows`` immediately preceded them."""
        joined = torch.cat((proj(left_rows), proj(rows)), dim=1)
        return conv(joined)[:, left_rows.shape[1] :]

    # -- context-scanning state pool -------------------------------------
    def init_context_state(
        self, max_slots: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        """Allocate the per-request running state for ``max_slots`` request slots."""
        if not self.scans_context:
            return
        if max_slots < 1:
            raise ValueError(f"DFLASH KDA scan needs max_slots >= 1, got {max_slots}.")
        self._ctx_state = torch.zeros(
            (int(max_slots), self.num_heads, self.head_dim, self.head_dim),
            dtype=torch.float32,
            device=device,
        )
        self._ctx_tail = torch.zeros(
            (int(max_slots), self.conv_window, self.hidden_size),
            dtype=dtype,
            device=device,
        )

    def _require_context_state(self) -> torch.Tensor:
        if not self.scans_context:
            raise RuntimeError("DFLASH KDA layer does not scan context.")
        if self._ctx_state is None or self._ctx_tail is None:
            raise RuntimeError(
                "DFLASH KDA scan state is not allocated; call init_context_state() "
                "with the request-slot count before serving."
            )
        return self._ctx_state

    def reset_context_state(self, slots: torch.Tensor) -> None:
        """Forget the running context of the given request slots."""
        state = self._require_context_state()
        slots = slots.to(device=state.device, dtype=torch.int64)
        state[slots] = 0
        self._ctx_tail[slots] = 0

    def advance_context_state(
        self,
        slots: torch.Tensor,
        rows: torch.Tensor,
        row_lens: torch.Tensor,
        reset_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Consume newly verified context rows for a batch of request slots.

        ``rows`` is ``[sum(row_lens), hidden_size]``: the rows of slot ``i``
        are contiguous, in ascending position order, and follow the rows of
        slot ``i - 1``. Slots flagged in ``reset_mask`` start a new request
        (their state and conv tail are zeroed first). States are updated in
        place; nothing is returned.
        """
        state = self._require_context_state()
        device = state.device
        slots = slots.to(device=device, dtype=torch.int64)
        lens = [int(n) for n in row_lens.tolist()]
        if slots.numel() != len(lens):
            raise ValueError(
                "DFLASH KDA scan: slots and row_lens disagree: "
                f"{slots.numel()} vs {len(lens)}."
            )
        if reset_mask is not None:
            reset_mask = reset_mask.to(device=device, dtype=torch.bool)
            if bool(reset_mask.any()):
                self.reset_context_state(slots[reset_mask])
        if rows.device != device:
            rows = rows.to(device)
        rows = rows.to(self._ctx_tail.dtype)

        keys, values, gates, betas, seq_slots, seq_lens = [], [], [], [], [], []
        start = 0
        for index, length in enumerate(lens):
            if length == 0:
                continue
            slice_rows = rows[start : start + length].unsqueeze(0)  # [1, n, hidden]
            start += length
            slot = slots[index]
            tail = self._ctx_tail[slot].unsqueeze(0)  # [1, window, hidden]
            shape = (1, length, self.num_heads, self.head_dim)
            keys.append(
                self._conv_with_left_rows(
                    self.k_conv1d, self.k_proj, tail, slice_rows
                ).reshape(shape)
            )
            values.append(
                self._conv_with_left_rows(
                    self.v_conv1d, self.v_proj, tail, slice_rows
                ).reshape(shape)
            )
            raw_gate, beta = self._gates(slice_rows)
            gates.append(raw_gate)
            betas.append(beta)
            seq_slots.append(slot)
            seq_lens.append(length)
            if self.conv_window:
                self._ctx_tail[slot] = torch.cat((tail[0], slice_rows[0]), dim=0)[
                    -self.conv_window :
                ]
        if start != int(rows.shape[0]):
            raise ValueError(
                f"DFLASH KDA scan: rows has {int(rows.shape[0])} entries but "
                f"row_lens sum to {start}."
            )
        if not seq_lens:
            return

        k = torch.cat(keys, dim=1)
        v = torch.cat(values, dim=1)
        raw_gate = torch.cat(gates, dim=1)
        beta = torch.cat(betas, dim=1)
        slot_index = torch.stack(seq_slots)
        if device.type == "cuda":
            from sglang.kernels.ops.attention.fla.kda import chunk_kda

            cu_seqlens = torch.tensor(
                [0, *itertools.accumulate(seq_lens)], device=device, dtype=torch.int64
            )
            # The state does not depend on queries; reuse k so the l2norm has
            # a finite input. With initial_state + initial_state_indices the
            # kernel reads h0 from, and writes the final state back into, the
            # addressed pool slots.
            chunk_kda(
                q=k,
                k=k,
                v=v,
                g=raw_gate,
                beta=_kda_kernel_beta(beta),
                use_qk_l2norm_in_kernel=True,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                lower_bound=self.lower_bound,
                initial_state=state,
                initial_state_indices=slot_index,
                cu_seqlens=cu_seqlens,
            )
            return
        offset = 0
        for slot, length in zip(seq_slots, seq_lens):
            sl = slice(offset, offset + length)
            offset += length
            # Reference layout is [B, H, K, V]; the pool is [slots, H, V, K].
            initial = state[slot].transpose(-1, -2).unsqueeze(0)
            _, final = reference_dflash_kda(
                k[:, sl],
                k[:, sl],
                v[:, sl],
                raw_gate[:, sl],
                beta[:, sl],
                self.A_log,
                self.dt_bias,
                self.lower_bound,
                initial_state=initial,
                output_final_state=True,
            )
            state[slot] = final[0].transpose(-1, -2)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        del positions
        if hidden_states.ndim != 2:
            raise ValueError(
                "DFLASH KDA expects flattened [batch * block_size, hidden_size] "
                f"states, got shape={tuple(hidden_states.shape)}."
            )
        token_count = int(hidden_states.shape[0])
        if token_count % self.block_size:
            raise ValueError(
                "DFLASH KDA token count must be divisible by block_size; "
                f"got token_count={token_count}, block_size={self.block_size}."
            )

        blocks = hidden_states.reshape(-1, self.block_size, self.hidden_size)
        block_count = int(blocks.shape[0])
        projection_shape = (
            block_count,
            self.block_size,
            self.num_heads,
            self.head_dim,
        )
        initial_state = None
        if self.scans_context:
            state = self._require_context_state()
            slots = forward_batch.req_pool_indices
            if slots is None:
                raise RuntimeError(
                    "DFLASH KDA scan needs forward_batch.req_pool_indices."
                )
            slots = slots.to(device=state.device, dtype=torch.int64).reshape(-1)
            if int(slots.numel()) != block_count:
                raise ValueError(
                    "DFLASH KDA scan expects one proposal block per request; got "
                    f"{block_count} blocks for {int(slots.numel())} requests."
                )
            tail = self._ctx_tail[slots]  # [B, window, hidden]
            q = self._conv_with_left_rows(self.q_conv1d, self.q_proj, tail, blocks)
            k = self._conv_with_left_rows(self.k_conv1d, self.k_proj, tail, blocks)
            v = self._conv_with_left_rows(self.v_conv1d, self.v_proj, tail, blocks)
            # Clone: the kernel writes the post-block state back into whatever
            # it was handed, and block tokens are speculative.
            initial_state = state[slots].clone()
        else:
            q = self.q_conv1d(self.q_proj(blocks))
            k = self.k_conv1d(self.k_proj(blocks))
            v = self.v_conv1d(self.v_proj(blocks))
        q = q.reshape(projection_shape)
        k = k.reshape(projection_shape)
        v = v.reshape(projection_shape)
        raw_gate, beta = self._gates(blocks)

        if hidden_states.device.type == "cuda":
            from sglang.kernels.ops.attention.fla.kda import chunk_kda

            state_kwargs = {}
            if initial_state is not None:
                state_kwargs = dict(
                    initial_state=initial_state,
                    initial_state_indices=torch.arange(
                        block_count, device=initial_state.device, dtype=torch.int64
                    ),
                )
            attention_output = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=raw_gate,
                beta=_kda_kernel_beta(beta),
                use_qk_l2norm_in_kernel=True,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                lower_bound=self.lower_bound,
                **state_kwargs,
            )
        else:
            attention_output = reference_dflash_kda(
                q,
                k,
                v,
                raw_gate,
                beta,
                self.A_log,
                self.dt_bias,
                self.lower_bound,
                initial_state=(
                    None if initial_state is None else initial_state.transpose(-1, -2)
                ),
            )

        output_gate = self._output_gate(blocks).reshape(projection_shape)
        attention_output = self.o_norm(attention_output, output_gate)
        attention_output = self.o_proj(attention_output.flatten(-2))
        return attention_output.reshape(token_count, self.hidden_size)


class DFlashMLP(nn.Module):
    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        intermediate_size = int(getattr(config, "intermediate_size", 0))
        if intermediate_size <= 0:
            raise ValueError(
                f"Invalid intermediate_size={intermediate_size} for DFlash MLP."
            )

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix="gate_up_proj" if not prefix else f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix="down_proj" if not prefix else f"{prefix}.down_proj",
        )
        hidden_act = getattr(config, "hidden_act", "silu")
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported DFlash activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


@torch.compile(dynamic=True, backend=get_compiler_backend(), disable=_is_npu)
def _grouped_conv(hidden_states, delta, base, block_size, num_groups, group_size, taps):
    blocks = hidden_states.unflatten(-1, (num_groups, group_size))
    coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
    out = coefficients[:, 0] * blocks
    position = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    if block_size & (block_size - 1) == 0:
        position = position & (block_size - 1)
    else:
        position = position % block_size
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        out = out + coefficients[:, tap] * shifted * (position >= tap).view(-1, 1, 1)
    return out.flatten(-2)


class DFlashGroupedConv(nn.Module):
    """Grouped dynamic depthwise K-tap convolution across one DFlash block.

    Each sublayer is wrapped: `prepare` convolves its input and returns the kernel
    for `finish` to convolve its output, both from one projection of the input.
    """

    def __init__(
        self, hidden_size: int, block_size: int, taps: int, group_size: int
    ) -> None:
        super().__init__()
        if hidden_size % group_size:
            raise ValueError(
                f"DFLASH conv_group_size={group_size} must divide "
                f"hidden_size={hidden_size}."
            )
        hidden_size = int(hidden_size)
        self.block_size = int(block_size)
        self.taps = int(taps)
        self.group_size = int(group_size)
        self.num_groups = hidden_size // self.group_size
        # [input/output, tap, channel], the layout training exports.
        base_kernel = torch.zeros(2, self.taps, hidden_size)
        base_kernel[:, 0] = 1.0
        self.base_kernel = nn.Parameter(base_kernel)
        self.kernel_projection = nn.Linear(
            hidden_size, 2 * self.taps * self.num_groups, bias=False
        )

    def _convolve(self, hidden_states, delta, side: int) -> torch.Tensor:
        # Marked here, not inside: by the time the compiled function traces, the dim
        # is symbolic and the group index costs an integer div and mod per element.
        torch._dynamo.mark_static(hidden_states, 1)
        torch._dynamo.mark_static(delta, 1)
        torch._dynamo.mark_static(delta, 2)
        return _grouped_conv(
            hidden_states,
            delta,
            self.base_kernel[side],
            self.block_size,
            self.num_groups,
            self.group_size,
            self.taps,
        )

    def prepare(self, hidden_states: torch.Tensor):
        coefficients = self.kernel_projection(hidden_states).reshape(
            *hidden_states.shape[:-1], 2, self.taps, self.num_groups
        )
        return (
            self._convolve(hidden_states, coefficients[..., 0, :, :], side=0),
            coefficients[..., 1, :, :],
        )

    def finish(self, hidden_states: torch.Tensor, coefficients) -> torch.Tensor:
        return self._convolve(hidden_states, coefficients, side=1)


class DFlashDecoderLayer(nn.Module):
    attention_cls = DFlashAttention

    def __init__(
        self,
        config,
        layer_id: int,
        attention_conv: Optional[DFlashGroupedConv] = None,
        mlp_conv: Optional[DFlashGroupedConv] = None,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        attention_prefix = f"{prefix}.self_attn" if prefix else ""
        attention_modes = get_dflash_attention_modes(config)
        layer_mode = attention_modes[layer_id]
        if layer_mode == "kda":
            attention_class = DFlashKDAAttention
        else:
            attention_class = self.attention_cls
        self.self_attn = attention_class(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=attention_prefix,
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        mlp_prefix = f"{prefix}.mlp" if prefix else ""
        self.mlp = DFlashMLP(
            config=config, quant_config=quant_config, prefix=mlp_prefix
        )

        self.attention_conv = attention_conv
        self.mlp_conv = mlp_conv

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.numel() == 0:
            # Keep return types consistent for upstream callers.
            if residual is None:
                residual = hidden_states
            return hidden_states, residual

        # Pre-norm attention with fused residual+norm when possible (Qwen3-style).
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        attention_kernel = None
        if self.attention_conv is not None:
            hidden_states, attention_kernel = self.attention_conv.prepare(hidden_states)

        attn_out = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        if attention_kernel is not None:
            attn_out = self.attention_conv.finish(attn_out, attention_kernel)

        hidden_states, residual = self.post_attention_layernorm(attn_out, residual)

        mlp_kernel = None
        if self.mlp_conv is not None:
            hidden_states, mlp_kernel = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if mlp_kernel is not None:
            hidden_states = self.mlp_conv.finish(hidden_states, mlp_kernel)
        return hidden_states, residual


class DFlashDraftModel(nn.Module):
    """SGLang DFlash draft model with an optional Nemotron embedding.

    The checkpoint provides:
      - transformer weights for `layers.*`
      - `fc.weight`, `hidden_norm.weight` for projecting target context features
      - `norm.weight` for final normalization
    """

    decoder_layer_cls = DFlashDecoderLayer
    supports_fused_context_kv = True

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        self.config = config

        hidden_size = int(config.hidden_size)
        num_layers = int(config.num_hidden_layers)
        self.attention_modes = get_dflash_attention_modes(config)
        self.supports_fused_context_kv = (
            bool(type(self).supports_fused_context_kv)
            and "kda" not in self.attention_modes
        )
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))
        draft_config = self.draft_config = parse_dflash_draft_config(
            draft_hf_config=config
        )
        self.block_size = draft_config.resolve_block_size(default=16)
        self.candidate_selector: Optional[nn.Module] = None
        self.is_nemotron_35_draft = is_nemotron_35_draft_config(config)
        self.embed_tokens: Optional[VocabParallelEmbedding] = None
        if self.is_nemotron_35_draft:
            embed_prefix = f"{prefix}.embed_tokens" if prefix else "embed_tokens"
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                hidden_size,
                quant_config=quant_config,
                prefix=embed_prefix,
            )

        def grouped_conv():
            if not draft_config.conv_kernel_size:
                return None
            return DFlashGroupedConv(
                hidden_size,
                self.block_size,
                draft_config.conv_kernel_size,
                draft_config.conv_group_size,
            )

        self.layers = nn.ModuleList(
            [
                self.decoder_layer_cls(
                    config=config,
                    layer_id=i,
                    attention_conv=grouped_conv(),
                    mlp_conv=grouped_conv(),
                    quant_config=quant_config,
                    prefix=(
                        (f"{prefix}.layers.{i}" if prefix else f"layers.{i}")
                        if self.is_nemotron_35_draft
                        else ""
                    ),
                )
                for i in range(num_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # Project per-token target context features:
        # concat(K * hidden_size) -> hidden_size, where K is the number of target-layer
        # feature tensors concatenated per token (not necessarily equal to num_layers).
        if draft_config.num_target_layers is not None:
            target_num_layers = int(draft_config.num_target_layers)
        elif draft_config.target_layer_ids is not None:
            target_num_layers = max(draft_config.target_layer_ids) + 1
        else:
            target_num_layers = num_layers
        target_layer_ids = draft_config.resolve_target_layer_ids(
            target_num_layers=target_num_layers, draft_num_layers=num_layers
        )
        num_context_features = len(target_layer_ids)

        self.num_context_features = int(num_context_features)
        if self.is_nemotron_35_draft:
            fc_prefix = f"{prefix}.fc" if prefix else "fc"
            self.fc = ReplicatedLinear(
                self.num_context_features * hidden_size,
                hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=fc_prefix,
            )
        else:
            self.fc = nn.Linear(
                self.num_context_features * hidden_size, hidden_size, bias=False
            )
        self.hidden_norm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def set_block_size(self, block_size: int) -> None:
        """Adopt the block size the worker resolved.

        The convolutions are built from the checkpoint's block_size, which
        --speculative-num-draft-tokens may override; the layout they index
        depends on it, so the resolved value has to reach them.
        """
        self.block_size = int(block_size)
        for layer in self.layers:
            set_attention_block_size = getattr(layer.self_attn, "set_block_size", None)
            if set_attention_block_size is not None:
                set_attention_block_size(self.block_size)
            for conv in (layer.attention_conv, layer.mlp_conv):
                if conv is not None:
                    conv.block_size = self.block_size

    def iter_context_attention_layers(self):
        """Yield only draft layers that own a target-context KV cache."""
        for layer in self.layers:
            if not getattr(layer.self_attn, "is_dflash_kda", False):
                yield layer

    def iter_scan_kda_layers(self):
        """Yield draft layers whose KDA recurrence scans the target context."""
        for layer in self.layers:
            if getattr(layer.self_attn, "is_dflash_kda_scan", False):
                yield layer

    @property
    def has_scan_kda_layers(self) -> bool:
        return any(True for _ in self.iter_scan_kda_layers())

    def init_kda_context_state(
        self, max_slots: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        """Allocate the per-request running state of context-scanning KDA layers."""
        for layer in self.iter_scan_kda_layers():
            layer.self_attn.init_context_state(max_slots, device, dtype)

    def advance_kda_context(
        self,
        slots: torch.Tensor,
        ctx_hidden: torch.Tensor,
        row_lens: torch.Tensor,
        reset_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Feed newly verified context rows to every context-scanning KDA layer."""
        for layer in self.iter_scan_kda_layers():
            layer_ctx_hidden = self.prepare_context_hidden_for_kv(layer, ctx_hidden)
            layer.self_attn.advance_context_state(
                slots, layer_ctx_hidden, row_lens, reset_mask
            )

    def get_attention_sliding_window_size(self) -> Optional[int]:
        return get_dflash_attention_sliding_window_size(self.config)

    def get_input_embeddings(self) -> Optional[VocabParallelEmbedding]:
        return self.embed_tokens

    def prepare_context_hidden_for_kv(
        self, layer: DFlashDecoderLayer, ctx_hidden: torch.Tensor
    ) -> torch.Tensor:
        return ctx_hidden

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        """Project concatenated target-layer hidden states into draft hidden_size."""
        expected = int(
            self.fc.input_size if self.is_nemotron_35_draft else self.fc.in_features
        )
        if target_hidden.ndim != 2 or int(target_hidden.shape[-1]) != expected:
            raise ValueError(
                "DFLASH target_hidden feature dim mismatch. "
                f"Expected shape [N, {expected}] "
                f"(num_context_features={self.num_context_features}, hidden_size={int(self.config.hidden_size)}), "
                f"but got shape={tuple(target_hidden.shape)}. "
                "This usually means the target model is capturing a different number of layer features than "
                "the draft checkpoint/config expects."
            )
        projected = self.fc(target_hidden)
        if self.is_nemotron_35_draft:
            projected = projected[0]
        return self.hidden_norm(projected)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ) -> LogitsProcessorOutput:
        if input_embeds is None:
            if self.embed_tokens is not None:
                input_embeds = self.embed_tokens(input_ids)
            elif hasattr(self, "forward_embed"):
                input_embeds = self.forward_embed(input_ids)
            else:
                raise ValueError(
                    "DFlashDraftModel requires `input_embeds` (use the target "
                    "embedding)."
                )
        hidden_states = input_embeds
        residual: Optional[torch.Tensor] = None

        for layer in self.layers:
            hidden_states, residual = layer(
                positions, hidden_states, forward_batch, residual
            )

        if hidden_states.numel() != 0:
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)

        return LogitsProcessorOutput(
            next_token_logits=None,
            hidden_states=hidden_states,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, weight_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())

        # Alias the native export's "encoder." names.
        _VENDOR_ENCODER_ALIASES = {
            "encoder.fc.weight": "fc.weight",
            "encoder.output_norm_enc.weight": "hidden_norm.weight",
        }

        def resolve_param_name(name: str) -> Optional[str]:
            if name in params_dict:
                return name
            if name.startswith("model."):
                stripped_name = name[len("model.") :]
                if stripped_name in params_dict:
                    return stripped_name
            else:
                prefixed_name = f"model.{name}"
                if prefixed_name in params_dict:
                    return prefixed_name
            aliased_name = _VENDOR_ENCODER_ALIASES.get(name)
            if aliased_name is not None and aliased_name in params_dict:
                return aliased_name
            return None

        loaded_names = set()
        ignored_names = []
        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if f".{weight_name}." not in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                resolved_name = resolve_param_name(mapped_name)
                if resolved_name is None:
                    continue
                param = params_dict[resolved_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                loaded_names.add(resolved_name)
                break
            else:
                resolved_name = resolve_param_name(name)
                if resolved_name is None:
                    # Ignore unexpected weights (e.g., HF rotary caches).
                    ignored_names.append(name)
                    continue
                loaded_names.add(resolved_name)
                param = params_dict[resolved_name]
                if resolved_name.endswith("fc.weight"):
                    if self.is_nemotron_35_draft:
                        expected_shape = (
                            int(self.config.hidden_size),
                            int(self.num_context_features * self.config.hidden_size),
                        )
                        loaded_shape = _logical_linear_weight_shape(
                            param,
                            loaded_weight,
                            output_features=expected_shape[0],
                        )
                        shape_matches = loaded_shape == expected_shape or (
                            getattr(param, "pack_factor", None) is None
                            and tuple(loaded_weight.shape) == tuple(param.shape)
                        )
                    else:
                        expected_shape = tuple(param.shape)
                        loaded_shape = tuple(loaded_weight.shape)
                        shape_matches = loaded_shape == expected_shape
                    if not shape_matches:
                        raise ValueError(
                            "DFLASH fc.weight shape mismatch. This usually means the draft checkpoint's "
                            "number of context features (K) does not match this config. "
                            f"Expected fc.weight.shape={expected_shape} "
                            f"(num_context_features={self.num_context_features}, hidden_size={int(self.config.hidden_size)}), "
                            f"but got {loaded_shape} for weight '{name}'."
                        )
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
        missing_names = sorted(set(params_dict) - loaded_names)
        logger.info(
            "DFLASH draft load_weights: loaded=%d params, ignored=%d checkpoint tensors%s, "
            "params without checkpoint tensor=%d%s",
            len(loaded_names),
            len(ignored_names),
            f" (e.g. {ignored_names[:6]})" if ignored_names else "",
            len(missing_names),
            f" (e.g. {missing_names[:6]})" if missing_names else "",
        )


class DFlashLagunaAttention(DFlashAttention):
    """Laguna DFlash attention with the trained Laguna softplus gate."""

    def __init__(
        self, config, layer_id: int, quant_config=None, prefix: str = ""
    ) -> None:
        super().__init__(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=prefix,
        )
        hidden_size = int(config.hidden_size)
        total_num_heads = self.total_num_heads
        gating = normalize_gating(getattr(config, "gating", True))
        self.gating = gating
        self.gate_per_head = gating == "per-head"
        if self.gating == "disabled":
            self.g_proj = None
        else:
            g_out = (
                total_num_heads
                if self.gate_per_head
                else total_num_heads * self.head_dim
            )
            self.g_proj = ColumnParallelLinear(
                hidden_size,
                g_out,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.g_proj" if prefix else "g_proj",
            )

    def apply_attention_output(
        self, attn_output: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        if self.g_proj is None:
            return attn_output

        gate, _ = self.g_proj(hidden_states)
        gate = F.softplus(gate.float()).to(attn_output.dtype)
        if self.gate_per_head:
            attn_shape = attn_output.shape
            return (
                attn_output.view(*attn_shape[:-1], self.num_heads, self.head_dim)
                * gate.unsqueeze(-1)
            ).view(attn_shape)
        else:
            return attn_output * gate


class DFlashLagunaDecoderLayer(DFlashDecoderLayer):
    attention_cls = DFlashLagunaAttention


class DFlashLagunaForCausalLM(DFlashDraftModel):
    """Laguna DFlash draft model matching the exported Speculators checkpoint."""

    decoder_layer_cls = DFlashLagunaDecoderLayer
    supports_fused_context_kv = False

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))
        hidden_size = int(config.hidden_size)
        self.aux_hidden_norms = nn.ModuleList(
            [
                RMSNorm(hidden_size, eps=rms_norm_eps)
                for _ in range(self.num_context_features)
            ]
        )

    def prepare_context_hidden_for_kv(
        self, layer: DFlashLagunaDecoderLayer, ctx_hidden: torch.Tensor
    ) -> torch.Tensor:
        return layer.input_layernorm(ctx_hidden)

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        expected = int(
            self.fc.input_size if self.is_nemotron_35_draft else self.fc.in_features
        )
        if target_hidden.ndim != 2 or int(target_hidden.shape[-1]) != expected:
            raise ValueError(
                "Laguna DFLASH target_hidden feature dim mismatch. "
                f"Expected shape [N, {expected}] "
                f"(num_context_features={self.num_context_features}, hidden_size={int(self.config.hidden_size)}), "
                f"but got shape={tuple(target_hidden.shape)}."
            )

        num_slices = int(self.num_context_features)
        slice_size = int(target_hidden.shape[-1]) // num_slices
        slices = target_hidden.view(target_hidden.shape[0], num_slices, slice_size)
        compute_dtype = self.fc.weight.dtype
        if slices.dtype != compute_dtype:
            slices = slices.to(compute_dtype)
        normed = torch.empty_like(slices)
        for i, norm in enumerate(self.aux_hidden_norms):
            normed[:, i, :] = norm(slices[:, i, :])
        fused = normed.reshape(target_hidden.shape[0], -1)
        projected = self.fc(fused)
        if self.is_nemotron_35_draft:
            projected = projected[0]
        return self.hidden_norm(projected)


@torch.compile(dynamic=True, backend=get_compiler_backend(), disable=_is_npu)
def _score_edges(
    *,
    predecessor_table: torch.Tensor,
    successor_table: torch.Tensor,
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    hidden: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    keys = successor_table[candidate_ids]
    # Concatenate the ids and look them up once. Concatenating the looked-up rows
    # instead moves a [b, slots, k, rank] float tensor where this moves one id per
    # candidate, and it costs a second gather for the anchor.
    predecessor_ids = torch.cat(
        [anchor_token_ids[:, None, None].expand(-1, 1, top_k), candidate_ids[:, :-1]],
        dim=1,
    )
    predecessors = predecessor_table[predecessor_ids]
    return unary_logits[:, :, None] + torch.einsum(
        "blpr,blcr->blpc", predecessors * hidden[:, :, None], keys
    )


@torch.compile(dynamic=True, backend=get_compiler_backend(), disable=_is_npu)
def _follow_maps(maps, initial_indices, edges: int):
    index = initial_indices
    path = [index]
    for edge in range(edges):
        index = maps[:, edge].gather(-1, index[:, None])[:, 0]
        path.append(index)
    return torch.stack(path, dim=1)


class CandidateSelector(nn.Module):
    """Scores the K x K transitions between adjacent proposal slots, then walks them.

    The [vocab, r] tables are replicated on every TP rank rather than sharded like
    the LM head: candidate ids are gathered globally, so any rank can need any row.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        vocab_size: int,
        state_rank: int,
        top_k: int,
    ) -> None:
        super().__init__()
        if _flashinfer_top_k is None:
            logger.warning(
                "flashinfer is unavailable; the DFlash2 selector falls back to "
                "torch.topk, which roughly halves end-to-end throughput on a large "
                "vocabulary."
            )
        state_rank = int(state_rank)
        self.top_k = int(top_k)
        self.predecessor_codebook = nn.Parameter(
            torch.zeros(int(vocab_size), state_rank), requires_grad=False
        )
        self.successor_codebook = nn.Parameter(
            torch.zeros(int(vocab_size), state_rank), requires_grad=False
        )
        self.hidden_projection = nn.Linear(hidden_size, state_rank, bias=False)

    def build_lattice(
        self,
        *,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """score[b,e,p,c] = unary[b,e,c] + <A[pred[b,e,p]] * project(h[b,e]), B[c]>

        pred is cand[b,e-1], and the verified anchor for slot 0.
        """
        # Everything but the batch is a model constant. Left symbolic, inductor
        # recovers indices with an integer division per element instead of folding.
        hidden = self.hidden_projection(hidden_states)
        for tensor in (candidate_ids, unary_logits, hidden):
            torch._dynamo.mark_static(tensor, 1)
            torch._dynamo.mark_static(tensor, 2)
        return _score_edges(
            predecessor_table=self.predecessor_codebook,
            successor_table=self.successor_codebook,
            candidate_ids=candidate_ids,
            unary_logits=unary_logits,
            hidden=hidden,
            anchor_token_ids=anchor_token_ids,
            top_k=self.top_k,
        )

    def sample_path(
        self,
        *,
        candidate_ids: torch.Tensor,
        scores: torch.Tensor,
        uniforms: torch.Tensor,
        temperatures: torch.Tensor,
        greedy_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Walk one path, with q over the K candidates for the verify. greedy_mask
        rows take the argmax, selected rather than branched, so one captured graph
        serves greedy and sampling batches alike."""
        if scores.is_cuda:
            return selector_walk_triton(
                candidate_ids=candidate_ids,
                scores=scores,
                uniforms=uniforms,
                temperatures=temperatures,
                greedy_mask=greedy_mask,
            )
        top_k = self.top_k
        temps = temperatures.view(-1, 1)
        initial_probs = torch.softmax(scores[:, 0, 0].float() / temps, dim=-1)
        initial_indices = (
            uniforms[:, :1]
            .ge(initial_probs.cumsum(dim=-1))
            .sum(dim=-1)
            .clamp_max(top_k - 1)
        )
        transition_probs = torch.softmax(
            scores[:, 1:].float() / temps[:, :, None, None], dim=-1
        )
        local_maps = (
            uniforms[:, 1:, None, None]
            .ge(transition_probs.cumsum(dim=-1))
            .sum(dim=-1)
            .clamp_max(top_k - 1)
        )
        initial_indices = torch.where(
            greedy_mask, scores[:, 0, 0].argmax(dim=-1), initial_indices
        )
        local_maps = torch.where(
            greedy_mask[:, None, None], scores[:, 1:].argmax(dim=-1), local_maps
        )
        torch._dynamo.mark_static(local_maps, 1)
        torch._dynamo.mark_static(local_maps, 2)
        path_indices = _follow_maps(
            local_maps, initial_indices, int(scores.shape[1]) - 1
        )
        tokens = candidate_ids.gather(-1, path_indices.unsqueeze(-1))[:, :, 0]
        realized_rows = transition_probs.gather(
            2, path_indices[:, :-1, None, None].expand(-1, -1, 1, top_k)
        )[:, :, 0]
        q_rows = torch.cat((initial_probs.unsqueeze(1), realized_rows), dim=1)
        # Greedy rows walk the argmax, so their q is the point mass there, not
        # the temperature-1 softmax above. The triton walk stores the same.
        q_rows = torch.where(
            greedy_mask[:, None, None], F.one_hot(path_indices, top_k).float(), q_rows
        )
        return tokens, q_rows


class DFlash2DraftModel(DFlashDraftModel):
    """DFlash backbone + candidate selector. Reuses the DFLASH speculative worker."""

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        draft_config = self.draft_config
        if not draft_config.selector_rank:
            raise ValueError(
                "DFlash selector draft requires dflash_config.selector_rank."
            )
        self.candidate_selector = CandidateSelector(
            hidden_size=int(config.hidden_size),
            vocab_size=int(config.vocab_size),
            state_rank=draft_config.selector_rank,
            top_k=draft_config.selector_top_k,
        )
        # The draft has no head of its own; the worker points this at the target's
        # before capture.
        self.lm_head: Optional[nn.Module] = None

    def _transform_unary_logits(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        if self.draft_config.output_multiplier != 1.0:
            logits.mul_(self.draft_config.output_multiplier)
        softcap = self.draft_config.final_logit_softcapping
        if softcap is not None:
            logits.div_(softcap).tanh_().mul_(softcap)
        return logits

    def compute_candidates(
        self, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Top-k base candidates via the target lm_head: hidden [N, H] -> global
        candidate_ids / unary_logits [N, K]. Under TP (vocab-sharded lm_head): local top-k
        per shard, all-gather K logits/ids (not the full vocab), then a global top-k --
        identical candidates at O(tp*K) instead of O(vocab) gather bandwidth."""
        assert self.lm_head is not None, "draft_model.lm_head unset before capture"
        k = self.candidate_selector.top_k
        # The worker screens the head before capture, but its eager fallback
        # (_propose_selector_block) attaches whatever the target has.
        weight = getattr(self.lm_head, "weight", None)
        quant_method = getattr(self.lm_head, "quant_method", None)
        use_quant_head = should_apply_lm_head_quant_method(self.lm_head, quant_method)
        if not use_quant_head and not is_dense_head_weight(weight):
            raise RuntimeError(
                "DFlash2 selector requires a dense FP16/BF16/FP32 target lm_head "
                "or a supported lm_head.quant_method."
            )
        if get_parallel().tp_size == 1:
            org = int(self.lm_head.org_vocab_size)
            vals, ids = _radix_topk(
                _project_candidate_logits(
                    hidden, self.lm_head, num_org=org, use_quant_head=use_quant_head
                ),
                k,
            )
            return ids.long(), self._transform_unary_logits(vals)
        shard = self.lm_head.shard_indices
        vals, ids = _radix_topk(
            _project_candidate_logits(
                hidden,
                self.lm_head,
                num_org=int(shard.num_org_elements),
                use_quant_head=use_quant_head,
            ),
            k,
        )
        global_ids = ids.long() + int(shard.org_vocab_start_index)
        gathered_vals = tensor_model_parallel_all_gather(vals.float(), dim=-1)
        gathered_ids = tensor_model_parallel_all_gather(global_ids, dim=-1)
        top_vals, sel = torch.topk(gathered_vals, k, dim=-1)
        return torch.gather(gathered_ids, -1, sel).long(), self._transform_unary_logits(
            top_vals
        )


class MuseGlimmerAssistantModel(DFlashDraftModel):
    """Alias for checkpoints declaring architectures=["MuseGlimmerAssistantModel"]."""


EntryClass = [
    DFlashDraftModel,
    DFlashLagunaForCausalLM,
    MuseGlimmerAssistantModel,
    DFlash2DraftModel,
]
