# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# Adapted from the DFlash reference implementation (HF) but implemented with
# SGLang primitives (RadixAttention + SGLang KV cache). Most drafts borrow the
# target embedding and LM head; Nemotron 3.5 drafts carry their own embedding.

from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from functools import partial
from typing import Iterable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from sglang.kernels.ops.communication import sp_collective
from sglang.kernels.ops.speculative.dflash import (
    dflash_grouped_conv,
    selector_walk_triton,
)
from sglang.kernels.ops.speculative.lilicorr import lilicorr_topk_lse
from sglang.srt.configs.draft_parallel import DraftParallelSpec, DraftTPExecution
from sglang.srt.configs.laguna import normalize_gating
from sglang.srt.distributed.device_communicators.triton_symm_mem_ag import (
    MultimemAllGatherer,
    recommended_max_tokens,
)
from sglang.srt.environ import envs
from sglang.srt.hf_transformers import get_rope_config
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.communication import k3_ar_fusion
from sglang.srt.layers.layer_boundary import AttentionInputs, get_attn_tp_context
from sglang.srt.layers.layernorm import GemmaRMSNorm, RMSNorm
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
from sglang.srt.layers.quantization.unquant import (
    bf16_gemm_dispatch,
    get_bf16_gemm_backend,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils.common import set_weight_attrs
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
from sglang.srt.models.utils import apply_qk_norm, bind_or_assign
from sglang.srt.runtime_context import get_parallel, get_spec
from sglang.srt.speculative.dflash_draft_collectives import (
    DraftAllReduce,
    DraftTopkGather,
    DraftTopkGathers,
    serving_deadline_ns,
)
from sglang.srt.runtime_context import get_forward
from sglang.srt.speculative import k31ht_draft_ar_site as k31ht_site
from sglang.srt.speculative.dflash_utils import (
    can_dflash_slice_qkv_weight,
    get_dflash_attention_sliding_window_size,
    get_dflash_layer_attention_params,
    is_dense_head_weight,
    is_dflash_mla_draft_config,
    is_nemotron_35_draft_config,
    parse_dflash_draft_config,
)
from sglang.srt.utils import BumpAllocator, is_npu
from sglang.srt.utils.device import get_compiler_backend

_is_npu = is_npu()
if _is_npu:
    from sgl_kernel_npu.norm.split_qkv_rmsnorm_rope import split_qkv_rmsnorm_rope
logger = logging.getLogger(__name__)


try:
    from flashinfer import top_k as _flashinfer_top_k
except ImportError:
    _flashinfer_top_k = None


class _UnusedZeroAllocator:
    """Stands in for the fp8 bmm scale allocator an unquantized MLA draft never uses."""

    def allocate(self, size: int):
        raise RuntimeError(
            "DFLASH MLA draft requested an fp8 bmm scale buffer; "
            "unset SGLANG_OPT_DFLASH_MLA_DRAFT_SKIP_ZERO_ALLOC."
        )


_UNUSED_ZERO_ALLOCATOR = _UnusedZeroAllocator()


def _radix_topk(scores: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # The selector's largest single cost: it reads the whole logits tensor.
    if _flashinfer_top_k is not None:
        return _flashinfer_top_k(scores, k, sorted=True, deterministic=True)
    return torch.topk(scores, k, dim=-1)


def _k31ht_topk_gather_pack_requested() -> bool:
    """K31HT_DRAFT_TOPK_GATHER_PACK (default off): fail-closed parse ('' / '0' off, '1' on, else ValueError)."""
    v = os.environ.get("K31HT_DRAFT_TOPK_GATHER_PACK", "")
    if v in ("", "0"):
        return False
    if v == "1":
        return True
    raise ValueError(f"K31HT_DRAFT_TOPK_GATHER_PACK must be unset, '', '0' or '1', got {v!r}")


_K31HT_TOPK_GATHER_PACK = _k31ht_topk_gather_pack_requested()
_K31HT_TOPK_GATHER_PACK_LOGGED = False


def k31ht_topk_gather_packed(group, vals: torch.Tensor, global_ids: torch.Tensor):
    """K31HT_DRAFT_TOPK_GATHER_PACK: the candidate values (fp32 bits) and ids (int32) of the vocab-sharded top-k go
    through ONE all-gather as an [N, K, 2] int32 tensor instead of two (fp32 values, int64 ids). Gathered along the K
    dim, rank-major, exactly like the two dim=-1 gathers; the unpacked values are bit-identical to theirs and the ids
    equal in value (int32 view, ids < 2**31; the caller's torch.gather(...).long() widens the selected K)."""
    packed = torch.stack(
        (vals.float().contiguous().view(torch.int32), global_ids.to(torch.int32)), dim=-1
    )
    g = group.all_gather(packed, dim=-2)
    global _K31HT_TOPK_GATHER_PACK_LOGGED
    if not _K31HT_TOPK_GATHER_PACK_LOGGED:
        _K31HT_TOPK_GATHER_PACK_LOGGED = True
        logger.info(
            "K31HT_DRAFT_TOPK_GATHER_PACK ENGAGED: candidate top-k values + ids in one all-gather (world %d, K %d)",
            group.world_size,
            int(vals.shape[-1]),
        )
    return g[..., 0].contiguous().view(torch.float32), g[..., 1]


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


class _HeadSplitK:
    """SGLANG_OPT_DFLASH_DRAFT_HEAD_SPLITK_ROWS: the plugin split-K projection of the org-vocab head rows [N, K] at the
    listed drafted-row counts, one K slice (one FP32 accumulation per output), M tile 64, N tile = rows rounded up to
    8, 9 stages (the T64s1 tactic). Prepared (JIT + TMA maps) at its first eager use."""

    TACTIC = (64, 1, 9)  # (M tile, split, stages); the N tile follows the row count

    def __init__(self, weight: torch.Tensor, rows):
        from sgl_nvidia_mega.kimi_k3.cutedsl.gemm.projection import PreparedProjection

        m_tile, split, stages = self.TACTIC
        self.n, self.k = weight.shape
        self.rows = frozenset(rows)
        policies = {r: (m_tile, -(-r // 8) * 8, split, stages) for r in self.rows}
        # The neighbours are stock kernels that do not fence their stores for TMA readers: the projection fences
        # around its TMA reads itself (as the draft split-K projections, speculative/dflash_draft_splitk.py).
        self._prepared = PreparedProjection(weight, policies, proxy_fences=True).prepare()
        self.engaged_rows = set()  # row counts that ran on this projection (logged once each)

    def covers(self, x: torch.Tensor) -> bool:
        return (
            x.dim() == 2
            and x.shape[0] in self.rows
            and x.shape[1] == self.k
            and x.dtype == torch.bfloat16
            and x.stride(1) == 1
            and x.stride(0) % 8 == 0
            and x.data_ptr() % 16 == 0
        )

    def run(self, x: torch.Tensor) -> torch.Tensor:
        rows = int(x.shape[0])
        if rows not in self.engaged_rows:
            self.engaged_rows.add(rows)
            logger.info("DFLASH draft head split-K ENGAGED at %d rows (N %d, K %d)", rows, self.n, self.k)
        return self._prepared.run(x, x.new_empty((rows, self.n)))


# (weight data_ptr, num_org) -> _HeadSplitK, or None when the head cannot take it.
_HEAD_SPLITK = {}


def _head_splitk(weight: torch.Tensor, num_org: int) -> Optional[_HeadSplitK]:
    rows = envs.SGLANG_OPT_DFLASH_DRAFT_HEAD_SPLITK_ROWS.get()
    if not rows:
        return None
    key = (weight.data_ptr(), int(num_org))
    if key in _HEAD_SPLITK:
        return _HEAD_SPLITK[key]
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "SGLANG_OPT_DFLASH_DRAFT_HEAD_SPLITK_ROWS: first head projection inside a CUDA graph capture "
            "(the projection compiles at its first eager use)"
        )
    head = weight[:num_org]
    reason = None
    if head.dtype != torch.bfloat16 or not head.is_contiguous():
        reason = f"head dtype {head.dtype} contiguous {head.is_contiguous()}"
    elif head.shape[0] % _HeadSplitK.TACTIC[0] or head.shape[1] % 128:
        reason = f"head shape {tuple(head.shape)} (N multiple of 64, K multiple of 128 needed)"
    if reason is not None:
        logger.warning("DFLASH draft head split-K NOT ENGAGED: %s", reason)
        _HEAD_SPLITK[key] = None
        return None
    _HEAD_SPLITK[key] = _HeadSplitK(head, (int(r) for r in rows))
    return _HEAD_SPLITK[key]


def _project_candidate_logits(
    hidden: torch.Tensor, lm_head: nn.Module, *, num_org: int, use_quant_head: bool
) -> torch.Tensor:
    """Project draft hiddens through the target head, restricted to the org vocab."""
    if not use_quant_head:
        weight = lm_head.weight
        hidden = hidden.to(weight.dtype)
        # SGLANG_OPT_DFLASH_DRAFT_HEAD_SPLITK_ROWS: the covered rows on the plugin's split-K head GEMM.
        split_k = _head_splitk(weight, num_org)
        if split_k is not None and split_k.covers(hidden):
            return split_k.run(hidden)
        # Tuned bf16 dispatch (kernel chosen by GEMM shape); bf16 CUDA only.
        backend = get_bf16_gemm_backend()
        if (
            (backend.is_cutedsl() or backend.is_gemv())
            and hidden.is_cuda
            and weight.dtype == torch.bfloat16
        ):
            return bf16_gemm_dispatch(hidden, weight[:num_org], None)
        return torch.matmul(hidden, weight[:num_org].T)
    # A packed weight can't be row-sliced to the org vocab like the dense path,
    # and flashinfer's radix top-k rejects the crop view (non-contiguous), so
    # mask the padded tail out of the top-k instead.
    logits = lm_head.quant_method.apply(lm_head, hidden, None).contiguous()
    if logits.shape[-1] > num_org:
        logits[:, num_org:] = float("-inf")
    return logits


# Symmetric-buffer name of the fused candidate all-gather's output.
_CANDIDATE_AG_BUFFER = "dflash_candidate_ag"
# None: not probed yet.
_CANDIDATE_AG_READY: Optional[bool] = None


def _candidate_ag_ready(group) -> bool:
    """Whether the direct all-gather can run on ``group``; registers its comm once."""
    global _CANDIDATE_AG_READY
    if _CANDIDATE_AG_READY is None:
        _CANDIDATE_AG_READY = _probe_candidate_ag(group)
    return _CANDIDATE_AG_READY


def _probe_candidate_ag(group) -> bool:
    # The kernel barriers on k3_ar_fusion's communicator, so only the TP group works.
    if (
        not envs.SGLANG_OPT_DFLASH_FUSED_CANDIDATE_AG.get()
        or not k3_ar_fusion.enabled()
        or group is not get_parallel().tp_group
    ):
        return False

    try:
        sp_collective.register_comm(group.ca_comm.obj)
    except AssertionError:
        logger.warning(
            "DFLASH fused candidate all-gather: another communicator owns "
            "TP%d; using NCCL.",
            group.world_size,
        )
        return False
    logger.info(
        "DFLASH fused candidate all-gather enabled (TP%d, one direct multicast "
        "all-gather of the top-k values and ids)",
        group.world_size,
    )
    return True


def _fused_candidate_all_gather(
    group, vals: torch.Tensor, global_ids: torch.Tensor
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """[N, K] vals/ids -> [N, tp*K] fp32 vals, int32 ids, rank-major; None for NCCL."""
    if not _candidate_ag_ready(group):
        return None

    rows, k = vals.shape
    tp = group.world_size
    # Each rank's [N, 2K] int32 shard must be whole 16 B vectors.
    if (rows * k) % 2 or rows > k3_ar_fusion._max_buffer_rows():
        return None
    payload = torch.empty((rows, 2 * k), dtype=torch.int32, device=vals.device)
    payload[:, :k].view(torch.float32).copy_(vals)
    # Global ids stay below the vocabulary size, so int32 carries them exactly.
    payload[:, k:].copy_(global_ids)
    # The kernel copies raw 16 B vectors; bf16 is only its host-side dtype check.
    out = k3_ar_fusion.symm_buffer(
        _CANDIDATE_AG_BUFFER,
        rows,
        tp * 4 * k,
        torch.bfloat16,
        group_name=group.cpu_group.group_name,
    )
    sp_collective.all_gather_direct(
        tp,
        payload.view(torch.bfloat16),
        out,
        output_mc_ptr=k3_ar_fusion.get_mc_ptr(out),
        # One CTA covers the few hundred bytes each rank sends.
        tuning=sp_collective.Tuning(num_blocks=1, block_size=128),
    )
    # [tp, N, 2K] rank-major -> [N, tp, 2K]; reshape copies each half contiguous.
    gathered = out.view(torch.int32).view(tp, rows, 2 * k).transpose(0, 1)
    gathered_vals = gathered[..., :k].view(torch.float32).reshape(rows, tp * k)
    gathered_ids = gathered[..., k:].reshape(rows, tp * k)
    return gathered_vals, gathered_ids


def candidate_topk(
    hidden: torch.Tensor,
    lm_head: nn.Module,
    k: int,
    *,
    with_partition: bool = False,
    topk_gather: Optional[Union[DraftTopkGather, DraftTopkGathers]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Global top-k candidates from the target head: hidden [N, H] -> ids / vals [N, K].

    Under TP (vocab-sharded lm_head): local top-k per shard, all-gather K logits/ids (not
    the full vocab), then a global top-k -- identical candidates at O(tp*K) instead of
    O(vocab) gather bandwidth. with_partition also returns the full-vocabulary lse [N] fp32.
    """
    # The worker screens the head before capture, but the eager fallbacks attach
    # whatever the target has.
    weight = getattr(lm_head, "weight", None)
    quant_method = getattr(lm_head, "quant_method", None)
    use_quant_head = should_apply_lm_head_quant_method(lm_head, quant_method)
    if not use_quant_head and not is_dense_head_weight(weight):
        raise RuntimeError(
            "DFlash candidate top-k requires a dense FP16/BF16/FP32 target lm_head "
            "or a supported lm_head.quant_method."
        )
    # Gather over the group the head is vocab-sharded on: under DP attention with
    # --enable-dp-lm-head that is the attention-TP group, whose ranks share this
    # DP rank's tokens (the full TP group mixes other DP ranks' rows).
    parallel = get_parallel()
    group = (
        parallel.attn_tp_group
        if getattr(lm_head, "use_attn_tp_group", False)
        else parallel.tp_group
    )
    tp = group.world_size == 1
    num_org = (
        int(lm_head.org_vocab_size)
        if tp
        else int(lm_head.shard_indices.num_org_elements)
    )
    logits = _project_candidate_logits(
        hidden, lm_head, num_org=num_org, use_quant_head=use_quant_head
    )
    lse = None
    if with_partition:
        vals, ids, lse = lilicorr_topk_lse(logits, k)
    else:
        vals, ids = _radix_topk(logits, k)
    if tp:
        return ids.long(), vals, lse
    global_ids = ids.long() + int(lm_head.shard_indices.org_vocab_start_index)
    # With both switches set the packed gather (topk_gather) runs;
    # SGLANG_OPT_DFLASH_FUSED_CANDIDATE_AG applies only without it.
    if topk_gather is not None:
        gathered_vals, gathered_ids = topk_gather(group, vals.float(), global_ids)
    else:
        gathered = _fused_candidate_all_gather(group, vals, global_ids)
        if gathered is not None:
            gathered_vals, gathered_ids = gathered
        else:
            if _K31HT_TOPK_GATHER_PACK:
                gathered_vals, gathered_ids = k31ht_topk_gather_packed(group, vals, global_ids)
            else:
                gathered_vals = group.all_gather(vals.float(), dim=-1)
                gathered_ids = group.all_gather(global_ids, dim=-1)
    top_vals, sel = torch.topk(gathered_vals, k, dim=-1)
    if lse is not None:
        lse = torch.logsumexp(group.all_gather(lse.unsqueeze(-1), dim=-1), dim=-1)
    return torch.gather(gathered_ids, -1, sel).long(), top_vals, lse


_ROPE_DIM_SUPERSEDED_KEYS = frozenset(
    {"rope_dim", "partial_rotary_factor", "mrope_section", "mrope_interleaved"}
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

        self.qkv_proj = self._build_qkv_proj(attention_bias, quant_config, prefix)
        self.o_proj = RowParallelLinear(
            self.total_num_heads * head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj" if prefix else "o_proj",
        )

        self.use_gemma_norm = bool(getattr(config, "use_gemma_norm", False))
        norm_cls = GemmaRMSNorm if self.use_gemma_norm else RMSNorm
        self.q_norm = norm_cls(head_dim, eps=rms_norm_eps)
        self.k_norm = norm_cls(head_dim, eps=rms_norm_eps)

        rope_theta, rope_scaling = get_rope_config(config)
        rotary_dim = head_dim
        partial_rotary_factor = float(getattr(config, "partial_rotary_factor", 1.0))
        if rope_scaling is not None and "rope_dim" in rope_scaling:
            # An explicit rope_dim is the trained rotary width (HF Qwen3 ignores the
            # copied target partial/M-RoPE fields); drafts only see 1-D positions.
            rotary_dim = int(rope_scaling["rope_dim"])
            partial_rotary_factor = 1.0
            rope_scaling = {
                k: v
                for k, v in rope_scaling.items()
                if k not in _ROPE_DIM_SUPERSEDED_KEYS
            }
        rope_is_neox_style = bool(
            getattr(
                config, "rope_is_neox_style", getattr(config, "is_neox_style", True)
            )
        )
        max_position_embeddings = int(getattr(config, "max_position_embeddings", 32768))
        self.rotary_emb = get_rope(
            head_dim,
            rotary_dim=rotary_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=rope_is_neox_style,
            partial_rotary_factor=partial_rotary_factor,
        )

        self.scaling = head_dim**-0.5
        rotary = self.rotary_emb
        self.use_table_qk_norm_rope = (
            not _is_npu
            and not self.use_gemma_norm
            and hasattr(rotary, "cos_sin_cache")
            and getattr(rotary, "rotary_dim", None) == head_dim
            and getattr(rotary, "is_neox_style", False)
        )
        self.sliding_window_size, self.attn_type = get_dflash_layer_attention_params(
            config, layer_id
        )
        draft_cfg = parse_dflash_draft_config(draft_hf_config=config)
        self.v_scale: Optional[float] = draft_cfg.attention_value_scale
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
                {"weight_loader": sharded_weight_loader(0, parallel_group="tp")},
            )
        elif draft_cfg.attention_sink_bias:
            # Per-head sink bias; each TP rank owns its slice of the
            # all-heads checkpoint tensor.
            self.attention_sink_bias = nn.Parameter(
                torch.empty(self.num_heads, dtype=torch.float32 if _is_npu else None),
                requires_grad=False,
            )
            set_weight_attrs(
                self.attention_sink_bias,
                {"weight_loader": sharded_weight_loader(0, parallel_group="tp")},
            )
        self.attn = RadixAttention(
            num_heads=self.num_heads,
            head_dim=head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=self.sliding_window_size,
            attn_type=self.attn_type,
            quant_config=quant_config,
        )

    def _build_qkv_proj(self, bias: bool, quant_config, prefix: str) -> nn.Module:
        return QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj" if prefix else "qkv_proj",
        )

    def forward_prepare_npu(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)

        if self.use_gemma_norm:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q, k = apply_qk_norm(
                q, k, self.q_norm, self.k_norm, self.head_dim, allow_inplace=False
            )
            q, k = self.rotary_emb(positions, q, k)
            return q, k, v

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
            from sglang.kernels.ops.attention.table_qk_norm_rope import (
                table_qk_norm_rope_,
            )

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
            # In-place QK kernels use weight directly; Gemma requires 1 + weight.
            q, k = apply_qk_norm(
                q,
                k,
                self.q_norm,
                self.k_norm,
                self.head_dim,
                allow_inplace=not self.use_gemma_norm,
            )
            q, k = self.rotary_emb(positions, q, k)
        if self.v_scale is not None:
            v = v * self.v_scale
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
        else:
            # Fallback: compute full QKV and discard Q (keeps compatibility with quantized weights).
            qkv, _ = self.qkv_proj(hidden_states)
            _, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # Keep V scaling consistent with forward() so ctx and self-generated V
        # align in the draft cache.
        if self.v_scale is not None:
            v = v * self.v_scale
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


# Default off: under the a2a / fi_a2a DCP reduce, project the draft's 512-wide
# head partials by w_vc before the exchange and combine them with
# dcp_push_combine, as the target's DSA layers do.
_DRAFT_DCP_V_PREPROJECT = os.environ.get(
    "SGLANG_K31_DRAFT_DCP_V_PREPROJECT", "0"
).strip().lower() in ("1", "true", "yes", "on")


class DFlashMLAAttention(DeepseekV2AttentionMLA):
    """SpecForge MLA draft attention served by the target's absorbed MLA path.

    SpecForge exports DeepSeek's MLA parameter names and RoPE conventions, so
    only the DFlash layer causality, context-KV writes and loading differ.
    """

    dcp_v_preproject = _DRAFT_DCP_V_PREPROJECT
    draft_dcp_v_preproject = _DRAFT_DCP_V_PREPROJECT

    def __init__(
        self,
        config,
        layer_id: int,
        quant_config=None,
        prefix: str = "",
        draft_all_reduce: Optional[DraftAllReduce] = None,
    ) -> None:
        if quant_config is not None:
            raise ValueError("DFLASH mla drafts require unquantized draft weights.")
        # Under DCP the draft pool keeps DCP owner rows like the target's
        # (KVCacheConfigurator.draft_mla_rows_are_physical), so the layer runs
        # the target's DCP decode path: gathered q, rank-local KV, LSE merge.
        sliding_window_size, attn_type = get_dflash_layer_attention_params(
            config, layer_id
        )
        rope_theta, rope_scaling = get_rope_config(config)
        if rope_scaling is not None and rope_scaling.get(
            "rope_type", rope_scaling.get("type")
        ) in (None, "default"):
            rope_scaling = None
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            # SpecForge omits q_lora_rank when Q has no low-rank path.
            q_lora_rank=getattr(config, "q_lora_rank", None),
            kv_lora_rank=config.kv_lora_rank,
            rope_theta=rope_theta,
            # The base class rewrites rope_type in place.
            rope_scaling=dict(rope_scaling) if rope_scaling else None,
            max_position_embeddings=config.max_position_embeddings,
            layer_id=layer_id,
            prefix=prefix,
            attn_type=attn_type,
            sliding_window_size=sliding_window_size,
            reduce_results=draft_all_reduce is None,
        )
        self._draft_all_reduce = draft_all_reduce
        # Unquantized MLA drafts never take the fp8 bmm scale buffer.
        self._skip_zero_alloc = envs.SGLANG_OPT_DFLASH_MLA_DRAFT_SKIP_ZERO_ALLOC.get()
        self.fused_kv_write = envs.SGLANG_OPT_DFLASH_MLA_FUSED_KV_WRITE.get()
        self.fused_kv_write_pdl = envs.SGLANG_OPT_DFLASH_DRAFT_PDL_GLUE.get()
        # SGLANG_OPT_DFLASH_DRAFT_MLAQ: set by install_draft_mlaq (speculative/dflash_draft_mlaq.py).
        self._draft_mlaq = None
        # SGLANG_OPT_DFLASH_DRAFT_ATTNBLOCK: the DraftMlaQuery path's attention + w_vc as one launch. Off: the class method runs as before.
        if envs.SGLANG_OPT_DFLASH_DRAFT_ATTNBLOCK.get():
            self._attnblock_declines = set()
            self.forward_absorb_core = self._forward_absorb_core_attnblock
        if self.q_lora_rank is not None:
            # SpecForge stores the two halves of the fused down projection.
            self.fused_qkv_a_proj_with_mqa.weight.weight_loader = (
                self._load_fused_a_proj_shard
            )
        kv_b_proj_loader = self.kv_b_proj.weight.weight_loader

        def load_kv_b_proj(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
            kv_b_proj_loader(param, loaded_weight)
            self.absorb_kv_b_proj()

        self.kv_b_proj.weight.weight_loader = load_kv_b_proj

    def _load_fused_a_proj_shard(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: int
    ) -> None:
        rows = param.data.split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim]
        )[shard_id]
        if rows.shape != loaded_weight.shape:
            raise ValueError(
                f"DFLASH mla fused a_proj shard {shard_id} expects "
                f"{tuple(rows.shape)}, got {tuple(loaded_weight.shape)}."
            )
        rows.copy_(loaded_weight)

    def route_projections_through_linears(self) -> None:
        """qkv_a and q_b through their modules' quant_method (no fused_a_gemm bypass)."""
        self._use_min_latency_fused_a_gemm = False
        self._use_min_latency_q_b_gemm = False

    def absorb_kv_b_proj(self) -> None:
        """Split kv_b_proj into the per-head w_kc / w_vc the absorbed forward reads."""
        w_kc, w_vc = self.kv_b_proj.weight.unflatten(
            0, (-1, self.qk_nope_head_dim + self.v_head_dim)
        ).split([self.qk_nope_head_dim, self.v_head_dim], dim=1)
        self.w_kc = bind_or_assign(
            self.w_kc, w_kc.transpose(1, 2).contiguous().transpose(1, 2)
        )
        self.w_vc = bind_or_assign(self.w_vc, w_vc.contiguous().transpose(1, 2))
        # weight reload: refresh the gathered DCP V weights in place, as the target does
        if self.draft_dcp_v_preproject and self._dcp_v_weights is not None:
            self._prepare_dcp_v_weights(get_parallel().dcp_group)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        get_attn_tp_context().set_attn_inputs(
            AttentionInputs(hidden_states, forward_batch, self.prepare_qkv_latent)
        )
        # Sized for the fp8 bmm scales, which unquantized drafts never take.
        zero_allocator = (
            _UNUSED_ZERO_ALLOCATOR
            if self._skip_zero_alloc
            else BumpAllocator(
                buffer_size=2, dtype=torch.float32, device=hidden_states.device
            )
        )
        output = super().forward(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            zero_allocator=zero_allocator,
        )
        get_attn_tp_context().clear_attn_inputs()
        if self._draft_all_reduce is not None:
            output = self._draft_all_reduce(output)
        return output

    def forward_absorb_prepare(
        self,
        positions,
        hidden_states,
        forward_batch,
        zero_allocator,
        llama_4_scaling=None,
        prev_topk_indices=None,
    ):
        """With SGLANG_OPT_DFLASH_DRAFT_MLAQ (install_draft_mlaq): q_a / kv_a RMSNorm, q_b, W_UK, RoPE and the FP8 KV-row store
        in one DraftMlaQuery launch on the Triton backend; forward_absorb_core then hands its q / k / v to the attention."""
        mlaq = self._draft_mlaq
        if mlaq is not None and self.current_attention_backend == "triton" and llama_4_scaling is None:
            qkv = get_attn_tp_context().fetch_qkv_latent()
            out = mlaq.run(qkv, positions, forward_batch.out_cache_loc)
            if out is not None:
                q, k, v = out
                self._q8kv8_qprep_overlap_pending = False
                self._kv_cache_saved_early = False
                self._fused_kv_write_rope_rows = None
                self._draft_mlaq_qk = (q, k, v)
                return (
                    q[..., self.kv_lora_rank :],
                    k[..., self.kv_lora_rank :],
                    q[..., : self.kv_lora_rank],
                    v,
                    forward_batch,
                    zero_allocator,
                    positions,
                    None,
                    llama_4_scaling,
                    None,
                )
        return super().forward_absorb_prepare(
            positions,
            hidden_states,
            forward_batch,
            zero_allocator,
            llama_4_scaling,
            prev_topk_indices,
        )

    def _forward_absorb_core_attnblock(
        self,
        q_pe,
        k_pe,
        q_nope_out,
        k_nope,
        forward_batch,
        zero_allocator,
        positions,
        topk_indices,
        llama_4_scaling,
        fusion_plan=None,
        attention_output_gate=None,
    ):
        """SGLANG_OPT_DFLASH_DRAFT_ATTNBLOCK: after DraftMlaQuery, the split-KV attention and w_vc in one launch, then o_proj.
        Any call the fused block does not cover runs the served forward_absorb_core unchanged (the reason is logged once)."""
        qkv = self._draft_mlaq_qk
        if (
            qkv is not None
            and topk_indices is None
            and llama_4_scaling is None
            and fusion_plan is None
            and attention_output_gate is None
        ):
            from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla import (
                _finish_absorb_output,
            )

            out = self._run_attnblock(qkv, forward_batch)
            if out is not None:
                self._draft_mlaq_qk = None
                return _finish_absorb_output(self, out, None, topk_indices)
        return type(self).forward_absorb_core(
            self,
            q_pe,
            k_pe,
            q_nope_out,
            k_nope,
            forward_batch,
            zero_allocator,
            positions,
            topk_indices,
            llama_4_scaling,
            fusion_plan,
            attention_output_gate,
        )

    def _run_attnblock(self, qkv, forward_batch) -> Optional[torch.Tensor]:
        """The served Triton draft call's inputs (forward_extend's split-KV branch: window rail, causal, no soft cap / sinks / custom
        mask) through dflash_draft_attnblock_fwd; None when this call is not that form."""
        from sglang.srt.layers.attention.graph_utils import padded_extend_real_tokens
        from sglang.srt.layers.radix_attention import AttentionType
        from sglang.srt.model_executor.forward_context import get_attn_backend
        from sglang.srt.speculative.dflash_draft_attnblock import (
            attnblock_covers,
            dflash_draft_attnblock_fwd,
        )

        backend = get_attn_backend()  # the backend RadixAttention.forward would call
        md = getattr(backend, "forward_metadata", None)
        layer = self.attn_mqa
        checks = (
            (padded_extend_real_tokens(qkv[0], forward_batch) is None, "padded extend rows"),
            (getattr(backend, "_draft_splitkv_mode", 0) in (1, 2), "backend is not the split-KV draft Triton backend"),
            (md is not None and getattr(md, "window_kv_indices", None) is not None, "no window rail"),
            (layer.sliding_window_size is not None and layer.sliding_window_size > -1, "not a sliding-window layer"),
            (getattr(self, "dcp_size", 1) == 1 and getattr(backend, "dcp_size", 1) == 1, "DCP"),
            (not getattr(backend, "enable_deterministic", False), "deterministic mode"),
            (getattr(backend, "dllm_attention", None) is None, "dLLM attention"),
            (md is not None and getattr(md, "custom_mask", None) is None, "custom mask"),
            (not layer.logit_cap and layer.xai_temperature_len <= 0, "soft cap / temperature"),
            (not layer.is_cross_attention and layer.attn_type == AttentionType.DECODER, "not causal decoder attention"),
        )
        why = "; ".join(w for ok, w in checks if not ok)
        if not why:
            q, k, v = qkv
            k_buffer = backend.token_to_kv_pool.get_key_buffer(layer.layer_id)
            k_scale = layer.k_scale_float if layer.k_scale is not None else 1.0
            v_scale = layer.v_scale_float if layer.k_scale is not None else 1.0
            why = attnblock_covers(q, k, v, k_buffer, md.qo_indptr, self.w_vc, md.max_extend_len, k_scale, v_scale)
        if why:
            if why not in self._attnblock_declines:
                self._attnblock_declines.add(why)
                logger.info("DFLASH draft attention block declined (layer %d): %s", layer.layer_id, why)
            return None
        out = torch.empty((q.shape[0], q.shape[1] * self.v_head_dim), dtype=torch.bfloat16, device=q.device)
        return dflash_draft_attnblock_fwd(
            q,
            k,
            v,
            k_buffer,
            md.qo_indptr,
            md.window_kv_indptr,
            md.window_kv_indices,
            self.w_vc,
            out,
            layer.scaling,
            k_scale=k_scale,
            sliding_window_size=layer.sliding_window_size,
            max_kv=backend._draft_splitkv_max_kv,
        )

    def project_context_kv(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cache rows of target-context tokens: normalized latent and rotated k_pe,
        each [n, 1, dim]."""
        if self.q_lora_rank is None:
            latent, _ = self.kv_a_proj_with_mqa(hidden_states)
        else:
            latent = F.linear(
                hidden_states,
                self.fused_qkv_a_proj_with_mqa.weight[self.q_lora_rank :],
            )
        k_nope = self.kv_a_layernorm(latent[..., : self.kv_lora_rank]).unsqueeze(1)
        k_pe = latent[..., self.kv_lora_rank :].unsqueeze(1)
        _, k_pe = self.rotary_emb(positions, torch.empty_like(k_pe), k_pe)
        return k_nope, k_pe

    def write_context_kv(
        self,
        *,
        pool,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_loc: torch.Tensor,
        swa_cache_loc: Optional[torch.Tensor] = None,
        cache_loc_2d: Optional[torch.Tensor] = None,
        commit_lens: Optional[torch.Tensor] = None,
    ) -> None:
        """Write this layer's rows for target-context tokens; with cache_loc_2d,
        only each request's first commit_lens rows."""
        k_nope, k_pe = self.project_context_kv(positions, hidden_states)
        if cache_loc_2d is None:
            pool.set_mla_kv_buffer(
                self.attn_mqa,
                KVWriteLoc(cache_loc, swa_loc=swa_cache_loc, physical=True),
                k_nope,
                k_pe,
            )
        else:
            pool.layer_router.layer(self.attn_mqa.layer_id).store_mla_prefix(
                self.attn_mqa,
                KVWriteLoc(
                    cache_loc_2d,
                    swa_loc=None
                    if swa_cache_loc is None
                    else swa_cache_loc.view_as(cache_loc_2d),
                    physical=True,
                ),
                commit_lens,
                k_nope,
                k_pe,
            )


class DFlashMLP(nn.Module):
    def __init__(
        self,
        config,
        quant_config=None,
        prefix: str = "",
        draft_all_reduce: Optional[DraftAllReduce] = None,
    ) -> None:
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
            reduce_results=draft_all_reduce is None,
            prefix="down_proj" if not prefix else f"{prefix}.down_proj",
        )
        self._draft_all_reduce = draft_all_reduce
        hidden_act = getattr(config, "hidden_act", "silu")
        if hidden_act == "silu":
            self.act_fn = SiluAndMul()
        elif hidden_act == "swigluoai":
            from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
                swiglu_no_interleaved_with_alpha_and_limit,
            )

            self.act_fn = partial(
                swiglu_no_interleaved_with_alpha_and_limit,
                gemm1_alpha=float(config.swiglu_alpha),
                gemm1_limit=float(config.swiglu_limit),
            )
        else:
            raise ValueError(
                f"Unsupported DFlash activation: {hidden_act}. "
                "Supported activations are silu and swigluoai."
            )

    # SGLANG_OPT_DFLASH_DRAFT_SPLITK_ACT: install_draft_splitk sets the fused gate_up + SiLU-mul projection.
    split_k_gate_up_act = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused = self.split_k_gate_up_act
        if fused is not None and fused.covers(x):
            x = fused.run(x)
        else:
            gate_up, _ = self.gate_up_proj(x)
            x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        if self._draft_all_reduce is not None:
            x = self._draft_all_reduce(x)
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
        shifted = torch.where((position >= tap).view(-1, 1, 1), shifted, 0)
        out = out + coefficients[:, tap] * shifted
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
        self.kernel_projection = ReplicatedLinear(
            hidden_size, 2 * self.taps * self.num_groups, bias=False
        )
        # Set by SGLANG_OPT_DFLASH_DRAFT_SPLITK (speculative/dflash_draft_splitk.py).
        self.split_k_projection = None
        # SGLANG_OPT_DFLASH_DRAFT_CONV_PROLOGUE (speculative/dflash_draft_conv_prologue.py): the consumer GEMM's conv-prologue form,
        # and (raw rows, conv projection) left by prepare_projection for that GEMM.
        self.conv_prologue = None
        self.conv_pending = None
        self._pdl_glue = envs.SGLANG_OPT_DFLASH_DRAFT_PDL_GLUE.get()
        # SGLANG_OPT_DFLASH_DRAFT_EARLY_TRIGGER: an int32 device flag set by the worker before capture.
        self._early_flag = None

    def _project(self, hidden_states: torch.Tensor) -> torch.Tensor:
        projection = self.split_k_projection
        if projection is not None and projection.covers(hidden_states):
            return projection.run(hidden_states)
        return self.kernel_projection(hidden_states)[0]

    def _convolve(self, hidden_states, delta, side: int) -> torch.Tensor:
        if hidden_states.is_cuda and hidden_states.dim() == 2:
            return dflash_grouped_conv(
                hidden_states,
                delta,
                self.base_kernel[side],
                self.block_size,
                self.group_size,
                self.taps,
                pdl=self._pdl_glue,
                early_flag=self._early_flag,
            )
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

    def _project_conv(self, hidden_states: torch.Tensor):
        """SGLANG_OPT_DFLASH_DRAFT_SPLITK_SK3B_CONV: (side-0 conv output, projection) from one kernel for the rows the split-K projection's
        run_conv covers (bitwise to _project + the side-0 _convolve), else None."""
        projection = self.split_k_projection
        if projection is None or not hidden_states.is_cuda or hidden_states.dim() != 2:
            return None
        covers_conv = getattr(projection, "covers_conv", None)
        if covers_conv is None or not covers_conv(hidden_states):
            return None
        projection_out, out = projection.run_conv(hidden_states, self.base_kernel[0])
        return out, projection_out

    def prepare(self, hidden_states: torch.Tensor):
        fused = self._project_conv(hidden_states)
        if fused is not None:
            out, projection = fused
            coefficients = projection.reshape(*hidden_states.shape[:-1], 2, self.taps, self.num_groups)
            return out, coefficients[..., 1, :, :]
        coefficients = self._project(hidden_states).reshape(
            *hidden_states.shape[:-1], 2, self.taps, self.num_groups
        )
        return (
            self._convolve(hidden_states, coefficients[..., 0, :, :], side=0),
            coefficients[..., 1, :, :],
        )

    def finish(self, hidden_states: torch.Tensor, coefficients) -> torch.Tensor:
        return self._convolve(hidden_states, coefficients, side=1)

    def prepare_projection(self, hidden_states: torch.Tensor):
        """prepare() returning the whole [tokens, 2 * taps * groups] projection instead of its side-1 view (the
        SGLANG_OPT_DFLASH_DRAFT_AR_SITE kernel reads side 1 from it)."""
        fused = self._project_conv(hidden_states)
        if fused is not None:
            return fused
        projection = self._project(hidden_states)
        prologue = self.conv_prologue
        if prologue is not None and prologue.covers(hidden_states):
            # conv prologue: the consumer GEMM convolves side 0 itself; it takes this record (take_pending) or raises.
            if self.conv_pending is not None:
                raise RuntimeError("conv prologue: the previous conv record was never consumed")
            self.conv_pending = (hidden_states, projection)
            return hidden_states, projection
        coefficients = projection.reshape(
            *hidden_states.shape[:-1], 2, self.taps, self.num_groups
        )
        return (
            self._convolve(hidden_states, coefficients[..., 0, :, :], side=0),
            projection,
        )


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
        draft_all_reduce: Optional[DraftAllReduce] = None,
    ) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        norm_cls = GemmaRMSNorm if getattr(config, "use_gemma_norm", False) else RMSNorm
        self.input_layernorm = norm_cls(hidden_size, eps=rms_norm_eps)
        attention_prefix = f"{prefix}.self_attn" if prefix else ""
        attention_cls = self.attention_cls
        attention_kwargs = {}
        if is_dflash_mla_draft_config(config):
            if attention_cls is not DFlashAttention:
                raise ValueError(
                    f"{type(self).__name__} does not support mla draft attention."
                )
            attention_cls = DFlashMLAAttention
            # DFlashAttention's o_proj keeps the group's all-reduce.
            attention_kwargs["draft_all_reduce"] = draft_all_reduce
        self.self_attn = attention_cls(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=attention_prefix,
            **attention_kwargs,
        )
        self.post_attention_layernorm = norm_cls(hidden_size, eps=rms_norm_eps)
        mlp_prefix = f"{prefix}.mlp" if prefix else ""
        self.mlp = DFlashMLP(
            config=config,
            quant_config=quant_config,
            prefix=mlp_prefix,
            draft_all_reduce=draft_all_reduce,
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

    def forward_ar_site(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: torch.Tensor,
        next_norm: RMSNorm,
        next_conv: Optional[DFlashGroupedConv],
        sites,
    ) -> torch.Tensor:
        """forward() with SGLANG_OPT_DFLASH_DRAFT_AR_SITE engaged (DraftArSites.engage): ``hidden_states`` is this
        layer's input norm output; each sublayer's partial goes through one site (all-reduce + conv finish + add +
        norm). Returns ``next_norm``(new residual); ``residual`` is updated in place. ``next_conv``: the next
        layer's attention conv, whose projection is the first kernel after this layer (None after the last)."""
        hidden_states, projection = self.attention_conv.prepare_projection(hidden_states)
        with sites.ring_scope(0, projection, self.attention_conv):
            partial = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
        if self.attention_conv.conv_pending is not None:
            raise RuntimeError("conv prologue: attention conv record not consumed by qkv_a")
        hidden_states = sites.run(
            0, partial, projection, self.attention_conv, residual, self.post_attention_layernorm,
            successor=self.mlp_conv,
        )  # fmt: skip
        hidden_states, projection = self.mlp_conv.prepare_projection(hidden_states)
        with sites.ring_scope(1, projection, self.mlp_conv):
            partial = self.mlp(hidden_states)
        if self.mlp_conv.conv_pending is not None:
            raise RuntimeError("conv prologue: mlp conv record not consumed by gate_up")
        return sites.run(1, partial, projection, self.mlp_conv, residual, next_norm, successor=next_conv)


class DFlashDraftModel(nn.Module):
    """SGLang DFlash draft model with an optional Nemotron embedding.

    The checkpoint provides:
      - transformer weights for `layers.*`
      - `fc.weight`, `hidden_norm.weight` for projecting target context features
      - `norm.weight` for final normalization
    """

    draft_parallel_spec = DraftParallelSpec(tp_execution=DraftTPExecution.SINGLE_GROUP)

    decoder_layer_cls = DFlashDecoderLayer
    supports_fused_context_kv = True
    # Layer prefixes and a quant_config-aware `fc`, for quantized checkpoints.
    supports_quantization = False
    # `project_target_hidden` gathers a TP-split `fc` (SGLANG_OPT_DFLASH_TP_AUX_FC).
    supports_fc_tp_split = True

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        self.config = config

        hidden_size = int(config.hidden_size)
        num_layers = int(config.num_hidden_layers)
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.rms_norm_eps = rms_norm_eps
        use_gemma_norm = bool(getattr(config, "use_gemma_norm", False))
        norm_cls = GemmaRMSNorm if use_gemma_norm else RMSNorm
        if use_gemma_norm or is_dflash_mla_draft_config(config):
            # Fused context KV materialization assumes ordinary RMSNorm weights,
            # and an MLA draft has no fused qkv_proj to slice (context KV goes
            # through project_context_kv()).
            self.supports_fused_context_kv = False
        draft_config = self.draft_config = parse_dflash_draft_config(
            draft_hf_config=config
        )
        self.block_size = draft_config.resolve_block_size(default=16)
        # One draft request per step all-reduces [block_size, hidden] partials.
        self.draft_all_reduce: Optional[DraftAllReduce] = None
        tp_group = get_parallel().tp_group
        if (
            envs.SGLANG_OPT_DFLASH_DRAFT_PK3_ALL_REDUCE.get()
            and tp_group.world_size > 1
        ):
            if get_parallel().attn_tp_size != tp_group.world_size:
                raise ValueError(
                    "SGLANG_OPT_DFLASH_DRAFT_PK3_ALL_REDUCE needs the draft's "
                    "attention TP group to be its TP group."
                )
            self.draft_all_reduce = DraftAllReduce(
                group=tp_group,
                rows=self.block_size,
                hidden_size=hidden_size,
                deadline_ns=serving_deadline_ns(),
            )
        # Set by install_draft_ar_site (SGLANG_OPT_DFLASH_DRAFT_AR_SITE).
        self.ar_sites = None
        # Set by DFlash2DraftModel; serves the selector's top-k gather.
        self.topk_gather: Optional[Union[DraftTopkGather, DraftTopkGathers]] = None
        self.candidate_selector: Optional[nn.Module] = None
        self.lilicorr: Optional[nn.Module] = None
        self.is_nemotron_35_draft = is_nemotron_35_draft_config(config)
        self.quantizable = self.supports_quantization or self.is_nemotron_35_draft
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
                        if self.quantizable
                        else ""
                    ),
                    draft_all_reduce=self.draft_all_reduce,
                )
                for i in range(num_layers)
            ]
        )
        self.norm = norm_cls(hidden_size, eps=rms_norm_eps)

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
        # [start, end) fc output columns this rank computes; None = replicated fc.
        self.fc_tp_columns = self._fc_tp_columns(hidden_size)
        if self.quantizable:
            fc_prefix = f"{prefix}.fc" if prefix else "fc"
            self.fc = ReplicatedLinear(
                self.num_context_features * hidden_size,
                hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=fc_prefix,
            )
        elif self.fc_tp_columns is not None:
            start, end = self.fc_tp_columns
            self.fc = nn.Linear(
                self.num_context_features * hidden_size, end - start, bias=False
            )
            logger.info(
                "DFLASH aux fc split across TP: columns [%d, %d) of %d "
                "(SGLANG_OPT_DFLASH_TP_AUX_FC)",
                start,
                end,
                hidden_size,
            )
        else:
            self.fc = nn.Linear(
                self.num_context_features * hidden_size, hidden_size, bias=False
            )
        # (first output row, rows) of this rank's fc slice once shard_context_fc ran.
        self._context_fc_shard: Optional[Tuple[int, int]] = None
        self._context_fc_gatherer: Optional[MultimemAllGatherer] = None
        self.hidden_norm = norm_cls(hidden_size, eps=rms_norm_eps)

        # The model loader calls load_weights() before set_block_size(). Build
        # Domino projector modules here so their parameters are present while
        # checkpoint weights are loaded.
        self.projector_type = draft_config.projector_type
        self.shift_label = draft_config.shift_label
        self.prefix_gru: Optional[nn.GRU] = None
        self.embed_proj: Optional[nn.Sequential] = None
        if draft_config.is_domino:
            assert draft_config.gru_hidden_dim is not None
            assert draft_config.emb_dim is not None
            self.prefix_gru = nn.GRU(
                input_size=hidden_size,
                hidden_size=int(draft_config.gru_hidden_dim),
                num_layers=1,
                batch_first=True,
                bias=False,
            )
            self.embed_proj = nn.Sequential(
                nn.Linear(
                    hidden_size + int(draft_config.gru_hidden_dim),
                    int(draft_config.emb_dim),
                    bias=False,
                ),
                nn.SiLU(),
                nn.Linear(
                    int(draft_config.emb_dim), int(config.vocab_size), bias=False
                ),
            )

    def set_block_size(self, block_size: int) -> None:
        """Adopt the block size the worker resolved.

        The convolutions are built from the checkpoint's block_size, which
        --speculative-num-draft-tokens may override; the layout they index
        depends on it, so the resolved value has to reach them.
        """
        self.block_size = int(block_size)
        if self.draft_all_reduce is not None:
            self.draft_all_reduce.set_rows(self.block_size)
        if self.topk_gather is not None:
            self.topk_gather.set_rows(self.block_size - 1)
        for layer in self.layers:
            for conv in (layer.attention_conv, layer.mlp_conv):
                if conv is not None:
                    conv.block_size = self.block_size

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
        expected = int(self.fc.input_size if self.quantizable else self.fc.in_features)
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
        if self.quantizable:
            projected = projected[0]
        elif self.fc_tp_columns is not None:
            projected = self._gather_fc_columns(projected)
        elif self._context_fc_gatherer is not None:
            # Each rank projects its hidden columns; the gather rebuilds the row.
            projected = self._context_fc_gatherer(projected)
        ctx_hidden = self.hidden_norm(projected)
        return ctx_hidden

    def shard_context_fc(self, tp_group) -> bool:
        """Keep this rank's slice of the context fc's output features.

        project_target_hidden then all-gathers the BF16 column slices, so each
        output feature is the replicated fc's dot product over all context
        features, at 1/world of its weight bytes. tp_group must be the group
        get_tp_group() names when the projection runs, and every rank must hold
        the same target rows. False (fc left replicated) when it cannot shard.
        """
        fc = self.fc
        world = int(tp_group.world_size)
        if (
            world <= 1
            or self.quantizable
            # Already split at construction (SGLANG_OPT_DFLASH_TP_AUX_FC).
            or self.fc_tp_columns is not None
            or self._context_fc_shard is not None
            or type(self).project_target_hidden
            is not DFlashDraftModel.project_target_hidden
            or not isinstance(fc, nn.Linear)
            or fc.bias is not None
            or fc.out_features % world != 0
        ):
            return False
        rows = fc.out_features // world
        start = int(tp_group.rank_in_group) * rows
        fc.weight = nn.Parameter(
            fc.weight.detach().narrow(0, start, rows).contiguous(),
            requires_grad=False,
        )
        fc.out_features = rows
        self._context_fc_shard = (start, rows)
        # Rows are a decode verify block per request; a prefill falls back to NCCL.
        self._context_fc_gatherer = MultimemAllGatherer(
            max_tokens=recommended_max_tokens(include_prefill=False, floor=512)
        )
        return True

    def _fc_tp_columns(self, hidden_size: int) -> Optional[Tuple[int, int]]:
        parallel = get_parallel()
        if (
            not envs.SGLANG_OPT_DFLASH_TP_AUX_FC.get()
            or not self.supports_fc_tp_split
            or self.quantizable
            or parallel.tp_size == 1
            or hidden_size % parallel.tp_size != 0
            # An attention-owning draft runs under a narrowed TP group (draft_tp_context).
            or parallel.attn_dp_enabled
            or parallel.attn_cp_size > 1
        ):
            return None
        width = hidden_size // parallel.tp_size
        return parallel.tp_rank * width, (parallel.tp_rank + 1) * width

    def _gather_fc_columns(self, local: torch.Tensor) -> torch.Tensor:
        # Each element is one rank's value plus zeros, so the all-reduce gathers
        # exactly (a -0.0 comes back as +0.0).
        start, end = self.fc_tp_columns
        full = local.new_zeros(local.shape[0], int(self.config.hidden_size))
        full[:, start:end] = local
        return get_parallel().tp_group.all_reduce(full)

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

        sites = self.ar_sites
        if sites is not None and sites.covers(hidden_states):
            hidden_states = self._forward_ar_sites(
                positions, hidden_states, forward_batch, sites
            )
            return LogitsProcessorOutput(
                next_token_logits=None,
                hidden_states=hidden_states,
            )

        if k31ht_site.k31ht_draft_site_epi_requested():
            sited = self._k31ht_forward_ar_sites(positions, hidden_states, forward_batch)
            if sited is not None:
                return LogitsProcessorOutput(next_token_logits=None, hidden_states=sited)
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

    def _forward_ar_sites(self, positions, hidden_states, forward_batch, sites):
        """The layer loop + final norm with SGLANG_OPT_DFLASH_DRAFT_AR_SITE: each layer's MLP site normalizes with
        the next layer's input norm (the final norm after the last layer), so those norms do not run here."""
        residual = hidden_states
        hidden_states = self.layers[0].input_layernorm(hidden_states)
        next_norms = [layer.input_layernorm for layer in self.layers[1:]] + [self.norm]
        next_convs = [layer.attention_conv for layer in self.layers[1:]] + [None]
        with sites.engage(self.layers):
            for layer, next_norm, next_conv in zip(self.layers, next_norms, next_convs):
                hidden_states = layer.forward_ar_site(
                    positions, hidden_states, forward_batch, residual, next_norm, next_conv, sites
                )
        return hidden_states

    def _k31ht_site_layout_reason(self, hidden_states: torch.Tensor) -> Optional[str]:
        """Why the K31HT_DRAFT_SITE_EPI layer loop cannot replace the stock one for this forward (None = it can)."""
        if hidden_states.numel() == 0 or hidden_states.dim() != 2:
            return f"hidden {tuple(hidden_states.shape)}"
        if hidden_states.dtype != torch.bfloat16 or hidden_states.shape[1] != k31ht_site.HIDDEN:
            return f"hidden dtype {hidden_states.dtype} width {hidden_states.shape[1]}"
        if get_forward().sp_active:
            return "sequence parallel active"
        norms = [self.norm] + [n for l in self.layers for n in (l.input_layernorm, l.post_attention_layernorm)]
        if any(type(n) is not RMSNorm or n.cast_x_before_out_mul or n.weight.dtype != torch.bfloat16 for n in norms):
            return "norms are not plain bf16 RMSNorm"
        for layer in self.layers:
            for conv in (layer.attention_conv, layer.mlp_conv):
                if conv is None or conv.taps != k31ht_site.TAPS or conv.group_size != k31ht_site.GROUP:
                    return "draft conv missing or not 2-tap / 16-group"
                if conv.base_kernel.dtype != torch.bfloat16:
                    return f"conv base dtype {conv.base_kernel.dtype}"
            for lin in (layer.self_attn.o_proj, layer.mlp.down_proj):
                if not lin.reduce_results or getattr(lin, "use_decode_attn_tp", False):
                    return "projection all-reduce is not the plain reduce_results path"
        return None

    def _k31ht_site(self, linear, partial, conv, kernel, residual, norm, lean=0):
        """One site: ``partial`` is the projection output, already all-reduced by the stock path, or (``lean``,
        K31HT_DRAFT_AR_LEAN) unreduced and reduced here by the stock 2shot_push kernel with ``lean`` threads per CTA; one
        epilogue kernel then replaces conv finish + fused add-RMSNorm (same arithmetic)."""
        group = k31ht_site.stock_group_for(linear) if lean else None
        if lean:
            partial = k31ht_site.lean_all_reduce(group, partial, lean)
        out = k31ht_site.draft_site_epilogue(
            partial, kernel, conv.base_kernel[1], residual, norm.weight, norm.variance_epsilon, conv.block_size
        )
        return out

    def _k31ht_forward_ar_sites(self, positions, hidden_states, forward_batch) -> Optional[torch.Tensor]:
        """K31HT_DRAFT_SITE_EPI=1: the stock layer loop with each projection's conv finish and the following add-RMSNorm
        (post-attention norm; next layer's input norm; the model norm after the last layer) done by one epilogue kernel after
        the projection's all-reduce. Bit-identical to the stock loop (same arithmetic); returns None to run the stock loop."""
        reason = self._k31ht_site_layout_reason(hidden_states)
        if reason is not None:
            k31ht_site.decline(reason)
            return None
        n = len(self.layers)
        h = hidden_states
        residual = None
        lean = k31ht_site.k31ht_draft_ar_lean_threads()  # K31HT_DRAFT_AR_LEAN
        for i, layer in enumerate(self.layers):
            if i == 0:
                residual = h
                x = layer.input_layernorm(h)
            else:
                x = h  # the previous MLP site applied this layer's input_layernorm
            x, attention_kernel = layer.attention_conv.prepare(x)
            with k31ht_site.no_reduce(layer.self_attn.o_proj) if lean else nullcontext():
                partial = layer.self_attn(positions=positions, hidden_states=x, forward_batch=forward_batch)
            h = self._k31ht_site(
                layer.self_attn.o_proj, partial, layer.attention_conv, attention_kernel, residual, layer.post_attention_layernorm,
                lean=lean,
            )
            x, mlp_kernel = layer.mlp_conv.prepare(h)
            with k31ht_site.no_reduce(layer.mlp.down_proj) if lean else nullcontext():
                partial = layer.mlp(x)
            norm = self.layers[i + 1].input_layernorm if i + 1 < n else self.norm
            h = self._k31ht_site(
                layer.mlp.down_proj, partial, layer.mlp_conv, mlp_kernel, residual, norm, lean=lean,
            )
        return h

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, weight_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            ("fused_qkv_a_proj_with_mqa", "q_a_proj", 0),
            ("fused_qkv_a_proj_with_mqa", "kv_a_proj_with_mqa", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params = set()

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

        for name, loaded_weight in weights:
            unprefixed_name = name.removeprefix("model.")
            if self.projector_type != "domino" and unprefixed_name.startswith(
                ("prefix_gru.", "embed_proj.")
            ):
                raise ValueError(
                    "DFLASH checkpoint contains Domino projector weights but "
                    f"projector_type={self.projector_type!r}."
                )
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
                loaded_params.add(resolved_name)
                break
            else:
                resolved_name = resolve_param_name(name)
                if resolved_name is None:
                    # Ignore unexpected weights (e.g., HF rotary caches).
                    continue
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
                    elif self.fc_tp_columns is not None:
                        expected_shape = (
                            int(self.config.hidden_size),
                            int(self.num_context_features * self.config.hidden_size),
                        )
                        loaded_shape = tuple(loaded_weight.shape)
                        shape_matches = loaded_shape == expected_shape
                        if shape_matches:
                            start, end = self.fc_tp_columns
                            loaded_weight = loaded_weight[start:end]
                    else:
                        if self._context_fc_shard is not None:
                            # A weight update after shard_context_fc: keep the slice.
                            loaded_weight = loaded_weight.narrow(
                                0, *self._context_fc_shard
                            )
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
                if resolved_name.startswith(("prefix_gru.", "embed_proj.")) and tuple(
                    loaded_weight.shape
                ) != tuple(param.shape):
                    raise ValueError(
                        "DFLASH Domino projector weight shape mismatch: "
                        f"expected {resolved_name}{tuple(param.shape)}, got "
                        f"{tuple(loaded_weight.shape)} from {name!r}."
                    )
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(resolved_name)

        if self.projector_type == "domino":
            required = {
                "prefix_gru.weight_ih_l0",
                "prefix_gru.weight_hh_l0",
                "embed_proj.0.weight",
                "embed_proj.2.weight",
            }
            missing = required - loaded_params
            if missing:
                raise ValueError(
                    "DFLASH Domino checkpoint is missing required projector weights: "
                    f"{sorted(missing)}."
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
    # Its own `project_target_hidden` reads the whole `fc` output.
    supports_fc_tp_split = False

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
        if (
            envs.SGLANG_OPT_DFLASH_PACKED_TOPK_GATHER.get()
            and get_parallel().tp_size > 1
        ):
            # One request per step selects among [block_size - 1, top_k] candidates per rank.
            gather_kwargs = dict(
                rows=self.block_size - 1,
                k=self.candidate_selector.top_k,
                vocab_size=int(config.vocab_size),
                deadline_ns=serving_deadline_ns(),
            )
            max_requests = envs.SGLANG_OPT_DFLASH_PACKED_TOPK_GATHER_MAX_BS.get()
            self.topk_gather = (
                DraftTopkGathers(max_requests=max_requests, **gather_kwargs)
                if max_requests > 1
                else DraftTopkGather(**gather_kwargs)
            )
            if max_requests > 1 and get_parallel().tp_rank == 0:
                logger.info(
                    "DFLASH packed top-k gather at draft batch sizes 1-%d (%d mailboxes).",
                    max_requests,
                    len(self.topk_gather.by_rows),
                )

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
        candidate_ids / unary_logits [N, K]."""
        assert self.lm_head is not None, "draft_model.lm_head unset before capture"
        ids, vals, _ = candidate_topk(
            hidden,
            self.lm_head,
            self.candidate_selector.top_k,
            topk_gather=self.topk_gather,
        )
        return ids, self._transform_unary_logits(vals)


# dflash_selector.cuh's selection reproduces torch.topk's 32-slot sort, so top_k is 16 (top-16 drafter: TP16 x 16 = 256
# gathered) or 32 (the public DFlash2 drafter: TP8 x 32 = 256 or TP16 x 32 = 512 gathered); selector rank 256.
_FUSED_SELECTOR_WIDTHS = {16: (256,), 32: (256, 512)}
_FUSED_SELECTOR_RANK = 256
# Rows (bs x draft slots) up to which dflash_selector_select is shown to keep torch.topk's tie order, per (gathered
# width, top_k): torch.topk orders the equal values of 256-wide top-16 rows differently from 224 rows on, while 512-wide
# top-32 rows matched up to 448. Above the bound the sampler runs the unfused head (FusedSelectorHead.covers).
_FUSED_SELECT_MAX_ROWS = {(256, 16): 112, (256, 32): 112, (512, 32): 448}


class FusedSelectorHead:
    """The DFlash2 draft head after its LM head in three launches (SGLANG_OPT_DFLASH_FUSED_SELECTOR): the candidate
    top-k and lattice operands (dflash_selector_select), the lattice GEMM, and the walk into the draft sampler's
    buffers. The hidden projection and the walk's draws run on a side stream behind the local top-k and the gather.
    Every output is bitwise equal to _SelectorDraftSampler's unfused path: the local top-k kernel runs only where
    flashinfer would take the radix path it copies, and the sampler uses the head only for rows it covers()."""

    def __init__(self, *, draft_model: DFlash2DraftModel, max_bs: int, device) -> None:
        # Imported here so the flag-off path never loads the module.
        from sglang.kernels.ops.speculative.dflash_selector import (
            dflash_selector_local_topk,
            dflash_selector_select,
            local_topk_matches_flashinfer,
            local_topk_supported,
            selector_walk_fused,
        )

        self._select, self._walk = dflash_selector_select, selector_walk_fused
        self._local_topk = dflash_selector_local_topk
        self._local_matches = local_topk_matches_flashinfer
        self._logged_fallback = set()
        lm_head = draft_model.lm_head
        # The local top-k kernel reproduces flashinfer's deterministic sorted top-k of BF16 logits; without flashinfer,
        # _radix_topk is torch.topk and candidate_topk's own ops run.
        self._local_fused = (
            _flashinfer_top_k is not None
            and lm_head.weight.dtype == torch.bfloat16
            and local_topk_supported(int(lm_head.shard_indices.num_org_elements))
        )
        self._model = draft_model
        selector = draft_model.candidate_selector
        self._top_k = selector.top_k
        self._rank = selector.hidden_projection.out_features
        # The target's ParallelLMHead (the worker attaches it before building the sampler).
        self._use_attn_tp_group = bool(draft_model.lm_head.use_attn_tp_group)
        self._check_supported(draft_model)
        self._select_max_rows = _FUSED_SELECT_MAX_ROWS[
            (self._group().world_size * self._top_k, self._top_k)
        ]
        rows = int(max_bs) * (draft_model.block_size - 1)
        self._stream = torch.cuda.Stream(device=device)
        self._unary = torch.empty((rows, self._top_k), dtype=torch.float32, device=device)
        self._pred_h = torch.empty(
            (rows, self._top_k, self._rank), dtype=torch.bfloat16, device=device
        )
        self._keys = torch.empty_like(self._pred_h)
        # SGLANG_OPT_DFLASH_DRAFT_HEAD_GATHER: the head reads one compact copy of the drafted rows (bs >= 2) made by a PDL kernel
        # instead of the plain reshape copy. The hidden projection keeps the served slice (F.linear on the compact view is not
        # bitwise to it). Bitwise equal to off.
        self._rows_buf = None
        self._gather_logged = set()
        if envs.SGLANG_OPT_DFLASH_DRAFT_HEAD_GATHER.get():
            self._rows_buf = torch.empty(
                (rows, int(lm_head.weight.shape[1])), dtype=lm_head.weight.dtype, device=device
            )
        # SGLANG_OPT_DFLASH_SELECTOR_ONE_LAUNCH: the select, the lattice GEMM and the walk in one kernel, bitwise equal.
        self._one_launch = None
        if envs.SGLANG_OPT_DFLASH_SELECTOR_ONE_LAUNCH.get():
            from sglang.kernels.ops.speculative.dflash_selector_one_launch import (
                dflash_selector_one_launch,
                one_launch_supported,
            )

            slots = int(draft_model.block_size) - 1
            if one_launch_supported(top_k=self._top_k, rank=self._rank, slots=slots):
                self._one_launch = dflash_selector_one_launch
                logger.info(
                    "SGLANG_OPT_DFLASH_SELECTOR_ONE_LAUNCH ENGAGED: select + lattice + walk in one launch "
                    f"(top_k {self._top_k}, rank {self._rank}, slots {slots})"
                )
            else:
                logger.warning(
                    "SGLANG_OPT_DFLASH_SELECTOR_ONE_LAUNCH DECLINED: needs top_k 16, rank % 64 == 0 and <= 8 slots "
                    f"(top_k {self._top_k}, rank {self._rank}, slots {slots}); the three-launch tail runs"
                )

    def _gather_rows(self, hidden: torch.Tensor) -> torch.Tensor:
        """SGLANG_OPT_DFLASH_DRAFT_HEAD_GATHER: hidden [bs, slots, H] -> the same rows as a compact view of the persistent
        buffer (one PDL copy), or hidden itself where the copy does not apply (the reason is logged once)."""
        from sglang.kernels.ops.speculative.dflash_draft_gather import (
            draft_rows_gather,
            draft_rows_gather_covers,
        )

        why = draft_rows_gather_covers(hidden, self._rows_buf)
        key = (int(hidden.shape[0]), why)
        if key not in self._gather_logged:
            self._gather_logged.add(key)
            if why:
                logger.info("SGLANG_OPT_DFLASH_DRAFT_HEAD_GATHER declined at bs %d: %s", hidden.shape[0], why)
            else:
                logger.info(
                    "SGLANG_OPT_DFLASH_DRAFT_HEAD_GATHER ENGAGED at bs %d (%d drafted rows, one PDL copy for the head)",
                    hidden.shape[0], hidden.shape[0] * hidden.shape[1],
                )
        if why:
            return hidden
        return draft_rows_gather(hidden, self._rows_buf).view(hidden.shape)

    def _check_supported(self, draft_model: DFlash2DraftModel) -> None:
        selector = draft_model.candidate_selector
        lm_head = draft_model.lm_head
        config = draft_model.draft_config
        group = self._group()
        problems = []
        if not is_dense_head_weight(lm_head.weight) or should_apply_lm_head_quant_method(
            lm_head, lm_head.quant_method
        ):
            problems.append("a dense BF16/FP16/FP32 target lm_head")
        widths = _FUSED_SELECTOR_WIDTHS.get(self._top_k)
        if widths is None or self._rank != _FUSED_SELECTOR_RANK:
            problems.append(
                f"selector top_k 16 or 32 and rank {_FUSED_SELECTOR_RANK} (got {self._top_k}, {self._rank})"
            )
        elif group.world_size * self._top_k not in widths:
            problems.append(
                f"tp x top_k in {widths} for top_k {self._top_k} (got {group.world_size} x {self._top_k})"
            )
        tables = (selector.predecessor_codebook, selector.successor_codebook)
        if any(t.dtype != torch.bfloat16 or not t.is_contiguous() for t in tables):
            problems.append("contiguous BF16 codebooks")
        if config.output_multiplier != 1.0 or config.final_logit_softcapping is not None:
            problems.append("no output multiplier or softcap on the unary logits")
        if problems:
            raise ValueError(
                "SGLANG_OPT_DFLASH_FUSED_SELECTOR needs " + "; ".join(problems) + "."
            )

    def _group(self):
        parallel = get_parallel()
        return parallel.attn_tp_group if self._use_attn_tp_group else parallel.tp_group

    def covers(self, bs: int) -> bool:
        """Whether the fused head reproduces the unfused one at this batch size (dflash_selector_select's tie order
        is shown equal to torch.topk's only up to _FUSED_SELECT_MAX_ROWS rows)."""
        rows = int(bs) * (self._model.block_size - 1)
        if rows <= self._select_max_rows:
            return True
        if ("select", rows) not in self._logged_fallback:
            self._logged_fallback.add(("select", rows))
            logger.info(
                "DFlash fused selector: %d draft rows > %d, unfused head runs at this batch size",
                rows,
                self._select_max_rows,
            )
        return False

    def _local_candidates(self, logits: torch.Tensor, offset: int):
        """This rank's top-k as (int64 global ids, values): the local top-k kernel where flashinfer would take the radix
        path whose tie order it copies, else flashinfer itself as candidate_topk calls it."""
        rows, length = logits.shape
        if self._local_fused and self._local_matches(rows, length):
            return self._local_topk(logits=logits, k=self._top_k, offset=offset)
        if self._local_fused and ("local", rows) not in self._logged_fallback:
            self._logged_fallback.add(("local", rows))
            logger.info(
                "DFlash fused selector: %d rows of %d logits take flashinfer's own top-k (its filtered path)",
                rows,
                length,
            )
        vals, ids = _radix_topk(logits, self._top_k)
        return ids.long() + offset, vals

    def _gather(self, group, vals: torch.Tensor, global_ids: torch.Tensor):
        """candidate_topk's gather choice: the packed gather, else the fused candidate all-gather
        (_fused_candidate_all_gather), else NCCL."""
        topk_gather = self._model.topk_gather
        if topk_gather is not None:
            return topk_gather(group, vals.float(), global_ids)
        gathered = _fused_candidate_all_gather(group, vals, global_ids)
        if gathered is not None:
            return gathered
        return (
            group.all_gather(vals.float(), dim=-1),
            group.all_gather(global_ids, dim=-1),
        )

    def __call__(
        self,
        *,
        hidden: torch.Tensor,
        anchors: torch.Tensor,
        uniforms: torch.Tensor,
        temperatures: torch.Tensor,
        greedy_mask: torch.Tensor,
        tokens: torch.Tensor,
        candidates: torch.Tensor,
        q_rows: torch.Tensor,
    ) -> None:
        """hidden [bs, slots, H] -> tokens [bs * slots], candidates and q_rows [bs, slots, top_k], in place."""
        bs, slots = hidden.shape[0], hidden.shape[1]
        rows = bs * slots
        model, selector = self._model, self._model.candidate_selector
        lm_head = model.lm_head
        group = self._group()
        num_org = int(lm_head.shard_indices.num_org_elements)
        head_in = hidden if self._rows_buf is None else self._gather_rows(hidden)
        logits = _project_candidate_logits(
            head_in.reshape(-1, hidden.shape[-1]), lm_head, num_org=num_org, use_quant_head=False
        )
        main = torch.cuda.current_stream(hidden.device)
        # After the LM head GEMM, so these never compete with it; joined before the selection reads them.
        self._stream.wait_stream(main)
        with torch.cuda.stream(self._stream):
            projected = selector.hidden_projection(hidden)
            uniforms.uniform_()
        offset = int(lm_head.shard_indices.org_vocab_start_index)
        global_ids, vals = self._local_candidates(logits, offset)
        gathered_vals, gathered_ids = self._gather(group, vals, global_ids)
        main.wait_stream(self._stream)
        self._select_and_walk(
            bs=bs,
            slots=slots,
            gathered_vals=gathered_vals,
            gathered_ids=gathered_ids,
            projected=projected,
            anchors=anchors,
            uniforms=uniforms,
            temperatures=temperatures,
            greedy_mask=greedy_mask,
            tokens=tokens,
            candidates=candidates,
            q_rows=q_rows,
        )

    def _select_and_walk(
        self,
        *,
        bs: int,
        slots: int,
        gathered_vals: torch.Tensor,
        gathered_ids: torch.Tensor,
        projected: torch.Tensor,
        anchors: torch.Tensor,
        uniforms: torch.Tensor,
        temperatures: torch.Tensor,
        greedy_mask: torch.Tensor,
        tokens: torch.Tensor,
        candidates: torch.Tensor,
        q_rows: torch.Tensor,
    ) -> None:
        """Everything after the candidate gather: candidates, tokens and q_rows in place."""
        rows = bs * slots
        selector = self._model.candidate_selector
        if self._one_launch is not None:
            self._one_launch(
                gathered_vals=gathered_vals.contiguous(),
                gathered_ids=gathered_ids.contiguous(),
                hidden=projected,
                anchors=anchors,
                pred_table=selector.predecessor_codebook,
                succ_table=selector.successor_codebook,
                uniforms=uniforms,
                temperatures=temperatures,
                greedy_mask=greedy_mask,
                candidates=candidates.view(rows, self._top_k),
                tokens=tokens,
                q_rows=q_rows,
            )
            return
        unary, pred_h, keys = self._unary[:rows], self._pred_h[:rows], self._keys[:rows]
        self._select(
            gathered_vals=gathered_vals.contiguous(),
            gathered_ids=gathered_ids.contiguous(),
            hidden=projected,
            anchors=anchors,
            pred_table=selector.predecessor_codebook,
            succ_table=selector.successor_codebook,
            candidates=candidates.view(rows, self._top_k),
            unary=unary,
            pred_h=pred_h,
            keys=keys,
        )
        shape = (bs, slots, self._top_k, self._rank)
        # _score_edges' contraction on the same operand layouts: the same bmm call.
        lattice = torch.einsum("blpr,blcr->blpc", pred_h.view(shape), keys.view(shape))
        self._walk(
            lattice=lattice,
            unary=unary,
            candidates=candidates,
            uniforms=uniforms,
            temperatures=temperatures,
            greedy_mask=greedy_mask,
            tokens=tokens,
            q_rows=q_rows,
        )


class MiniMaxDFlash2DraftModel(DFlash2DraftModel):
    """Compatibility alias for existing MiniMax DFlash2 checkpoint configs."""


class MuseGlimmerAssistantModel(DFlashDraftModel):
    """Alias for checkpoints declaring architectures=["MuseGlimmerAssistantModel"]."""


class DFlash2BeamDraftModel(DFlash2DraftModel):
    """Alias for checkpoints declaring architectures=["DFlash2BeamDraftModel"]."""


EntryClass = [
    DFlashDraftModel,
    DFlashLagunaForCausalLM,
    MuseGlimmerAssistantModel,
    DFlash2DraftModel,
    DFlash2BeamDraftModel,
    MiniMaxDFlash2DraftModel,
]
