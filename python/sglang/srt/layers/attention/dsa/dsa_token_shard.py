# Copyright 2023-2026 SGLang Team
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

"""Runtime gate and tensor plumbing for DSA token sharding.

Only the query path is sharded: after ``q_b_proj`` and the ``w_kc`` absorb a
rank swaps "my heads for every token" for "every head for my tokens", and the
swap is undone after attention, before ``w_vc``. The KV write, the DCP context
gather and the indexer stay full width.

Composes with the indexer's query sharding
(``SGLANG_NPU_ENABLE_DSA_INDEXER_QUERY_SHARDING``): both pick the same rows,
which ``test_dsa_token_shard_indexer_row_agreement.py`` pins.
"""

from functools import lru_cache
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_token_shard_layout import (
    DsaTokenShardPlan,
    cumulative,
    plan_dsa_token_shard,
)
from sglang.srt.layers.dcp.layout import dcp_crop_free_extend
from sglang.srt.layers.layer_boundary import get_attn_tp_context
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_npu, print_info_once

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@lru_cache(maxsize=1)
def _dsa_token_shard_flag() -> bool:
    """Resolved once: it decides at model construction whether the full-head
    RadixAttention exists, so it must not change mid-run. Not a module-level
    read, which would run before any test could set it; tests call
    ``reset_dsa_token_shard_flags()``."""
    enabled = envs.SGLANG_NPU_ENABLE_DSA_TOKEN_SHARD.get()
    if enabled and envs.SGLANG_NPU_USE_MLAPO.get():
        # MLAPO writes the KV cache itself, at a slot mapping this has already
        # sliced. This defaults on, so it yields unless both were set.
        if envs.SGLANG_NPU_ENABLE_DSA_TOKEN_SHARD.is_set():
            raise ValueError(
                "SGLANG_NPU_ENABLE_DSA_TOKEN_SHARD does not compose with "
                "SGLANG_NPU_USE_MLAPO: the fused MLA preprocess writes the KV "
                "cache at a slot mapping the token shard has already sliced."
            )
        enabled = False
        print_info_once(
            "DSA token-shard is off because SGLANG_NPU_USE_MLAPO is on. Unset "
            "SGLANG_NPU_USE_MLAPO to get the sharded attention back"
        )
    return enabled


@lru_cache(maxsize=1)
def _dsa_token_shard_multi_request_flag() -> bool:
    return envs.SGLANG_NPU_ENABLE_DSA_TOKEN_SHARD_MULTI_REQUEST.get()


@lru_cache(maxsize=1)
def _dsa_token_shard_narrow_a2a_flag() -> bool:
    return envs.SGLANG_NPU_ENABLE_DSA_TOKEN_SHARD_NARROW_A2A.get()


def reset_dsa_token_shard_flags() -> None:
    """Re-read the env flags. For tests; never call this while serving."""
    _dsa_token_shard_flag.cache_clear()
    _dsa_token_shard_multi_request_flag.cache_clear()
    _dsa_token_shard_narrow_a2a_flag.cache_clear()


def dsa_token_shard_enabled() -> bool:
    """Asked at model build AND at forward, so both agree.

    ``is_npu()``: ``deepseek_v2.py`` is shared by every backend, and a true
    answer builds an extra full-head ``RadixAttention`` only NPU ever uses.
    """
    return _dsa_token_shard_flag() and is_npu() and get_parallel().attn_tp_size > 1


def dsa_token_shard_multi_request_enabled() -> bool:
    """Shard multi-request extends too, by passing full per-request KV lengths
    with ``sparse_mode=0`` instead of shortening them -- lengths are cumulative,
    so shortening one moves the next request's start. Safe only where the causal
    crop is not load-bearing, which ``dcp_crop_free_extend`` decides."""
    return _dsa_token_shard_multi_request_flag()


def dsa_token_shard_narrow_a2a_enabled() -> bool:
    """Exchange q before the ``w_kc`` absorb and the head output after ``w_vc``,
    so the wire carries 256-wide tensors instead of the 512-wide latent, in two
    collectives instead of three. Bitwise identical.

    Off by default: on A3 tp16 it is 4-8% faster at 6k tokens but 8-13% SLOWER
    below 3k, and costs 4.06 GiB of KV pool (2.13 GiB of weights plus
    2 x HCCL_BUFFSIZE, charged to the attention-TP communicator).
    """
    return _dsa_token_shard_narrow_a2a_flag() and dsa_token_shard_enabled()


def dsa_token_shard_attach_full_kv_b(self_attn) -> None:
    """Give this layer the whole attention-TP group's ``w_kc`` and ``w_vc``.

    The narrow all-to-all absorbs after the exchange, so a rank needs the
    weights of the heads it ends up with. Declines non-bf16 weights, whose
    scale tensors would have to travel too.
    """
    if not dsa_token_shard_narrow_a2a_enabled() or not getattr(
        self_attn, "use_dsa", False
    ):
        return
    w_kc = getattr(self_attn, "w_kc", None)
    w_vc = getattr(self_attn, "w_vc", None)
    if w_kc is None or w_vc is None:
        return
    wanted = [("w_kc", w_kc), ("w_vc", w_vc)]
    not_bf16 = [(n, w.dtype) for n, w in wanted if w.dtype != torch.bfloat16]
    if not_bf16:
        print_info_once(
            "DSA token-shard narrow all-to-all is off: it needs bf16 weights, got "
            + ", ".join(f"{n}={d}" for n, d in not_bf16)
        )
        return

    from sglang.srt.layers.dp_attention import attn_tp_all_gather_into_tensor

    tp = get_parallel().attn_tp_size
    gathered_bytes = 0
    for name, w in wanted:
        # Gather in the layout the tensor is physically in: the loader stores
        # w_kc transposed and the batched matmul is tuned for that, so
        # transposing round the gather avoids a second full-size copy.
        transposed = not w.is_contiguous()
        send = w.transpose(1, 2) if transposed else w
        if not send.is_contiguous():
            send = send.contiguous()
        full = w.new_empty((send.shape[0] * tp, *send.shape[1:]))
        attn_tp_all_gather_into_tensor(full, send)
        if transposed:
            full = full.transpose(1, 2)
        setattr(self_attn, f"{name}_full", full)
        gathered_bytes += full.numel() * full.element_size()
    print_info_once(
        "DSA token-shard narrow all-to-all is ON: every rank holds the full w_kc "
        f"and w_vc, {gathered_bytes / (1 << 20):.1f} MB per layer"
    )


# Distinguishes "no plan cached yet" from "cached, and it is None".
_MISSING = object()


def get_dsa_token_shard_plan(
    forward_batch: "ForwardBatch",
    index_topk: Optional[int] = None,
) -> Optional[DsaTokenShardPlan]:
    """This forward's token slice for this rank, or None. Cached on the batch.

    Extend only: decode is already fast under graph capture. Every refusal is
    logged once -- the indexer-sharding bug this superseded cost four weeks
    because its refusal was silent.
    """
    if not _dsa_token_shard_flag():
        return None
    cached = getattr(forward_batch, "npu_dsa_token_shard_plan", _MISSING)
    if cached is not _MISSING:
        return cached
    plan = _build_dsa_token_shard_plan(forward_batch, index_topk)
    forward_batch.npu_dsa_token_shard_plan = plan
    return plan


def _build_dsa_token_shard_plan(
    forward_batch, index_topk=None
) -> Optional[DsaTokenShardPlan]:
    parallel = get_parallel()
    if parallel.attn_tp_size <= 1:
        print_info_once("DSA token-shard is off: attention TP size is 1")
        return None

    mode = forward_batch.forward_mode
    if not mode.is_extend():
        return None
    if mode.is_draft_extend_v2() or mode.is_target_verify():
        print_info_once(
            f"DSA token-shard is off for {mode}: the draft and verify paths carry "
            "their own per-step sequence-length tables, which this does not build"
        )
        return None

    if get_attn_tp_context().input_scattered:
        # A scattered input is already sliced; slicing it again cuts twice.
        print_info_once(
            "DSA token-shard is off: the attention input is scattered across the "
            "TP group"
        )
        return None

    extend_lens = forward_batch.extend_seq_lens_cpu
    prefix_lens = forward_batch.extend_prefix_lens_cpu
    if not extend_lens or prefix_lens is None:
        print_info_once(
            "DSA token-shard is off: this extend carries no CPU length metadata"
        )
        return None

    plan = plan_dsa_token_shard(
        extend_lens,
        [p + e for p, e in zip(prefix_lens, extend_lens)],
        parallel.attn_tp_size,
        parallel.attn_tp_rank,
    )
    num_requests = sum(1 for n in extend_lens if n > 0)
    # Evaluated even for one request: dcp_crop_free_extend caches its answer on
    # the batch and the first caller's index_topk wins, and this is the caller
    # that holds the real one.
    lift_applies = _dsa_token_shard_multi_request_flag() and dcp_crop_free_extend(
        forward_batch, index_topk
    )
    if num_requests > 1 and not lift_applies:
        print_info_once(
            f"DSA token-shard is off for multi-request extends ({num_requests} "
            "requests here). SGLANG_NPU_ENABLE_DSA_TOKEN_SHARD_MULTI_REQUEST "
            "lifts this"
        )
        return None

    if plan.num_tokens < parallel.attn_tp_size:
        print_info_once(
            f"DSA token-shard is off for batches under {parallel.attn_tp_size} "
            f"tokens (this one has {plan.num_tokens}): the slice would be mostly "
            "padding"
        )
        return None

    # Logged on engage too: an unset flag returns before every refusal, so the
    # absence of a refusal proves nothing.
    print_info_once(
        f"DSA token-shard is ON: the attention query is sharded across "
        f"{parallel.attn_tp_size} ranks at extend"
    )
    return plan


def dsa_token_shard_cumulative_lens(
    forward_batch: "ForwardBatch", plan: DsaTokenShardPlan, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(cumulative query lens, cumulative key lens)`` as int32 device tensors,
    built once per forward. Per layer it was 156 blocking H2D copies a forward."""
    cached = getattr(forward_batch, "npu_dsa_token_shard_cu_lens", None)
    if cached is None:
        cached = (
            torch.tensor(cumulative(plan.query_lens), dtype=torch.int32, device=device),
            torch.tensor(cumulative(plan.key_lens), dtype=torch.int32, device=device),
        )
        forward_batch.npu_dsa_token_shard_cu_lens = cached
    return cached


def dsa_token_shard_slice(x: torch.Tensor, plan: DsaTokenShardPlan) -> torch.Tensor:
    """This rank's ``plan.rows`` rows, zero-padded at the tail: every rank must
    hand the all-to-all the same row count."""
    sliced = x[plan.local_start : plan.local_end]
    missing = plan.rows - sliced.shape[0]
    if missing <= 0:
        return sliced
    return torch.cat([sliced, sliced.new_zeros((missing, *x.shape[1:]))], dim=0)


def dsa_token_shard_redistribute_heads(
    x: torch.Tensor, plan: DsaTokenShardPlan
) -> torch.Tensor:
    """``[num_tokens, h, d]`` -> ``[rows, h * tp_size, d]``, by one all-to-all.

    Heads come out in global order: chunk ``s`` comes from rank ``s``, which a
    ColumnParallelLinear gives heads ``[s*h, (s+1)*h)``.
    """
    parallel = get_parallel()
    tp = parallel.attn_tp_size
    h, d = x.shape[1], x.shape[2]
    assert x.shape[0] <= plan.num_tokens_pad, (
        f"DSA token-shard was handed {x.shape[0]} rows but planned for at most "
        f"{plan.num_tokens_pad}. The batch is padded by a wider rule than "
        "attn_tp_size; the plan must be built from the padded width."
    )
    missing = plan.num_tokens_pad - x.shape[0]
    if missing > 0:
        x = torch.cat([x, x.new_zeros((missing, h, d))], dim=0)
    # reshape, not view: q_nope_out arrives non-contiguous from
    # npu_transpose_batchmatmul.
    send = x.reshape(tp, plan.rows, h, d).contiguous()
    recv = torch.empty_like(send)
    parallel.attn_tp_group.all_to_all_single(recv, send)
    return recv.permute(1, 0, 2, 3).reshape(plan.rows, tp * h, d)


def dsa_token_shard_restore_tokens(
    x: torch.Tensor, plan: DsaTokenShardPlan, num_rows: int
) -> torch.Tensor:
    """``[rows, h * tp_size, d]`` -> ``[num_rows, h, d]``, the inverse.

    ``num_rows`` must be the width handed to the redistribute: a width that
    arrived padded must leave padded, or the next layer's KV write indexes a
    narrower tensor than its slot map. Required, because defaulting it is what
    broke this. Rows past ``num_tokens`` are zeroed -- the operator never wrote
    them, and untouched device memory can be NaN.
    """
    parallel = get_parallel()
    tp = parallel.attn_tp_size
    h, d = x.shape[1] // tp, x.shape[2]
    assert x.shape[1] == h * tp, (
        f"DSA token-shard restore expects a full head set, got {x.shape[1]} for "
        f"tp_size {tp}"
    )
    assert plan.num_tokens <= num_rows <= plan.num_tokens_pad, (
        f"DSA token-shard restore asked for {num_rows} rows, outside "
        f"[{plan.num_tokens}, {plan.num_tokens_pad}]"
    )
    send = x.reshape(plan.rows, tp, h, d).permute(1, 0, 2, 3).contiguous()
    recv = torch.empty_like(send)
    parallel.attn_tp_group.all_to_all_single(recv, send)
    out = recv.reshape(plan.num_tokens_pad, h, d)[:num_rows]
    if num_rows > plan.num_tokens:
        out[plan.num_tokens :].zero_()
    return out
