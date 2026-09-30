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

"""Runtime gate and tensor plumbing for DSA-CP token sharding.

The arithmetic lives in ``dsa_token_shard_layout`` on purpose: everything here needs the
runtime topology and real tensors, while the layout is plain integers and can
be replayed against a brute-force reference with no torch at all.

**Only the query path is sharded here, not the whole attention block.** The
exchange happens after ``q_b_proj`` and the absorb through ``w_kc``: the rank
swaps "my heads for every token" for "every head for my tokens"
(``dsa_token_shard_redistribute_heads``) and undoes it after attention
(``dsa_token_shard_restore_tokens``), before ``w_vc``. Nothing slices ``q_lora`` -- an
earlier version of this docstring said it did, which hid the cost below.

What travels is therefore the 512-wide absorbed latent and its output, each a
fixed linear function of the narrower tensor beside it (``q`` is 192 wide,
the head output 128). Exchanging on the narrow side instead would move 3.4x
fewer bytes in 2 collectives rather than 3, at the price of a full ``w_kc`` and
``w_vc`` per rank -- W1 in ``DSA_CP_HANDOFF_2026-09-24.md``, which asks for a
probe before any code.

The KV cache write, the DCP context gather and the indexer all keep running at
full width, so the older indexer-only query sharding
(``SGLANG_NPU_ENABLE_DSA_INDEXER_QUERY_SHARDING``) composes with this rather
than conflicting: it shards the indexer and gathers the top-k back, and this
takes its own slice of that result. Both pick the *same* rows, so that gather
is pure overhead whenever the two run together (W2). The agreement they rely on
is pinned by ``test/registered/dcp/test_dsa_token_shard_indexer_row_agreement.py``.

What is given up is the saving on the K-side projections, which run full width
today anyway. What is kept is the reason for the exercise: sparse attention is
bound by a per-query top-k KV read, which falls by ``attn_tp_size`` when the
queries do.
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
    """``SGLANG_NPU_ENABLE_DSA_TOKEN_SHARD``, resolved against MLAPO. Cached per process.

    Not a module-level read. That ran at import, before any test could set the
    variable, so no test could reach the off path -- and the conditional gates
    this feature may grow will need exactly that. Still resolved once: the flag
    decides whether the full-head RadixAttention is built, which happens at model
    construction, so it must not change mid-run. Tests call
    ``reset_dsa_token_shard_flags()``.
    """
    enabled = envs.SGLANG_NPU_ENABLE_DSA_TOKEN_SHARD.get()
    if enabled and envs.SGLANG_NPU_USE_MLAPO.get():
        # The fused MLA preprocess writes the KV cache at a slot mapping DSA-CP
        # has already sliced. DSA-CP defaults on, so it yields unless both were
        # set explicitly.
        if envs.SGLANG_NPU_ENABLE_DSA_TOKEN_SHARD.is_set():
            raise ValueError(
                "SGLANG_NPU_ENABLE_DSA_TOKEN_SHARD does not compose with "
                "SGLANG_NPU_USE_MLAPO. The fused MLA preprocess writes the KV "
                "cache itself, at a slot mapping DSA-CP has already sliced."
            )
        enabled = False
        print_info_once(
            "DSA-CP is off because SGLANG_NPU_USE_MLAPO is on: the fused MLA "
            "preprocess writes the KV cache at a slot mapping DSA-CP would have "
            "sliced. Unset SGLANG_NPU_USE_MLAPO to get the sharded attention back"
        )
    return enabled


@lru_cache(maxsize=1)
def _dsa_token_shard_multi_request_flag() -> bool:
    return envs.SGLANG_NPU_ENABLE_DSA_TOKEN_SHARD_MULTI_REQUEST.get()


@lru_cache(maxsize=1)
def _dsa_token_shard_narrow_a2a_flag() -> bool:
    return envs.SGLANG_NPU_ENABLE_DSA_TOKEN_SHARD_NARROW_A2A.get()


def dsa_token_shard_narrow_a2a_enabled() -> bool:
    """Whether to exchange the query before the ``w_kc`` absorb, not after it.

    DSA-CP swaps heads for tokens after ``q_b_proj`` and the absorb, so the wire
    carries the 512-wide absorbed latent in and the 512-wide attention output
    back. Both are linear functions of narrower tensors beside them: q is 256
    wide out of ``q_b_proj`` (``qk_nope`` 192 + rope 64) and the head output is
    ``v_head_dim`` 256. Exchanging on the narrow side moves 512 values per
    row-head instead of 1088, in two collectives instead of three.

    Exact either way: ``npu_transpose_batchmatmul`` is a per-head, per-row
    product and the all-to-all only permutes (token, head) pairs, so absorbing
    then permuting equals permuting then absorbing -- provided each rank holds
    the ``w_kc``/``w_vc`` of the head it ends up with, which is what the full
    weights buy. Verified bitwise at 2018 / 6021 / 16022 tokens.

    Off by default. Measured on A3 tp16: 3.7-7.8% faster at 6007 tokens and
    3-4.5% at 16007, but **8-13% slower below 3000**, and it costs 4.06 GiB of
    KV pool -- 2.13 GiB of weights plus 2 x ``HCCL_BUFFSIZE``, which HCCL
    charges the attention-TP communicator on its first sizeable collective.
    ``DSA_CP_HANDOFF_2026-09-24.md`` section 8 has the tables.
    """
    return _dsa_token_shard_narrow_a2a_flag() and dsa_token_shard_enabled()


def dsa_token_shard_attach_full_kv_b(self_attn) -> None:
    """Give this layer the whole attention-TP group's ``w_kc`` and ``w_vc``.

    Called once per layer from the weight loader, after it has split
    ``kv_b_proj`` and applied the NPU layout; the loader itself does not change.
    Both legs of the narrow all-to-all absorb after the exchange, so they need
    the weights of the head the rank ends up holding rather than the head it was
    assigned. Does nothing when the feature is off. Declines on non-bf16
    weights: the quantized layouts carry scale tensors that would have to be
    gathered with them.
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
            "DSA-CP narrow all-to-all is off: it needs bf16 weights, got "
            + ", ".join(f"{n}={d}" for n, d in not_bf16)
        )
        return

    from sglang.srt.layers.dp_attention import attn_tp_all_gather_into_tensor

    tp = get_parallel().attn_tp_size
    gathered_bytes = 0
    for name, w in wanted:
        # Gather in the layout the tensor is physically in, then view back. The
        # loader stores w_kc as [h, qk_nope, kv_lora] laid out transposed, and
        # the batched matmul is tuned for that, so transposing here is free both
        # ways and no second full-size copy is needed. w_vc is plain contiguous.
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
        "DSA-CP narrow all-to-all is ON: every rank holds the full w_kc and "
        f"w_vc, {gathered_bytes / (1 << 20):.1f} MB per layer"
    )


def reset_dsa_token_shard_flags() -> None:
    """Re-read the DSA-CP env flags. For tests; never call this while serving."""
    _dsa_token_shard_flag.cache_clear()
    _dsa_token_shard_multi_request_flag.cache_clear()
    _dsa_token_shard_narrow_a2a_flag.cache_clear()


def dsa_token_shard_multi_request_enabled() -> bool:
    """Whether DSA-CP also shards batches carrying more than one request.

    The restriction was never about the query arithmetic -- ``plan_dsa_token_shard``
    already handles a slice that straddles request boundaries. It was about
    ``actual_seq_lengths_kv``: DSA-CP shortened each request's entry so the
    operator's causal crop would align, and because those lengths are
    cumulative, shortening one moves where the next request starts.

    So do not shorten them. Pass the full per-request lengths, the same ones
    the unsharded path passes, and set ``sparse_mode`` to 0 so there is no crop
    to align. That is safe only above a bound: below ``index_topk`` the top-k
    selects every key it is offered, so the crop is what makes the result
    causal. ``dcp_crop_free_extend`` decides that once per forward and the
    attention backend reads the same answer.

    When on, this applies to single-request extends too, deliberately -- one
    convention rather than two, and it can be tested on an ordinary tail.
    """
    return _dsa_token_shard_multi_request_flag()


class _Missing:
    """Distinguishes "no plan cached yet" from "cached, and it is None"."""


_MISSING = _Missing()


def dsa_token_shard_enabled() -> bool:
    """Whether DSA-CP is on for this process. Read at model build AND forward.

    Both sides must agree, so both ask here rather than reading the env var
    twice.

    ``is_npu()`` guards it because this is asked from ``deepseek_v2.py``, which
    every backend shares, and a true answer builds an extra full-head
    ``RadixAttention`` at the same ``layer_id``. The forward path that would use
    it exists only under ``hardware_backend/npu``, so on any other device that
    module is dead weight registered with the attention backend. Harmless while
    the flag defaulted off; not something to discover by flipping the default.
    """
    return _dsa_token_shard_flag() and is_npu() and get_parallel().attn_tp_size > 1


def get_dsa_token_shard_plan(
    forward_batch: "ForwardBatch",
    index_topk: Optional[int] = None,
) -> Optional[DsaTokenShardPlan]:
    """This forward's token slice for this rank, or None if DSA-CP is off here.

    Built once per forward and cached on the batch: the split is by token
    position and does not vary by layer.

    **Extend only, deliberately, for the first version.** The measured 34% is
    entirely in the tail prefill, and decode is already fast from graph capture,
    so leaving decode on the unsharded path keeps the capture untouched and
    removes most of the risk. vLLM-Ascend does slice at decode too
    (``sfa_cp.py``), and composes with DCP by MRO; that is a later step, not a
    prerequisite.

    Every refusal is logged once. The indexer-sharding bug this supersedes cost
    four weeks precisely because its refusal was silent.
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
        print_info_once("DSA-CP is off: attention TP size is 1, nothing to shard")
        return None

    mode = forward_batch.forward_mode
    if not mode.is_extend():
        # Decode and the speculative modes keep the unsharded path. Not a
        # defect: see the docstring.
        return None
    if mode.is_draft_extend_v2() or mode.is_target_verify():
        print_info_once(
            f"DSA-CP is off for {mode}: the draft and verify paths carry their "
            "own per-step sequence-length tables, which this does not build yet"
        )
        return None

    if get_attn_tp_context().input_scattered:
        # The slice assumes this rank was handed the whole batch; a
        # scattered attention input would cut a slice twice. Upstream
        # replaced LayerScatterModes with the layer-boundary contracts,
        # which ask this of the forward rather than of the layer.
        print_info_once(
            "DSA-CP is off: the attention input is scattered across the TP group"
        )
        return None

    extend_lens = forward_batch.extend_seq_lens_cpu
    prefix_lens = forward_batch.extend_prefix_lens_cpu
    if not extend_lens or prefix_lens is None:
        print_info_once("DSA-CP is off: this extend carries no CPU length metadata")
        return None

    plan = plan_dsa_token_shard(
        extend_lens,
        [p + e for p, e in zip(prefix_lens, extend_lens)],
        parallel.attn_tp_size,
        parallel.attn_tp_rank,
    )
    multi_request = sum(1 for n in extend_lens if n > 0) > 1
    # The lift applies only where the causal crop is not load-bearing.
    lift_applies = _dsa_token_shard_multi_request_flag() and dcp_crop_free_extend(
        forward_batch, index_topk
    )
    if not lift_applies and multi_request:
        # The restriction is the KV layout, not the query arithmetic: the
        # buffer's cumulative KV lengths double as its request boundaries.
        print_info_once(
            "DSA-CP is off for multi-request extends "
            f"({sum(1 for n in extend_lens if n > 0)} requests here); the "
            "cumulative KV lengths it would need describe the buffer's own "
            "request boundaries. SGLANG_NPU_ENABLE_DSA_TOKEN_SHARD_MULTI_REQUEST lifts this"
        )
        return None

    if plan.num_tokens < parallel.attn_tp_size:
        # Fewer tokens than ranks: most ranks would hold nothing but padding and
        # the all-gathers would move more than the attention saves.
        print_info_once(
            f"DSA-CP is off for batches under {parallel.attn_tp_size} tokens "
            f"(this one has {plan.num_tokens}); the slice would be mostly padding"
        )
        return None

    # Log when it engages, not only when it refuses: an unset flag leaves this
    # function before every refusal, so no refusal line proves nothing.
    print_info_once(
        f"DSA-CP is ON: the attention query is sharded across "
        f"{parallel.attn_tp_size} ranks at extend"
    )
    return plan


def dsa_token_shard_cumulative_lens(
    forward_batch: "ForwardBatch", plan: DsaTokenShardPlan, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The operator's two length vectors as device tensors, built ONCE per forward.

    Returns ``(cumulative query lengths, cumulative key lengths)``. Both carry
    one entry per request, and neither varies by layer -- the split is by token
    position. Building them at the point of use meant
    ``torch.tensor(list, device=npu)`` twice per layer, which at 78 layers is
    **156 host-to-device copies per forward**. Each one drains the queue before
    the next kernel is enqueued, so the cost is a synchronisation rather than a
    copy, and it lands squarely inside the win this feature exists to produce.

    Cached on the batch beside the plan, and built as a pair because a forward
    that needs one needs the other.

    ``int32`` on the query's device, which is what both call sites ask for, so
    their ``.to()`` is a no-op rather than another copy.
    """
    cached = getattr(forward_batch, "npu_dsa_token_shard_cu_lens", None)
    if cached is None:
        cached = (
            torch.tensor(cumulative(plan.query_lens), dtype=torch.int32, device=device),
            torch.tensor(cumulative(plan.key_lens), dtype=torch.int32, device=device),
        )
        forward_batch.npu_dsa_token_shard_cu_lens = cached
    return cached


def dsa_token_shard_slice(x: torch.Tensor, plan: DsaTokenShardPlan) -> torch.Tensor:
    """This rank's ``plan.rows`` rows of a full-width, token-major tensor.

    Always returns exactly ``rows`` rows, padding the tail when the slice runs
    past the real tokens. Every rank must produce the same row count or the
    all-gather that rebuilds full width has nothing regular to work with, and
    the padded rows are cheap: they are the last rank's only, at most
    ``tp_size - 1`` of them, and their attention output is discarded.
    """
    sliced = x[plan.local_start : plan.local_end]
    missing = plan.rows - sliced.shape[0]
    if missing <= 0:
        return sliced
    pad = sliced.new_zeros((missing, *x.shape[1:]))
    return torch.cat([sliced, pad], dim=0)


def dsa_token_shard_redistribute_heads(
    x: torch.Tensor, plan: DsaTokenShardPlan
) -> torch.Tensor:
    """``[num_tokens, h, d]`` -> ``[rows, h * tp_size, d]``, by all-to-all.

    Turns "my heads for every token" into "every head for my tokens". Nothing
    is duplicated and nothing is dropped: the group holds the same
    (token, head) pairs before and after, they just move to a different owner.

    The head order that comes out is the global one: ``all_to_all_single``
    fills output chunk ``s`` from rank ``s``, and a ColumnParallelLinear gives
    rank ``s`` the contiguous head block ``[s*h, (s+1)*h)``.

    ``num_tokens`` is rounded up to ``num_tokens_pad`` with zero rows so every
    rank sends the same width, which is what makes one all-to-all enough.
    SGLang usually pads the batch to exactly that width already.
    """
    parallel = get_parallel()
    tp = parallel.attn_tp_size
    h, d = x.shape[1], x.shape[2]
    assert x.shape[0] <= plan.num_tokens_pad, (
        f"DSA-CP was handed {x.shape[0]} rows but planned for at most "
        f"{plan.num_tokens_pad} ({plan.num_tokens} tokens aligned to {tp}). "
        "The batch is padded by a wider rule than attn_tp_size; the plan must "
        "be built from the padded width, not from extend_seq_lens_cpu."
    )
    missing = plan.num_tokens_pad - x.shape[0]
    if missing > 0:
        x = torch.cat([x, x.new_zeros((missing, h, d))], dim=0)
    # reshape, not view: q_nope_out arrives non-contiguous from
    # npu_transpose_batchmatmul and view() refuses that.
    send = x.reshape(tp, plan.rows, h, d).contiguous()
    recv = torch.empty_like(send)
    parallel.attn_tp_group.all_to_all_single(recv, send)
    return recv.permute(1, 0, 2, 3).reshape(plan.rows, tp * h, d)


def dsa_token_shard_restore_tokens(
    x: torch.Tensor, plan: DsaTokenShardPlan, num_rows: int
) -> torch.Tensor:
    """``[rows, h * tp_size, d]`` -> ``[num_rows, h, d]``. Inverse of the above.

    Everything downstream of attention -- ``w_vc``, o_proj, the layer
    communicator -- expects this rank's own heads for the whole batch, so the
    sharding is undone here rather than propagated.

    ``num_rows`` must be the width handed to ``dsa_token_shard_redistribute_heads``: a
    width that arrives padded must leave padded, or the next layer's KV write
    indexes a narrower tensor than its slot map describes. It is required
    rather than defaulted because defaulting it is what broke this.

    Rows past ``plan.num_tokens`` are zeroed: the operator was told to process
    only ``query_lens`` queries, so its output there is untouched device
    memory, which can be NaN.
    """
    parallel = get_parallel()
    tp = parallel.attn_tp_size
    h, d = x.shape[1] // tp, x.shape[2]
    assert x.shape[1] == h * tp, (
        f"DSA-CP restore expects a full head set, got {x.shape[1]} for tp_size {tp}"
    )
    assert plan.num_tokens <= num_rows <= plan.num_tokens_pad, (
        f"DSA-CP restore asked for {num_rows} rows, outside "
        f"[{plan.num_tokens}, {plan.num_tokens_pad}]"
    )
    send = x.reshape(plan.rows, tp, h, d).permute(1, 0, 2, 3).contiguous()
    recv = torch.empty_like(send)
    parallel.attn_tp_group.all_to_all_single(recv, send)
    out = recv.reshape(plan.num_tokens_pad, h, d)[:num_rows]
    if num_rows > plan.num_tokens:
        out[plan.num_tokens :].zero_()
    return out
