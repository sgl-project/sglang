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

The arithmetic is in ``dsa_cp_layout``, split off on purpose: everything here
needs the runtime topology and real tensors, while the layout is plain integers
and can be replayed against a brute-force reference with no torch at all. Keep
it that way -- the clamp arithmetic is where a ragged batch goes wrong, and
being able to check it locally is worth the extra file.

**Only the query path is sharded here, not the whole attention block.**
vLLM-Ascend slices ``hidden_states`` at the block's entry and then has to
all-gather the latent KV, the rope key and the indexer's K back to full width
(``sfa_cp.py:388-426``). This slices later -- after ``fused_qkv_a_proj_with_mqa``
and the QK norm, which were already replicated per rank and so cost nothing
extra -- and only for ``q_lora``. The K side and the indexer keep running at
full width, which means:

- the KV cache write and the DCP context gather are **untouched**: they still
  see every token, at the unsliced ``out_cache_loc``;
- the indexer is **untouched**, and the older indexer-only query sharding
  (``SGLANG_NPU_ENABLE_DSA_INDEXER_QUERY_SHARDING``) still applies to it and
  still divides its work by ``attn_tp_size``. The two compose rather than
  conflict: that flag shards the indexer's queries and gathers the top-k back
  to full width, and this then takes its own slice of that result.

What is given up is the saving on the K-side projections and norms. They run
full width today, so this is not a regression -- just not an additional win.
What is kept is the entire reason for the exercise: ``SparseFlashAttention``,
3,174 ms of an 8,696 ms forward, bound by a per-query top-k KV read that falls
by ``attn_tp_size`` when the queries do.
"""

from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_cp_layout import (
    DsaCpPlan,
    cumulative,
    plan_dsa_cp_shard,
)
from sglang.srt.layers.communicator import ScatterMode
from sglang.srt.layers.dcp.layout import dcp_crop_free_extend
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_npu, print_info_once

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

# Read once, at import. This is not merely a fast path: the flag decides whether
# q_b_proj and kv_b_proj are built full-width or TP-sharded, which happens when
# the model is constructed, so flipping it later would leave the weights
# disagreeing with the forward.
_enable_dsa_cp = envs.SGLANG_NPU_ENABLE_DSA_CP.get()

if _enable_dsa_cp and envs.SGLANG_NPU_USE_MLAPO.get():
    # vLLM-Ascend refuses the same pair ("Fused preprocessing does not support
    # DSA-CP", sfa_cp.py:309). Their fused preprocess writes the KV cache from
    # inside the operator, at the batch's own slot mapping, which under DSA-CP
    # is the sliced one -- so the rows the slice does not own never get
    # written. Untested here either way; refuse rather than find out in an
    # accuracy run.
    #
    # Which one yields depends on who asked. DSA-CP defaults ON, so a user who
    # set only MLAPO never asked for this pair and must not be met with a hard
    # failure for a flag they did not touch -- DSA-CP steps aside and says so.
    # A user who set BOTH explicitly gets the error, because silently dropping
    # one of two things someone deliberately turned on is worse.
    if envs.SGLANG_NPU_ENABLE_DSA_CP.is_set():
        raise ValueError(
            "SGLANG_NPU_ENABLE_DSA_CP does not compose with "
            "SGLANG_NPU_USE_MLAPO. The fused MLA preprocess writes the KV cache "
            "itself, at a slot mapping DSA-CP has already sliced."
        )
    _enable_dsa_cp = False
    print_info_once(
        "DSA-CP is off because SGLANG_NPU_USE_MLAPO is on: the fused MLA "
        "preprocess writes the KV cache at a slot mapping DSA-CP would have "
        "sliced. Unset SGLANG_NPU_USE_MLAPO to get the sharded attention back"
    )


_enable_dsa_cp_multi_request = envs.SGLANG_NPU_ENABLE_DSA_CP_MULTI_REQUEST.get()


def dsa_cp_multi_request_enabled() -> bool:
    """Whether DSA-CP also shards batches carrying more than one request.

    **The lift is smaller than the restriction made it look.** The original
    refusal was not about the query arithmetic -- ``plan_dsa_cp_shard`` already
    computes a correct per-request query length for a slice that straddles
    request boundaries, and that is the part with 1,980 exhaustive cases behind
    it. It was about ``actual_seq_lengths_kv``: DSA-CP SHORTENED each request's
    entry so the operator's right-down causal crop would align query ``j`` to
    key ``K - Q + j``, and because those lengths are cumulative, shortening one
    moves where the next request starts in the buffer.

    So do not shorten them. Pass the full per-request lengths -- which is
    exactly what the non-DSA-CP path already passes, ``dcp_kv_indptr[1:]`` --
    and set ``sparse_mode`` to 0 so there is no crop to align.

    **That is only safe above a bound, which the first version of this got
    wrong.** The crop is not redundant: below ``index_topk`` the top-k selects
    every key it is offered (dsa_indexer.py:402), so it is not causal and the
    crop is what makes the result so. Stage A measured the cost of assuming
    otherwise -- prefill logprobs off by up to 2.09 against a 0.354 noise floor.
    So the lift applies only when every request in the batch has a prefix of at
    least ``index_topk``; ``dcp_crop_free_extend`` decides that once per forward
    and the attention backend reads the same answer. Below the bound the
    one-request refusal stands, exactly as before this flag existed.

    The DCP **decode** branch really does run with ``sparse_mode`` 0 -- but for a
    different reason than the one this used to claim. At decode every key in the
    buffer is already a past token, so there is nothing to mask. At extend the
    buffer holds the chunk's own later tokens.

    What the rank does not touch costs nothing: a request outside its slice gets
    query length 0, its KV rows stay in the buffer, and no query names them.

    **When this is on it applies to single-request extends too**, deliberately.
    One convention is easier to reason about than two, the shortening it
    replaces was never doing anything useful there, and -- the practical point
    -- it means the flag can be tested on an ordinary one-request tail with
    ``p12_logprob_cross_config.sh`` before anyone has to arrange a batch that
    carries several. A single-request run is the FIRST test of this flag, not an
    unaffected control.

    **Measured, 2026-09-21** (``glm5.2_testing/p13_concurrent_lift.sh``): three
    ~4k tails on a 958k cached prefix, sent as one batch, came out bitwise
    identical to the pre-lift path at all 12,311 tail positions -- on a server
    with the flag on against one with it off, whose single-request control also
    matched bitwise. That covers the query-length-0 entries most ranks get here.
    The batch took ~6.5 s against 9.21 s without the lift.

    Estimated, not measured: ~31-40 s of AISBench phase 2 (110.2 s), where
    84.7% of prefill tokens decline without this. Not composable with the packed read, which refuses
    multi-request batches for a different reason -- the all-gather is rank-major
    over the whole send, so no request is contiguous in it -- so those batches
    take the permuting gather and this takes the query sharding.
    """
    return _enable_dsa_cp_multi_request


class _Missing:
    """Distinguishes "no plan cached yet" from "cached, and it is None"."""


_MISSING = _Missing()


def dsa_cp_enabled() -> bool:
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
    return _enable_dsa_cp and is_npu() and get_parallel().attn_tp_size > 1


def get_dsa_cp_plan(
    forward_batch: "ForwardBatch",
    layer_scatter_modes=None,
    index_topk: Optional[int] = None,
) -> Optional[DsaCpPlan]:
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
    if not _enable_dsa_cp:
        return None

    cached = getattr(forward_batch, "npu_dsa_cp_plan", _MISSING)
    if cached is not _MISSING:
        return cached

    plan = _build_dsa_cp_plan(forward_batch, layer_scatter_modes, index_topk)
    forward_batch.npu_dsa_cp_plan = plan
    return plan


def _build_dsa_cp_plan(
    forward_batch, layer_scatter_modes, index_topk=None
) -> Optional[DsaCpPlan]:
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

    if (
        layer_scatter_modes is not None
        and layer_scatter_modes.attn_mode != ScatterMode.TP_ATTN_FULL
    ):
        # The slice assumes this rank was handed the whole batch, which is what
        # TP_ATTN_FULL means. Under any other mode it already holds a piece and
        # this would cut it twice.
        print_info_once(
            "DSA-CP is off: attention scatter mode is "
            f"{layer_scatter_modes.attn_mode}, not TP_ATTN_FULL"
        )
        return None

    extend_lens = forward_batch.extend_seq_lens_cpu
    prefix_lens = forward_batch.extend_prefix_lens_cpu
    if not extend_lens or prefix_lens is None:
        print_info_once("DSA-CP is off: this extend carries no CPU length metadata")
        return None

    plan = plan_dsa_cp_shard(
        extend_lens,
        [p + e for p, e in zip(prefix_lens, extend_lens)],
        parallel.attn_tp_size,
        parallel.attn_tp_rank,
    )
    multi_request = sum(1 for n in extend_lens if n > 0) > 1
    # The lift only applies where the causal crop is not load-bearing. It works
    # by NOT shortening the per-request KV lengths and dropping the crop
    # instead, and stage A measured what dropping that crop costs when the
    # top-k is not causal: prefill logprobs off by up to 2.09 against a 0.354
    # floor. Below the bound the refusal stands, exactly as before the lift.
    lift_applies = _enable_dsa_cp_multi_request and dcp_crop_free_extend(
        forward_batch, index_topk
    )
    if not lift_applies and multi_request:
        # ONE REQUEST PER EXTEND FORWARD unless the lift is enabled, and the
        # reason is the KV layout rather than the query arithmetic.
        #
        # At DCP extend the operator reads a non-paged TND buffer whose
        # ``actual_seq_lengths_kv`` is CUMULATIVE, so those offsets double as the
        # request boundaries inside the buffer. A rank's slice sees only part of
        # the last request it touches, and shortening that request's cumulative
        # entry would move where the NEXT request starts -- reading real KV from
        # the wrong offset, which this model answers fluently. With one request
        # there is no next request: its shortened length is a true prefix of the
        # buffer and the read is exact.
        #
        # ``SGLANG_NPU_ENABLE_DSA_CP_MULTI_REQUEST`` lifts it by not shortening
        # anything: see ``dsa_cp_multi_request_enabled``. Measured on AISBench
        # 2026-09-18, this refusal costs 84.7% of phase 2's tokens.
        print_info_once(
            "DSA-CP is off for multi-request extends "
            f"({sum(1 for n in extend_lens if n > 0)} requests here); the "
            "cumulative KV lengths it would need describe the buffer's own "
            "request boundaries. SGLANG_NPU_ENABLE_DSA_CP_MULTI_REQUEST lifts this"
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

    # Say so when it engages, not only when it refuses. Absence of a refusal
    # line proves nothing: an unset flag leaves this function before any of
    # them. That gap is exactly how indexer query sharding stayed silently off
    # for fifteen token counts in sixteen, for four weeks. The message is fixed
    # rather than per-shape so it lands once; the shapes are in the refusals.
    print_info_once(
        f"DSA-CP is ON: the attention query is sharded across "
        f"{parallel.attn_tp_size} ranks at extend"
    )
    return plan


def dsa_cp_cumulative_lens(
    forward_batch: "ForwardBatch", plan: DsaCpPlan, device: torch.device
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
    cached = getattr(forward_batch, "npu_dsa_cp_cu_lens", None)
    if cached is None:
        cached = (
            torch.tensor(cumulative(plan.query_lens), dtype=torch.int32, device=device),
            torch.tensor(cumulative(plan.key_lens), dtype=torch.int32, device=device),
        )
        forward_batch.npu_dsa_cp_cu_lens = cached
    return cached


def dsa_cp_slice(x: torch.Tensor, plan: DsaCpPlan) -> torch.Tensor:
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


def dsa_cp_redistribute_heads(x: torch.Tensor, plan: DsaCpPlan) -> torch.Tensor:
    """``[num_tokens, h, d]`` -> ``[rows, h * tp_size, d]``, by all-to-all.

    Turns "my heads for every token" into "every head for my tokens". Nothing
    is duplicated and nothing is dropped: the group holds the same
    (token, head) pairs before and after, they just move to a different owner.

    The head order that comes out is the global one. ``all_to_all_single``
    fills chunk ``s`` of the output from rank ``s``, and a ColumnParallelLinear
    gives rank ``s`` the contiguous head block ``[s*h, (s+1)*h)``, so laying the
    chunks out in rank order lays the heads out in head order.

    Padding: ``num_tokens`` is rounded up to ``num_tokens_pad`` with zero rows
    so every rank sends the same width, which is what makes a *single*
    all-to-all enough.

    **The batch usually arrives already padded to exactly that width.** SGLang
    rounds the token count up to a multiple of ``attn_tp_size``
    (``forward_batch_info.py:1454``, ``ceil_align``), which is the same
    arithmetic as ``num_tokens_pad``, so ``missing`` is normally 0 and the rows
    past ``num_tokens`` are SGLang's padding rather than ours. Either way the
    caller must be handed back the width it gave -- see
    ``dsa_cp_restore_tokens``.
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
    # reshape, not view: q_nope_out arrives from npu_transpose_batchmatmul
    # with a permuted output layout, and view() refuses a non-contiguous
    # tensor outright.
    send = x.reshape(tp, plan.rows, h, d).contiguous()
    recv = torch.empty_like(send)
    parallel.attn_tp_group.all_to_all_single(recv, send)
    return recv.permute(1, 0, 2, 3).reshape(plan.rows, tp * h, d)


def dsa_cp_restore_tokens(
    x: torch.Tensor, plan: DsaCpPlan, num_rows: int
) -> torch.Tensor:
    """``[rows, h * tp_size, d]`` -> ``[num_rows, h, d]``. Inverse of the above.

    Everything downstream of attention -- the ``w_vc`` batch-matmul, o_proj,
    the layer communicator -- expects this rank's own heads for the whole
    batch, exactly as it gets them without DSA-CP. So the sharding is undone
    here rather than propagated, which is what keeps the change contained to
    the attention call itself.

    ``num_rows`` MUST be the width handed to ``dsa_cp_redistribute_heads``, and
    it is a required argument because defaulting it is what broke this. SGLang
    pads the batch to ``ceil_align(tokens, attn_tp_size)``
    (``forward_batch_info.py:1454``), so on a ragged chunk the real tensors
    carry **more** rows than ``extend_seq_lens_cpu`` sums to -- 12,912 against
    12,911 in the crash that found this. Returning ``plan.num_tokens`` rows
    silently narrowed every layer's output by one row, while
    ``out_cache_loc`` kept its padded length; the next layer's KV write then
    indexed a 12,911-row tensor with an index built from a 12,912-entry slot
    map and the gather asserted on device.

    This is the same lesson as the indexer's ``gather()``, which takes
    ``num_tokens`` and returns that many rows for exactly this reason
    (``63adee22a5``). A width that arrives padded must leave padded.

    The rows past ``plan.num_tokens`` are zeroed rather than left as the
    all-to-all delivered them. The operator was told to process only
    ``query_lens`` queries, so its output there is untouched device memory,
    which can be NaN; the unsharded path leaves the same rows equally
    undefined, but there it never travels through a collective. Zero is
    finite, deterministic, and costs at most ``tp_size - 1`` rows.
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
