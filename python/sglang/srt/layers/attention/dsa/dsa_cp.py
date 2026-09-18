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

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_cp_layout import DsaCpPlan, plan_dsa_cp_shard
from sglang.srt.layers.communicator import ScatterMode
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import print_info_once

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
    raise ValueError(
        "SGLANG_NPU_ENABLE_DSA_CP does not compose with SGLANG_NPU_USE_MLAPO. "
        "The fused MLA preprocess writes the KV cache itself, at a slot mapping "
        "DSA-CP has already sliced."
    )


class _Missing:
    """Distinguishes "no plan cached yet" from "cached, and it is None"."""


_MISSING = _Missing()


def dsa_cp_enabled() -> bool:
    """Whether DSA-CP is on for this process. Read at model build AND forward.

    Both sides must agree, so both ask here rather than reading the env var
    twice.
    """
    return _enable_dsa_cp and get_parallel().attn_tp_size > 1


def get_dsa_cp_plan(
    forward_batch: "ForwardBatch",
    layer_scatter_modes=None,
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

    plan = _build_dsa_cp_plan(forward_batch, layer_scatter_modes)
    forward_batch.npu_dsa_cp_plan = plan
    return plan


def _build_dsa_cp_plan(forward_batch, layer_scatter_modes) -> Optional[DsaCpPlan]:
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
    if sum(1 for n in extend_lens if n > 0) > 1:
        # ONE REQUEST PER EXTEND FORWARD, for this first version, and the reason
        # is the KV layout rather than the query arithmetic.
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
        # vLLM-Ascend does not hit this because it reads PA_BSND with a block
        # table, where the per-request length is independent of the layout
        # (``sfa_cp.py`` passes ``actual_seq_lengths_key`` un-cumulated). Lifting
        # the restriction means either that layout or gathering only the KV each
        # rank can see -- both real work, neither needed for the served shape,
        # which the scheduler already runs one request at a time ("#new-seq: 1").
        print_info_once(
            "DSA-CP is off for multi-request extends "
            f"({sum(1 for n in extend_lens if n > 0)} requests here); the "
            "cumulative KV lengths it would need describe the buffer's own "
            "request boundaries"
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
    all-to-all enough. The padded rows produce attention output that
    ``dsa_cp_restore_tokens`` trims off again.
    """
    parallel = get_parallel()
    tp = parallel.attn_tp_size
    h, d = x.shape[1], x.shape[2]
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


def dsa_cp_restore_tokens(x: torch.Tensor, plan: DsaCpPlan) -> torch.Tensor:
    """``[rows, h * tp_size, d]`` -> ``[num_tokens, h, d]``. Inverse of the above.

    Everything downstream of attention -- the ``w_vc`` batch-matmul, o_proj,
    the layer communicator -- expects this rank's own heads for the whole
    batch, exactly as it gets them without DSA-CP. So the sharding is undone
    here rather than propagated, which is what keeps the change contained to
    the attention call itself.
    """
    parallel = get_parallel()
    tp = parallel.attn_tp_size
    h, d = x.shape[1] // tp, x.shape[2]
    assert x.shape[1] == h * tp, (
        f"DSA-CP restore expects a full head set, got {x.shape[1]} for tp_size {tp}"
    )
    send = x.reshape(plan.rows, tp, h, d).permute(1, 0, 2, 3).contiguous()
    recv = torch.empty_like(send)
    parallel.attn_tp_group.all_to_all_single(recv, send)
    return recv.reshape(plan.num_tokens_pad, h, d)[: plan.num_tokens]
