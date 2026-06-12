# Copyright 2023-2024 SGLang Team
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

"""Context-parallel free functions and per-forward metadata.

This module is the relocated home of the pre-strategy ``layers/utils/cp_utils.py``
(it carries the MLA-prefill-CP and batch>1 logic merged in #23292 / #23269,
verbatim). The :class:`ContextParallelStrategy` classes in ``cp/strategy.py``
delegate to these functions, and the per-layer communicator + attention
backends call them directly.

``ContextParallelMetadata`` is re-exported here for compatibility; the
strategy-specific metadata dataclasses live with their concrete strategies.
"""

from itertools import accumulate
from typing import TYPE_CHECKING, Callable, List

import torch

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.cp.collectives import (
    cp_all_gather_reorganized_into_tensor,
    cp_all_gather_reorganized_into_tensor_kv_cache,
)
from sglang.srt.layers.cp.interleave import InterleaveContextParallelMetadata
from sglang.srt.layers.cp.zigzag import ContextParallelMetadata
from sglang.srt.layers.dp_attention import (
    attn_cp_all_gather_into_tensor,
    get_attention_cp_group,
    get_attention_cp_rank,
    get_attention_cp_size,
    is_allocation_symmetric,
)
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
from sglang.srt.server_args import get_global_server_args

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def is_prefill_context_parallel_enabled():
    return get_global_server_args().enable_prefill_context_parallel


def is_prefill_cp_in_seq_split():
    return (
        is_prefill_context_parallel_enabled()
        and get_global_server_args().prefill_cp_mode == "in-seq-split"
    )


def get_cp_padding_align_size() -> int:
    """Token-count alignment for CP padding of global_num_tokens: 2 * cp_size
    for zigzag (in-seq-split) CP, otherwise cp_size (1 when CP is off, so the
    padding is a no-op; extra padding breaks EAGLE/MTP draft prefill, see
    #23269). Keep prepare_mlp_sync_batch and cal_padded_tokens consistent
    through this helper.
    """
    from sglang.srt.layers.attention.dsa.utils import is_dsa_prefill_cp_in_seq_split

    attn_cp_size = get_attention_cp_size()
    if is_prefill_cp_in_seq_split() or is_dsa_prefill_cp_in_seq_split():
        return attn_cp_size * 2
    return attn_cp_size


def is_mla_prefill_cp_enabled() -> bool:
    sa = get_global_server_args()
    return sa.enable_prefill_context_parallel and sa.use_mla_backend


def is_dsa_enable_prefill_cp():
    """True when DSA prefill CP is enabled through the normalized server args."""
    from sglang.srt.layers.cp.strategy import is_cp_enabled

    try:
        sa = get_global_server_args()
    except ValueError:
        return False
    return is_cp_enabled() and bool(getattr(sa, "_is_dsa_model_arch", False))


def is_dsa_prefill_cp_in_seq_split():
    from sglang.srt.layers.cp.strategy import is_zigzag

    return is_dsa_enable_prefill_cp() and is_zigzag()


def is_dsa_prefill_cp_round_robin_split():
    from sglang.srt.layers.cp.strategy import is_interleave

    return is_dsa_enable_prefill_cp() and is_interleave()


def can_dsa_prefill_cp_round_robin_split(forward_batch: "ForwardBatch"):
    from sglang.srt.layers.cp.strategy import _get_cp_strategy, is_interleave

    if not is_dsa_enable_prefill_cp() or not is_interleave():
        return False
    strategy = _get_cp_strategy()
    if strategy is None:
        return False
    # ForwardBatch.prepare_mlp_sync_batch pads input_ids/out_cache_loc/positions
    # to the CP-aligned token count before attention metadata is built. Use that
    # padded count so DSA metadata makes the same split/no-split decision as the
    # model-entry CP path.
    input_ids = getattr(forward_batch, "input_ids", None)
    num_tokens = (
        len(input_ids)
        if input_ids is not None
        else sum(forward_batch.extend_seq_lens_cpu)
    )
    return strategy.can_apply(num_tokens, forward_batch)


def mla_use_prefill_cp(forward_batch, mla_enable_prefill_cp=None):
    if mla_enable_prefill_cp is None:
        mla_enable_prefill_cp = is_mla_prefill_cp_enabled()
    return (
        forward_batch.attn_cp_metadata is not None
        and mla_enable_prefill_cp
        and forward_batch.forward_mode.is_context_parallel_extend()
    )


def can_cp_split(seq_len: int, cp_size: int, forward_batch):
    # Base conditions: CP must be enabled, size > 1, and this must be a
    # CP-extend (prefill) step. The seq_len // (cp_size * 2) check ensures
    # the load-balancing split into 2 * cp_size blocks is non-degenerate.
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    cur_cp_seq_len = seq_len // (cp_size * 2)
    if not (
        cur_cp_seq_len != 0
        and cp_size > 1
        # prepare_context_parallel_metadata hard-codes bs_per_cp_group = 1;
        # guard explicitly to avoid silent mis-partitioning under continuous batching.
        and forward_batch.forward_mode.is_context_parallel_extend()
        # is_context_parallel_extend() returns True for MIXED (prefill+decode
        # in one step), but the zigzag split only makes sense on pure extend.
        and forward_batch.forward_mode != ForwardMode.MIXED
        and is_prefill_context_parallel_enabled()
    ):
        return False

    # Per-sequence guards for bs > 1. Every sequence must be long enough for
    # the 2*cp_size-way split. A sub-threshold request reaching this point
    # means the scheduler failed to filter it out and a silent non-CP
    # fallback would have masked the bug -- raise instead. Per-sequence
    # radix-cache prefix is supported: prefix is baked into kv_len_prev/next
    # via prefix_offsets[s] inside prepare_context_parallel_metadata.
    extend_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
    if extend_lens is None:
        return True

    cp_min = cp_size * 2
    for L in extend_lens:
        if L < cp_min:
            # A sub-threshold request cannot be zigzag-split into 2*cp_size
            # blocks; fall back to a normal (non-CP) prefill for this batch
            # instead of failing. Happens e.g. when a radix-cache prefix hit
            # leaves only a few unique extend tokens.
            return False

    return True


def maybe_prepare_cp_forward(
    forward_batch,
    attn_backend,
    input_embeds: torch.Tensor | None = None,
) -> None:
    from sglang.srt.layers.cp.strategy import get_cp_strategy

    strategy = get_cp_strategy()
    if strategy is None or not forward_batch.forward_mode.is_context_parallel_extend():
        return

    token_source = (
        forward_batch.input_ids if forward_batch.input_ids is not None else input_embeds
    )
    if token_source is None:
        return

    num_tokens = len(token_source)
    if not strategy.can_apply(num_tokens, forward_batch):
        forward_batch.attn_cp_metadata = None
        return

    seqs_len = forward_batch.seq_lens_cpu
    if hasattr(seqs_len, "tolist"):
        seqs_len = seqs_len.tolist()
    extend_seqs_len = getattr(forward_batch, "extend_seq_lens_cpu", None)
    if hasattr(extend_seqs_len, "tolist"):
        extend_seqs_len = extend_seqs_len.tolist()
    forward_batch.attn_cp_metadata = strategy.build_metadata(
        num_tokens, seqs_len, extend_seqs_len
    )

    metadata = getattr(attn_backend, "forward_metadata", None)
    core_meta = getattr(metadata, "core_attn_metadata", None)
    if core_meta is None:
        return
    strategy.reindex_attn_metadata(core_meta)
    if getattr(metadata, "indexer_metadata", None) is not None and hasattr(
        attn_backend, "init_forward_metadata_indexer"
    ):
        metadata.indexer_metadata = attn_backend.init_forward_metadata_indexer(
            core_meta
        )


def maybe_cp_split_before_forward(model, forward_batch, kwargs: dict) -> None:
    from sglang.srt.layers.cp.strategy import get_cp_strategy

    strategy = get_cp_strategy()
    if (
        strategy is None
        or forward_batch.attn_cp_metadata is None
        or not getattr(model, "use_cp_v2_model_runner_split", False)
    ):
        return

    input_embeds = kwargs.get("input_embeds")
    if (
        input_embeds is None
        and forward_batch.input_ids is not None
        and hasattr(model, "get_input_embeddings")
    ):
        embed_layer = model.get_input_embeddings()
        input_embeds = embed_layer(forward_batch.input_ids)

    sharded_input_embeds = strategy.split_before_forward(
        forward_batch,
        forward_batch.input_ids,
        forward_batch.positions,
        input_embeds,
    )
    if sharded_input_embeds is not None:
        kwargs["input_embeds"] = sharded_input_embeds


def cp_split_and_rebuild_data(forward_batch, input_: torch.Tensor):
    from sglang.srt.layers.attention.dsa.utils import dsa_cp_round_robin_split_data

    if is_dsa_prefill_cp_round_robin_split():
        cp_size = get_attention_cp_size()
        assert (
            input_.shape[0] % cp_size == 0
        ), f"Expect input shape 0 can divided by cp size, but got input shape {input_.shape}, cp size {cp_size}"
        return dsa_cp_round_robin_split_data(input_)

    input_list = list(
        torch.split(input_, forward_batch.attn_cp_metadata.split_list, dim=0)
    )
    result = torch.cat(
        [input_list[i] for i in forward_batch.attn_cp_metadata.zigzag_index], dim=0
    )
    if input_.ndim == 1:
        return result.contiguous()
    result = result.view(-1, input_.shape[-1])
    return result


def cp_split_and_rebuild_position(forward_batch, positions: torch.Tensor):
    from sglang.srt.layers.attention.dsa.utils import dsa_cp_round_robin_split_data

    if is_dsa_prefill_cp_round_robin_split():
        cp_size = get_attention_cp_size()
        assert positions.shape[0] % cp_size == 0, (
            f"Expect positions shape 0 can divided by cp size, but got positions shape {positions.shape}, "
            f"cp size {cp_size}"
        )
        return dsa_cp_round_robin_split_data(positions)

    position_id_list = list(
        torch.split(positions, forward_batch.attn_cp_metadata.split_list, dim=-1)
    )
    positions = torch.cat(
        [position_id_list[i] for i in forward_batch.attn_cp_metadata.zigzag_index],
        dim=-1,
    )
    return positions


def cp_round_robin_input_ids(input_ids):
    """
    input input_ids:
    rank0~7: 0,1,2,3,4,5,...

    output input_ids:
    a2a none:
    rank0~7: 0,8,16,...,1,9,17,...,2,10,18,...

    not a2a none:
    rank0: 0,8,16,...
    rank1: 1,9,17,...
    rank2: 2,10,18,...
    ...
    """
    cp_size = get_attention_cp_size()
    cp_rank = get_attention_cp_rank()
    if get_moe_a2a_backend().is_none():
        input_ids = input_ids.reshape(-1, cp_size).T.flatten()
    else:
        input_ids = input_ids[cp_rank::cp_size].contiguous()
    return input_ids


def cp_all_gather_rerange_output(input_tensor, cp_size, forward_batch, stream):
    """
    # for in-seq-split
    |   +-----------before allgather------------+|
    |   | dp_atten_tp0: block0, block7 |
    |   | dp_atten_tp1: block1, block6 |
    |   | dp_atten_tp2: block2, block5 |
    |   | dp_atten_tp3: block3, block4 |
    |
    |   +----------before rerange---------------+|
    | block0 | block7 | block1 | block6 | block2 | block5 | block3 | block4 |
    |
    |   +--------------result-------------------+
    | block0 | block1 | block2 | block3 | block4 | block5 | block6 | block7 |
    |   +-------------------------+

    # for round-robin-split
    |   +-----------before allgather------------+|
    | dp_atten_tp0: token0, token4, token8, token12, token16, ... |
    | dp_atten_tp1: token1, token5, token9, token13, token17, ... |
    | dp_atten_tp2: token2, token6, token10, token14, token18, ... |
    | dp_atten_tp3: token3, token7, token11, token15, token19, ... |
    |
    |   +--------------result-------------------+
    | token0, token1, token2, token3, token4, token5, token6, token7, ...
    |   +-------------------------+
    """
    if is_dsa_prefill_cp_round_robin_split():
        with use_symmetric_memory(
            get_attention_cp_group(), disabled=not is_allocation_symmetric()
        ):
            output_tensor = input_tensor.new_empty(
                (input_tensor.shape[0] * cp_size, *input_tensor.shape[1:]),
            )
        attn_cp_all_gather_into_tensor(
            output_tensor,
            input_tensor,
        )
        out_shape = output_tensor.shape
        output_tensor = (
            output_tensor.view(cp_size, -1, *out_shape[1:])
            .transpose(0, 1)
            .reshape(out_shape)
        )
        return output_tensor

    # TODO: Do we need to remove the padding here?
    bs_seq_len, hidden_size = input_tensor.shape
    output_tensor = cp_all_gather_reorganized_into_tensor(
        input_tensor,
        cp_size,
        forward_batch,
        stream,
    )
    outputs_list = list(
        torch.split(
            output_tensor, forward_batch.attn_cp_metadata.reverse_split_len, dim=0
        )
    )
    output_tensor = torch.cat(
        [outputs_list[i] for i in forward_batch.attn_cp_metadata.cp_reverse_index],
        dim=0,
    )
    output_tensor = output_tensor.view(-1, hidden_size)
    return output_tensor


def cp_all_gather_rerange_kv_cache(input_tensor, cp_size, forward_batch, stream):
    """
    Allgather and reorganize KV cache from all ranks in context parallel group.

    # for in-seq-split
    |   +-----------before allgather------------+|
    |   | dp_atten_tp0: block0, block7 |
    |   | dp_atten_tp1: block1, block6 |
    |   | dp_atten_tp2: block2, block5 |
    |   | dp_atten_tp3: block3, block4 |
    |
    |   +----------before rerange---------------+|
    | block0 | block7 | block1 | block6 | block2 | block5 | block3 | block4 |
    |
    |   +--------------result-------------------+
    | block0 | block1 | block2 | block3 | block4 | block5 | block6 | block7 |
    |   +-------------------------+
    """
    output_tensor = cp_all_gather_reorganized_into_tensor_kv_cache(
        input_tensor,
        cp_size,
        forward_batch,
        stream,
    )
    outputs_list = list(
        torch.split(
            output_tensor, forward_batch.attn_cp_metadata.reverse_split_len, dim=0
        )
    )
    output_tensor = torch.cat(
        [outputs_list[i] for i in forward_batch.attn_cp_metadata.cp_reverse_index],
        dim=0,
    )
    # No need to reshape - output_tensor already has the correct shape [seq_len, ...]
    return output_tensor


def cp_allgather_and_save_kv_cache(forward_batch, layer, k, v, cp_size, swa_loc=None):
    """
    Allgather KV cache from all CP ranks and write the full result
    into each rank's local memory pool.

    swa_loc is the pre-translated full->SWA write target for hybrid SWA pools.
    """
    cache_loc = (
        forward_batch.out_cache_loc
        if not layer.is_cross_attention
        else forward_batch.encoder_out_cache_loc
    )

    k = k.contiguous()
    v = v.contiguous()

    key_cache_full = cp_all_gather_rerange_kv_cache(
        k, cp_size, forward_batch, torch.cuda.current_stream()
    )
    value_cache_full = cp_all_gather_rerange_kv_cache(
        v, cp_size, forward_batch, torch.cuda.current_stream()
    )

    get_token_to_kv_pool().set_kv_buffer(
        layer,
        KVWriteLoc(cache_loc, swa_loc),
        key_cache_full,
        value_cache_full,
        layer.k_scale,
        layer.v_scale,
    )


def cp_attn_forward_extend(
    forward_batch,
    q: torch.Tensor,
    device: torch.device,
    attn_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor],
) -> torch.Tensor:
    """
    Split q into prev/next zigzag halves based on CP metadata, call the
    backend-specific attention function twice with appropriate per-half
    metadata, and concatenate the results.

    For bs > 1, q is laid out as [all_prev_tokens_across_seqs,
    all_next_tokens_across_seqs]; the split point is total_q_prev_tokens.
    cu_seqlens_q_prev/next tensors have shape [bs+1] and carry the
    per-sequence boundaries through FlashAttention's variable-length API.

    attn_fn signature:
        attn_fn(q, cu_seqlens_q, cache_seqlens, max_seqlen_q) -> result
    where only these four CP-varying parameters differ between halves.
    All other backend-specific args should be captured in the closure.
    """
    cp_meta = forward_batch.attn_cp_metadata

    q_prev = q[: cp_meta.total_q_prev_tokens]
    q_next = q[cp_meta.total_q_prev_tokens :]

    result_prev = attn_fn(
        q_prev,
        cp_meta.cu_seqlens_q_prev_tensor,
        cp_meta.kv_len_prev_tensor,
        cp_meta.max_seqlen_q_prev,
    )
    result_next = attn_fn(
        q_next,
        cp_meta.cu_seqlens_q_next_tensor,
        cp_meta.kv_len_next_tensor,
        cp_meta.max_seqlen_q_next,
    )

    return torch.concat([result_prev, result_next], dim=0)


def prepare_context_parallel_metadata(
    kv_len,
    cp_rank,
    cp_size,
    seqs_len,
    extend_seqs_len=None,
    device="cuda",
):
    if is_dsa_prefill_cp_round_robin_split():
        return InterleaveContextParallelMetadata(
            total_seq_lens=int(kv_len),
            bs=len(extend_seqs_len) if extend_seqs_len is not None else 1,
        )

    """prepare_input_dp_with_cp_dsa-zigzag index
    Example (DP_ATTENT_TP == CP_SIZE == 4, single sequence):
        block0 | block1 | block2 | block3 | block4 | block5 | block6 | block7
        rank 0: block0, block7
        rank 1: block1, block6
        rank 2: block2, block5
        rank 3: block3, block4
    For bs > 1, each sequence is split into cp_segment_num = 2 * cp_size
    blocks independently; per-rank layout becomes:
        [s0.block_r, s1.block_r, ..., s_{bs-1}.block_r,
         s0.block_{2*cp_size-1-r}, ..., s_{bs-1}.block_{2*cp_size-1-r}]
    i.e. all prev blocks first, then all next blocks -- so torch.split at
    total_q_prev_tokens cleanly separates them.
    """
    assert extend_seqs_len is not None
    extend_seqs_len = [int(x) for x in extend_seqs_len]

    # Update the extend_seqs_len to the padded length.
    pad_len = int(kv_len) - sum(extend_seqs_len)
    if pad_len > 0:
        extend_seqs_len[-1] += pad_len
        if seqs_len is not None and len(seqs_len) == len(extend_seqs_len):
            seqs_len = list(seqs_len)
            seqs_len[-1] += pad_len

    bs = len(extend_seqs_len)
    cp_segment_num = cp_size * 2

    # Prefix offset (radix cache hit length) per sequence. For non-NSA
    # (FlashAttention) the prefix is baked into kv_len_prev/next via
    # prefix_offsets[s] below, so cache_seqlens correctly covers the cached
    # prefix. NSA leaves bare cumulatives so its indexer can re-add the
    # offset itself.
    if seqs_len is not None and len(seqs_len) == bs:
        prefix_offsets = [
            max(int(seqs_len[s]) - extend_seqs_len[s], 0) for s in range(bs)
        ]
    else:
        prefix_offsets = [0] * bs

    # Per-sequence block sizes: first (L % cp_segment_num) blocks get +1.
    per_seq_block_sizes: List[List[int]] = []
    split_list: List[int] = []
    for s in range(bs):
        L = extend_seqs_len[s]
        base = L // cp_segment_num
        rem = L % cp_segment_num
        blk = [base + 1 if i < rem else base for i in range(cp_segment_num)]
        per_seq_block_sizes.append(blk)
        split_list.extend(blk)

    # Per-rank aggregate: this rank owns block r and block (2*cp_size-1-r)
    # of every sequence.
    per_rank_actual_token = [0] * cp_size
    for r in range(cp_size):
        total = 0
        for s in range(bs):
            total += (
                per_seq_block_sizes[s][r]
                + per_seq_block_sizes[s][cp_segment_num - 1 - r]
            )
        per_rank_actual_token[r] = total
    max_single_rank = max(per_rank_actual_token) if per_rank_actual_token else 0
    # Kept as cp_size copies so downstream torch.split(x, max_rank_len) still
    # works directly. All entries intentionally identical.
    max_rank_len = [max_single_rank] * cp_size

    # Zigzag index selecting which of split_list's bs * cp_segment_num pieces
    # this rank owns, in the order [all_prevs, all_nexts].
    zigzag_index = list(
        range(cp_rank, cp_rank + bs * cp_segment_num, cp_segment_num)
    ) + list(
        range(
            cp_segment_num - cp_rank - 1,
            bs * cp_segment_num,
            cp_segment_num,
        )
    )

    # Reverse index: given the post-allgather concatenation
    #   [rank0_prevs_all_seqs, rank0_nexts_all_seqs,
    #    rank1_prevs_all_seqs, rank1_nexts_all_seqs, ...]
    # produce a permutation that restores [s0_b0..s0_bN, s1_b0..s1_bN, ...].
    cp_reverse_index: List[int] = []
    for batch_id in range(bs):
        cp_reverse_index.extend(
            list(range(batch_id, cp_segment_num * bs, 2 * bs))
            + list(
                range(
                    (cp_segment_num - 1) * bs + batch_id,
                    0,
                    -2 * bs,
                )
            )
        )

    # Split sizes matching the post-allgather concatenation order above.
    reverse_split_len: List[int] = []
    for r in range(cp_size):
        for s in range(bs):
            reverse_split_len.append(per_seq_block_sizes[s][r])
        for s in range(bs):
            reverse_split_len.append(per_seq_block_sizes[s][cp_segment_num - 1 - r])

    # Per-sequence cumulatives used for FA cache_seqlens.
    #   kv_len_prev[s] = sum of seq s's blocks [0..cp_rank] (inclusive).
    #   kv_len_next[s] = sum of seq s's blocks [0..cp_segment_num-cp_rank-1] (inclusive).
    nsa_mode = is_dsa_enable_prefill_cp()
    kv_len_prev_list: List[int] = []
    kv_len_next_list: List[int] = []
    actual_seq_q_prev_list: List[int] = []
    actual_seq_q_next_list: List[int] = []
    for s in range(bs):
        blk = per_seq_block_sizes[s]
        cum_prev = sum(blk[: cp_rank + 1])
        cum_next = sum(blk[: cp_segment_num - cp_rank])
        # NSA indexer re-adds prefix offset itself; leave bare cumulative.
        # For non-NSA (FlashAttention), bake prefix into cache_seqlens.
        if nsa_mode:
            kv_len_prev_list.append(cum_prev)
            kv_len_next_list.append(cum_next)
        else:
            kv_len_prev_list.append(prefix_offsets[s] + cum_prev)
            kv_len_next_list.append(prefix_offsets[s] + cum_next)
        actual_seq_q_prev_list.append(blk[cp_rank])
        actual_seq_q_next_list.append(blk[cp_segment_num - cp_rank - 1])

    # FlashAttention CUDA tensors (device parameterized for unit tests).
    kv_len_prev_tensor = torch.tensor(
        kv_len_prev_list, device=device, dtype=torch.int32
    )
    kv_len_next_tensor = torch.tensor(
        kv_len_next_list, device=device, dtype=torch.int32
    )
    actual_seq_q_prev_tensor = torch.tensor(
        actual_seq_q_prev_list, device=device, dtype=torch.int32
    )
    actual_seq_q_next_tensor = torch.tensor(
        actual_seq_q_next_list, device=device, dtype=torch.int32
    )
    cu_prev = [0] + list(accumulate(actual_seq_q_prev_list))
    cu_next = [0] + list(accumulate(actual_seq_q_next_list))
    cu_seqlens_q_prev_tensor = torch.tensor(cu_prev, device=device, dtype=torch.int32)
    cu_seqlens_q_next_tensor = torch.tensor(cu_next, device=device, dtype=torch.int32)

    total_q_prev_tokens = cu_prev[-1]
    total_q_next_tokens = cu_next[-1]
    max_seqlen_q_prev = max(actual_seq_q_prev_list) if actual_seq_q_prev_list else 0
    max_seqlen_q_next = max(actual_seq_q_next_list) if actual_seq_q_next_list else 0
    total_seq_lens = sum(extend_seqs_len)

    # Cheap invariants: metadata must be a valid permutation spec.
    # - split_list has bs * cp_segment_num pieces (all blocks, all seqs).
    # - zigzag_index has 2 * bs entries (this rank's prev + next per seq).
    # - cp_reverse_index has bs * cp_segment_num entries (reorders the
    #   full allgathered stream back to per-seq-original order).
    assert len(split_list) == bs * cp_segment_num
    assert sum(split_list) == total_seq_lens
    assert len(zigzag_index) == 2 * bs
    assert len(cp_reverse_index) == bs * cp_segment_num
    assert sorted(cp_reverse_index) == list(range(bs * cp_segment_num))
    assert sum(per_rank_actual_token) == total_seq_lens

    return ContextParallelMetadata(
        split_list=split_list,
        zigzag_index=zigzag_index,
        cp_reverse_index=cp_reverse_index,
        reverse_split_len=reverse_split_len,
        per_rank_actual_token=per_rank_actual_token,
        max_rank_len=max_rank_len,
        kv_len_prev_tensor=kv_len_prev_tensor,
        kv_len_next_tensor=kv_len_next_tensor,
        actual_seq_q_prev_tensor=actual_seq_q_prev_tensor,
        actual_seq_q_next_tensor=actual_seq_q_next_tensor,
        cu_seqlens_q_prev_tensor=cu_seqlens_q_prev_tensor,
        cu_seqlens_q_next_tensor=cu_seqlens_q_next_tensor,
        total_q_prev_tokens=total_q_prev_tokens,
        total_q_next_tokens=total_q_next_tokens,
        max_seqlen_q_prev=max_seqlen_q_prev,
        max_seqlen_q_next=max_seqlen_q_next,
        kv_len_prev_list=kv_len_prev_list,
        kv_len_next_list=kv_len_next_list,
        actual_seq_q_prev_list=actual_seq_q_prev_list,
        actual_seq_q_next_list=actual_seq_q_next_list,
        total_seq_lens=total_seq_lens,
        bs=bs,
    )
