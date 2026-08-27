from functools import lru_cache
from typing import TYPE_CHECKING, List, Tuple, Union

import torch
import torch.nn.functional as F
import triton

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    attn_cp_all_gather_into_tensor,
    attn_cp_reduce_scatter_tensor,
    is_allocation_symmetric,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.runtime_context import (
    get_disagg,
    get_memory,
    get_parallel,
    process_model_config,
)
from sglang.srt.utils import get_bool_env_var, is_cuda, is_hip
from sglang.srt.utils.common import ceil_align, ceil_div


@lru_cache(maxsize=1)
def aiter_can_use_preshuffle_paged_mqa() -> bool:
    """Whether aiter's preshuffle paged MQA / cache kernels can be used on this runtime.

    aiter's ``deepgemm_fp8_paged_mqa_logits`` only supports ``KVBlockSize > 1`` and
    ``Preshuffle=True`` on its gluon kernel path. The gluon path is enabled when
    Triton >= 3.5.0, OR when ``AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS=1`` is set
    (which additionally requires that the AOT gluon kernel artifacts ship inside
    the aiter wheel/image). Otherwise aiter asserts ``KVBlockSize == 1`` and
    refuses ``Preshuffle=True``.

    sglang's DSA indexer uses this single decision to pick:
      * ``page_size``: 64 (preshuffle) vs 1 (legacy) on ROCm
      * ``Preshuffle`` / ``preshuffle`` flags on the aiter MQA + cache kernels
      * ``get_page_table_64`` vs ``get_page_table_1`` on the metadata
      * whether ``GetKAndS.execute`` uses the aiter or the triton implementation

    The result is cached so the cost is paid once per process.

    Set ``SGLANG_DSA_HIP_DISABLE_PRESHUFFLE=1`` to force the legacy path even when
    the gluon kernel would otherwise be available (useful for CI bisection).
    ``SGLANG_NSA_HIP_DISABLE_PRESHUFFLE`` is a deprecated alias.
    """
    if not is_hip():
        return False
    if not get_bool_env_var("SGLANG_USE_AITER"):
        return False
    if envs.SGLANG_DSA_HIP_DISABLE_PRESHUFFLE.get():
        return False
    if get_bool_env_var("AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS"):
        return True
    try:
        from packaging.version import Version

        return Version(Version(triton.__version__).base_version) >= Version("3.5.0")
    except Exception:
        return False


# Tile size for the indexer FP8 K-cache preshuffle layout. Store and gather
# kernels reorganize each page into (tile x tile) blocks so the aiter preshuffle
# paged-MQA gather can consume the cache directly.
INDEXER_K_CACHE_PRESHUFFLE_TILE = 16


if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def compute_dsa_seqlens(original_seq_lens, dsa_index_topk: int, index_kpool: int = 1):
    if index_kpool <= 1:
        return original_seq_lens.clamp(max=dsa_index_topk)

    # kpool: history tokens are selected at pool granularity (multiples of
    # index_kpool), while the ragged tail (the partial trailing pool) is always
    # kept. Round the history part down to a pool boundary, clamp it to topk,
    # then add back the tail.
    full_pool_tokens = (
        torch.div(original_seq_lens, index_kpool, rounding_mode="floor") * index_kpool
    )
    selected_history_tokens = full_pool_tokens.clamp(max=dsa_index_topk)
    tail_tokens = original_seq_lens - full_pool_tokens
    return selected_history_tokens + tail_tokens


def should_remap_pd_dsa_seed_to_local_slots() -> bool:
    """Whether a PD seed should enter the allocator-local fused TopK domain."""
    return (
        is_cuda()
        and envs.SGLANG_DSA_FUSE_TOPK.get()
        and get_disagg().disaggregation_mode == "decode"
        and not get_memory().enable_hisparse
        and not get_parallel().dcp_enabled
    )


def should_use_dsa_fused_topk(seed_dsa_topk_from_draft_extend: bool) -> bool:
    """Select fused TopK for PD IndexShare.

    PD Prefill worker:
    - Target prefill: fused TopK enabled.
    - Draft extend: fused TopK disabled.

    PD Decode worker:
    - Draft decode / target verify / draft extend: fused TopK enabled.
    """
    pd_index_share_seed = (
        get_disagg().disaggregation_mode != "null" and seed_dsa_topk_from_draft_extend
    )
    return envs.SGLANG_DSA_FUSE_TOPK.get() and (
        not pd_index_share_seed or should_remap_pd_dsa_seed_to_local_slots()
    )


def is_dsa_enable_prefill_cp():
    if not envs.SGLANG_ENABLE_CP_V2.get():
        return get_parallel().enable_dsa_prefill_context_parallel

    # Derive from the runtime CP topology + model arch rather than the legacy
    # flag under CP-v2: DSA prefill CP is active when the CP group is on for a
    # DeepSeek Sparse Attention model.
    if get_parallel().attn_cp_size <= 1:
        return False
    from sglang.srt.configs.model_config import is_deepseek_dsa, is_deepseek_v4

    hf_config = process_model_config().hf_config
    return is_deepseek_dsa(hf_config) or is_deepseek_v4(hf_config)


def is_dsa_prefill_cp_in_seq_split():
    return (
        is_dsa_enable_prefill_cp()
        and get_parallel().dsa_prefill_cp_mode == "in-seq-split"
    )


def is_dsa_prefill_cp_round_robin_split():
    return (
        is_dsa_enable_prefill_cp()
        and get_parallel().dsa_prefill_cp_mode == "round-robin-split"
    )


# Structural surface where the graph DSA split-op dispatch (DSA indexer) and the
# MLA BMM-into-attention fusion apply: a non-speculative extend (prefill) running
# inside a piecewise/breakable CUDA graph. Both fusions are now on by default on
# this surface (no feature flag); each adds its own extra carve-outs at its call
# site (e.g. the indexer also excludes DSA prefill context parallelism).
def is_graph_dsa_split_op_surface(forward_batch: "ForwardBatch") -> bool:
    return (
        is_cuda()
        and (is_in_tc_piecewise_cuda_graph() or is_in_breakable_cuda_graph())
        and forward_batch.forward_mode.is_extend_without_speculative()
    )


def can_dsa_prefill_cp_round_robin_split(forward_batch: "ForwardBatch"):
    if not forward_batch.forward_mode.is_context_parallel_extend():
        return False
    cp_size = get_parallel().attn_cp_size
    seq_len = sum(forward_batch.extend_seq_lens_cpu)
    return (
        is_dsa_prefill_cp_round_robin_split()
        and seq_len > 0
        and seq_len >= cp_size
        and cp_size > 1
    )


def cp_zigzag_full_plan_rows(
    forward_batch: "ForwardBatch", device: torch.device
) -> torch.Tensor | None:
    cp_meta = getattr(forward_batch, "attn_cp_metadata", None)
    if cp_meta is None or getattr(cp_meta, "zigzag_index", None) is None:
        return None
    if (
        getattr(forward_batch, "extend_seq_lens_cpu", None) is None
        or getattr(cp_meta, "split_list", None) is None
    ):
        return None

    extend_lens = [int(x) for x in forward_batch.extend_seq_lens_cpu]
    bs = len(extend_lens)
    split_list = [int(x) for x in cp_meta.split_list]
    if bs == 0 or len(split_list) % bs != 0:
        return None
    cp_segment_num = len(split_list) // bs

    q_offsets = [0]
    for q_len in extend_lens:
        q_offsets.append(q_offsets[-1] + q_len)

    rows: List[int] = []
    for seg_idx in cp_meta.zigzag_index:
        seg_idx = int(seg_idx)
        batch_id = seg_idx // cp_segment_num
        block_id = seg_idx % cp_segment_num
        if batch_id >= bs:
            return None
        block_base = batch_id * cp_segment_num
        block_start = sum(split_list[block_base : block_base + block_id])
        block_len = split_list[seg_idx]
        row_start = q_offsets[batch_id] + block_start
        rows.extend(range(row_start, row_start + block_len))

    return torch.tensor(rows, dtype=torch.long, device=device)


def dsa_cp_round_robin_split_data(input_: Union[torch.Tensor, List]):
    """
    # for round-robin-split, split the tokens evenly according to the rule of token_idx % cp_size.
    |   +-----------before split------------+|
    | token0, token1, token2, token3, token4, token5, token6, token7, ...
    |
    |   +--------------result-------------------+
    | dp_atten_tp0: token0, token4, token8, token12, token16, ... |
    | dp_atten_tp1: token1, token5, token9, token13, token17, ... |
    | dp_atten_tp2: token2, token6, token10, token14, token18, ... |
    | dp_atten_tp3: token3, token7, token11, token15, token19, ... |
    |   +-------------------------+
    """
    cp_size = get_parallel().attn_cp_size
    cp_rank = get_parallel().attn_cp_rank
    if isinstance(input_, (tuple, list)):
        indices = range(cp_rank, len(input_), cp_size)
        return input_[indices]

    tokens = len(input_)
    if tokens % cp_size != 0:
        cur_len = tokens // cp_size + (tokens % cp_size > cp_rank)
        if cur_len == 0:
            return input_.new_empty(0, *input_.shape[1:])
        indices = torch.arange(cp_rank, tokens, cp_size, device=input_.device)
        return input_[indices]

    # for torch device tensor
    return input_.view(-1, cp_size, *input_.shape[1:])[:, cp_rank].contiguous()


def cal_padded_tokens(forward_batch: "ForwardBatch"):
    # Consistent with the padding calculation logic in ForwardBatch.prepare_mlp_sync_batch,
    # calculate the actual token length after padding when attn_tp_size > 1 or in the MAX_LEN padding mode.
    from sglang.srt.layers.cp.padding import get_cp_padding_align_size
    from sglang.srt.layers.cp.utils import enable_cp_v2, is_cp_v2_active

    # CP-v2 already pads each rank-local shard to its physical size
    if is_cp_v2_active(forward_batch):
        return forward_batch.attn_cp_metadata.per_rank_actual_token[
            get_parallel().attn_cp_rank
        ]

    global_num_tokens = forward_batch.global_num_tokens_cpu.copy()
    sync_group_size = len(global_num_tokens)
    attn_cp_size = get_parallel().attn_cp_size
    # Must mirror ForwardBatch.prepare_mlp_sync_batch, which applies cp_align_size only when
    # CP-v2 is disabled. Under enable_cp_v2() the speculative forwards (TARGET_VERIFY /
    # DRAFT_EXTEND_V2) reach here with is_cp_v2_active False, and q is padded to attn_tp_size only
    # (not cp-aligned). Applying cp_align here over-pads the flashmla metadata past q, so
    # num_splits ends up longer than q -> fwd_kvcache_mla fails "num_splits must have shape (b+1)".
    # (attn_cp analog of the attn_tp fix in PR #30642 / issue #30296.)
    if not enable_cp_v2():
        cp_align_size = get_cp_padding_align_size()
        for i in range(sync_group_size):
            global_num_tokens[i] = ceil_align(global_num_tokens[i], cp_align_size)
    # Reuse the mode selected when the DP buffer was prepared.
    dp_padding_mode = forward_batch.dp_padding_mode
    if dp_padding_mode is None:
        dp_padding_mode = DpPaddingMode.get_dp_padding_mode(
            forward_batch.is_extend_in_batch, global_num_tokens
        )
    if dp_padding_mode.is_max_len():
        tokens = max(global_num_tokens)
    elif len(global_num_tokens) > 1:
        tokens = global_num_tokens[get_parallel().attn_dp_rank]
    else:
        tokens = global_num_tokens[0]
    if can_dsa_prefill_cp_round_robin_split(forward_batch):
        tokens = ceil_div(tokens, attn_cp_size)
    return tokens


def pad_dsa_cache_seqlens(forward_batch: "ForwardBatch", dsa_cache_seqlens):
    if dsa_use_prefill_cp(forward_batch) and not can_dsa_prefill_cp_round_robin_split(
        forward_batch
    ):
        return dsa_cache_seqlens

    attn_cp_size = get_parallel().attn_cp_size
    needs_cp_pad = attn_cp_size > 1 and can_dsa_prefill_cp_round_robin_split(
        forward_batch
    )
    needs_dp_pad = forward_batch.global_num_tokens_cpu is not None
    if not needs_cp_pad and not needs_dp_pad:
        return dsa_cache_seqlens
    tokens = cal_padded_tokens(forward_batch)
    pad_len = tokens - dsa_cache_seqlens.shape[0]
    if pad_len > 0:
        dsa_cache_seqlens = torch.cat(
            [
                dsa_cache_seqlens,
                dsa_cache_seqlens.new_zeros(pad_len, *dsa_cache_seqlens.shape[1:]),
            ]
        )
    return dsa_cache_seqlens


def can_dsa_cp_split(seq_len: int, cp_size: int, use_dsa: bool, forward_batch):
    if (
        cp_size <= 1
        or not use_dsa
        or not forward_batch.forward_mode.is_context_parallel_extend()
        or not is_dsa_enable_prefill_cp()
        or sum(forward_batch.extend_seq_lens_cpu) < cp_size
    ):
        return False

    if is_dsa_prefill_cp_round_robin_split():
        cur_cp_seq_len = seq_len // cp_size
        assert (
            seq_len % cp_size == 0
        ), f"seq_len {seq_len} is not divisible by cp_size {cp_size} when dsa_prefill_cp_mode is round-robin-split"
    else:
        # TODO current just support prefill batch=1 and len(input_ids) > self.cp_size * 2
        # Note: (self.cp_size * 2) To achieve load balancing for seq computation,
        # the seq data needs to be divided and recombined at twice the size of cp_size.
        cur_cp_seq_len = seq_len // (cp_size * 2)
    return cur_cp_seq_len != 0


from sglang.kernels.ops.attention.dsa.cp_split import (
    dsa_cp_round_robin_split_q_seqs_kernel,
)


def dsa_cp_round_robin_split_q_seqs_cpu(extend_seqs):
    cp_size = get_parallel().attn_cp_size
    cp_rank = get_parallel().attn_cp_rank
    extra_seq = 0
    q_seqs = []
    for bs, cur_len in enumerate(extend_seqs):
        cur_len += extra_seq
        cur_seq = cur_len // cp_size + int(cur_len % cp_size > cp_rank)
        q_seqs.append(cur_seq)
        extra_seq = cur_len - cur_seq * cp_size
    bs_idx = list([i for i, x in enumerate(q_seqs) if x > 0])
    q_seqs = [q_len for q_len in q_seqs if q_len > 0]
    return q_seqs, bs_idx


def dsa_cp_round_robin_split_q_seqs(
    extend_seqs_cpu, extend_seqs
) -> Tuple[List, torch.Tensor, List, torch.Tensor]:
    """
    round-robin-split distributes tokens across ranks based on token_idx % cp_size.

    Return:
    ret_q_lens_cpu(List) and ret_q_lens(torch.Tensor): the partitioned length (excluding zeros) on the current cp rank
        for each sequence after distribution across cp ranks.
    bs_idx_cpu(List) and bs_idx(torch.Tensor): marks which sequences are ultimately selected,
        i.e., those with a partitioned length greater than zero.
    """
    cp_size = get_parallel().attn_cp_size
    cp_rank = get_parallel().attn_cp_rank
    # len(ret_q_lens_cpu) == len(bs_idx_cpu)
    ret_q_lens_cpu, bs_idx_cpu = dsa_cp_round_robin_split_q_seqs_cpu(extend_seqs_cpu)
    ret_q_lens = torch.empty(
        (len(bs_idx_cpu),), device=extend_seqs.device, dtype=extend_seqs.dtype
    )
    bs_idx = torch.empty(
        (len(bs_idx_cpu),), device=extend_seqs.device, dtype=torch.int32
    )
    grid = (1,)
    dsa_cp_round_robin_split_q_seqs_kernel[grid](
        extend_seqs, ret_q_lens, bs_idx, len(extend_seqs), cp_size, cp_rank
    )
    return ret_q_lens_cpu, ret_q_lens, bs_idx_cpu, bs_idx


def dsa_use_prefill_cp(forward_batch, dsa_enable_prefill_cp=None):
    if dsa_enable_prefill_cp is None:
        dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
    if (
        forward_batch.attn_cp_metadata is not None
        and dsa_enable_prefill_cp
        and forward_batch.forward_mode.is_context_parallel_extend()
    ):
        return True
    else:
        return False


def cp_split_and_rebuild_data(forward_batch, input_: torch.Tensor):
    if is_dsa_prefill_cp_round_robin_split():
        cp_size = get_parallel().attn_cp_size
        assert (
            input_.shape[0] % cp_size == 0
        ), f"Expect input shape 0 can divided by cp size, but got input shape {input_.shape}, cp size {cp_size}"
        return dsa_cp_round_robin_split_data(input_)

    input_list = list(
        torch.split(input_, forward_batch.attn_cp_metadata.split_list, dim=0)
    )
    result = torch.cat(
        [input_list[i] for i in forward_batch.attn_cp_metadata.zigzag_index], dim=0
    ).view(-1, input_.shape[-1])
    return result


def cp_split_and_rebuild_position(forward_batch, positions: torch.Tensor):
    if is_dsa_prefill_cp_round_robin_split():
        cp_size = get_parallel().attn_cp_size
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


def cp_attn_tp_all_gather_reorganazied_into_tensor(
    input_: torch.Tensor, attn_tp_size, forward_batch
):
    """
    Allgather communication for context_parallel(kv_cache, index_k, hidden_states).
    This implementation mainly consists of three parts:
    Step 1, padding the input shape to unify the shape for allgather communication (the shape must be the same).
    Step 2, synchronized allgather communication.
    Step 3, removing the padding and reassembling the data according to the actual tokens.
    """
    # The metadata is the source of truth for the per-rank physical extent.
    # ceil(total_len / cp_size) can be smaller for multi-request batches whose
    # per-sequence remainders accumulate on the same rank.
    max_rank_len = forward_batch.attn_cp_metadata.max_rank_len
    max_len = max_rank_len[0]
    assert len(max_rank_len) == attn_tp_size and all(
        rank_len == max_len for rank_len in max_rank_len
    ), f"all-gather requires equal physical rank lengths, got {max_rank_len}"
    assert input_.shape[0] <= max_len, (
        f"local CP input has {input_.shape[0]} rows, exceeding the metadata "
        f"physical extent {max_len}"
    )
    pad_size = max_len - input_.shape[0]
    if pad_size > 0:
        input_ = F.pad(input_, (0, 0, 0, pad_size), mode="constant", value=0)
    input_ = input_.contiguous()
    group = get_parallel().attn_cp_group
    with use_symmetric_memory(group, disabled=not is_allocation_symmetric()):
        input_tensor_all = torch.empty(
            max_len * attn_tp_size,
            input_.shape[1],
            device=input_.device,
            dtype=input_.dtype,
        )
    attn_cp_all_gather_into_tensor(input_tensor_all, input_)
    outputs_list_max = list(
        torch.split(
            input_tensor_all, forward_batch.attn_cp_metadata.max_rank_len, dim=0
        )
    )
    outputs = torch.cat(
        [
            outputs_list_max[index][:per_rank_len]
            for index, per_rank_len in enumerate(
                forward_batch.attn_cp_metadata.per_rank_actual_token
            )
        ],
        dim=0,
    )
    return outputs


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
            get_parallel().attn_cp_group, disabled=not is_allocation_symmetric()
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

    bs_seq_len, hidden_size = input_tensor.shape
    output_tensor = cp_attn_tp_all_gather_reorganazied_into_tensor(
        input_tensor,
        cp_size,
        forward_batch,
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


# "Plain" CP layout: rank i holds the contiguous token slice
# [i*S/cp, (i+1)*S/cp). This is the natural layout produced by a vanilla
# all_gather on rank-major buffers, so KDA layers (which need natural-sequential
# tokens for causal_conv1d / chunk_kda) can use plain all_gather / plain
# reduce_scatter with zero rerange permute. At MLA boundaries the layout is
# converted to/from the configured scattered mode (round-robin or zigzag) that
# MLA's CP attention expects for load balance.
#
# Residual streams stay in plain throughout the layer stack and never need
# conversion -- they are re-derived from hidden_states at each layer entry by
# `mhc.attn_split`, so they inherit hidden_states' layer-input layout, which
# under the plain cross-layer contract is always plain.


def cp_plain_split(input_tensor: torch.Tensor) -> torch.Tensor:
    """Model entry scatter under the plain layout contract.

    Slice the global [S_total, ...] tensor into this rank's contiguous chunk
    [cp_rank * K, (cp_rank+1) * K). Pure local op; replaces
    `cp_split_and_rebuild_data` at model entry when the plain contract is on.
    """
    cp_size = get_parallel().attn_cp_size
    cp_rank = get_parallel().attn_cp_rank
    assert input_tensor.shape[0] % cp_size == 0, (
        f"cp_plain_split expects total tokens divisible by cp_size, "
        f"got {input_tensor.shape[0]} % {cp_size} != 0"
    )
    chunk = input_tensor.shape[0] // cp_size
    return input_tensor[cp_rank * chunk : (cp_rank + 1) * chunk].contiguous()


def cp_plain_all_gather(input_tensor: torch.Tensor, cp_size: int) -> torch.Tensor:
    """Inverse of cp_plain_split: gather plain per-rank slices into the full
    [S, ...] tensor in natural sequential order.

    Under the plain contract the all_gather output [rank0_chunk | rank1_chunk |
    ...] *is* the natural sequential ordering, so no rerange permute is needed.
    Used at model exit and at KDA prepare_attn entry.
    """
    out_shape = (input_tensor.shape[0] * cp_size,) + tuple(input_tensor.shape[1:])
    with use_symmetric_memory(
        get_parallel().attn_cp_group, disabled=not is_allocation_symmetric()
    ):
        output_tensor = input_tensor.new_empty(out_shape)
    attn_cp_all_gather_into_tensor(output_tensor, input_tensor)
    return output_tensor


def cp_plain_reduce_scatter(input_tensor: torch.Tensor, cp_size: int) -> torch.Tensor:
    """Inverse of cp_plain_all_gather for KDA o_proj output.

    Takes a CP-partial-sum [S, H] in natural sequential order and emits this
    rank's contiguous slice [S/cp, H] via a single reduce_scatter. The plain
    layout is rank-major contiguous, so reduce_scatter's default contiguous
    split is exactly what we want -- no permute or view+transpose.

    Comm cost: (N-1)/N * D, vs all_reduce + split which is 2*(N-1)/N * D
    -- ~33% cheaper in NCCL ring traffic than the all_reduce path, and saves
    the full-tensor permute that a round-robin reduce_scatter would need.
    """
    S = input_tensor.shape[0]
    assert S % cp_size == 0, (
        f"cp_plain_reduce_scatter expects S divisible by cp_size, "
        f"got S={S}, cp_size={cp_size}"
    )
    out_shape = (S // cp_size,) + tuple(input_tensor.shape[1:])
    with use_symmetric_memory(
        get_parallel().attn_cp_group, disabled=not is_allocation_symmetric()
    ):
        output_tensor = input_tensor.new_empty(out_shape)
    attn_cp_reduce_scatter_tensor(output_tensor, input_tensor.contiguous())
    return output_tensor


def cp_plain_to_scattered(
    input_tensor: torch.Tensor,
    forward_batch,
    cp_size: int,
) -> torch.Tensor:
    """Convert a plain per-rank slice [S/cp, H] into the configured CP scatter
    layout (round-robin or zigzag), suitable for MLA's CP attention kernel.

    Used at MLA prepare_attn. Composes existing primitives:
      1. plain all_gather: rank-major output IS natural sequential under the
         plain contract, so no rerange needed.
      2. cp_split_and_rebuild_data: local split per the active mode
         (round-robin via stride-cp_size, or zigzag via metadata indices).

    Fast path: for round-robin mode with K = S/cp divisible by cp, replace
    AG + local stride-cp split with a single all_to_all_single. Each rank
    sends K/cp rows to each destination instead of broadcasting all K rows --
    cp x less NCCL traffic and no [S, H] intermediate.
    """
    K = input_tensor.shape[0]
    if is_dsa_prefill_cp_round_robin_split() and K % cp_size == 0:
        tail = input_tensor.shape[1:]
        send = (
            input_tensor.view(K // cp_size, cp_size, *tail).transpose(0, 1).contiguous()
        )
        recv = torch.empty_like(send)
        torch.distributed.all_to_all_single(
            recv, send, group=get_parallel().attn_cp_group.device_group
        )
        return recv.flatten(0, 1)

    full = cp_plain_all_gather(input_tensor, cp_size)
    return cp_split_and_rebuild_data(forward_batch, full)


def cp_scattered_to_plain(
    input_tensor: torch.Tensor,
    forward_batch,
    cp_size: int,
) -> torch.Tensor:
    """Inverse of cp_plain_to_scattered. Takes a scattered (round-robin or
    zigzag) per-rank slice [S/cp, H] (MLA's natural output layout) and emits
    a plain per-rank slice.

    Used at MLA prepare_mlp. Composes:
      1. cp_all_gather_rerange_output: gather + rerange to natural sequential.
      2. local contiguous slice [cp_rank * K, (cp_rank+1) * K).

    Fast path: for round-robin mode with K = S/cp divisible by cp, replace
    AG + rerange + slice with a single all_to_all_single. Each rank sends
    K/cp contiguous rows to each destination -- send buffer is already laid
    out correctly (no pre-copy), recv needs one transpose-contiguous to
    interleave by source rank. cp x less NCCL traffic and no [S, H] alloc.
    """
    K = input_tensor.shape[0]
    if is_dsa_prefill_cp_round_robin_split() and K % cp_size == 0:
        tail = input_tensor.shape[1:]
        send = input_tensor.view(cp_size, K // cp_size, *tail)
        recv = torch.empty_like(send)
        torch.distributed.all_to_all_single(
            recv, send, group=get_parallel().attn_cp_group.device_group
        )
        # recv[s, n] = global token at position r*K + n*cp + s on this rank r;
        # plain out[n*cp + s] = recv[s, n], i.e. transpose dims 0 and 1.
        return recv.transpose(0, 1).contiguous().view(K, *tail)

    full = cp_all_gather_rerange_output(
        input_tensor, cp_size, forward_batch, torch.cuda.current_stream()
    )
    cp_rank = get_parallel().attn_cp_rank
    chunk = full.shape[0] // cp_size
    return full[cp_rank * chunk : (cp_rank + 1) * chunk].contiguous()


def fp8_mqa_logits_ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def fp8_mqa_logits_make_fused_kv(
    kv_fp8: torch.Tensor,
    kv_scales: torch.Tensor,
    block_kv: int,
    head_dim: int,
) -> torch.Tensor:
    num_phys_blocks = kv_fp8.shape[0]
    per_token_size = head_dim + 4
    block_bytes = block_kv * per_token_size
    scale_offset = block_kv * head_dim

    fused = torch.zeros(
        num_phys_blocks, block_bytes, dtype=torch.uint8, device=kv_fp8.device
    )
    for blk in range(num_phys_blocks):
        fused[blk, :scale_offset] = kv_fp8[blk].view(torch.uint8).reshape(-1)
        fused[blk, scale_offset:] = (
            kv_scales[blk].float().contiguous().view(torch.uint8).reshape(-1)
        )
    return fused.view(num_phys_blocks, block_kv, 1, per_token_size)
