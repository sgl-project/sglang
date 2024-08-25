"""Util functions for sequence parallel layout and runtime metadata."""

import itertools
from typing import TYPE_CHECKING, Sequence, Union

import numpy as np
import torch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import InputMetadata
    from sglang.srt.model_executor.model_runner import ModelRunner


#### Offset of a sequence parallel shard under the sequence parallel layout.
def _seq_parallel_offset_extend(sp_rank, sp_size, extend_seq_lens: np.ndarray):
    return np.sum(np.ceil(extend_seq_lens / sp_size).astype(np.int32)) * sp_rank


def _seq_parallel_offset_decode(sp_rank, sp_size, seq_lens: np.ndarray):
    return np.sum((seq_lens % sp_size) < sp_rank)


#### Indices from sequence parallel layout to normal layout
def _sp_to_normal_indices_extend(sp_size, extend_seq_lens: np.ndarray):
    """
    Indices from the Sequence Parallel layout (padded) to the normal layout.
    """
    sp_seq_lens = np.ceil(extend_seq_lens / sp_size).astype(np.int32)
    sp_len = np.sum(sp_seq_lens)
    sp_seq_offset = np.concatenate(
        [np.asarray([0], dtype=np.int32), np.cumsum(sp_seq_lens[:-1])]
    )
    sp_arange = np.arange(sp_size).reshape(-1, 1)
    indices = []
    for i in range(len(extend_seq_lens)):
        sp_idx = np.arange(sp_seq_lens[i]).reshape(1, -1).repeat(sp_size, axis=0)
        sp_idx = (sp_idx + sp_seq_offset[i] + sp_len * sp_arange).reshape(-1)
        sp_idx = sp_idx[: extend_seq_lens[i]]
        indices.append(sp_idx)
    indices = np.concatenate(indices)
    return indices


def _sp_to_normal_indices_decode(sp_size, seq_lens: np.ndarray):
    """
    Indices from the Sequence Parallel layout (padded) to the normal layout.
    """
    req_sp_rank = seq_lens % sp_size
    sp_rank_size = [np.sum(req_sp_rank == r) for r in range(sp_size)]
    req_sp_offset = np.cumsum(np.asarray([0] + sp_rank_size[:-1]))
    req_sp_offset = req_sp_offset[req_sp_rank]
    for sp_rank in range(sp_size):
        local_reqs = req_sp_rank == sp_rank
        req_sp_index = np.cumsum(local_reqs) - 1
        req_sp_offset += req_sp_index * local_reqs  # mask out reqs not here.
    return req_sp_offset


#### From normal layout to sequence parallel layout. Only for debug purpose
def _debug_normal_to_sp_indices_decode(sp_size, seq_lens):
    """(Debug only) Indices from normal layout to the SP layout (padded)."""
    indices = [
        seq_parallel_decode_indices(sp_rank, sp_size, seq_lens)
        for sp_rank in range(sp_size)
    ]
    indices = [(np.arange(len(idxs)), idxs) for idxs in indices]
    return indices


def _debug_normal_to_sp_indices_extend(sp_size, seq_lens):
    """(Debug only) Indices from normal layout to the SP layout (padded)."""
    indices = []
    sp_seq_lens = np.ceil(seq_lens / sp_size).astype(np.int32)
    seq_offset = np.concatenate(
        [np.asarray([0], dtype=np.int32), np.cumsum(seq_lens[:-1])]
    )
    sp_seq_offset = np.concatenate(
        [np.asarray([0], dtype=np.int32), np.cumsum(sp_seq_lens[:-1])]
    )
    for sp_rank in range(sp_size):
        start_idx = seq_offset + sp_seq_lens * sp_rank
        end_idx = np.minimum(seq_offset + sp_seq_lens * (sp_rank + 1), seq_lens)
        normal_layout_idx = np.concatenate(
            [np.arange(start_idx[i], end_idx[i]) for i in range(len(seq_lens))]
        )
        if sp_rank == sp_size - 1:
            length = end_idx - start_idx
            target_layout_idx = np.concatenate(
                [
                    np.arange(sp_seq_offset[i], sp_seq_offset[i] + length[i])
                    for i in range(len(seq_lens))
                ]
            )
        else:
            target_layout_idx = np.arange(len(normal_layout_idx))
        indices.append((target_layout_idx, normal_layout_idx))
    return indices


def _debug_normal_to_sp(indices, output_tensor, tensor):
    """
    Use the indices generated above to translate from a normal layout to a
    SP layout (padded). Due to the padding, `output_tensor`'s shape is different
    from the input `tensor`'s.
    """
    for idxs in indices:
        output_tensor[idxs] = tensor
    output_tensor = output_tensor.contiguous()
    return output_tensor


#### Padding
def seq_parallel_pad_zeros(
    indices: torch.Tensor, seq_lens, sp_size: int, only_last_shard: bool = False
):
    """
    Add padding zeros to SP-layout indices (must be a 1D tensor) so that the last
    SP shard will have its sequences padded after each sequence and all SP shards
    can have the same length.

    This function is used to (1) adjust the positions tensor to align input_ids with
    their positions during positional encoding and (2) adjust the output cache location
    to write KV cache of padded tokens to slot 0 (reserved for dummy output).
    """
    sp_seq_lens = np.ceil(seq_lens / sp_size).astype(np.int32)
    last_sp_seq_lens = seq_lens - sp_seq_lens * (sp_size - 1)
    padded_num_tokens = np.sum(sp_seq_lens).astype(np.int32)
    if only_last_shard:
        padded_indices = torch.zeros(
            padded_num_tokens, dtype=indices.dtype, device=indices.device
        )
        padded_stt = stt = 0
    else:
        padded_indices = torch.zeros(
            sp_size * padded_num_tokens, dtype=indices.dtype, device=indices.device
        )
        # All non-last shards do not need padding and hence can be copied.
        padded_stt = padded_num_tokens * (sp_size - 1)
        stt = padded_stt
        padded_indices[:padded_stt] = indices[:stt]

    bs = seq_lens.size
    for i in range(bs):
        padded_end = padded_stt + sp_seq_lens[i]
        end = stt + last_sp_seq_lens[i]
        padded_indices[padded_stt : padded_stt + last_sp_seq_lens[i]] = indices[stt:end]
        padded_stt = padded_end
        stt = end
    return padded_indices


def _get_num_padding_tokens(sp_size, extend_seq_lens: np.ndarray):
    """Get the number of tokens padded for SP."""
    padded_size = np.ceil(extend_seq_lens / sp_size).astype(np.int32)
    return sp_size * padded_size - extend_seq_lens


#### Get length/indices of sequence parallel local tokens within a batch
def seq_parallel_local_len_extend(
    sp_rank, sp_size, extend_seq_lens: Union[int, np.ndarray]
):
    """Get the number of tokens in this SP. Padding is not considered."""
    padded_size = np.ceil(extend_seq_lens / sp_size).astype(np.int32)
    return (
        padded_size
        if sp_rank != sp_size - 1
        else extend_seq_lens - (sp_size - 1) * padded_size
    )


def seq_parallel_extend_local_token_slice(sp_rank, sp_size, seq_len: int):
    """Get the SP local slice for a single request's extended input ids."""
    start = int(np.ceil(seq_len / sp_size) * sp_rank)
    length = seq_parallel_local_len_extend(sp_rank, sp_size, seq_len)
    return slice(start, start + length)


def seq_parallel_decode_indices(sp_rank, sp_size, seq_lens: np.ndarray):
    """Get Indices from the normal layout to the sequence parallel layout."""
    return np.nonzero((seq_lens % sp_size) == sp_rank)[0]


#### Transpose to sequence parallel layout
def seq_parallel_input_ids_extend(
    input_ids: Sequence[Sequence[int]], sp_size: int, bs: int
):
    # Note: The flatten input ids with Sequence Parallel is in form of:
    # [req_0_sp_0, req_1_sp_0, ... req_n_sp_0,
    #  req_0_sp_1, req_1_sp_1, ..., req_n_sp_1,
    #   ...
    #  req_0_sp_m, req_0_padding, req_1_sp_m, req_1_padding, ...]
    # ]
    # The padding is for collection primitives which needs each candidate to
    # have the same size. Since we don't expect too many requests in SP,
    # the extra compute caused by this is affordable.
    flatten_input_ids = [[] for _ in range(sp_size)]
    num_padding_tokens = _get_num_padding_tokens(
        sp_size, np.asarray([len(ids) for ids in input_ids])
    )
    for i in range(bs):
        for sp_rank in range(sp_size):
            ids = input_ids[i]
            local_slice = seq_parallel_extend_local_token_slice(
                sp_rank, sp_size, len(ids)
            )
            flatten_input_ids[sp_rank].extend(ids[local_slice])
        flatten_input_ids[-1].extend([0] * num_padding_tokens[i])
    flatten_input_ids = list(itertools.chain(*flatten_input_ids))
    return flatten_input_ids


def seq_parallel_input_ids_decode(
    input_ids: Sequence[int], sp_size: int, seq_lens: np.ndarray
):
    input_ids_sp = [[] for _ in range(sp_size)]
    # NOTE: in the extend phase, we evenly do sequence partition on extended
    # tokens (extend_len). However, since prefix lens is cleaned, we instead
    # use the whole sequence length (seq_lens) for the round-robin KV-cache.
    for sp_rank in range(sp_size):
        input_ids_sp[sp_rank].extend(
            input_ids[seq_parallel_decode_indices(sp_rank, sp_size, seq_lens)]
        )
    flatten_input_ids = list(itertools.chain(*input_ids_sp))
    return flatten_input_ids


#### Handle metadata
def init_sequence_parallel_args(
    model_runner: "ModelRunner", batch: "ScheduleBatch", forward_mode
):
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    sp_rank = model_runner.sp_rank
    sp_size = model_runner.sp_size
    seq_lens = batch.seq_lens
    extend_seq_lens_cpu = batch.prefill_extend_lens
    num_tokens = batch.input_ids.numel()
    if sp_size > 1:
        # During the runtime, we should use positions[local_token_indices]
        # to get positions for each SP shard.
        if forward_mode == ForwardMode.DECODE:
            seq_lens_cpu = seq_lens.cpu().numpy()
            sp_to_normal_indices = _sp_to_normal_indices_decode(sp_size, seq_lens_cpu)
            sp_local_token_length = seq_parallel_decode_indices(
                sp_rank, sp_size, seq_lens_cpu
            ).size
            sp_local_token_offset = _seq_parallel_offset_decode(
                sp_rank, sp_size, seq_lens_cpu
            )
            # Convert positions to SP layout and add padding zeros.
            normal_to_sp_indices = np.argsort(sp_to_normal_indices)
            # positions = positions[normal_to_sp_indices]
        else:
            sp_to_normal_indices = _sp_to_normal_indices_extend(
                sp_size, extend_seq_lens_cpu
            )
            sp_local_token_length = seq_parallel_local_len_extend(
                sp_rank, sp_size, extend_seq_lens_cpu
            )
            sp_local_token_offset = _seq_parallel_offset_extend(
                sp_rank, sp_size, extend_seq_lens_cpu
            )
            # Convert positions to SP layout and add padding zeros.
            normal_to_sp_indices = np.argsort(sp_to_normal_indices)
            # positions = positions[normal_to_sp_indices]
            # positions = seq_parallel_pad_zeros(positions, extend_seq_lens_cpu, sp_size)
            # Add padding zeros to out_cache_loc and write KV of padded tokens that may
            # exist in the last SP shard to slot 0 (reserved for dummy output).
            if sp_rank == sp_size - 1:
                out_cache_loc = seq_parallel_pad_zeros(
                    batch.out_cache_loc, extend_seq_lens_cpu, sp_size, True
                )
    else:
        sp_to_normal_indices = np.arange(num_tokens)
        normal_to_sp_indices = np.arange(num_tokens)
        sp_local_token_length = num_tokens
        sp_local_token_offset = 0

    _debug_normal_to_sp_metadata = None
    if False and sp_size > 1:
        if forward_mode == ForwardMode.DECODE:
            _debug_normal_to_sp_metadata = _debug_normal_to_sp_indices_decode(
                sp_size, seq_lens_cpu
            )
        else:
            _debug_normal_to_sp_metadata = _debug_normal_to_sp_indices_extend(
                sp_size, extend_seq_lens_cpu
            )

    init_args = {
        "sp_size": sp_size,
        "sp_rank": sp_rank,
        "sp_to_normal_indices": sp_to_normal_indices,
        "sp_local_token_length": sp_local_token_length,
        "sp_local_token_offset": sp_local_token_offset,
        "_debug_normal_to_sp_metadata": _debug_normal_to_sp_metadata,
        "flashinfer_prefill_wrapper_sp_full": model_runner.flashinfer_prefill_wrapper_sp_full,
        "flashinfer_prefill_wrapper_sp_causal": model_runner.flashinfer_prefill_wrapper_sp_causal,
    }
    aux_args = {"normal_to_sp_indices": normal_to_sp_indices}
    return init_args, aux_args
