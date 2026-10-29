"""Integer-only M4 verify layout and N-gram history snapshot.

The history is read into graph-owned output, never aliased to the mutable pool.
This kernel does not write pool state or replace hash/gather/commit consumers.
"""

import logging
from typing import NamedTuple

import torch
import triton
import triton.language as tl

from . import enabled

from sglang.kernels.ops.pdl_small import ENABLED as _PDL_SMALL

logger = logging.getLogger(__name__)
_logged = False


class VerifyLayout(NamedTuple):
    lengths: torch.Tensor
    req_indices: torch.Tensor
    token_offsets: torch.Tensor
    valid_tokens: torch.Tensor
    state_indices: torch.Tensor
    ngram_context: torch.Tensor


@triton.jit
def _q38_mk_mtp_ple_prepare(
    Tokens, Lengths, Requests, Locations, SlotMap, History,
    OutLengths, OutRequests, OutOffsets, OutValid, OutSlots, OutContext,
    EOS: tl.constexpr, HAS_LENGTHS: tl.constexpr, BLOCK: tl.constexpr,
    PDL_SMALL: tl.constexpr = False,
):
    # Protect dynamic reads and reused-buffer writes before relaxing this edge.
    if PDL_SMALL:
        tl.extra.cuda.gdc_wait()
    col = tl.arange(0, BLOCK)
    length = tl.load(Lengths) if HAS_LENGTHS else 4
    req = tl.load(Requests).to(tl.int64)
    slot = tl.load(SlotMap + req).to(tl.int64)
    live_loc = tl.load(Locations + col, col < 4, other=0) != 0
    live_row = (length != 0) & (tl.sum(live_loc.to(tl.int32), 0) != 0)
    slot = tl.where(live_row, slot, 0)
    valid = col < length
    token = tl.load(Tokens + col, col < 4, other=EOS)
    tl.store(OutLengths, length)
    tl.store(OutRequests + col, 0, col < 4)
    tl.store(OutOffsets + col, col, col < 4)
    tl.store(OutValid + col, valid, col < 4)
    tl.store(OutSlots, slot)
    history = tl.load(History + slot * 2 + col, col < 2, other=EOS)
    tl.store(OutContext + col, history, col < 2)
    tl.store(OutContext + 2 + col, tl.where(valid, token, EOS), col < 4)
    if PDL_SMALL:
        tl.extra.cuda.gdc_launch_dependents()


def prepare_verify_layout(tokens, lengths, requests, locations, slot_map, history, eos):
    """Low-level entry; caller guarantees one M4 row and contiguous pool inputs."""
    device = tokens.device
    result = VerifyLayout(
        torch.empty((1,), dtype=torch.int64, device=device),
        torch.empty((4,), dtype=torch.int64, device=device),
        torch.empty((4,), dtype=torch.int64, device=device),
        torch.empty((4,), dtype=torch.bool, device=device),
        torch.empty((1,), dtype=torch.int64, device=device),
        torch.empty((1, 6), dtype=torch.int64, device=device),
    )
    _q38_mk_mtp_ple_prepare[(1,)](
        tokens, lengths if lengths is not None else requests, requests,
        locations, slot_map, history, *result,
        EOS=eos, HAS_LENGTHS=lengths is not None, BLOCK=32, num_warps=4,
        launch_pdl=_PDL_SMALL,
        PDL_SMALL=_PDL_SMALL,
    )
    return result


def try_prepare_verify_layout(input_ids, forward_batch, mode, pool, *,
                              ngram_size, ngram_eos_token_id, processed_tokens):
    """Return the opt-in layout, or None to execute the original PLE prepare."""
    if not enabled():
        return None
    from sglang.srt.layers.cp.utils import is_cp_active
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
    from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
        is_in_breakable_cuda_graph,
    )
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    if not (
        mode.is_target_verify()
        and forward_batch.forward_mode.is_target_verify()
        and not is_in_breakable_cuda_graph()
        and parallel.tp_size in (1, 4) and parallel.pp_size == 1
        and not is_cp_active(forward_batch)
        and input_ids.is_cuda and input_ids.dtype == torch.int64
        and input_ids.numel() == processed_tokens == 4
        and forward_batch.spec_info.draft_token_num == 4
        and ngram_size == 3 and ngram_eos_token_id is not None
        and type(pool) is HybridReqToTokenPool
        and get_token_to_kv_pool().layer_transfer_counter is None
    ):
        return None
    requests = forward_batch.req_pool_indices
    locations = forward_batch.out_cache_loc
    lengths = forward_batch.extend_seq_lens
    slot_map, history = pool.req_index_to_mamba_index_mapping, pool.ngram_pool.context
    if not (
        requests.numel() == 1 and requests.dtype in (torch.int32, torch.int64)
        and locations is not None and locations.numel() >= 4
        and slot_map.dtype == torch.int32
        and history is not None and history.dtype == torch.int64
        and history.ndim == 2 and history.shape[1] == 2
        and (lengths is None or (lengths.numel() == 1
             and lengths.dtype in (torch.int32, torch.int64)))
        and all(t.is_contiguous() and t.device == input_ids.device
                for t in (input_ids, requests, locations, slot_map, history))
    ):
        return None
    global _logged
    if not _logged:
        logger.info("Q38_MK_MTP PLE M4 integer prepare enabled; native hash/gather/commit")
        _logged = True
    return prepare_verify_layout(input_ids, lengths, requests, locations,
                                 slot_map, history, ngram_eos_token_id)
