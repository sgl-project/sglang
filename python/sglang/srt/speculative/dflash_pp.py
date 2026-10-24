"""DFLASH verify forwards for the stages before the last one under PP."""

from __future__ import annotations

import torch

from sglang.kernels.ops.speculative.cache_locs import assign_extend_cache_locs_func
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.dflash_info import DFlashVerifyInput


def build_relayed_verify(batch, target_worker, tokens: torch.Tensor):
    """Verify ``tokens`` ([bs, block], column 0 the bonus) at the batch's
    committed lengths, laid out as the DFLASH worker lays out its own block."""
    runner = target_worker.model_runner
    bs, block = tokens.shape
    prefix_lens = batch.seq_lens
    offsets = torch.arange(block, dtype=prefix_lens.dtype, device=prefix_lens.device)
    positions = (prefix_lens.unsqueeze(1) + offsets).reshape(-1).to(torch.int64)
    verify_out_cache_loc = assign_extend_cache_locs_func(
        req_pool_indices=batch.req_pool_indices,
        req_to_token=runner.req_to_token_pool.req_to_token,
        start_offset=prefix_lens,
        end_offset=prefix_lens + block,
        batch_size=bs,
        draft_token_num=block,
        device=prefix_lens.device,
    )
    seq_lens_cpu = batch.seq_lens_cpu
    if seq_lens_cpu is None:
        seq_lens_cpu = prefix_lens.to("cpu")
    kv_loc_plan = runner.kv_index_translator.plan(
        req_pool_indices=batch.req_pool_indices,
        seq_lens=prefix_lens,
        seq_lens_cpu=seq_lens_cpu,
        write_virtual=verify_out_cache_loc,
        read_extent=block,
    )
    verify_input = DFlashVerifyInput(
        draft_token=tokens.reshape(-1),
        positions=positions,
        draft_token_num=block,
        custom_mask=None,
        capture_hidden_mode=CaptureHiddenMode.FULL,
        kv_loc_plan=kv_loc_plan,
    )
    batch.out_cache_loc = verify_out_cache_loc
    backup = (batch.seq_lens_cpu, batch.seq_lens_sum)
    # Verify host bound = committed prefix + one verify block.
    batch.seq_lens_cpu = seq_lens_cpu + block
    batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
    try:
        forward_batch, can_run_cuda_graph = verify_input.prepare_for_verify(
            batch, target_worker
        )
    finally:
        batch.seq_lens_cpu, batch.seq_lens_sum = backup
    return forward_batch, can_run_cuda_graph, verify_out_cache_loc


def run_relayed_verify(batch, target_worker, pp_proxy_tensors):
    """An earlier stage's model chunk over the relayed block."""
    tokens = batch.spec_info.tokens.to(device=batch.seq_lens.device, dtype=torch.int64)
    forward_batch, can_run_cuda_graph, verify_out_cache_loc = build_relayed_verify(
        batch, target_worker, tokens
    )
    result = target_worker.forward_batch_generation(
        batch=None,
        forward_batch=forward_batch,
        pp_proxy_tensors=pp_proxy_tensors,
        is_verify=True,
        skip_attn_backend_init=True,
    )
    result.can_run_cuda_graph = can_run_cuda_graph
    return result, verify_out_cache_loc
