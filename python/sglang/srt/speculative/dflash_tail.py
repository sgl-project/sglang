"""Synchronous target-only steps at the actual model position boundary.

Every target DP owner chooses the same phase, including idle owners. Physical
KV headroom never authorizes evaluating RoPE at positions beyond model context.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def select_eager_phase(worker, batch):
    from sglang.srt.runtime_context import get_parallel

    limit = worker.model_runner.model_config.context_len
    decode = not (batch.forward_mode.is_extend() or batch.is_extend_in_batch)
    unsupported_sampling = (
        decode
        and batch.sampling_info is not None
        and not batch.sampling_info.is_all_greedy
    )
    tail = decode and any(
        int(r.kv.kv_committed_len) + worker.block_size > limit for r in batch.reqs
    )
    flags = torch.tensor(
        [int(unsupported_sampling), int(tail)], dtype=torch.int32, device="cpu"
    )
    parallel = get_parallel()
    if parallel.tp_group.world_size > 1:
        torch.distributed.all_reduce(
            flags, op=torch.distributed.ReduceOp.MAX, group=parallel.tp_group.cpu_group
        )
    if flags[0].item():
        raise ValueError(
            "The experimental independent DFlash execution path supports greedy requests only"
        )
    worker._target_only_tail_this_step = bool(flags[1].item())
    if decode:
        phase = "B1" if worker._target_only_tail_this_step else f"B{worker.block_size}"
        if getattr(worker, "_last_target_verify_phase", None) != phase:
            worker._last_target_verify_phase = phase
            logger.info(
                "DFLASH target verify phase=%s gpu=%s active_requests=%s "
                "context_limit=%s B1_graph=False",
                phase,
                getattr(worker, "gpu_id", None),
                len(batch.reqs),
                limit,
            )


def needs_target_only_tail(worker, batch):
    return worker._target_only_tail_this_step


def forward_target_only_tail(worker, batch, *, on_publish=None, grammar_barrier=None):
    from sglang.srt.layers.logprob_processor import compute_spec_logprobs
    from sglang.srt.managers.utils import GenerationBatchResult
    from sglang.srt.model_executor.forward_mode import CaptureHiddenMode
    from sglang.srt.speculative.dflash_info import DFlashVerifyInput
    from sglang.srt.speculative.dflash_utils import (
        apply_dflash_verify_logits_adjustments,
    )
    from sglang.srt.speculative.dflash_worker_v2 import _chain_accept_indices
    from sglang.srt.speculative.spec_tp_sync import SpecTpSyncSite
    from sglang.srt.speculative.spec_utils import GrammarTree, build_grammar_vocab_mask

    idle = batch.forward_mode.is_idle()
    device = worker.device
    count = 0 if idle else len(batch.reqs)
    prefix = batch.seq_lens
    tokens = (
        torch.empty((0, 1), dtype=torch.int64, device=device)
        if idle
        else batch.spec_info.bonus_tokens.reshape(count, 1)
    )
    positions = (
        torch.empty((0,), dtype=torch.int64, device=device)
        if idle
        else prefix.to(torch.int64)
    )
    if (
        count
        and torch.any(positions >= worker.model_runner.model_config.context_len).item()
    ):
        raise RuntimeError("Target-only tail attempted to exceed model position range")
    locations = (
        worker.model_runner.req_to_token_pool.req_to_token[
            batch.req_pool_indices, positions
        ].to(torch.int64)
        if count
        else torch.empty((0,), dtype=torch.int64, device=device)
    )
    host = batch.seq_lens_cpu
    plan = None
    if count:
        plan = worker.model_runner.runner_ids.plan(
            req_pool_indices=batch.req_pool_indices,
            seq_lens=prefix,
            seq_lens_cpu=host + 1,
            write_virtual=locations,
            read_extent=1,
        )
    batch.out_cache_loc = locations
    if count and batch.hisparse_coordinator is not None:
        batch.hisparse_coordinator.prepare_spec_verify(batch)
    verify = DFlashVerifyInput(
        draft_token=tokens.reshape(-1),
        positions=positions,
        draft_token_num=1,
        custom_mask=None,
        capture_hidden_mode=CaptureHiddenMode.FULL,
        iteration_ids=plan,
    )
    total = batch.seq_lens_sum
    if host is not None:
        batch.seq_lens_cpu = host + 1
        batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
    forward, _ = verify.prepare_for_verify(
        batch, worker.target_worker, defer_metadata_init=True
    )
    batch.seq_lens_cpu, batch.seq_lens_sum = host, total
    forward.can_run_decode_cuda_graph = False
    result = worker.target_worker.forward_batch_generation(
        batch=None, forward_batch=forward, is_verify=True
    )
    if result.can_run_cuda_graph:
        raise RuntimeError("DFLASH B1 context tail must execute eagerly")
    logits = result.logits_output
    commits = torch.ones((count,), dtype=torch.int32, device=device)
    if count:
        processor = worker._target_logits_processor
        if (
            processor is not None
            and processor.is_deferred_shard(logits.next_token_logits)
            and not worker._can_skip_verify_logits(batch, batch.sampling_info)
        ):
            logits.next_token_logits = processor.finish_deferred_gather(
                logits.next_token_logits
            )
        if processor is not None and processor.is_deferred_shard(
            logits.next_token_logits
        ):
            bonus = processor.deferred_argmax(logits.next_token_logits)
            logits.next_token_logits = None
        else:
            if batch.sampling_info is not None:
                apply_dflash_verify_logits_adjustments(
                    next_token_logits=logits.next_token_logits,
                    sampling_info=batch.sampling_info,
                    draft_token_num=1,
                )
            if batch.has_grammar:
                mask = build_grammar_vocab_mask(
                    reqs=batch.reqs,
                    tree=GrammarTree.from_linear_chain(tokens),
                    sampling_info=batch.sampling_info,
                    device=logits.next_token_logits.device,
                    barrier=grammar_barrier,
                )
                mask.apply(logits.next_token_logits)
            bonus = logits.next_token_logits.argmax(dim=-1)
        worker._tp_sync.sync(SpecTpSyncSite.DFLASH_ACCEPT_GREEDY, bonus)
        if batch.hisparse_coordinator is not None:
            batch.hisparse_coordinator.commit_spec_accept_tokens(
                batch, _chain_accept_indices(commits, 1)
            )
        hidden = logits.hidden_states
        if hidden is None:
            raise RuntimeError("Target-only tail lost captured draft context")
    else:
        bonus = torch.empty((0,), dtype=torch.int64, device=device)
        hidden = None
    if count:
        slots = batch.req_pool_indices
        full, window, _ = worker._independent_cache.locations(slots, positions, commits)
        worker._append_target_hidden_to_draft_kv_by_loc(
            target_hidden=hidden,
            cache_loc=full,
            swa_cache_loc=window,
            positions=positions,
            extend_lens=commits,
        )
    if count and batch.return_logprob:
        compute_spec_logprobs(batch, logits, bonus, chain_stride=1)
    if logits is not None:
        logits.hidden_states = None
    new_lengths = prefix + 1 if count else prefix
    if on_publish is not None:
        on_publish(new_lengths)
    next_input = worker._make_next_draft_input_decode(
        bonus_tokens=bonus, new_seq_lens=new_lengths
    )
    return GenerationBatchResult(
        logits_output=logits,
        next_token_ids=bonus,
        accept_lens=commits,
        next_draft_input=next_input,
        can_run_cuda_graph=False,
        speculative_num_draft_tokens=1,
        new_seq_lens=new_lengths,
        routed_experts_output=result.routed_experts_output,
        indexer_topk_output=result.indexer_topk_output,
        kv_weight_version_record=result.kv_weight_version_record,
    )
