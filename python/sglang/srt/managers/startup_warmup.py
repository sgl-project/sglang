"""Decode warm-up before a scheduler reports ready (``--enable-decode-warmup``).

A process's first speculative decode round pays one-time costs: lazy JIT builds
and loads, first-call setup in the draft, verify and post-verify kernels. The
warm-up runs one prefill and one decode round on dummy requests at each of
``--decode-warmup-batch-sizes`` before the ready message, so those costs land at
startup instead of in the first served decode.

The dummy requests sample with the server's ``--preferred-sampling-params`` (the
SamplingParams defaults when unset, as a request that sends no sampling parameters
on the native or Engine path), never enter the radix tree or HiCache, and their
request, KV and mamba slots are freed. The CPU, CUDA, Python and numpy RNG states,
the forced-acceptance generator (SGLANG_SIMULATE_ACC_LEN) and the staged next
draft's note are restored, so served sampling draws and staging decisions are
unchanged. What stays is compiled and loaded kernels, captured graphs, post-verify
graph keys at the warm-up's sampling mode (at most one per batch size, each with one
eager round counted toward its capture), allocator cache, and the free-list order of
the pools (placement only). The rounds run without autograd, as the event loop does.
A failed round or a pool that does not come back stops the server.
"""

from __future__ import annotations

import logging
import random
import time
from array import array
from contextlib import nullcontext
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)

# Arbitrary ordinary token ids for the dummy prompts.
_TOKEN_BASE = 1000
# Used when the server sets neither --max-prefill-tokens nor --chunked-prefill-size.
_DEFAULT_PREFILL_BUDGET = 8192


def run_decode_warmup(
    scheduler: Scheduler,
    *,
    batch_sizes: List[int],
    prompt_len: int,
    sampling: Optional[dict],
) -> None:
    """Run the warm-up rounds; raise if one fails or the pools do not come back."""
    worker = scheduler.model_worker
    refusal = _refusal(scheduler, worker)
    if refusal is not None:
        logger.warning("Decode warm-up skipped: %s", refusal)
        return
    rounds = _plan(
        batch_sizes=batch_sizes,
        prompt_len=prompt_len,
        max_running_requests=scheduler.max_running_requests,
        max_prefill_tokens=scheduler.max_prefill_tokens,
        chunked_prefill_size=scheduler.chunked_prefill_size,
        page_size=scheduler.page_size,
        max_req_input_len=scheduler.max_req_input_len,
    )
    import torch

    pools_before = _pool_text(scheduler)
    saved = _save_rng(worker)
    started = time.perf_counter()
    done, error = 0, None
    stream_ctx = (
        scheduler.forward_stream_ctx if scheduler.enable_overlap else nullcontext()
    )
    try:
        with stream_ctx, torch.no_grad():
            for bs, round_prompt_len in rounds:
                prefill_s, decode_s = _round(
                    scheduler,
                    worker,
                    bs=bs,
                    prompt_len=round_prompt_len,
                    sampling=sampling or {},
                    tag=str(done),
                )
                if done == 0:
                    logger.info(
                        "Decode warm-up: first round bs=%d prompt=%d prefill %.2f s, "
                        "decode %.2f s",
                        bs,
                        round_prompt_len,
                        prefill_s,
                        decode_s,
                    )
                done += 1
    except Exception as exc:
        error = exc
        logger.exception(
            "Decode warm-up failed in round %d of %d", done + 1, len(rounds)
        )
    finally:
        _restore(worker, saved)
    pools_after = _pool_text(scheduler)
    if error is None and pools_after != pools_before:
        error = RuntimeError(
            f"pools before the rounds: {pools_before}; after: {pools_after}"
        )
    if error is not None:
        # A failed round may leave pool or worker state behind, so do not serve.
        raise RuntimeError(
            "Decode warm-up failed; restart without --enable-decode-warmup"
        ) from error
    logger.info(
        "Decode warm-up done: %d rounds at bs %s in %.1f s; pools unchanged",
        done,
        [bs for bs, _ in rounds],
        time.perf_counter() - started,
    )


def _refusal(scheduler: Scheduler, worker) -> Optional[str]:
    from sglang.srt.disaggregation.mode import DisaggregationMode
    from sglang.srt.runtime_context import get_parallel
    from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2

    if scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
        return "a PD prefill server runs no decode round"
    if not isinstance(worker, EAGLEWorkerV2):
        return "the round drives the EAGLE v2 speculative worker only"
    parallel = get_parallel()
    if parallel.pp_size > 1 or parallel.attn_dp_enabled:
        return "pipeline or DP-attention ranks would not run the same rounds"
    if scheduler.hisparse_coordinator is not None:
        return "HiSparse stages its rows on the scheduler's batch path the rounds skip"
    return None


def _plan(
    *,
    batch_sizes: List[int],
    prompt_len: int,
    max_running_requests: int,
    max_prefill_tokens: Optional[int],
    chunked_prefill_size: Optional[int],
    page_size: int,
    max_req_input_len: int,
) -> List[Tuple[int, int]]:
    """(batch size, prompt tokens per request) per round, in descending batch size;
    each batch fits the prefill budget and max_running_requests."""
    budgets = [
        b for b in (max_prefill_tokens, chunked_prefill_size) if b is not None and b > 0
    ]
    prefill_budget = min(budgets) if budgets else _DEFAULT_PREFILL_BUDGET
    page = max(1, page_size)
    rounds = []
    for bs in sorted(
        {min(bs, max_running_requests) for bs in batch_sizes}, reverse=True
    ):
        round_prompt_len = min(prompt_len, prefill_budget // bs, max_req_input_len - 1)
        if round_prompt_len >= page:
            round_prompt_len -= round_prompt_len % page
        if round_prompt_len >= 1:
            rounds.append((bs, round_prompt_len))
    return rounds


def _round(
    scheduler: Scheduler, worker, *, bs: int, prompt_len: int, sampling: dict, tag: str
) -> Tuple[float, float]:
    """One dummy prefill and one decode round on ``bs`` fresh requests;
    (prefill seconds, decode seconds)."""
    import torch

    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.mem_cache.common import checkpoint_kv_cache

    tree_cache = scheduler.tree_cache
    reqs = []
    try:
        for i in range(bs):
            req = _dummy_req(
                scheduler,
                tree_cache,
                i=i,
                prompt_len=prompt_len,
                sampling=sampling,
                tag=tag,
            )
            reqs.append(req)
        batch = ScheduleBatch.init_new(
            reqs,
            scheduler.req_to_token_pool,
            scheduler.token_to_kv_pool_allocator,
            tree_cache,
            scheduler.model_config,
            scheduler.enable_overlap,
            scheduler.spec_algorithm,
        )
        batch.prepare_for_extend()
        if batch.input_ids is None and batch.prefill_input_ids_cpu is not None:
            batch.input_ids = batch.prefill_input_ids_cpu.to(
                batch.device, non_blocking=True
            )
            batch.prefill_input_ids_cpu = None
        started = time.perf_counter()
        # Scheduler.run_batch's non-overlap V2 branch, then the prefill result's
        # commit, without streaming, finish checks or a tree checkpoint.
        result = _forward(scheduler, worker, batch)
        batch.spec_info = result.next_draft_input
        if (
            result.new_seq_lens is not None
            and result.new_seq_lens is not batch.seq_lens
        ):
            batch.seq_lens = result.new_seq_lens
            if batch.seq_lens_cpu is not None:
                batch.seq_lens_cpu = result.new_seq_lens.to("cpu")
                batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
        batch.input_ids = None
        next_token_ids = result.next_token_ids.tolist()
        prefilled = time.perf_counter()
        for req, token in zip(reqs, next_token_ids):
            req.output_ids.append(token)
            checkpoint_kv_cache(req, tree_cache)
        batch.prepare_for_decode()
        _forward(scheduler, worker, batch)
        torch.cuda.synchronize()
        return prefilled - started, time.perf_counter() - prefilled
    finally:
        _release(scheduler, worker, reqs)


def _dummy_req(scheduler: Scheduler, tree_cache, *, i, prompt_len, sampling, tag):
    """A request walked through the match, lock and alloc lifecycle a scheduled one
    takes, barred from the radix tree."""
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    vocab = scheduler.model_config.vocab_size
    params = SamplingParams(**{"max_new_tokens": 64, "ignore_eos": True, **sampling})
    params.normalize(scheduler.tokenizer)
    params.verify(vocab)
    ids = array("q", [(_TOKEN_BASE + 7 * i + j) % vocab for j in range(prompt_len)])
    req = Req(
        rid=f"decode-warmup-{tag}-{i}",
        origin_input_text="",
        origin_input_ids=ids,
        sampling_params=params,
        vocab_size=vocab,
    )
    req.skip_radix_cache_insert = True
    req.init_next_round_input(tree_cache)
    req.lock = tree_cache.lock(req.last_node)
    req.extend_end = len(req.full_untruncated_fill_ids)
    return req


def _pool_text(scheduler: Scheduler) -> str:
    """Free request rows and the tree's available and evictable sizes per pool:
    equal before and after when every dummy slot came back and nothing entered the
    tree."""
    tree_text = (
        scheduler.tree_cache.available_and_evictable_str().strip().replace("\n", "; ")
    )
    return f"request rows {scheduler.req_to_token_pool.available_size()}; {tree_text}"


def _forward(scheduler: Scheduler, worker, batch):
    with scheduler._forward_isolation(batch, overlap=False):
        return worker.forward_batch_generation(batch)


def _release(scheduler: Scheduler, worker, reqs) -> None:
    import torch

    # A pending side-stream mamba commit may still write these rows; join it first.
    if worker._post_verify_graphs is not None:
        worker._post_verify_graphs.join()
    torch.cuda.synchronize()
    for req in reqs:
        # The retirer's uncheckpointed release, as for any request that ends.
        scheduler.request_retirer.release(req, checkpoint=False)
        if req.lock is not None:
            scheduler.tree_cache.unlock(req.lock)
            req.lock = None


def _save_rng(worker):
    import numpy as np
    import torch

    # The forced-acceptance lengths come from the worker's own generator when it
    # has one (the post-verify graphs' stream), else from the CPU generator.
    sim_rng = worker._simulated_acc_len_rng
    return (
        torch.default_generator.get_state(),
        torch.cuda.get_rng_state(),
        random.getstate(),
        np.random.get_state(),
        None if sim_rng is None else sim_rng.get_state(),
    )


def _restore(worker, saved) -> None:
    """Undo what the rounds changed that a served request can observe."""
    import numpy as np
    import torch

    pv = worker._post_verify_graphs
    if pv is not None:
        pv.join()
        # The last round's staging; the next served round sets both before reading.
        pv.next_draft = None
        pv._graphs.last_slot = None
    torch.cuda.synchronize()
    if worker._staged_draft is not None:
        # A served first decode whose request-pool ids match the last round's must
        # not take its staging.
        worker._staged_draft.forget()
    cpu_state, cuda_state, py_state, np_state, sim_state = saved
    torch.default_generator.set_state(cpu_state)
    torch.cuda.set_rng_state(cuda_state)
    random.setstate(py_state)
    np.random.set_state(np_state)
    if sim_state is not None:
        # Drawn on demand, one length per round: the first served draw is the one
        # a server without the warm-up makes.
        worker._simulated_acc_len_rng.set_state(sim_state)
        from sglang.srt.speculative import spec_utils

        # A pre-drawn block holds draws past the restored state.
        spec_utils._SIM_STREAMS.clear()
