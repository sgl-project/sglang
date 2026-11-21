"""Startup-only preparation of DeepSeek-V4 decoder tail row graphs."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
import tqdm

from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.attention.dsv4.unified_kv_gate import is_unified_kv_triton
from sglang.srt.layers.cp.base import ContextParallelStrategyKind, get_cp_strategy
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.forward_mode import CaptureHiddenMode, ForwardMode
from sglang.srt.model_executor.runner_utils import capture_mode
from sglang.srt.runtime_context import get_disagg, get_flags, get_parallel, get_schedule

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.runner.eager_runner import EagerRunner
    from sglang.srt.models.deepseek_v4_replay_tail import DecoderReplayTail

logger = logging.getLogger(__name__)
# Decoder bounded replay retains one SWA window per request.
_TAIL_WINDOW = 128


@contextmanager
def _startup_batch(model_runner: ModelRunner, lengths: list[int]):
    """Lease distinct real request slots and paged KV, never the graph dummy row."""
    pool = model_runner.req_to_token_pool
    allocator = model_runner.token_to_kv_pool_allocator
    device = model_runner.device
    indices = pool.alloc_rows(len(lengths))
    if indices is None:
        raise RuntimeError("Decoder tail startup: insufficient temporary request slots")
    slots = None
    saved_rows = []
    try:
        seq_cpu = torch.tensor(lengths, dtype=torch.int64)
        seq = seq_cpu.to(device)
        zeros_cpu = torch.zeros_like(seq_cpu)
        zeros = zeros_cpu.to(device)
        req_cpu = torch.tensor(indices, dtype=torch.int64)
        req = req_cpu.to(device)
        slots = allocator.alloc_extend(
            prefix_lens=zeros,
            prefix_lens_cpu=zeros_cpu,
            seq_lens=seq,
            seq_lens_cpu=seq_cpu,
            last_loc=torch.full_like(seq, -1),
            extend_num_tokens=sum(lengths),
        )
        if slots is None:
            raise RuntimeError("Decoder tail startup: insufficient temporary KV pages")
        pool.alloc_aux_to_lengths(
            req_pool_indices_cpu=req_cpu, target_seq_lens_cpu=seq_cpu
        )
        offset = 0
        for index, length in zip(indices, lengths):
            saved_rows.append((index, pool.req_to_token[index, :length].clone()))
            pool.write((index, slice(0, length)), slots[offset : offset + length])
            offset += length
        n = sum(lengths)
        count = torch.tensor(n, dtype=torch.int32, device=device)
        batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=len(lengths),
            input_ids=torch.zeros(n, dtype=torch.int64, device=device),
            req_pool_indices=req,
            req_pool_indices_cpu=req_cpu,
            seq_lens=seq,
            seq_lens_cpu=seq_cpu,
            orig_seq_lens=seq,
            out_cache_loc=slots,
            seq_lens_sum=n,
            positions=torch.cat([torch.arange(v, device=device) for v in lengths]),
            extend_num_tokens=n,
            extend_seq_lens=seq,
            extend_seq_lens_cpu=list(lengths),
            extend_prefix_lens=zeros,
            extend_prefix_lens_cpu=[0] * len(lengths),
            extend_start_loc=(seq.cumsum(0) - seq).to(torch.int32),
            extend_logprob_start_lens_cpu=list(lengths),
            return_logprob=False,
            spec_algorithm=model_runner.spec_algorithm,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            is_extend_in_batch=True,
            # This synthetic DP=1 prefill is identical on all participating ranks.
            can_run_dp_prefill_cuda_graph=True,
            global_forward_mode=ForwardMode.EXTEND,
            global_num_tokens_cpu=[n],
            global_num_tokens_gpu=torch.tensor([n], dtype=torch.int64, device=device),
            global_num_tokens_for_logprob_cpu=[n],
            global_num_tokens_for_logprob_gpu=torch.tensor(
                [n], dtype=torch.int64, device=device
            ),
            global_num_token_non_padded=count,
            global_num_token_non_padded_cpu=n,
            num_token_non_padded=count.clone(),
        )
        model_runner.runner_ids.bind_runner_slots(batch)
        model_runner._prepare_eager_forward_batch(batch)
        yield batch
    finally:
        # Complete the temporary KV writes before returning pages to the allocator.
        torch.get_device_module(device).synchronize()
        for index, value in saved_rows:
            pool.write((index, slice(0, value.numel())), value)
        if slots is not None:
            allocator.free(slots)
        pool.free_rows(indices)


def _skip_reason(tail: DecoderReplayTail, model_runner: ModelRunner) -> str | None:
    p = get_parallel()
    if tail.runner.max_rows == 0:
        return "row graphs disabled"
    if model_runner.is_draft_worker:
        return "draft worker"
    if get_disagg().disaggregation_mode == "decode":
        return "decode worker"
    if p.pp_size != 1 or p.attn_dp_size != 1:
        return "startup preparation requires PP=1 and attention DP=1"
    if is_unified_kv_triton() or envs.SGLANG_ENABLE_GRAPH_POOL_BORROW.get():
        return "incompatible graph/KV backend"
    if get_global_expert_distribution_recorder().recording:
        return "expert distribution recording"
    if p.attn_cp_size > 1:
        cp = get_cp_strategy()
        if (
            cp is None
            or cp.kind != ContextParallelStrategyKind.INTERLEAVE
            or p.attn_tp_size != 1
            or _TAIL_WINDOW % p.attn_cp_size
            or not (
                get_moe_a2a_backend().is_megamoe() or get_moe_a2a_backend().is_none()
            )
        ):
            return (
                "CP tail graphs require interleave CP, attention TP=1 "
                "and moe-a2a-backend=megamoe or none"
            )
    return None


def capture_decoder_tail_graphs(
    tail: DecoderReplayTail,
    model_runner: ModelRunner,
    eager_runner: EagerRunner,
    prefill_runner=None,
) -> tuple[float, float]:
    """Run a full eager chunk, then capture all reachable buckets largest first."""
    if tail._startup_prepared:
        return 0.0, 0.0
    reason = _skip_reason(tail, model_runner)
    if reason is not None:
        logger.info(
            "Decoder tail graph startup skipped: %s; runtime capture disabled", reason
        )
        return 0.0, 0.0
    assert not get_flags().dp.capturing_prefill_graph
    assert not capture_mode.is_capture_mode
    cp_size = get_parallel().attn_cp_size
    rows = tail.runner.max_rows
    # A request contributes at most one tail window. Do not fabricate batches
    # exceeding the server's request limit or the eager input registry.
    request_limit = min(
        model_runner.max_running_requests,
        model_runner.req_to_token_pool.available_size(),
    )
    rows = min(rows, request_limit * _TAIL_WINDOW // cp_size)
    rows -= rows % tail.runner.row_step
    if rows == 0:
        raise RuntimeError("Decoder tail startup: no reachable row buckets")
    configured_chunk = get_schedule().chunked_prefill_size
    chunk = configured_chunk if configured_chunk and configured_chunk > 0 else 8192
    chunk = min(chunk, model_runner.model_config.context_len)
    if chunk < _TAIL_WINDOW:
        raise RuntimeError("Decoder tail startup requires at least one SWA window")
    # Replay a supported outer BCG shape during startup as well. Its fused
    # boundary can present a different HcState than the eager prefix.
    bcg_runner = (
        prefill_runner
        if getattr(prefill_runner, "prefill_backend_name", None) == Backend.BREAKABLE
        else None
    )
    capture_tokens = (
        max((n for n in bcg_runner.capture_num_tokens if n <= chunk), default=chunk)
        if bcg_runner is not None
        else chunk
    )
    rows = min(rows, capture_tokens // _TAIL_WINDOW * _TAIL_WINDOW // cp_size)
    rows -= rows % tail.runner.row_step
    if rows == 0:
        raise RuntimeError("Decoder tail startup: no buckets fit the prefill shape")

    def lengths_for(bucket):
        requests = bucket * cp_size // _TAIL_WINDOW
        length, remainder = divmod(capture_tokens, requests)
        return [length + (rank < remainder) for rank in range(requests)]

    device_module = torch.get_device_module(model_runner.device)
    backend = model_runner.attn_backend
    saved = {
        name: getattr(backend, name)
        for name in (
            "forward_metadata",
            "tail_forward_metadata",
            "prefill_graph_replay_batch",
        )
        if hasattr(backend, name)
    }
    start = time.perf_counter()
    device_module.synchronize()
    free_before = device_module.mem_get_info()[0]
    initial_graphs = len(tail.runner._graphs)
    bcg_buckets = 0
    logger.info(
        "Decoder tail graph startup: eager warmup=%s global tokens, "
        "CP=%s, local buckets=%s..%s step=%s",
        chunk,
        cp_size,
        tail.runner.row_step,
        rows,
        tail.runner.row_step,
    )
    try:
        with (
            torch.no_grad(),
            forward_context(ForwardContext(attn_backend=backend)),
        ):
            tail._startup_warming = True
            with _startup_batch(model_runner, [chunk]) as batch:
                eager_runner.execute(batch)
            # Also warm the largest request-axis shape before any tail capture.
            lengths = lengths_for(rows)
            with _startup_batch(model_runner, lengths) as batch:
                eager_runner.execute(batch)
            tail._startup_warming = False
            tail.runner._warm = True
            tail._startup_capture = True
            with tqdm.tqdm(
                range(rows, 0, -tail.runner.row_step),
                disable=get_parallel().tp_rank != 0,
            ) as progress:
                for bucket in progress:
                    if get_parallel().tp_rank == 0:
                        num_tokens = bucket * cp_size
                        local_rows = bucket
                        avail_mem = device_module.mem_get_info()[0] / 2**30
                        progress.set_description(
                            f"Capturing decoder tail shape ({num_tokens=} "
                            f"{local_rows=} {avail_mem=:.2f} GB)"
                        )
                    lengths = lengths_for(bucket)
                    before = len(tail.runner._graphs)
                    with _startup_batch(model_runner, lengths) as batch:
                        eager_runner.execute(batch)
                    if (
                        not tail._last_run_replayed
                        or len(tail.runner._graphs) <= before
                    ):
                        raise RuntimeError(
                            f"Decoder tail startup did not capture local bucket {bucket}"
                        )
                    if bcg_runner is not None:
                        with _startup_batch(model_runner, lengths) as batch:
                            if bcg_runner.can_run_graph(batch):
                                tail._last_run_replayed = False
                                bcg_runner.execute(batch)
                                if tail._last_run_replayed:
                                    bcg_buckets += 1
                                elif getattr(
                                    bcg_runner, "enable_cp_bcg_capture", False
                                ):
                                    raise RuntimeError(
                                        f"CP BCG startup did not replay tail bucket {bucket}"
                                    )
                            else:
                                logger.info(
                                    "Decoder tail startup: outer BCG ineligible for "
                                    "local bucket %s; eager variant prepared",
                                    bucket,
                                )
    finally:
        tail._startup_capture = False
        tail._startup_warming = False
        for name, value in saved.items():
            setattr(backend, name, value)
    device_module.synchronize()
    used_gb = (free_before - device_module.mem_get_info()[0]) / 2**30
    seconds = time.perf_counter() - start
    logger.info(
        "Decoder tail graph startup complete: CP=%s, %s buckets, %s new graphs, "
        "local_max_rows=%s, bcg_buckets=%s/%s, %.2f s, %.3f GiB; runtime capture disabled",
        cp_size,
        rows // tail.runner.row_step,
        len(tail.runner._graphs) - initial_graphs,
        rows,
        bcg_buckets,
        rows // tail.runner.row_step if bcg_runner is not None else 0,
        seconds,
        used_gb,
    )
    tail._startup_prepared = True
    return used_gb, seconds
