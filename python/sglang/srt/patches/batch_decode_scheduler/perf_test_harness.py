"""In-process driver for a real SGLang ``Scheduler``, for GPU E2E latency bench.

The harness reproduces exactly what ``Scheduler.event_loop_normal`` does, one
step at a time, so the numbers are the scheduler's own numbers::

    plan   = scheduler.get_next_batch_to_run(running_batch, last_batch)
    result = scheduler.run_batch(plan.batch_to_run)
             scheduler.process_batch_result(plan.batch_to_run, result)

Measurement semantics are aligned with RTP-LLM ``grid_perf_test`` and the vLLM
``batch_decode_scheduler`` patch so all three can be put in one table:

* ``cost_ms``      -> RTP-LLM ``cost_time``
* ``prefill_ms``   -> RTP-LLM ``first_token_cost_time``
* ``per_token_ms`` -> RTP-LLM ``decode_time_per_token``

Only three device synchronizations happen per measured round
(``mark_batch_start`` / ``mark_lap`` / ``mark_batch_end``); the measurement loop
itself never synchronizes, so decode step overlap is preserved. The
``run_step_timed`` path exists purely for ``--per-step-timing`` diagnostics and
its numbers are reported in a separate section.
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from array import array
from typing import Iterator, List, Optional

import msgspec
import torch

from sglang.srt.configs.hybrid_arch import mambaish_config
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import AbortReq, TokenizedGenerateReqInput
from sglang.srt.managers.overlap_utils import resolve_forward_inputs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import is_cuda, is_hip

logger = logging.getLogger(__name__)

# Token id fed to the model as the "sampled" token of a fake prefill. Must be a
# valid vocab index or Req._check_vocab_boundary_finish aborts the request with
# "NaN happened". 1 matches the vLLM patch.
FAKE_TOKEN_ID = 1

# Upper bound on the prefill token budget when fake-KV registration is on. The
# budget still sizes runner-side host/device staging buffers even though no
# prefill forward runs, and a single pinned allocation above ~1 GiB is rejected
# by the host allocator. Matches vLLM's _FAKE_KV_TOKEN_BUDGET_CAP.
FAKE_KV_TOKEN_BUDGET_CAP = 65536

# How many consecutive non-extend schedule decisions we tolerate while
# registering fake-KV requests before declaring the scheduler stuck.
_MAX_REGISTRATION_STALLS = 16

# Safety bound on drain(): a drain step frees at least one request slot, so any
# sane batch drains well within this.
_MAX_DRAIN_STEPS = 4096

PHASE_EMPTY = "empty"
PHASE_IDLE = "idle"
PHASE_PREFILL = "prefill"
PHASE_DECODE = "decode"
PHASE_MIXED = "mixed"


class StepStat(msgspec.Struct, frozen=True):
    """Immutable snapshot of one scheduler step. ``forward_ms`` is only
    populated by ``run_step_timed`` (the diagnostic path)."""

    phase: str
    batch_size: int
    extend_num_tokens: int
    forward_ms: float = 0.0


def _phase_of(batch: Optional[ScheduleBatch]) -> str:
    if batch is None:
        return PHASE_EMPTY
    mode = batch.forward_mode
    if mode.is_idle():
        return PHASE_IDLE
    if mode.is_decode():
        return PHASE_DECODE
    if mode.is_mixed():
        return PHASE_MIXED
    if mode.is_extend():
        return PHASE_PREFILL
    return mode.name.lower()


def _stat_of(batch: Optional[ScheduleBatch], forward_ms: float = 0.0) -> StepStat:
    if batch is None:
        return StepStat(
            phase=PHASE_EMPTY, batch_size=0, extend_num_tokens=0, forward_ms=forward_ms
        )
    return StepStat(
        phase=_phase_of(batch),
        batch_size=batch.batch_size(),
        extend_num_tokens=batch.extend_num_tokens or 0,
        forward_ms=forward_ms,
    )


def synthetic_input_ids(req_index: int, seq_len: int) -> array:
    """Per-request-distinct prompt token ids.

    Radix cache is disabled by the runner, but distinct prompts are a second
    line of defence: identical prompts would make every request after the first
    a pure prefix-cache hit and the prefill number meaningless.
    """
    return array("q", [((req_index * 7919 + j) % 4096) + 1 for j in range(seq_len)])


class BenchHarness:
    """Drives one in-process ``Scheduler`` (one TP/DP rank) step by step."""

    def __init__(
        self,
        *,
        server_args: ServerArgs,
        port_args: PortArgs,
        gpu_id: int,
        tp_rank: int,
        moe_ep_rank: int = 0,
        pp_rank: int = 0,
        attn_cp_rank: int = 0,
        moe_dp_rank: int = 0,
        dp_rank: Optional[int] = None,
    ):
        self.scheduler = Scheduler(
            server_args,
            port_args,
            gpu_id,
            tp_rank,
            moe_ep_rank,
            pp_rank,
            attn_cp_rank,
            moe_dp_rank,
            dp_rank,
        )
        self._mute_ipc_senders()

        scheduler = self.scheduler
        self.model_runner = scheduler.tp_worker.model_runner
        self.model_config = scheduler.model_config
        self.vocab_size = self.model_config.vocab_size
        self.device = scheduler.device
        self.device_module = torch.get_device_module(self.device)
        self.is_mambaish_model = mambaish_config(self.model_config) is not None

        self._req_counter = 0
        self._batch_start = 0.0
        self._sync_count = 0
        self._exit_stack: Optional[contextlib.ExitStack] = None

        self._init_schedule_stream()

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------

    def _mute_ipc_senders(self) -> None:
        """Drop scheduler -> tokenizer / detokenizer output.

        ``SenderWrapper.send_output`` short-circuits on ``socket is None`` — that
        is the officially supported no-op path (non-rank-zero schedulers are
        constructed that way). We mutate the wrappers in place rather than
        rebuilding ``SchedulerIpcChannels`` (a frozen dataclass) because several
        components captured the wrapper objects at construction time.
        """
        channels = self.scheduler.ipc_channels
        channels.send_to_tokenizer.socket = None
        channels.send_to_detokenizer.socket = None

    def _init_schedule_stream(self) -> None:
        """Replicate the ``Scheduler.run_event_loop`` prologue.

        ``run_batch``'s overlap path and ``_apply_war_barrier`` both read
        ``schedule_stream`` / ``_war_barrier_enabled``, which are only set up by
        ``run_event_loop`` — which we never call.
        """
        scheduler = self.scheduler
        scheduler.schedule_stream = self.device_module.Stream(priority=0)
        if self.device == "cpu":
            scheduler.schedule_stream.synchronize = lambda: None
        elif is_cuda() or is_hip():
            redraws = 0
            while (
                scheduler.schedule_stream.cuda_stream
                == scheduler.forward_stream.cuda_stream
                and redraws < 64
            ):
                scheduler.schedule_stream = self.device_module.Stream(priority=0)
                redraws += 1
        scheduler._war_barrier_enabled = (
            is_cuda() or envs.SGLANG_ENABLE_WAR_BARRIER.get()
        )

    def __enter__(self) -> "BenchHarness":
        # Same ambient state the real event loops run under: DynamicGradMode
        # decorates event_loop_normal/overlap, and run_event_loop wraps the
        # dispatch in the schedule stream context.
        from sglang.srt.utils import DynamicGradMode

        stack = contextlib.ExitStack()
        stack.enter_context(DynamicGradMode())
        stack.enter_context(
            self.device_module.StreamContext(self.scheduler.schedule_stream)
        )
        self._exit_stack = stack
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        stack, self._exit_stack = self._exit_stack, None
        if stack is not None:
            stack.close()

    def get_init_info(self) -> dict:
        return self.scheduler.get_init_info()

    # ------------------------------------------------------------------
    # Request submission
    # ------------------------------------------------------------------

    def submit(
        self,
        batch_size: int,
        seq_len: int,
        *,
        max_new_tokens: int = 1,
        ignore_eos: bool = True,
    ) -> List[str]:
        """Enqueue ``batch_size`` synthetic requests of ``seq_len`` prompt tokens."""
        rids = []
        for _ in range(batch_size):
            rid = f"bench-{self._req_counter}"
            input_ids = synthetic_input_ids(self._req_counter, seq_len)
            self._req_counter += 1

            sampling_params = SamplingParams(
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                ignore_eos=ignore_eos,
            )
            # handle_generate_request does NOT normalize/verify; TokenizerManager
            # normally does. Skipping normalize() leaves stop_strs as None and
            # blows up in the finish-condition check at the first decode step.
            sampling_params.normalize(None)
            sampling_params.verify(self.vocab_size)

            self.scheduler.handle_generate_request(
                TokenizedGenerateReqInput(
                    rid=rid,
                    input_text=None,
                    input_ids=input_ids,
                    input_embeds=None,
                    mm_inputs=None,
                    token_type_ids=None,
                    sampling_params=sampling_params,
                    return_logprob=False,
                    logprob_start_len=-1,
                    top_logprobs_num=0,
                    token_ids_logprob=None,
                    stream=False,
                )
            )
            rids.append(rid)

        self._assert_admitted(rids=rids, seq_len=seq_len)
        return rids

    def _assert_admitted(self, *, rids: List[str], seq_len: int) -> None:
        queued = {req.rid for req in self.scheduler.waiting_queue}
        missing = [rid for rid in rids if rid not in queued]
        if missing:
            raise AssertionError(
                f"{len(missing)}/{len(rids)} requests were rejected at admission "
                f"(seq_len={seq_len}). Most likely the prompt exceeds "
                f"max_req_input_len={self.scheduler.max_req_input_len} or the KV "
                f"pool is too small (max_total_num_tokens="
                f"{self.scheduler.max_total_num_tokens}). Lower --seq-lens or "
                f"raise --mem-fraction-static / --max-total-tokens."
            )
        for req in self.scheduler.waiting_queue:
            if req.rid in set(rids) and len(req.origin_input_ids) != seq_len:
                raise AssertionError(
                    f"request {req.rid} was truncated to "
                    f"{len(req.origin_input_ids)} tokens (wanted {seq_len})"
                )

    # ------------------------------------------------------------------
    # Step driving
    # ------------------------------------------------------------------

    def _advance_one_step(self) -> Optional[ScheduleBatch]:
        """The one and only step primitive. Byte-for-byte the body of
        ``Scheduler.event_loop_normal``, minus request receiving and the
        idle/self-check branch.

        ``cur_batch_for_debug`` is deliberately left unset so the scheduler
        watchdog stays inert during the bench.
        """
        scheduler = self.scheduler
        plan = scheduler.get_next_batch_to_run(
            running_batch=scheduler.running_batch, last_batch=scheduler.last_batch
        )
        scheduler.running_batch = plan.running_batch
        batch = plan.batch_to_run
        if batch:
            result = scheduler.run_batch(batch)
            scheduler.process_batch_result(batch, result)
        scheduler.last_batch = batch
        return batch

    def run_step_no_timing(self) -> StepStat:
        """Measurement / warmup / profile path. Never synchronizes."""
        return _stat_of(self._advance_one_step())

    def run_step_timed(self) -> StepStat:
        """Diagnostic path only (``--per-step-timing``).

        The two extra synchronizations drain the GPU between steps, which
        destroys decode step overlap and inflates the total. Never feed these
        numbers into the headline table.
        """
        self._sync()
        t0 = time.perf_counter()
        batch = self._advance_one_step()
        self._sync()
        return _stat_of(batch, forward_ms=(time.perf_counter() - t0) * 1e3)

    # ------------------------------------------------------------------
    # Timing marks (the only three synchronization points per round)
    # ------------------------------------------------------------------

    def _sync(self) -> None:
        self._sync_count += 1
        self.device_module.synchronize()

    @property
    def sync_count(self) -> int:
        """Number of device synchronizations issued so far (unit tests use this
        to prove the measurement path does not sync per step)."""
        return self._sync_count

    def mark_batch_start(self) -> None:
        self._sync()
        self._batch_start = time.perf_counter()

    def mark_lap(self) -> float:
        """Elapsed ms since ``mark_batch_start``. Does NOT reset the clock, so
        prefill and cost share one origin (same as the vLLM harness)."""
        self._sync()
        return (time.perf_counter() - self._batch_start) * 1e3

    def mark_batch_end(self) -> float:
        self._sync()
        return (time.perf_counter() - self._batch_start) * 1e3

    # ------------------------------------------------------------------
    # Assertions — a split batch must fail loudly, never produce a number
    # ------------------------------------------------------------------

    @staticmethod
    def assert_phase(stat: StepStat, expected: str) -> None:
        if stat.phase != expected:
            raise AssertionError(
                f"expected a {expected!r} step, scheduler produced {stat.phase!r} "
                f"(batch_size={stat.batch_size}, "
                f"extend_num_tokens={stat.extend_num_tokens})"
            )

    @staticmethod
    def assert_batch(stat: StepStat, expected_bs: int) -> None:
        if stat.batch_size != expected_bs:
            raise AssertionError(
                f"batch was split: scheduler ran batch_size={stat.batch_size}, "
                f"expected {expected_bs}. Raise --max-running-requests / "
                f"--mem-fraction-static, or lower the grid point."
            )

    @staticmethod
    def assert_prefill_tokens(stat: StepStat, batch_size: int, seq_len: int) -> None:
        expected = batch_size * seq_len
        if stat.extend_num_tokens != expected:
            raise AssertionError(
                f"prefill was chunked: extend_num_tokens={stat.extend_num_tokens}, "
                f"expected {expected}. Check --chunked-prefill-size (-1) and "
                f"--max-prefill-tokens."
            )

    def assert_scheduler_clean(self) -> None:
        """SGLang-specific: nothing left queued or half-chunked."""
        scheduler = self.scheduler
        if scheduler.chunked_req is not None:
            raise AssertionError(
                f"scheduler.chunked_req is set ({scheduler.chunked_req.rid}); "
                "prefill got chunked"
            )
        if scheduler.waiting_queue:
            raise AssertionError(
                f"{len(scheduler.waiting_queue)} request(s) stayed in the waiting "
                "queue; the batch did not fit in one step"
            )

    # ------------------------------------------------------------------
    # Draining
    # ------------------------------------------------------------------

    def drain(self) -> None:
        """Abort everything and step until the scheduler is fully idle.

        Real steps are required: request slots and KV blocks are only reclaimed
        by ``filter_batch`` / ``release_kv_cache`` inside a step.
        """
        self.scheduler.abort_request(AbortReq(rid="", abort_all=True))
        steps = 0
        while not self.scheduler.is_fully_idle():
            self._advance_one_step()
            steps += 1
            if steps > _MAX_DRAIN_STEPS:
                raise AssertionError(
                    f"scheduler did not go idle after {steps} drain steps "
                    f"(running={self.scheduler.running_batch.batch_size()}, "
                    f"waiting={len(self.scheduler.waiting_queue)})"
                )
        self.scheduler.last_batch = None
        self.scheduler._prev_decode_launch_ts = None

    # ------------------------------------------------------------------
    # Fake-KV registration (--partial 1)
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def _relaxed_prefill_budget(self, budget: int) -> Iterator[None]:
        """Temporarily raise the prefill token budget during fake-KV registration.

        ``get_next_batch_to_run`` runs prefill first whenever
        ``get_new_batch_prefill`` can return a batch. The only way already
        registered requests start decoding mid-registration is that call
        returning None — typically because of the prefill token budget. Since we
        never run the prefill forward, that budget has no physical cost here, so
        we lift it and let the whole batch register in a single step.
        """
        previous = self.scheduler.max_prefill_tokens
        self.scheduler.max_prefill_tokens = max(previous, budget)
        try:
            yield
        finally:
            self.scheduler.max_prefill_tokens = previous

    def submit_decode_only(
        self, batch_size: int, seq_len: int, num_decode_steps: int
    ) -> List[str]:
        """Register ``batch_size`` requests with real KV slots but zero KV content.

        Equivalent to RTP-LLM's ``setIsContextStream(false)``: KV blocks are
        really allocated and ``req_to_token`` is really written by
        ``prepare_for_extend``; only the prefill forward is skipped.
        """
        if self.is_mambaish_model:
            self._assert_mamba_fake_kv_supported()

        rids = self.submit(
            batch_size,
            seq_len,
            # +batch_size+1 of headroom so no request can finish during the
            # measured decode steps and shrink the batch.
            max_new_tokens=num_decode_steps + batch_size + 1,
        )
        pending = set(rids)
        stalled = 0
        scheduler = self.scheduler
        with self._relaxed_prefill_budget(batch_size * seq_len):
            while pending:
                plan = scheduler.get_next_batch_to_run(
                    running_batch=scheduler.running_batch,
                    last_batch=scheduler.last_batch,
                )
                batch = plan.batch_to_run
                if batch is None or not batch.forward_mode.is_extend():
                    # Do not execute: a decode step here would silently advance
                    # already-registered requests and split the KV lengths.
                    stalled += 1
                    if stalled > _MAX_REGISTRATION_STALLS:
                        raise AssertionError(
                            "fake-KV registration stalled: the scheduler stopped "
                            f"accepting new extends with {len(pending)} request(s) "
                            "left. Lower --batch-sizes / --seq-lens, or raise "
                            "--mem-fraction-static / --max-total-tokens."
                        )
                    continue
                scheduler.running_batch = plan.running_batch
                self._fake_process_batch_result(batch)
                scheduler.last_batch = batch
                pending -= {req.rid for req in batch.reqs}
                stalled = 0

        self._assert_uniform_registration(batch_size, seq_len)
        return rids

    def _assert_mamba_fake_kv_supported(self) -> None:
        pool = self.model_runner.req_to_token_pool
        if not isinstance(pool, HybridReqToTokenPool):
            raise AssertionError(
                "model reports a mamba-ish config but req_to_token_pool is "
                f"{type(pool).__name__}; --partial 1 cannot guarantee clean "
                "recurrent state. Use --partial 0."
            )

    def _fake_process_batch_result(self, batch: ScheduleBatch) -> None:
        """Feed ``process_batch_result`` a fabricated extend result.

        Skipping ``run_batch`` skips five things it would have done, all of
        which we redo here:

        1. deferred mamba clear/COW (normally executed inside the forward),
        2. ``forward_iter`` / ``launch_ts`` (dereferenced by
           ``_record_step_counters``),
        3. ``resolve_forward_inputs`` (consumes the pinned prefill staging),
        4. producing ``next_token_ids`` — a device tensor, because the prefill
           result path calls ``.tolist()`` on it unconditionally,
        5. ``_relay_forward_payload`` — without it the next decode step gathers
           ``future_map.output_tokens_buf`` at slots that were never written and
           feeds garbage (or the -1 sentinel) into the embedding.

        ``extend_input_len_per_req`` / ``extend_logprob_start_len_per_req`` are
        left None, matching what ``run_batch`` does when
        ``batch.return_logprob`` is False.
        """
        self._apply_deferred_mamba_clear(batch)

        scheduler = self.scheduler
        scheduler.forward_ct += 1
        batch.forward_iter = scheduler.forward_ct
        batch.launch_ts = time.monotonic()

        resolve_forward_inputs(batch, scheduler.future_map)

        result = GenerationBatchResult(
            next_token_ids=torch.full(
                (len(batch.reqs),),
                FAKE_TOKEN_ID,
                dtype=torch.long,
                device=self.device,
            ),
            can_run_cuda_graph=False,
        )
        scheduler._relay_forward_payload(batch.req_pool_indices, result)
        batch.input_ids = None
        scheduler.process_batch_result(batch, result)

    def _apply_deferred_mamba_clear(self, batch: ScheduleBatch) -> None:
        """Mirror ``ModelRunner._maybe_execute_deferred_mamba_cow_and_clear``.

        ``prepare_for_extend`` consumes ``req.mamba_needs_clear`` and parks the
        slot ids on the batch; the actual zeroing happens inside the forward. A
        skipped forward would leave the previous request's recurrent state in
        the slot — silently wrong numbers, not a crash.
        """
        pool = self.model_runner.req_to_token_pool
        if not isinstance(pool, HybridReqToTokenPool):
            return
        if batch.mamba_clear_indices is not None and len(batch.mamba_clear_indices) > 0:
            # mamba_pool is a physical store: translate before zeroing.
            pool.mamba_pool.clear_slots(
                pool.translate_mamba_indices(batch.mamba_clear_indices)
            )
        if (
            batch.mamba_cow_src_indices is not None
            and len(batch.mamba_cow_src_indices) > 0
        ):
            raise AssertionError(
                "fake-KV hit the mamba copy-on-write path, which only happens on "
                "a prefix match. Radix cache must be disabled for --partial 1."
            )
        batch.mamba_clear_indices = None
        batch.mamba_cow_src_indices = None
        batch.mamba_cow_dst_indices = None

    def _registered_reqs(self) -> List:
        """Requests that are alive after fake-KV registration.

        An extend batch is only merged into ``running_batch`` by the *next*
        ``get_next_batch_to_run``, so the most recent extend still sits in
        ``last_batch``. Earlier extends have already been merged, and a request
        never appears in both, so the two lists are disjoint.
        """
        scheduler = self.scheduler
        reqs = list(scheduler.running_batch.reqs)
        last_batch = scheduler.last_batch
        if last_batch is not None and last_batch.forward_mode.is_extend():
            reqs += list(last_batch.reqs)
        return reqs

    def _assert_uniform_registration(self, expected_bs: int, seq_len: int) -> None:
        """Every request must sit at exactly the same KV length before the first
        measured decode step, otherwise the decode batch is ragged and the
        per-token number is not comparable across engines."""
        self.assert_scheduler_clean()
        running = self._registered_reqs()
        if len(running) != expected_bs:
            raise AssertionError(
                f"fake-KV registered {len(running)} requests, expected "
                f"{expected_bs}"
            )
        prompt_lens = {len(req.origin_input_ids) for req in running}
        if prompt_lens != {seq_len}:
            raise AssertionError(
                f"fake-KV prompt lengths are not uniform: {sorted(prompt_lens)}, "
                f"expected all == {seq_len}"
            )
        output_lens = {len(req.output_ids) for req in running}
        if output_lens != {1}:
            raise AssertionError(
                f"fake-KV requests hold {sorted(output_lens)} output tokens, "
                "expected exactly 1 (the fake token). Some request decoded "
                "during registration."
            )
        seq_lens = {req.seqlen for req in running}
        if len(seq_lens) != 1:
            raise AssertionError(
                f"fake-KV KV lengths diverged: {sorted(seq_lens)}. The decode "
                "batch would be ragged."
            )

    # ------------------------------------------------------------------
    # Profiling
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def torch_profile(self, output_dir: str, trace_name: str) -> Iterator[None]:
        """Raw ``torch.profiler`` window exporting a plain (ungzipped) chrome
        trace.

        We deliberately do not go through ``SchedulerProfilerManager``: it gzips
        the trace and issues a ``torch.distributed.barrier``, and the shared
        timeline analyzer (``vllm/patches/batch_decode_scheduler/
        perf_test_timeline.py``) reads plain JSON and parses the step count out
        of the file name.
        """
        os.makedirs(output_dir, exist_ok=True)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if is_cuda() or is_hip():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities) as prof:
            yield
        path = os.path.join(output_dir, trace_name + ".json")
        prof.export_chrome_trace(path)
        logger.info("Wrote trace %s", path)

    def profile_run(
        self,
        *,
        mode: str,
        batch_size: int,
        seq_len: int,
        num_steps: int,
        output_dir: str,
        skip_prefill_forward: bool,
        trace_suffix: str = "",
    ) -> str:
        """One warmup round, then one profiled round.

        For decode modes the prefill happens outside the profiler window, so the
        trace holds exactly ``num_steps`` decode steps and per-step averages are
        exact.

        ``trace_suffix`` (e.g. ``_tp1``) keeps concurrent ranks from overwriting
        each other. It goes after ``steps{N}`` because the shared timeline
        analyzer recovers the step count with a ``re.search``, not a full match.
        """
        trace_name = (
            f"sglang_{mode}_bs{batch_size}_seq{seq_len}_steps{num_steps}{trace_suffix}"
        )
        self._profile_warmup(
            mode=mode,
            batch_size=batch_size,
            seq_len=seq_len,
            num_steps=num_steps,
            skip_prefill_forward=skip_prefill_forward,
        )

        if mode == PHASE_PREFILL:
            self.submit(batch_size, seq_len, max_new_tokens=1)
            with self.torch_profile(output_dir, trace_name):
                self.run_step_no_timing()
        else:
            if skip_prefill_forward:
                self.submit_decode_only(batch_size, seq_len, num_steps)
            else:
                self.submit(batch_size, seq_len, max_new_tokens=num_steps + 1)
                self.run_step_no_timing()
            with self.torch_profile(output_dir, trace_name):
                for _ in range(num_steps):
                    self.run_step_no_timing()
        self.drain()
        return os.path.join(output_dir, trace_name + ".json")

    def _profile_warmup(
        self,
        *,
        mode: str,
        batch_size: int,
        seq_len: int,
        num_steps: int,
        skip_prefill_forward: bool,
    ) -> None:
        if mode == PHASE_PREFILL:
            self.submit(batch_size, seq_len, max_new_tokens=1)
            self.run_step_no_timing()
        elif skip_prefill_forward:
            self.submit_decode_only(batch_size, seq_len, num_steps)
            for _ in range(min(2, num_steps)):
                self.run_step_no_timing()
        else:
            self.submit(batch_size, seq_len, max_new_tokens=num_steps + 1)
            self.run_step_no_timing()
            for _ in range(min(2, num_steps)):
                self.run_step_no_timing()
        self.drain()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.drain()
        with contextlib.suppress(Exception):
            self.scheduler.release_host_resources()
