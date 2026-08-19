"""CLI for the SGLang GPU E2E latency bench.

Runs a ``batch_size x seq_len`` grid through an in-process ``Scheduler`` and
aggregates it the way RTP-LLM ``grid_perf_test`` does (sort the per-round
values, drop min and max, average the rest). Column names and CSV layout are
byte-identical to the vLLM ``batch_decode_scheduler`` patch so the three
engines can be concatenated into one table.

    python3 -m sglang.srt.patches.batch_decode_scheduler.perf_test_runner \\
        --model-path /path/to/model --partial 1 \\
        --batch-sizes 1,4 --seq-lens 128,1024 --num-decode-steps 20

``--partial`` matches RTP-LLM: 0 = PD, 1 = decode-only (fake KV), 2 =
prefill-only. See ``sglang/srt/patches/RUN_GUIDE.md``.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import multiprocessing as mp
import os
import signal
import sys
import tempfile
import time
import traceback
from typing import Callable, Dict, List, Optional, Tuple

import msgspec
import numpy as np

from sglang.srt.entrypoints.engine import (
    Engine,
    _calculate_rank_ranges,
    _compute_parallelism_ranks,
    _set_envs_and_config,
    _wait_for_scheduler_ready,
)
from sglang.srt.layers.dp_attention import compute_dp_attention_world_info
from sglang.srt.managers.scheduler import configure_scheduler_process
from sglang.srt.patches.batch_decode_scheduler.perf_test_harness import (
    FAKE_KV_TOKEN_BUDGET_CAP,
    PHASE_DECODE,
    PHASE_PREFILL,
    BenchHarness,
)
from sglang.srt.plugins import load_plugins
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import maybe_reindex_device_id, numa_utils
from sglang.srt.utils.network import bind_port, get_free_port
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = logging.getLogger(__name__)

# Grace period for a rank process to exit after its siblings are done. A wedged
# NCCL collective never returns on its own, so the parent force-kills past this.
_RANK_JOIN_GRACE_S = 60

# The global warmup submits one real prefill regardless of --partial, so its
# length is capped: kernel warmup does not need the grid's (possibly huge)
# seq_len, and per-cell --num-warmup-iters already warm the exact shapes.
_GLOBAL_WARMUP_MAX_SEQ_LEN = 512


class BenchResult(msgspec.Struct):
    """One grid cell, aggregated RTP-LLM style.

    Every ``*_ms`` is a trimmed mean over rounds (sort the per-round values,
    drop min and max, average the rest — see ``_trimmed_mean``), matching
    RTP-LLM ``batch_perf_impl.run``'s ``measurements[1:-1]``. Every
    ``*_mean_ms`` is the raw mean over the same rounds, for reference.
    The sample unit is the round (one per ``num_iters``), not the step.

    Field names and order match the vLLM patch's ``BenchResult`` exactly.
    """

    mode: str
    batch_size: int
    seq_len: int
    num_rounds: int = 0
    # per-round total wall time, begin -> last token (~ RTP cost_time)
    cost_ms: float = 0.0
    cost_mean_ms: float = 0.0
    # per-round prefill / first-token wall time (~ RTP first_token_cost_time)
    prefill_ms: float = 0.0
    prefill_mean_ms: float = 0.0
    # per-round (cost - prefill) / decode_steps (~ RTP decode_time_per_token)
    per_token_ms: float = 0.0
    per_token_mean_ms: float = 0.0

    @property
    def primary_ms(self) -> float:
        """Headline metric: decode_time_per_token for decode, else prefill."""
        return self.per_token_ms if self.mode == PHASE_DECODE else self.prefill_ms


class StepDiag(msgspec.Struct):
    """``--per-step-timing`` diagnostic for one grid cell.

    Produced by a dedicated extra round that synchronizes around every step.
    Those synchronizations drain the GPU between steps and inflate the total,
    so these numbers are reported in their own section and never enter the
    headline table or the CSV.
    """

    mode: str
    batch_size: int
    seq_len: int
    num_steps: int
    prefill_step_ms: float = 0.0
    decode_step_mean_ms: float = 0.0
    decode_step_max_ms: float = 0.0
    sum_step_ms: float = 0.0


class RankPayload(msgspec.Struct):
    """What one rank process writes to the shared result dir."""

    tp_rank: int
    dp_rank: int
    status: str
    elapsed_s: float = 0.0
    results: List[BenchResult] = []
    diags: List[StepDiag] = []
    error: str = ""


def _trimmed_mean(samples: List[float]) -> float:
    """RTP-LLM aggregation: sort, drop min and max, average the rest.

    Falls back to plain mean when fewer than 3 samples (nothing to trim).
    Mirrors batch_perf_impl.run: measurements.sort(); measurements[1:-1].
    """
    if not samples:
        return 0.0
    if len(samples) < 3:
        return float(np.mean(samples))
    return float(np.mean(sorted(samples)[1:-1]))


def _agg(samples: List[float]) -> Tuple[float, float]:
    """Return (trimmed_mean, raw_mean) over rounds."""
    if not samples:
        return 0.0, 0.0
    return _trimmed_mean(samples), float(np.mean(samples))


# ----------------------------------------------------------------------
# Bench loops
# ----------------------------------------------------------------------


def run_prefill_bench(
    harness: BenchHarness,
    batch_size: int,
    seq_len: int,
    num_iters: int,
    num_warmup_iters: int = 1,
) -> BenchResult:
    cost_times: List[float] = []
    for i in range(num_warmup_iters + num_iters):
        harness.submit(batch_size, seq_len, max_new_tokens=1)
        harness.mark_batch_start()
        stat = harness.run_step_no_timing()
        harness.assert_phase(stat, PHASE_PREFILL)
        harness.assert_batch(stat, batch_size)
        harness.assert_prefill_tokens(stat, batch_size, seq_len)
        # Wall time of the single prefill step (~ RTP first_token_cost_time).
        cost_ms = harness.mark_batch_end()
        if i >= num_warmup_iters:
            cost_times.append(cost_ms)
        harness.drain()

    cost_trim, cost_mean = _agg(cost_times)
    return BenchResult(
        mode=PHASE_PREFILL,
        batch_size=batch_size,
        seq_len=seq_len,
        num_rounds=len(cost_times),
        cost_ms=cost_trim,
        cost_mean_ms=cost_mean,
        prefill_ms=cost_trim,  # prefill == total for a single step
        prefill_mean_ms=cost_mean,
    )


def run_decode_bench(
    harness: BenchHarness,
    batch_size: int,
    kv_len: int,
    num_decode_steps: int,
    num_iters: int,
    num_warmup_iters: int = 1,
    skip_prefill_forward: bool = False,
) -> BenchResult:
    cost_times: List[float] = []
    prefill_times: List[float] = []
    per_token_times: List[float] = []
    for i in range(num_warmup_iters + num_iters):
        prefill_ms = 0.0
        if skip_prefill_forward:
            harness.submit_decode_only(batch_size, kv_len, num_decode_steps)
            harness.mark_batch_start()
        else:
            harness.submit(
                batch_size, kv_len, max_new_tokens=num_decode_steps + 1, ignore_eos=True
            )
            harness.mark_batch_start()
            setup_stat = harness.run_step_no_timing()
            harness.assert_phase(setup_stat, PHASE_PREFILL)
            harness.assert_batch(setup_stat, batch_size)
            harness.assert_prefill_tokens(setup_stat, batch_size, kv_len)
            # Wall time to first token (~ RTP first_token_cost_time), on the
            # same batch-start clock as cost below.
            prefill_ms = harness.mark_lap()

        for _ in range(num_decode_steps):
            stat = harness.run_step_no_timing()
            harness.assert_phase(stat, PHASE_DECODE)
            harness.assert_batch(stat, batch_size)

        # Whole round, begin -> last token (~ RTP cost_time).
        cost_ms = harness.mark_batch_end()
        if i >= num_warmup_iters:
            cost_times.append(cost_ms)
            prefill_times.append(prefill_ms)
            # Per-round decode_time_per_token = (cost - prefill) / steps,
            # matching RTP ResponseInfo.decode_time_per_token.
            per_token_times.append((cost_ms - prefill_ms) / num_decode_steps)
        harness.drain()

    cost_trim, cost_mean = _agg(cost_times)
    prefill_trim, prefill_mean = _agg(prefill_times)
    pt_trim, pt_mean = _agg(per_token_times)
    return BenchResult(
        mode=PHASE_DECODE,
        batch_size=batch_size,
        seq_len=kv_len,
        num_rounds=len(cost_times),
        cost_ms=cost_trim,
        cost_mean_ms=cost_mean,
        prefill_ms=prefill_trim,
        prefill_mean_ms=prefill_mean,
        per_token_ms=pt_trim,
        per_token_mean_ms=pt_mean,
    )


def run_step_diag(
    harness: BenchHarness,
    *,
    mode: str,
    batch_size: int,
    seq_len: int,
    num_decode_steps: int,
    skip_prefill_forward: bool,
) -> StepDiag:
    """One extra round driven through ``run_step_timed``.

    Diagnostic only: the per-step synchronizations serialize CPU and GPU, so
    ``sum_step_ms`` is systematically larger than the round's ``cost_ms``.
    """
    prefill_step_ms = 0.0
    decode_step_ms: List[float] = []

    if mode == PHASE_PREFILL:
        harness.submit(batch_size, seq_len, max_new_tokens=1)
        prefill_step_ms = harness.run_step_timed().forward_ms
        steps = 1
    else:
        if skip_prefill_forward:
            harness.submit_decode_only(batch_size, seq_len, num_decode_steps)
        else:
            harness.submit(
                batch_size,
                seq_len,
                max_new_tokens=num_decode_steps + 1,
                ignore_eos=True,
            )
            prefill_step_ms = harness.run_step_timed().forward_ms
        for _ in range(num_decode_steps):
            decode_step_ms.append(harness.run_step_timed().forward_ms)
        steps = num_decode_steps
    harness.drain()

    return StepDiag(
        mode=mode,
        batch_size=batch_size,
        seq_len=seq_len,
        num_steps=steps,
        prefill_step_ms=prefill_step_ms,
        decode_step_mean_ms=float(np.mean(decode_step_ms)) if decode_step_ms else 0.0,
        decode_step_max_ms=max(decode_step_ms) if decode_step_ms else 0.0,
        sum_step_ms=prefill_step_ms + sum(decode_step_ms),
    )


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------


def print_table(results: List[BenchResult]) -> None:
    prefill = [r for r in results if r.mode == PHASE_PREFILL]
    decode = [r for r in results if r.mode == PHASE_DECODE]

    if prefill:
        print(
            "=== Prefill (trimmed mean over rounds, "
            "~ RTP-LLM first_token_cost_time) ==="
        )
        header = (
            f"{'Mode':<8} {'BS':>4} {'SeqLen':>7} "
            f"{'prefill(ms)':>12} {'mean(ms)':>9} {'rounds':>7}"
        )
        print(header)
        print("-" * len(header))
        for r in prefill:
            print(
                f"{r.mode:<8} {r.batch_size:>4} {r.seq_len:>7} "
                f"{r.prefill_ms:>12.2f} {r.prefill_mean_ms:>9.2f} "
                f"{r.num_rounds:>7}"
            )

    if decode:
        if prefill:
            print()
        print("=== Decode (trimmed mean over rounds, ~ RTP-LLM grid_perf_test) ===")
        header = (
            f"{'Mode':<8} {'BS':>4} {'SeqLen':>7} "
            f"{'cost(ms)':>9} {'prefill(ms)':>12} "
            f"{'per_token(ms)':>14} {'mean(ms)':>9} {'rounds':>7}"
        )
        print(header)
        print("-" * len(header))
        for r in decode:
            print(
                f"{r.mode:<8} {r.batch_size:>4} {r.seq_len:>7} "
                f"{r.cost_ms:>9.2f} {r.prefill_ms:>12.2f} "
                f"{r.per_token_ms:>14.2f} {r.per_token_mean_ms:>9.2f} "
                f"{r.num_rounds:>7}"
            )


def print_step_diag_table(diags: List[StepDiag]) -> None:
    if not diags:
        return
    print()
    print("=== Per-step diagnostics (--per-step-timing) ===")
    print(
        "WARNING: every step is bracketed by a device synchronize, which drains "
        "the GPU\n         between steps. sum_step(ms) is NOT comparable with "
        "cost(ms) above and\n         must not be summed into it."
    )
    header = (
        f"{'Mode':<8} {'BS':>4} {'SeqLen':>7} {'steps':>6} "
        f"{'prefill_step(ms)':>17} {'decode_step(ms)':>16} "
        f"{'decode_max(ms)':>15} {'sum_step(ms)':>13}"
    )
    print(header)
    print("-" * len(header))
    for d in diags:
        print(
            f"{d.mode:<8} {d.batch_size:>4} {d.seq_len:>7} {d.num_steps:>6} "
            f"{d.prefill_step_ms:>17.2f} {d.decode_step_mean_ms:>16.2f} "
            f"{d.decode_step_max_ms:>15.2f} {d.sum_step_ms:>13.2f}"
        )


def write_csv(results: List[BenchResult], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "mode",
                "batch_size",
                "seq_len",
                "num_rounds",
                "cost_ms",
                "cost_mean_ms",
                "prefill_ms",
                "prefill_mean_ms",
                "per_token_ms",
                "per_token_mean_ms",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.mode,
                    r.batch_size,
                    r.seq_len,
                    r.num_rounds,
                    f"{r.cost_ms:.2f}",
                    f"{r.cost_mean_ms:.2f}",
                    f"{r.prefill_ms:.2f}",
                    f"{r.prefill_mean_ms:.2f}",
                    f"{r.per_token_ms:.2f}",
                    f"{r.per_token_mean_ms:.2f}",
                ]
            )


# ----------------------------------------------------------------------
# ServerArgs construction
# ----------------------------------------------------------------------


def resolve_parallelism(
    *,
    cli_tp_size: int,
    cli_dp_size: int,
    cli_ep_size: int,
    enable_dp_attention: bool,
) -> Dict[str, int]:
    """Translate vLLM/RTP-style CLI parallelism into SGLang ``ServerArgs``.

    vLLM and RTP-LLM read ``--tp-size`` as "TP inside one DP replica", so the
    world size is ``tp x dp``. SGLang has two different conventions:

    * ``--enable-dp-attention``: ``server_args.tp_size`` IS the world size and
      DP is carved out of it (``attn_tp_size = tp_size // dp_size // cp_size``),
      so we must pass ``tp x dp``.
    * plain DP: each replica is an independent TP group of ``tp_size`` GPUs and
      the launcher repeats it ``dp_size`` times, so ``tp_size`` passes through.

    There is no ``attention_tp_size`` server arg — it is derived. Returns the
    ServerArgs-level values.
    """
    world_tp = cli_tp_size * cli_dp_size if enable_dp_attention else cli_tp_size
    ep_size = cli_ep_size if cli_ep_size > 0 else 1
    # _compute_parallelism_ranks divides by (tp_size // moe_dp_size // ep_size);
    # we never set --moe-dp-size, so moe_dp_size is 1 and the check is on tp_size.
    if world_tp % ep_size != 0:
        raise ValueError(
            f"--ep-size {ep_size} does not divide the SGLang world tp_size "
            f"{world_tp} (= --tp-size {cli_tp_size} x --dp-size {cli_dp_size} "
            f"under dp-attention). The moe_ep_rank math would silently skew."
        )
    return {"tp_size": world_tp, "dp_size": cli_dp_size, "ep_size": ep_size}


def _prefill_token_budget(
    *, max_batch_size: int, max_seq_len: int, fake_kv: bool
) -> int:
    """Prefill token budget: one step must swallow the whole batch.

    Under fake KV no prefill forward runs, but the budget still sizes the
    runner's staging buffers and the activation reservation, so clamp it the
    way the vLLM patch does.
    """
    budget = max(max_batch_size * max_seq_len, 16384)
    if fake_kv:
        budget = max(
            min(budget, FAKE_KV_TOKEN_BUDGET_CAP),
            max_seq_len + max_batch_size + 64,
        )
    return budget


def build_server_args(
    args: argparse.Namespace, batch_sizes: List[int], seq_lens: List[int]
) -> ServerArgs:
    """Build the ServerArgs the bench pins. Deviating from these breaks either
    exact batching or the timing semantics, so they are not CLI-overridable."""
    max_batch_size = max(batch_sizes)
    max_seq_len = max(seq_lens)
    parallelism = resolve_parallelism(
        cli_tp_size=args.tp_size,
        cli_dp_size=args.dp_size,
        cli_ep_size=args.ep_size,
        enable_dp_attention=args.enable_dp_attention,
    )

    kwargs = dict(
        model_path=args.model_path,
        tokenizer_path=args.model_path,
        # We feed token ids directly; no tokenizer/detokenizer in the loop.
        skip_tokenizer_init=True,
        dtype=args.dtype,
        load_format=args.load_format,
        trust_remote_code=args.trust_remote_code,
        random_seed=args.random_seed,
        log_level=args.log_level,
        base_gpu_id=args.base_gpu_id,
        enable_metrics=False,
        # Exact batching: one step must take the whole batch, unchunked, with
        # no prefix reuse across the synthetic prompts. Under dp-attention
        # SGLang treats max_running_requests as a global budget and each rank
        # gets max_running_requests // attn_dp_size (pool_configurator.py:529),
        # so scale it up to keep --batch-sizes per-DP-rank.
        max_running_requests=max_batch_size
        * (args.dp_size if args.enable_dp_attention else 1),
        chunked_prefill_size=-1,
        max_prefill_tokens=_prefill_token_budget(
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            fake_kv=args.partial == 1,
        ),
        disable_radix_cache=True,
        # event_loop_normal semantics: one step = one schedule + one forward +
        # one result process, so phase assertions and the three sync points are
        # meaningful.
        disable_overlap_schedule=not args.enable_overlap,
        disable_cuda_graph=args.disable_cuda_graph,
        pp_size=args.pp_size,
        enable_dp_attention=args.enable_dp_attention,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
        **parallelism,
    )
    for name in (
        "mem_fraction_static",
        "max_total_tokens",
        "context_length",
        "attention_backend",
    ):
        value = getattr(args, name)
        if value is not None:
            kwargs[name] = value

    return ServerArgs(**kwargs)


# ----------------------------------------------------------------------
# Grid driving (runs inside every rank process)
# ----------------------------------------------------------------------


def _global_warmup(*, harness: BenchHarness, min_seq_len: int, prefix: str) -> None:
    """Warmup layer 2 of 3: one tiny prefill + one decode before the grid.

    Layer 1 is engine init (memory pool, attention backend plan, CUDA graph
    capture — all inside ``Scheduler.__init__``); layer 3 is the per-grid-cell
    ``--num-warmup-iters`` rounds inside the bench loops.
    """
    # "tiny" must stay tiny even when the grid holds a single long-seq cell:
    # this prefill runs for real in every --partial mode, so an uncapped
    # min(seq_lens) (e.g. 65536) can OOM a fake-KV run whose measured path
    # never executes a prefill forward at all.
    warmup_seq_len = min(min_seq_len, _GLOBAL_WARMUP_MAX_SEQ_LEN)
    print(f"{prefix}Global warmup (bs=1, seq_len={warmup_seq_len}) ...", flush=True)
    harness.submit(1, warmup_seq_len, max_new_tokens=2, ignore_eos=True)
    harness.run_step_no_timing()
    harness.run_step_no_timing()
    harness.drain()
    print(f"{prefix}Global warmup done", flush=True)


def run_bench_grid(
    *,
    harness: BenchHarness,
    args: argparse.Namespace,
    batch_sizes: List[int],
    seq_lens: List[int],
    trace_suffix: str = "",
    rank_tag: str = "",
) -> Tuple[List[BenchResult], List[StepDiag]]:
    """Run the whole grid on this rank. bs-major row order, same as vLLM."""
    # RTP-aligned partial: 2 = prefill only; 1 = fake-KV decode; 0 = PD.
    mode = PHASE_PREFILL if args.partial == 2 else PHASE_DECODE
    skip_prefill = args.partial == 1
    # Every rank shares one stdout, so untagged progress lines cannot be
    # attributed back to a rank.
    prefix = f"[{rank_tag}] " if rank_tag else ""

    _global_warmup(harness=harness, min_seq_len=min(seq_lens), prefix=prefix)

    results: List[BenchResult] = []
    diags: List[StepDiag] = []
    for bs in batch_sizes:
        for seq_len in seq_lens:
            label = f"{mode} bs={bs} seq_len={seq_len}"
            print(f"{prefix}Running {label} ...", flush=True)

            if mode == PHASE_PREFILL:
                r = run_prefill_bench(
                    harness, bs, seq_len, args.num_iters, args.num_warmup_iters
                )
            else:
                r = run_decode_bench(
                    harness,
                    bs,
                    seq_len,
                    args.num_decode_steps,
                    args.num_iters,
                    args.num_warmup_iters,
                    skip_prefill_forward=skip_prefill,
                )
            results.append(r)
            metric = "per_token" if mode == PHASE_DECODE else "prefill"
            print(
                f"{prefix}  {label}: "
                f"{metric}={r.primary_ms:.2f}ms cost={r.cost_ms:.2f}ms",
                flush=True,
            )

            if args.per_step_timing:
                diags.append(
                    run_step_diag(
                        harness,
                        mode=mode,
                        batch_size=bs,
                        seq_len=seq_len,
                        num_decode_steps=args.num_decode_steps,
                        skip_prefill_forward=skip_prefill,
                    )
                )

            if args.profile:
                steps = args.num_decode_steps if mode == PHASE_DECODE else 1
                trace = harness.profile_run(
                    mode=mode,
                    batch_size=bs,
                    seq_len=seq_len,
                    num_steps=steps,
                    output_dir=args.profile_dir,
                    skip_prefill_forward=skip_prefill,
                    trace_suffix=trace_suffix,
                )
                print(f"{prefix}  Trace: {trace}", flush=True)

    return results, diags


# ----------------------------------------------------------------------
# Rank worker
# ----------------------------------------------------------------------


def _rank_payload_path(result_dir: str, dp_rank: int, tp_rank: int) -> str:
    # Plain DP gives every replica its own tp_rank 0, so the dp rank has to be
    # part of the name or the replicas overwrite each other.
    return os.path.join(result_dir, f"rank_dp{dp_rank}_tp{tp_rank}.json")


def run_bench_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    attn_cp_rank: int,
    moe_dp_rank: int,
    moe_ep_rank: int,
    pp_rank: int,
    dp_rank: Optional[int],
    pipe_writer,
    display_tp_rank: Optional[int] = None,
    display_dp_rank: Optional[int] = None,
    display_moe_ep_rank: Optional[int] = None,
    *,
    bench_args: argparse.Namespace,
    result_dir: str,
    batch_sizes: List[int],
    seq_lens: List[int],
) -> None:
    """Drop-in replacement for ``run_scheduler_process``.

    Same positional signature, so it can be handed to either the official
    TP-only launcher or our DP fork; it just runs the bench grid instead of
    entering the event loop.
    """
    load_plugins()
    dp_rank = configure_scheduler_process(
        server_args,
        gpu_id,
        tp_rank,
        attn_cp_rank,
        moe_dp_rank,
        moe_ep_rank,
        pp_rank,
        dp_rank,
        display_tp_rank=display_tp_rank,
        display_dp_rank=display_dp_rank,
        display_moe_ep_rank=display_moe_ep_rank,
    )

    harness = None
    t_start = time.time()
    try:
        harness = BenchHarness(
            server_args=server_args,
            port_args=port_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            moe_ep_rank=moe_ep_rank,
            pp_rank=pp_rank,
            attn_cp_rank=attn_cp_rank,
            moe_dp_rank=moe_dp_rank,
            dp_rank=dp_rank,
        )
    except BaseException:
        # The parent's _wait_for_scheduler_ready keys off this dict; anything
        # other than "ready" makes it raise instead of hanging on the pipe.
        pipe_writer.send({"status": "error"})
        _write_rank_payload(
            result_dir=result_dir,
            payload=RankPayload(
                tp_rank=tp_rank,
                dp_rank=dp_rank if dp_rank is not None else 0,
                status="error",
                error=traceback.format_exc(),
            ),
        )
        raise

    pipe_writer.send(harness.get_init_info())

    suffix = ""
    tag_parts = []
    if server_args.dp_size > 1:
        suffix += f"_dp{dp_rank}"
        tag_parts.append(f"DP{dp_rank}")
    if server_args.tp_size > 1:
        suffix += f"_tp{tp_rank}"
        tag_parts.append(f"TP{tp_rank}")
    try:
        with harness:
            results, diags = run_bench_grid(
                harness=harness,
                args=bench_args,
                batch_sizes=batch_sizes,
                seq_lens=seq_lens,
                trace_suffix=suffix,
                rank_tag=" ".join(tag_parts),
            )
        payload = RankPayload(
            tp_rank=tp_rank,
            dp_rank=dp_rank if dp_rank is not None else 0,
            status="ok",
            elapsed_s=time.time() - t_start,
            results=results,
            diags=diags,
        )
    except BaseException:
        payload = RankPayload(
            tp_rank=tp_rank,
            dp_rank=dp_rank if dp_rank is not None else 0,
            status="error",
            elapsed_s=time.time() - t_start,
            error=traceback.format_exc(),
        )
        _write_rank_payload(result_dir=result_dir, payload=payload)
        harness.close()
        raise

    _write_rank_payload(result_dir=result_dir, payload=payload)
    harness.close()


def _write_rank_payload(*, result_dir: str, payload: RankPayload) -> None:
    path = _rank_payload_path(result_dir, payload.dp_rank, payload.tp_rank)
    with open(path + ".tmp", "wb") as f:
        f.write(msgspec.json.encode(payload))
    os.replace(path + ".tmp", path)


def _read_rank_payloads(result_dir: str) -> List[RankPayload]:
    payloads = []
    for path in sorted(glob.glob(os.path.join(result_dir, "rank_dp*_tp*.json"))):
        with open(path, "rb") as f:
            payloads.append(msgspec.json.decode(f.read(), type=RankPayload))
    return sorted(payloads, key=lambda p: (p.dp_rank, p.tp_rank))


# ----------------------------------------------------------------------
# Process launching
# ----------------------------------------------------------------------


def _spawn_tp_group(
    *,
    server_args: ServerArgs,
    port_args: PortArgs,
    base_gpu_id: int,
    dp_rank: Optional[int],
    worker_ports: Optional[List[int]],
    target: Callable,
) -> Tuple[List, List]:
    """One TP(xPP) group of bench workers.

    Structurally the body of ``DataParallelController.launch_tensor_parallel_group``
    minus the ZMQ wiring, the status bookkeeping and the blocking recv. All rank
    math is delegated to the official helpers so a SGLang upgrade that changes
    the hierarchy does not silently skew our ranks.
    """
    procs, readers = [], []
    memory_saver_adapter = TorchMemorySaverAdapter.create(
        enable=server_args.enable_memory_saver
    )
    pp_rank_range, tp_rank_range, pp_size_per_node, tp_size_per_node = (
        _calculate_rank_ranges(
            server_args.nnodes,
            server_args.pp_size,
            server_args.tp_size,
            server_args.node_rank,
        )
    )

    for pp_rank in pp_rank_range:
        for tp_rank in tp_rank_range:
            rank_port_args = port_args
            rank_dp_rank = dp_rank
            if server_args.enable_dp_attention:
                # DP attention shards differently: dp_rank follows from tp_rank.
                _, _, rank_dp_rank, _ = compute_dp_attention_world_info(
                    server_args.enable_dp_attention,
                    tp_rank,
                    server_args.tp_size,
                    server_args.dp_size,
                    server_args.attn_cp_size,
                )
                rank_port_args = PortArgs.init_new(
                    server_args, rank_dp_rank, worker_ports
                )
                # DP attention reuses the tensor parallel group, so every dp
                # rank must share one nccl port and instance id.
                rank_port_args.nccl_port = port_args.nccl_port
                rank_port_args.instance_id = port_args.instance_id

            gpu_id = (
                server_args.base_gpu_id
                + base_gpu_id
                + ((pp_rank % pp_size_per_node) * tp_size_per_node)
                + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
            )
            attn_cp_rank, moe_dp_rank, moe_ep_rank = _compute_parallelism_ranks(
                server_args, tp_rank
            )
            reader, writer = mp.Pipe(duplex=False)
            with maybe_reindex_device_id(gpu_id) as gpu_id:
                proc = mp.Process(
                    target=target,
                    args=(
                        server_args,
                        rank_port_args,
                        gpu_id,
                        tp_rank,
                        attn_cp_rank,
                        moe_dp_rank,
                        moe_ep_rank,
                        pp_rank,
                        rank_dp_rank,
                        writer,
                    ),
                )
                # SGLANG_NUMA_BIND_V2 defaults on and materially affects the
                # numbers, so keep the production binding.
                with (
                    memory_saver_adapter.configure_subprocess(),
                    numa_utils.configure_subprocess(server_args, gpu_id),
                ):
                    proc.start()
            procs.append(proc)
            readers.append(reader)
    return procs, readers


def _fork_dp_workers(
    *, server_args: ServerArgs, port_args: PortArgs, target: Callable
) -> Tuple[List, List]:
    """Launch DP ranks without a ``DataParallelController``.

    The real controller enters ``event_loop()`` (a ``while True`` ZMQ recv)
    right after spawning, and never returns — the bench never sends it a
    request, so it would hang and nobody would reap the workers.
    """
    if server_args.enable_dp_attention:
        # Nothing binds these: the schedulers only *connect* to
        # scheduler_input_ipc_name, and the bench never sends them a request.
        worker_ports = [get_free_port() for _ in range(server_args.dp_size)]
        return _spawn_tp_group(
            server_args=server_args,
            port_args=port_args,
            base_gpu_id=0,
            dp_rank=None,
            worker_ports=worker_ports,
            target=target,
        )

    # Plain DP: dp_size independent replicas of a tp_size group. Each needs its
    # own nccl port; hold them until every replica has one so the next
    # PortArgs.init_new cannot hand out the same port.
    rank_port_args, held = [], []
    for _ in range(server_args.dp_size):
        rank_args = PortArgs.init_new(server_args)
        rank_args.instance_id = port_args.instance_id
        held.append(bind_port(rank_args.nccl_port))
        rank_port_args.append(rank_args)
    for sock in held:
        sock.close()

    procs, readers = [], []
    base_gpu_id = 0
    for dp_rank, rank_args in enumerate(rank_port_args):
        group_procs, group_readers = _spawn_tp_group(
            server_args=server_args,
            port_args=rank_args,
            base_gpu_id=base_gpu_id,
            dp_rank=dp_rank,
            worker_ports=None,
            target=target,
        )
        procs.extend(group_procs)
        readers.extend(group_readers)
        base_gpu_id += (
            server_args.tp_size * server_args.pp_size * server_args.gpu_id_step
        )
    return procs, readers


def launch_bench_processes(
    *, server_args: ServerArgs, port_args: PortArgs, target: Callable
) -> Tuple[List, Callable[[], None]]:
    """Returns (processes, wait_for_ready)."""
    # Mirrors Engine._launch_scheduler_processes's own use_dp_controller test:
    # anything that would route through DataParallelController must not, because
    # the controller blocks in event_loop() forever and never reaps the workers.
    if server_args.dp_size == 1 and server_args.ep_join_mode != "scale":
        # The official TP-only launcher takes an injectable worker function and
        # does exactly what we need, so use it verbatim.
        init_result, procs = Engine._launch_scheduler_processes(
            server_args, port_args, target
        )
        return procs, init_result.wait_for_ready

    procs, readers = _fork_dp_workers(
        server_args=server_args, port_args=port_args, target=target
    )
    return procs, lambda: _wait_for_scheduler_ready(readers, procs)


def wait_for_workers(procs: List, timeout_s: float) -> List[str]:
    """Join every rank, tearing the rest down if one dies. Returns failures.

    A rank that dies mid-collective leaves its siblings blocked in NCCL
    forever, so a dead rank triggers an immediate teardown rather than a
    ``timeout_s`` wait.
    """
    deadline = time.time() + timeout_s
    failures: List[str] = []
    while True:
        alive = [p for p in procs if p.is_alive()]
        crashed = [p for p in procs if not p.is_alive() and p.exitcode not in (0, None)]
        if crashed and alive:
            failures.append(
                f"rank process {crashed[0].pid} exited with {crashed[0].exitcode} "
                f"while {len(alive)} sibling(s) were still running; tearing down"
            )
            break
        if not alive:
            break
        if time.time() > deadline:
            failures.append(f"timed out after {timeout_s:.0f}s waiting for ranks")
            break
        time.sleep(0.5)

    for proc in procs:
        if proc.is_alive():
            proc.terminate()
    for proc in procs:
        proc.join(timeout=_RANK_JOIN_GRACE_S)
        if proc.is_alive():
            os.kill(proc.pid, signal.SIGKILL)
            proc.join(timeout=_RANK_JOIN_GRACE_S)

    for i, proc in enumerate(procs):
        if proc.exitcode != 0:
            failures.append(f"rank process #{i} exit code is {proc.exitcode}")
    return failures


def failure_reasons(
    *,
    payloads: List[RankPayload],
    expected_ranks: int,
    process_failures: List[str],
) -> List[str]:
    """Reasons this run must not be reported as successful."""
    reasons = [
        f"dp{p.dp_rank} tp{p.tp_rank} failed:\n{p.error.rstrip()}"
        for p in payloads
        if p.status != "ok"
    ]
    if len(payloads) < expected_ranks:
        reported = sorted((p.dp_rank, p.tp_rank) for p in payloads)
        reasons.append(
            f"only {len(payloads)} of {expected_ranks} rank(s) reported results "
            f"(got {reported})"
        )
    reasons.extend(process_failures)
    return reasons


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SGLang GPU E2E bench (prefill-only / decode-only / PD)"
    )
    parser.add_argument("--model-path", required=True, help="Local model path")
    parser.add_argument(
        "--partial",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="RTP-LLM-aligned mode switch. RTP itself only accepts 1/2 "
        "(choices=[1, 2]); 0 is this patch's extension (same as the vLLM "
        "patch). "
        "0 = PD (real prefill + decode in one round, both metrics; NOT an RTP "
        "value), "
        "1 = decode only (fake KV: blocks really allocated and req_to_token "
        "really written, but zero content and no prefill forward — matches RTP "
        "setIsContextStream(false)), "
        "2 = prefill only.",
    )
    parser.add_argument(
        "--batch-sizes",
        default="1,4,16",
        help="Comma-separated batch sizes. With --dp-size N this is the "
        "PER-DP-RANK batch size (total = batch_size x dp_size), matching "
        "RTP-LLM GridRunner semantics.",
    )
    parser.add_argument(
        "--seq-lens",
        default="128,512,1024",
        help="Comma-separated sequence lengths (prefill=seq_len, decode=kv_len)",
    )
    parser.add_argument("--num-iters", type=int, default=5)
    parser.add_argument("--num-decode-steps", type=int, default=20)
    parser.add_argument(
        "--num-warmup-iters",
        type=int,
        default=1,
        help="Warmup iterations per grid point (results discarded)",
    )

    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallel size PER DP REPLICA (vLLM/RTP semantics). Under "
        "--enable-dp-attention this is multiplied by --dp-size to get "
        "server_args.tp_size, which SGLang treats as the world size.",
    )
    parser.add_argument("--pp-size", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument(
        "--dp-size",
        type=int,
        default=1,
        help="Data parallel size (spawns N replicas; each runs batch_size "
        "requests — RTP-aligned per-rank semantics, total = batch_size x N)",
    )
    parser.add_argument(
        "--ep-size", type=int, default=1, help="Expert parallel size for MoE models"
    )
    parser.add_argument(
        "--enable-dp-attention",
        action="store_true",
        help="DP attention + cross-DP EP (SGLang carves DP out of tp_size)",
    )
    parser.add_argument(
        "--disable-custom-all-reduce",
        action="store_true",
        help="Fall back to NCCL all-reduce. Needed on hosts where the JIT "
        "custom all-reduce kernel cannot be built (it needs a C++20 host "
        "compiler; GCC 10 fails on std::bit_cast).",
    )

    parser.add_argument(
        "--per-step-timing",
        action="store_true",
        help="Run one extra diagnostic round per grid point with a device "
        "synchronize around every step. Reported in its own section; never "
        "enters the headline table or the CSV.",
    )
    parser.add_argument(
        "--enable-overlap",
        action="store_true",
        help="Use the overlap scheduler (SGLang production default). CPU/GPU "
        "overlap makes per-step numbers meaningless; only cost_ms is reported.",
    )
    parser.add_argument(
        "--disable-cuda-graph", action="store_true", help="~ vLLM --enforce-eager"
    )
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=None,
        help="~ vLLM --gpu-memory-utilization",
    )
    parser.add_argument(
        "--max-total-tokens", type=int, default=None, help="KV pool size in tokens"
    )
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--load-format", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--attention-backend", default=None)
    parser.add_argument("--base-gpu-id", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--log-level", default="info")

    parser.add_argument(
        "--profile",
        action="store_true",
        help="Also run a dedicated torch.profiler pass per grid point. Each "
        "rank dumps its own chrome trace; prefill stays outside the window in "
        "decode modes so the trace holds exactly --num-decode-steps steps.",
    )
    parser.add_argument("--profile-dir", default="./profile_output")
    parser.add_argument(
        "--rank-timeout",
        type=float,
        default=7200.0,
        help="Seconds to wait for the rank processes before tearing them down",
    )
    parser.add_argument("--output", default=None, help="CSV output path")
    return parser.parse_args(argv)


def _report(
    *, payloads: List[RankPayload], args: argparse.Namespace, dp_size: int
) -> None:
    """Print the table for the representative rank of each DP replica."""
    by_dp: Dict[int, RankPayload] = {}
    for payload in sorted(payloads, key=lambda p: p.tp_rank):
        by_dp.setdefault(payload.dp_rank, payload)

    head = by_dp[min(by_dp)]
    if len(by_dp) > 1:
        for dp_rank, payload in sorted(by_dp.items())[1:]:
            for base, peer in zip(head.results, payload.results):
                diff_pct = (
                    abs(peer.primary_ms - base.primary_ms)
                    / max(base.primary_ms, 1e-6)
                    * 100
                )
                if diff_pct > 10:
                    print(
                        f"WARNING: dp rank {dp_rank} primary={peer.primary_ms:.2f}ms "
                        f"vs dp rank {head.dp_rank} primary={base.primary_ms:.2f}ms "
                        f"({diff_pct:.0f}% diff) for bs={base.batch_size} "
                        f"seq={base.seq_len}"
                    )

    print()
    if dp_size > 1:
        print(
            f"=== DP={dp_size} (showing dp rank {head.dp_rank}; batch_size is "
            f"PER-DP-RANK, RTP-aligned — total = bs x {dp_size}) ==="
        )
    if args.enable_overlap:
        print(
            "WARNING: --enable-overlap is on. CPU/GPU overlap makes the "
            "prefill/decode split approximate; only cost(ms) is trustworthy."
        )
    print_table(head.results)
    print_step_diag_table(head.diags)

    if args.output:
        write_csv(head.results, args.output)
        print(f"\nCSV written to {args.output}")


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    seq_lens = [int(x) for x in args.seq_lens.split(",")]

    server_args = build_server_args(args, batch_sizes, seq_lens)
    print(
        f"CLI tp={args.tp_size} dp={args.dp_size} ep={args.ep_size} "
        f"pp={args.pp_size} dp_attention={args.enable_dp_attention} "
        f"-> ServerArgs tp_size={server_args.tp_size} "
        f"dp_size={server_args.dp_size} ep_size={server_args.ep_size} "
        f"(attn_tp_size="
        f"{server_args.tp_size // (server_args.dp_size if args.enable_dp_attention else 1) // server_args.attn_cp_size}"
        f")",
        flush=True,
    )

    _set_envs_and_config(server_args)
    port_args = PortArgs.init_new(server_args)
    os.makedirs(args.profile_dir, exist_ok=True) if args.profile else None

    with tempfile.TemporaryDirectory(prefix="sglang-bench-") as result_dir:
        from functools import partial

        target = partial(
            run_bench_process,
            bench_args=args,
            result_dir=result_dir,
            batch_sizes=batch_sizes,
            seq_lens=seq_lens,
        )
        t0 = time.time()
        procs, wait_for_ready = launch_bench_processes(
            server_args=server_args, port_args=port_args, target=target
        )
        try:
            wait_for_ready()
            print(f"All {len(procs)} rank(s) ready in {time.time() - t0:.1f}s")
        except BaseException:
            for proc in procs:
                if proc.is_alive():
                    proc.kill()
            raise

        process_failures = wait_for_workers(procs, args.rank_timeout)
        payloads = _read_rank_payloads(result_dir)

    reasons = failure_reasons(
        payloads=payloads,
        expected_ranks=len(procs),
        process_failures=process_failures,
    )
    if reasons:
        print("ERROR: the benchmark did not complete cleanly:", file=sys.stderr)
        for reason in reasons:
            print(f"  - {reason}", file=sys.stderr)
        sys.exit(1)

    _report(payloads=payloads, args=args, dp_size=server_args.dp_size)


if __name__ == "__main__":
    main()
