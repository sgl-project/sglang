"""Measure replicated versus row-sharded GLM pooled-index selection on real GPUs.

Run with torchrun --standalone --nproc-per-node=8 and the same SGLang environment
as the server. This is an isolated operator experiment, not end-to-end serving.
It uses the production FP8 paged MQA, pooled Top-K and graph-safe collectives.
Query/key projections and cache writes are deliberately outside the experiment.
The cache stays replicated; each rank scores a disjoint set of request rows.
"""

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import statistics
import subprocess
from pathlib import Path

import deep_gemm
import torch
import torch.distributed as dist

from sglang.srt.distributed import parallel_state as ps
from sglang.srt.layers.attention.dsa.dsa_indexer import (
    _broadcast_indexer_topk_from_rank0,
)
from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
    build_kpool_topk_v2_plan,
    refresh_kpool_topk_v2_plan,
    topk_from_pooled_history_logits,
)
from sglang.test.test_utils import publish_build_topology


def make_inputs(rows, input_length):
    # Each request has distinct physical cache pages, as in the serving sweep.
    torch.manual_seed(43735)
    cols = input_length // 4
    pages_per_row = (cols + 63) // 64
    pages = rows * pages_per_row
    table = torch.arange(pages, dtype=torch.int32, device="cuda").view(rows, -1)
    lengths = torch.full((rows,), cols, dtype=torch.int32, device="cuda")
    # Include three unfinished-pool tail tokens in the 2051-column output.
    token_lengths = lengths * 4 + 3
    q = torch.randn(rows, 1, 32, 128, device="cuda", dtype=torch.bfloat16)
    q_scale = q.float().abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448
    q_fp8 = (q.float() / q_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    weights = torch.rand(rows, 32, device="cuda") * q_scale.view(rows, 32)
    cache = torch.empty(pages, 64 * 132, device="cuda", dtype=torch.uint8)
    # Bound temporary storage when testing long contexts and large batches.
    for lo in range(0, pages, 1024):
        hi = min(lo + 1024, pages)
        keys = torch.randn(hi - lo, 64, 128, device="cuda", dtype=torch.bfloat16)
        scale = keys.float().abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448
        fp8 = (keys.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        cache[lo:hi, : 64 * 128].copy_(fp8.view(torch.uint8).view(hi - lo, -1))
        cache[lo:hi, 64 * 128 :].copy_(
            scale.contiguous().view(torch.uint8).view(hi - lo, -1)
        )
    return q_fp8, cache.view(pages, 64, 1, 132), weights, table, lengths, token_lengths


def measure_graph(fn, group, repeats, replays):
    dist.barrier()
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with group.graph_capture() as capture:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture.stream):
            for _ in range(5):
                output = fn()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        dist.barrier()
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / (5 * replays))
    times = torch.tensor(samples, device="cuda")
    dist.all_reduce(times, op=dist.ReduceOp.MAX)
    samples = times.cpu().tolist()
    return {
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "samples_us": samples,
    }, output


def assert_selection(selected, scores, lengths, token_lengths, group):
    rows, cols = scores.shape
    assert selected.shape == (rows, 2051)
    # An owner may order tied groups differently. Check membership and the
    # score cutoff, rather than requiring an arbitrary Top-K output ordering.
    history = selected[:, :2048].reshape(rows, 512, 4)
    groups = history[:, :, 0] // 4
    expected_tokens = groups[:, :, None] * 4 + torch.arange(
        4, device="cuda", dtype=torch.int32
    )
    torch.testing.assert_close(history, expected_tokens, atol=0, rtol=0)
    ordered = groups.sort(-1).values
    assert (ordered[:, 1:] != ordered[:, :-1]).all().item()
    assert ((groups >= 0) & (groups < lengths[:, None])).all().item()
    threshold = scores.topk(512, dim=-1).values[:, -1]
    picked_scores = scores.gather(1, groups.long())
    assert (picked_scores.amin(-1) >= threshold - 1e-4).all().item()
    tails = lengths[:, None] * 4 + torch.arange(3, device="cuda", dtype=torch.int32)
    torch.testing.assert_close(selected[:, 2048:], tails, atol=0, rtol=0)
    assert (selected < token_lengths[:, None]).all().item()
    reference = selected.clone()
    group.broadcast(reference, src=0)
    torch.testing.assert_close(selected, reference, atol=0, rtol=0)


def check_edge_cases(group):
    """Check ragged ownership, empty owners, ties and changing captured inputs."""
    checked = []
    for rows in (1, 7, 9, 65):
        per = (rows + group.world_size - 1) // group.world_size
        lo = min(group.rank_in_group * per, rows)
        hi = min((group.rank_in_group + 1) * per, rows)
        cols = 2048
        scores = torch.arange(cols, dtype=torch.float32, device="cuda").repeat(rows, 1)
        lengths = (torch.arange(rows, dtype=torch.int32, device="cuda") * 271) % cols
        token_lengths = (
            lengths * 4 + torch.arange(rows, dtype=torch.int32, device="cuda") % 4
        )
        plan = build_kpool_topk_v2_plan(lengths[lo:hi]) if hi > lo else None

        def gather():
            if hi > lo:
                local = topk_from_pooled_history_logits(
                    scores[lo:hi],
                    lengths[lo:hi],
                    kpool=4,
                    topk=2048,
                    seq_lens=token_lengths[lo:hi],
                    topk_v2_plan=plan,
                    out_rows=per,
                )
            else:
                local = torch.full((per, 2051), -1, dtype=torch.int32, device="cuda")
            output = torch.empty(
                (per * group.world_size, 2051), dtype=torch.int32, device="cuda"
            )
            group.all_gather_into_tensor(output, local)
            return output[:rows]

        def validate(output):
            reference = output.clone()
            group.broadcast(reference, src=0)
            torch.testing.assert_close(output, reference, atol=0, rtol=0)
            for selected, row_scores, length, token_length in zip(
                output.cpu().tolist(),
                scores.cpu().tolist(),
                lengths.cpu().tolist(),
                token_lengths.cpu().tolist(),
            ):
                valid = [x for x in selected if x >= 0]
                history_count = min(length, 512) * 4
                assert len(valid) == history_count + token_length % 4
                assert len(valid) == len(set(valid))
                assert all(0 <= x < token_length for x in valid)
                history, tail = valid[:history_count], valid[history_count:]
                assert tail == list(range(length * 4, token_length))
                groups = {x // 4 for x in history}
                assert len(groups) == min(length, 512)
                assert set(history) == {
                    g * 4 + offset for g in groups for offset in range(4)
                }
                if groups:
                    cutoff = sorted(row_scores[:length], reverse=True)[
                        min(length, 512) - 1
                    ]
                    assert min(row_scores[g] for g in groups) >= cutoff

        validate(gather())
        with group.graph_capture() as capture:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture.stream):
                captured = gather()
        for phase in (1, 2):
            # The second replay has all tied scores. Both update lengths and
            # tails to ensure capture does not accidentally keep stale metadata.
            if phase == 1:
                scores.neg_()
            else:
                scores.zero_()
            lengths.add_(97).remainder_(cols)
            token_lengths.copy_(
                lengths * 4
                + (torch.arange(rows, dtype=torch.int32, device="cuda") + phase) % 4
            )
            if plan is not None:
                refresh_kpool_topk_v2_plan(plan=plan, pool_lens=lengths[lo:hi])
            graph.replay()
            validate(captured)
        checked.append(
            {"rows": rows, "eager": True, "changed_graph_replays": 2, "ties": True}
        )
    return checked


def run_case(rows, input_length, group, args):
    rank, world = group.rank_in_group, group.world_size
    q, cache, weights, table, lengths, token_lengths = make_inputs(rows, input_length)
    per = (rows + world - 1) // world
    lo, hi = min(rank * per, rows), min((rank + 1) * per, rows)
    full_context = lengths.view(-1, 1)
    local_context = full_context[lo:hi]
    full_schedule = deep_gemm.get_paged_mqa_logits_metadata(
        full_context, 64, deep_gemm.get_num_sms()
    )
    local_schedule = (
        deep_gemm.get_paged_mqa_logits_metadata(
            local_context, 64, deep_gemm.get_num_sms()
        )
        if hi > lo
        else None
    )
    full_plan = build_kpool_topk_v2_plan(lengths)
    local_plan = build_kpool_topk_v2_plan(lengths[lo:hi]) if hi > lo else None

    def logits(start, end, schedule):
        return deep_gemm.fp8_paged_mqa_logits(
            q[start:end],
            cache,
            weights[start:end],
            full_context[start:end],
            table[start:end],
            schedule,
            table.shape[1] * 64,
            clean_logits=False,
        )

    def select(scores, start, end, plan, out_rows=None):
        return topk_from_pooled_history_logits(
            scores,
            lengths[start:end],
            kpool=4,
            topk=2048,
            seq_lens=token_lengths[start:end],
            topk_v2_plan=plan,
            out_rows=out_rows,
        )

    full_scores = logits(0, rows, full_schedule)
    local_scores = logits(lo, hi, local_schedule) if hi > lo else None
    if hi > lo:
        torch.testing.assert_close(
            local_scores, full_scores[lo:hi], atol=1e-4, rtol=1e-5
        )

    def replicated():
        result = select(logits(0, rows, full_schedule), 0, rows, full_plan)
        return _broadcast_indexer_topk_from_rank0(result, force=True)

    def sharded(with_planning=False, topk_only=False):
        if hi > lo:
            schedule = (
                deep_gemm.get_paged_mqa_logits_metadata(
                    local_context, 64, deep_gemm.get_num_sms()
                )
                if with_planning
                else local_schedule
            )
            plan = (
                build_kpool_topk_v2_plan(lengths[lo:hi])
                if with_planning
                else local_plan
            )
            scores = local_scores if topk_only else logits(lo, hi, schedule)
            local = select(scores, lo, hi, plan, out_rows=per)
        else:
            local = torch.full((per, 2051), -1, dtype=torch.int32, device="cuda")
        output = torch.empty((per * world, 2051), dtype=torch.int32, device="cuda")
        group.all_gather_into_tensor(output, local)
        return output[:rows]

    expanded = select(full_scores, 0, rows, full_plan)
    compact = torch.empty((rows, 512), dtype=torch.int32, device="cuda")
    methods = {
        "replicated_logits": lambda: logits(0, rows, full_schedule),
        "replicated_topk": lambda: select(full_scores, 0, rows, full_plan),
        "expanded_broadcast_only": lambda: _broadcast_indexer_topk_from_rank0(
            expanded, force=True
        ),
        "compact_broadcast_only": lambda: _broadcast_indexer_topk_from_rank0(
            compact, force=True
        ),
        "replicated_index_broadcast": replicated,
        "row_sharded_index_allgather": sharded,
        "row_sharded_with_planning": lambda: sharded(with_planning=True),
        "row_sharded_topk_allgather": lambda: sharded(topk_only=True),
    }
    result = {
        "rows": rows,
        "input_tokens": input_length,
        "pool_tokens": input_length // 4,
        "world_size": world,
        "row_stride": per,
        "methods": {},
    }
    for name, fn in methods.items():
        if rank == 0:
            print(
                json.dumps(
                    {
                        "phase": "measure",
                        "rows": rows,
                        "input_tokens": input_length,
                        "method": name,
                    }
                ),
                flush=True,
            )
        timed, output = measure_graph(fn, group, args.repeats, args.replays)
        result["methods"][name] = timed
        if name in (
            "replicated_index_broadcast",
            "row_sharded_index_allgather",
            "row_sharded_with_planning",
            "row_sharded_topk_allgather",
        ):
            assert_selection(output, full_scores, lengths, token_lengths, group)
    result["selection_checks_passed"] = True
    base = result["methods"]["replicated_index_broadcast"]["median_us"]
    for name in ("row_sharded_index_allgather", "row_sharded_with_planning"):
        result["methods"][name]["speedup_vs_replicated"] = (
            base / result["methods"][name]["median_us"]
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", type=int, default=[16, 64, 256, 512])
    parser.add_argument(
        "--input-lengths", nargs="+", type=int, default=[8192, 32768, 131072, 262144]
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    rank, world, local_rank = (
        int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    )
    torch.cuda.set_device(local_rank)
    ps.set_custom_all_reduce(False)
    ps.init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
        timeout=600,
    )
    publish_build_topology(tp_size=world, world_rank=rank)
    ps.initialize_model_parallel()
    group = ps.get_tp_group()
    report = {
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "running",
        "world_size": world,
        "rank_max_timing": True,
        "scope": "Isolated graph-replay paged logits + pooled Top-K + collective. Synthetic full-length rows. Excludes projections, cache writes, attention, scheduling and model execution; speedups are not serving throughput gains.",
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("torch", "triton", "flashinfer-python")
        },
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,uuid", "--format=csv"], text=True
        ),
        "cases": [],
    }

    def save():
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            tmp = args.output.with_suffix(".tmp")
            tmp.write_text(json.dumps(report, indent=2) + "\n")
            tmp.replace(args.output)

    try:
        save()
        report["edge_checks"] = check_edge_cases(group)
        save()
        for length in [] if args.check_only else args.input_lengths:
            assert length >= 8192 and length % 256 == 0
            for rows in args.rows:
                assert rows > 0
                report["current"] = {"rows": rows, "input_tokens": length}
                save()
                result = run_case(rows, length, group, args)
                report["cases"].append(result)
                save()
                torch.cuda.empty_cache()
        report.update(
            status="complete",
            finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
        report.pop("current", None)
        save()
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        save()
        raise
    finally:
        ps.destroy_model_parallel()
        ps.destroy_distributed_environment()


if __name__ == "__main__":
    main()
