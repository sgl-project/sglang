"""Compare physical pooled-index layouts with real FP8 MQA on a single DCP node.

Example: torchrun --standalone --nproc-per-node=8 bench_kpool_page_sharding.py
--output /tmp/pooled-index-pages.json
"""

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import deep_gemm
import torch
import torch.distributed as dist
from bench_kpool_index_sharding import assert_selection, make_inputs, measure_graph

from sglang.kernels.ops.communication.kpool_candidates import exchange_kpool_candidates
from sglang.srt.distributed import parallel_state as ps
from sglang.srt.layers.attention.dsa.dsa_dcp import gather_kpool_topk
from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
    build_kpool_topk_v2_plan,
    kpool_assemble_softmax_rotate_write_cache,
    kpool_decode_update_and_maybe_write_cache,
    kpool_softmax_rotate_write_cache,
    topk_from_pooled_history_logits,
)
from sglang.srt.layers.attention.dsa.kpool_page_shard import KPoolPageShard, padded_rows
from sglang.test.test_utils import publish_build_topology


def check_writers(layout):
    """Every local byte and tail must match the corresponding reference slice."""
    torch.manual_seed(8123)
    device = "cuda"
    rows = 12
    page_count = rows * 16
    pool = SimpleNamespace(
        page_size=256,
        index_page_size=64,
        index_head_dim=128,
        index_kpool=4,
        tail_extra_slots=0,
    )
    compact = SimpleNamespace(**vars(pool), kpool_page_shard=layout)
    full = torch.full((page_count, 64 * 132), 13, dtype=torch.uint8, device=device)
    local = full[layout.rank :: layout.size].clone()
    keys = torch.randn(rows, 4, 128, device=device, dtype=torch.bfloat16)
    scores = torch.randn_like(keys)
    ape = torch.randn(4, 128, device=device)
    reqs = torch.arange(rows, device=device)
    locs = (reqs * 67 + 31).long()
    for state, buf in ((pool, full), (compact, local)):
        kpool_softmax_rotate_write_cache(state, buf, keys, scores, ape, locs)
    torch.testing.assert_close(local, full[layout.rank :: layout.size], atol=0, rtol=0)
    tails = torch.randn_like(keys)
    tail_scores = torch.randn_like(keys)
    for state, buf in ((pool, full), (compact, local)):
        kpool_assemble_softmax_rotate_write_cache(
            pool=state,
            buf=buf,
            chunk_k=keys.flatten(0, 1),
            chunk_score=scores.flatten(0, 1),
            tail_k=tails,
            tail_score=tail_scores,
            req_pool_idx=reqs.int(),
            n_from_tail=(reqs % 4).int(),
            chunk_src_start=(4 * reqs).int(),
            tail_logical_base=torch.zeros_like(reqs).int(),
            ape=ape,
            loc=locs + 1,
            round_scale=False,
        )
    torch.testing.assert_close(local, full[layout.rank :: layout.size], atol=0, rtol=0)
    positions = 256 * (reqs % 8) + (reqs % 3) * 4 + 3
    positions[1] -= 1
    table = torch.arange(rows * 64, device=device, dtype=torch.int32).view(rows, 64)
    outputs = []
    for state, buf in ((pool, full), (compact, local)):
        tail_k, tail_s = tails.clone(), tail_scores.clone()
        kpool_decode_update_and_maybe_write_cache(
            pool=state,
            buf=buf,
            tail_k=tail_k,
            tail_score=tail_s,
            key=keys[:, 0].contiguous(),
            slot_score=scores[:, 0].contiguous(),
            ape=ape,
            block_tables=table,
            req_pool_indices=reqs,
            positions=positions,
            seq_lens=positions + 1,
            out_cache_loc=1 + reqs * 4096 + positions,
        )
        outputs.append((tail_k, tail_s))
    torch.testing.assert_close(local, full[layout.rank :: layout.size], atol=0, rtol=0)
    for actual, expected in zip(outputs[1], outputs[0]):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def run_case(length, rows, group, layouts, args, report):
    world, rank = group.world_size, group.rank_in_group
    q, full_cache, weights, table, lens, seq_lens = make_inputs(rows, length)
    per = (rows + world - 1) // world

    def schedule(lengths):
        return deep_gemm.get_paged_mqa_logits_metadata(
            lengths.view(-1, 1).clamp(min=1), 64, deep_gemm.get_num_sms()
        )

    def score(keys, queries, gates, lengths, pages, plan):
        return deep_gemm.fp8_paged_mqa_logits(
            queries,
            keys,
            gates,
            lengths.view(-1, 1),
            pages,
            plan,
            pages.shape[1] * 64,
            clean_logits=False,
        )

    def select(logits, lengths, plan):
        return topk_from_pooled_history_logits(
            logits,
            lengths,
            kpool=4,
            topk=2048,
            seq_lens=lengths * 4 + 3,
            topk_v2_plan=plan,
        )

    full_sched = schedule(lens)
    full_plan = build_kpool_topk_v2_plan(lens)
    ref_scores = score(full_cache, q, weights, lens, table, full_sched)
    owner_start = rank * per
    owner_lens = padded_rows(lens, owner_start, per)
    owner_seq = padded_rows(seq_lens, owner_start, per)
    owner_plan = build_kpool_topk_v2_plan(owner_lens)

    def record(name, buffers, op):
        def cycle():
            for i, buf in enumerate(buffers):
                result = op(buf, i)
            return result

        for i, buf in enumerate(buffers):
            result = op(buf, i)
            assert_selection(result, ref_scores, lens, seq_lens, group)
        result, _ = measure_graph(cycle, group, args.repeats, args.replays)
        timings = {k: v / len(buffers) for k, v in result.items() if k != "samples_us"}
        timings["samples_us"] = [x / len(buffers) for x in result["samples_us"]]
        if rank == 0:
            report["methods"][name] = timings
            print(length, rows, name, timings["median_us"], flush=True)

    # Three independent caches exceed L2 across consecutive layers. The reference
    # remains available solely to check mass, tails and rank consensus outside timing.
    buffers = [full_cache, full_cache.clone(), full_cache.clone()]

    def layer_owner(buf, layer):
        root = layer % world
        if rank == root:
            result = select(
                score(buf, q, weights, lens, table, full_sched), lens, full_plan
            )
        else:
            result = torch.empty((rows, 2051), device=q.device, dtype=torch.int32)
        return group.broadcast(result, src=root)

    record("layer_owner_broadcast", buffers, layer_owner)
    local_q = padded_rows(q, owner_start, per)
    local_w = padded_rows(weights, owner_start, per)
    local_table = padded_rows(table, owner_start, per)
    local_sched = schedule(owner_lens)

    def replicated(buf, _):
        output = select(
            score(buf, local_q, local_w, owner_lens, local_table, local_sched),
            owner_lens,
            owner_plan,
        )
        return gather_kpool_topk(group, output, rows, rows)

    record("replicated_row_shard", buffers, replicated)
    del buffers
    for layout in layouts:
        stride, team, local_lens, pages = layout.decode_layout(table, lens, 64)
        team_q = padded_rows(q, team, layout.size * stride)
        team_w = padded_rows(weights, team, layout.size * stride)
        sched = schedule(local_lens)
        local_cache = full_cache[layout.rank :: layout.size].contiguous()
        buffers = [local_cache, local_cache.clone(), local_cache.clone()]
        for exchange in ("logits", "candidates"):
            method = replace(layout, exchange=exchange)

            def sharded(buf, _):
                part = score(buf, team_q, team_w, local_lens, pages, sched)[
                    :, : pages.shape[1] * 64
                ]
                if exchange == "candidates":
                    tokens = exchange_kpool_candidates(
                        part,
                        local_lens,
                        owner_lens,
                        owner_seq,
                        None,
                        None,
                        group=layout.group,
                        shards=layout.size,
                        rank=layout.rank,
                        page_size=64,
                    )
                else:
                    merged = method.transpose_scores(part, local_lens, stride, 64, 512)
                    tokens = select(merged, owner_lens, owner_plan)
                return gather_kpool_topk(group, tokens, rows, rows)

            # Compare every valid FP8 logit against the *same* queries and weights.
            partial = score(local_cache, team_q, team_w, local_lens, pages, sched)
            expected = (
                padded_rows(ref_scores, team, layout.size * stride)
                .view(layout.size * stride, -1, 64)[:, layout.rank :: layout.size]
                .reshape(layout.size * stride, -1)
            )
            torch.testing.assert_close(
                partial[:, : expected.shape[1]], expected, atol=0, rtol=0
            )
            record(f"pages{layout.size}_{exchange}", buffers, sharded)
        del buffers


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--points", nargs="+", default=["32768/256", "131072/64", "262144/64"]
    )
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--replays", type=int, default=10)
    args = p.parse_args()
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
        timeout=240,
    )
    publish_build_topology(tp_size=world, world_rank=rank)
    ps.initialize_model_parallel()
    group = ps.get_tp_group()
    report = {
        "world_size": world,
        "gpu": torch.cuda.get_device_name(),
        "cases": [],
        "status": "running",
    }

    def save():
        if rank == 0:
            tmp = args.output.with_suffix(".tmp")
            tmp.write_text(json.dumps(report, indent=2) + "\n")
            tmp.replace(args.output)

    try:
        layouts = []
        for size in (2, 4, 8):
            if world % size:
                continue
            start = rank // size * size
            pg = (
                group.device_group
                if size == world
                else dist.new_group(
                    ranks=group.ranks[start : start + size],
                    backend="nccl",
                    use_local_synchronization=True,
                )
            )
            layout = KPoolPageShard(size, rank % size, world, rank, pg, "logits")
            dist.all_reduce(torch.ones(1, device="cuda"), group=pg)
            check_writers(layout)
            layouts.append(layout)
        report["writers"] = "passed"
        save()
        for point in args.points:
            length, rows = map(int, point.split("/"))
            if rows % world or length % (256 * world):
                raise ValueError(
                    "Timing points must have whole DCP superpages and equal row partitions"
                )
            case = {"input_tokens": length, "concurrency": rows, "methods": {}}
            report["cases"].append(case)
            run_case(length, rows, group, layouts, args, case)
            save()
            torch.cuda.empty_cache()
        report["status"] = "complete"
        save()
    finally:
        ps.destroy_model_parallel()
        ps.destroy_distributed_environment()


if __name__ == "__main__":
    main()
