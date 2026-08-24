"""Microbenchmark the C++ unified-tree shape used by DSPARK long prefixes."""

import argparse
from array import array
from time import perf_counter_ns

import torch

from sglang.srt.mem_cache.cpp_radix_tree.radix_tree import RadixTreeCpp


def percentile(values: list[int], fraction: float) -> float:
    return sorted(values)[int((len(values) - 1) * fraction)] / 1e3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=1024)
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--window-size", type=int, default=4096)
    parser.add_argument("--shared-len", type=int, default=61440)
    parser.add_argument("--branch-len", type=int, default=5120)
    args = parser.parse_args()

    tree = RadixTreeCpp(False, None, args.page_size, 256, args.window_size)
    shared = array("q", (i % 32000 for i in range(args.shared_len)))
    match_ns: list[int] = []
    insert_ns: list[int] = []
    finish_ns: list[int] = []

    for branch in range(args.rounds):
        key = array("q", shared)
        key.extend([100000 + branch] * args.page_size)
        key.extend(
            (token + branch) % 32000
            for token in range(args.branch_len - args.page_size)
        )
        value = torch.arange(len(key), dtype=torch.int64)

        start = perf_counter_ns()
        _, _, full_hit = tree.match_prefix_swa_flat(key, len(key))
        match_ns.append(perf_counter_ns() - start)

        start = perf_counter_ns()
        prefix_len, node_id, _, rebuilds, _ = tree.writing_through_swa(
            key,
            value,
            min(full_hit, args.shared_len),
            len(key) - args.window_size,
            len(key),
        )
        insert_ns.append(perf_counter_ns() - start)

        start = perf_counter_ns()
        for rebuild_node_id, source in rebuilds:
            tree.set_swa_value(rebuild_node_id, source)
        overlap, locked_node, _, _, swa_uuid, skipped = (
            tree.match_node_range_and_lock_flat(
                node_id,
                min(full_hit, args.shared_len),
                prefix_len,
                True,
                True,
            )
        )
        if overlap is not None:
            value[min(full_hit, args.shared_len) : prefix_len].copy_(overlap)
        tree.unlock_ref_swa(locked_node, True, True, swa_uuid, skipped)
        finish_ns.append(perf_counter_ns() - start)
        assert prefix_len <= len(key)

    quarter = max(1, args.rounds // 4)
    print(
        {
            "rounds": args.rounds,
            "match_mean_us": sum(match_ns) / len(match_ns) / 1e3,
            "match_p50_us": percentile(match_ns, 0.50),
            "match_p99_us": percentile(match_ns, 0.99),
            "insert_mean_us": sum(insert_ns) / len(insert_ns) / 1e3,
            "insert_first_quarter_us": sum(insert_ns[:quarter]) / quarter / 1e3,
            "insert_last_quarter_us": sum(insert_ns[-quarter:]) / quarter / 1e3,
            "insert_p99_us": percentile(insert_ns, 0.99),
            "finish_mean_us": sum(finish_ns) / len(finish_ns) / 1e3,
            "stats": tree.debug_stats(),
        }
    )


if __name__ == "__main__":
    main()
