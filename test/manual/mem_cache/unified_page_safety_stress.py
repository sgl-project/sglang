"""Randomized page-safety stress of the unified KV pool (manual, opt-in).

Runs seeded random operation sequences against a unified chain on the CPU
device model, on two simulated ranks with different completion timing, and
checks every operation (see `sglang.test.unified_page_safety_harness`). A
failing seed is shrunk to a minimal operation sequence.

Usage:
    python test/manual/mem_cache/unified_page_safety_stress.py \\
        --seeds 0:200 --ops 400 --jobs 32 --minimize --out /tmp/stress
    python test/manual/mem_cache/unified_page_safety_stress.py --replay case.json
    python test/manual/mem_cache/unified_page_safety_stress.py --device cuda \\
        --seeds 0:8 --ops 200
"""

import argparse
import collections
import dataclasses
import json
import multiprocessing
import os
import sys
import time


def _parse_seeds(text):
    if ":" in text:
        a, b = text.split(":")
        return list(range(int(a), int(b)))
    return [int(x) for x in text.split(",")]


def _op_json(op):
    from sglang.test.unified_page_safety_harness import op_to_json

    return op_to_json(op)


def _one(job):
    seed, n_ops, minimize_failures, device = job
    from sglang.test.unified_page_safety_harness import minimize, run_seed

    t0 = time.time()
    if device == "cuda":
        from sglang.test.unified_page_safety_harness import run_seed_cuda

        result = run_seed_cuda(seed, n_ops)
    else:
        result = run_seed(seed, n_ops)
    out = {
        "seed": seed,
        "cfg": dataclasses.asdict(result.cfg),
        "ops_run": len(result.ops),
        "counts": result.counts,
        "stats": result.stats,
        "max_dead_fences": result.max_dead_fences,
        "step_reads": [[list(k), n] for k, n in result.step_reads.items()],
        "seconds": time.time() - t0,
        "failure": None,
    }
    if result.failure is not None:
        out["failure"] = {
            "check": result.failure.check,
            "message": str(result.failure),
            "at": result.failed_at,
        }
        if minimize_failures and device == "cpu":
            from sglang.test.unified_page_safety_harness import signature

            sig = signature(result.failure)
            small = minimize(result.cfg, result.ops, sig)
            out["failure"]["signature"] = sig
            from sglang.test.unified_page_safety_harness import run_ops

            again = run_ops(result.cfg, small)
            out["failure"]["minimal_ops"] = [_op_json(op) for op in small]
            from sglang.test.unified_page_safety_harness import case_to_json

            out["failure"]["case"] = case_to_json(result.cfg, small)
            out["failure"]["minimal_message"] = (
                str(again.failure) if again.failure is not None else None
            )
        out["failure"]["ops"] = [_op_json(op) for op in result.ops]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0:20")
    parser.add_argument("--ops", type=int, default=400)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--minimize", action="store_true")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--out", default=None)
    parser.add_argument("--replay", default=None, help="a case JSON to replay")
    args = parser.parse_args()

    if args.replay:
        from sglang.test.unified_page_safety_harness import replay_case

        with open(args.replay) as f:
            failure = replay_case(json.load(f))
        print(failure)
        return 1 if failure is not None else 0

    seeds = _parse_seeds(args.seeds)
    jobs = [(s, args.ops, args.minimize, args.device) for s in seeds]
    t0 = time.time()
    if args.jobs > 1:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(args.jobs) as pool:
            results = list(pool.imap_unordered(_one, jobs))
    else:
        results = [_one(j) for j in jobs]
    results.sort(key=lambda r: r["seed"])
    total_ops = sum(r["ops_run"] for r in results)
    counts = collections.Counter()
    for r in results:
        counts.update(r["counts"])
    failures = [r for r in results if r["failure"]]
    by_check = collections.Counter(r["failure"]["check"] for r in failures)
    print(
        f"seeds={len(results)} ops={total_ops} failures={len(failures)} "
        f"{dict(by_check)} in {time.time() - t0:.0f}s"
    )
    for r in failures:
        print(f"seed {r['seed']}: {r['failure']['message'][:400]}")
        if "minimal_ops" in r["failure"]:
            print(f"  minimal ({len(r['failure']['minimal_ops'])} ops):")
            for op in r["failure"]["minimal_ops"]:
                print(f"    {op}")
            print(f"  -> {str(r['failure']['minimal_message'])[:400]}")
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        for r in failures:
            if "case" in r["failure"]:
                name = f"case-{r['failure']['check']}-seed{r['seed']}.json"
                with open(os.path.join(args.out, name), "w") as f:
                    json.dump(r["failure"]["case"], f, indent=1)
        with open(os.path.join(args.out, "results.json"), "w") as f:
            json.dump(
                {
                    "total_ops": total_ops,
                    "counts": dict(counts),
                    "by_check": dict(by_check),
                    "results": results,
                },
                f,
                indent=1,
            )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
