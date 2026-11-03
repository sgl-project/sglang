"""Compare unified allocator implementations on recorded operation traces.

The oracle is another implementation's output, so this is a verification
script whose result a PR reports, not a unit test:

    # on the base checkout
    python test/manual/mem_cache/unified_trace_compare.py record --out base.jsonl
    # on the candidate checkout
    python test/manual/mem_cache/unified_trace_compare.py compare --trace base.jsonl
    # two implementations in one checkout
    python test/manual/mem_cache/unified_trace_compare.py pair --a builder --b <name>

Each run is one configuration (chain, page size, lazy compaction) driven by a
seed or by a scenario file from unified_trace_scenarios/. `compare` and `pair`
print the first divergence per run with the operations before it, and exit
non-zero on any divergence.
"""

import argparse
import collections
import json
import sys
from pathlib import Path

from sglang.test.unified_allocator_trace import (
    CHAINS,
    FORMAT_VERSION,
    TraceConfig,
    record_run,
    replay_run,
)

_SCENARIOS = Path(__file__).with_name("unified_trace_scenarios")


def _seeds(spec: str):
    lo, _, hi = spec.partition("-")
    return range(int(lo), int(hi or lo) + 1)


def _configs(args):
    for chain in args.chains.split(","):
        assert chain in CHAINS, f"unknown chain {chain!r}; choose from {CHAINS}"
        for ps in (int(x) for x in args.page_sizes.split(",")):
            for lazy in (bool(int(x)) for x in args.lazy.split(",")):
                yield TraceConfig(chain=chain, page_size=ps, lazy=lazy)


def _scenarios(directory: str):
    if not directory:
        return []
    out = []
    for path in sorted(Path(directory).glob("*.json")):
        sc = json.loads(path.read_text())
        config = TraceConfig(
            chain=sc["chain"], page_size=sc["page_size"], lazy=sc["lazy"]
        )
        out.append((sc["name"], config, sc["ops"]))
    return out


def _runs(args):
    """(label, config, seed, ops) for every run the arguments select."""
    configs = tuple(_configs(args))
    for config in configs:
        for seed in _seeds(args.seeds):
            yield f"{config.key()}/seed{seed}", config, seed, None
    selected = {config.key() for config in configs}
    for name, config, ops in _scenarios(args.scenarios):
        if config.key() in selected:
            yield f"{config.key()}/{name}", config, None, ops


def _report(label, divergence) -> None:
    print(f"DIVERGENCE {label} at step {divergence.step}: {divergence.op}")
    print(f"  differs: {', '.join(divergence.keys)}")
    for op in divergence.prefix:
        print(f"  after: {op}")


def cmd_record(args) -> int:
    coverage = collections.Counter()
    ops = 0
    with open(args.out, "w") as f:
        header = {
            "format_version": FORMAT_VERSION,
            "adapter": args.adapter,
            "full": args.full,
        }
        f.write(json.dumps(header) + "\n")
        for label, config, seed, scenario_ops in _runs(args):
            records = record_run(
                config,
                seed=seed,
                ops=scenario_ops,
                steps=args.steps,
                adapter_name=args.adapter,
                full=args.full,
                coverage=coverage,
            )
            ops += len(records)
            run = {
                "label": label,
                "config": {
                    "chain": config.chain,
                    "page_size": config.page_size,
                    "lazy": config.lazy,
                },
                "records": records,
            }
            f.write(json.dumps(run) + "\n")
    print(f"recorded {ops} operations to {args.out}")
    for key in sorted(coverage):
        print(f"  coverage {key}: {coverage[key]}")
    return 0


def cmd_compare(args) -> int:
    lines = Path(args.trace).read_text().splitlines()
    header = json.loads(lines[0])
    assert header["format_version"] == FORMAT_VERSION, header
    runs = diverged = ops = 0
    for line in lines[1:]:
        run = json.loads(line)
        config = TraceConfig(**run["config"])
        divergence = replay_run(
            config,
            run["records"],
            adapter_name=args.adapter,
            full=header["full"],
        )
        runs += 1
        ops += len(run["records"])
        if divergence is not None:
            diverged += 1
            _report(run["label"], divergence)
    print(
        f"compared {runs} runs, {ops} operations ({header['adapter']} -> "
        f"{args.adapter}): {diverged} diverged"
    )
    return 1 if diverged else 0


def cmd_pair(args) -> int:
    runs = diverged = ops = 0
    for label, config, seed, scenario_ops in _runs(args):
        records = record_run(
            config,
            seed=seed,
            ops=scenario_ops,
            steps=args.steps,
            adapter_name=args.a,
            full=True,
        )
        divergence = replay_run(config, records, adapter_name=args.b, full=True)
        runs += 1
        ops += len(records)
        if divergence is not None:
            diverged += 1
            _report(label, divergence)
    print(
        f"paired {runs} runs, {ops} operations ({args.a} vs {args.b}): {diverged} diverged"
    )
    return 1 if diverged else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("record", "pair"):
        p = sub.add_parser(name)
        p.add_argument("--chains", default=",".join(CHAINS))
        p.add_argument("--page-sizes", default="1,16")
        p.add_argument("--lazy", default="0,1")
        p.add_argument("--seeds", default="0-15")
        p.add_argument("--steps", type=int, default=200)
        p.add_argument("--scenarios", default=str(_SCENARIOS))
    record = sub.choices["record"]
    record.add_argument("--out", required=True)
    record.add_argument("--adapter", default="builder")
    record.add_argument(
        "--full", action="store_true", help="store observables in full, not digests"
    )
    pair = sub.choices["pair"]
    pair.add_argument("--a", required=True)
    pair.add_argument("--b", required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("--trace", required=True)
    compare.add_argument("--adapter", default="builder")
    args = parser.parse_args()
    return {"record": cmd_record, "compare": cmd_compare, "pair": cmd_pair}[args.cmd](
        args
    )


if __name__ == "__main__":
    sys.exit(main())
