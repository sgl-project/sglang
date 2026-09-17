"""Search dense LoRA plans for explicit, TP-local linear-site workloads."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lora_tuning import compare, measure


@dataclass(frozen=True)
class Case:
    name: str
    in_features: int
    slices: tuple[int, ...]
    tokens: int
    rank: int
    pool_rank: int
    slots: int
    request_lengths: tuple[int, ...]
    request_slots: tuple[int, ...]
    phase: str = "decode"
    mode: str = "graph"
    kind: str = "linear"
    tp: int = 1

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        for key in ("slices", "request_lengths", "request_slots"):
            value[key] = tuple(value[key])
        case = cls(**value)
        dimensions = (
            case.in_features,
            case.tokens,
            case.rank,
            case.pool_rank,
            case.slots,
            case.tp,
            *case.slices,
            *case.request_lengths,
        )
        if not case.name or any(type(x) is not int or x <= 0 for x in dimensions):
            raise ValueError("names and integer dimensions must be nonempty/positive")
        if not case.slices or not case.request_lengths:
            raise ValueError("slices and request lengths cannot be empty")
        if case.rank > case.pool_rank:
            raise ValueError("rank exceeds pool_rank")
        if sum(case.request_lengths) != case.tokens:
            raise ValueError("request lengths must sum to tokens")
        if len(case.request_slots) != len(case.request_lengths) or any(
            type(x) is not int or not -1 <= x < case.slots for x in case.request_slots
        ):
            raise ValueError("one valid slot per request is required (-1 means base)")
        if case.phase not in ("decode", "prefill") or case.mode not in (
            "eager",
            "graph",
        ):
            raise ValueError("invalid phase or execution mode")
        if case.phase == "decode" and any(x != 1 for x in case.request_lengths):
            raise ValueError("decode has one token per request")
        if case.kind not in ("linear", "lm_head"):
            raise ValueError(
                "this fixture supports linear/lm_head, not windowed or absorbed sites"
            )
        return case


def plan_dict(plan):
    return json.loads(json.dumps(asdict(plan)))


def candidate_specs(incumbent):
    """Bounded family/block/split search plus one-axis tile changes."""
    result = [incumbent]
    pairs = [
        ("grouped", "grouped"),
        ("grouped", "per_row"),
        ("per_row", "per_row"),
        ("all_slots", "per_row"),
    ]
    for (a, b), overlap in itertools.product(pairs, ("none", "a", "ab_delta")):
        for block in (16, 32, 64, 128) if "grouped" in (a, b) else (16,):
            for split in (1, 4, 8) if a == "grouped" else (1,):
                spec = {
                    **incumbent,
                    "a_family": a,
                    "b_family": b,
                    "overlap": overlap,
                    "block_size": block,
                }
                spec["a_tiles"] = {
                    **incumbent["a_tiles"],
                    "SPLIT_K": split,
                    "SPLIT_MODE": "serial",
                }
                result.append(spec)
    for site, key, values in (
        ("a_tiles", "BLOCK_SIZE_N", (16, 32, 64, 128)),
        ("a_tiles", "BLOCK_SIZE_K", (32, 64, 128)),
        ("b_tiles", "BLOCK_SIZE_N", (32, 64, 128)),
        ("b_tiles", "BLOCK_SIZE_K", (32, 64, 128, 256)),
        ("a_tiles", "num_warps", (4, 8)),
        ("b_tiles", "num_warps", (4, 8)),
        ("a_tiles", "num_stages", (2, 3)),
        ("a_tiles", "SPLIT_MODE", ("serial", "planes")),
    ):
        if site == "a_tiles" and incumbent["a_family"] == "all_slots":
            continue
        if key == "SPLIT_MODE" and (
            incumbent["a_family"] != "grouped"
            or incumbent["a_tiles"].get("SPLIT_K", 1) == 1
        ):
            continue
        for value in values:
            result.append({**incumbent, site: {**incumbent[site], key: value}})
    return list({json.dumps(p, sort_keys=True): p for p in result}.values())


def execution_key(spec, pool_rank):
    """Exclude ignored knobs and rank-clamped B tiles from candidate identity."""
    effective = json.loads(json.dumps(spec))
    a, b = effective["a_tiles"], effective["b_tiles"]
    if effective["a_family"] == "all_slots":
        a.clear()
    elif effective["a_family"] != "grouped":
        for key in ("GROUP_SIZE_M", "SPLIT_K", "SPLIT_MODE"):
            a.pop(key, None)
    else:
        a["SPLIT_K"] = int(a.get("SPLIT_K", 1))
        if a["SPLIT_K"] == 1:
            a.pop("SPLIT_MODE", None)
        else:
            a.setdefault("SPLIT_MODE", "serial")
    if effective["b_family"] != "grouped":
        b.pop("GROUP_SIZE_M", None)
    if "grouped" not in (effective["a_family"], effective["b_family"]):
        effective.pop("block_size", None)
    if "BLOCK_SIZE_K" in b:
        b["BLOCK_SIZE_K"] = max(
            16, min(int(b["BLOCK_SIZE_K"]), 1 << (pool_rank - 1).bit_length())
        )
    return json.dumps(effective, sort_keys=True)


class Workload:
    def __init__(self, case, seed, device):
        import torch

        self.case, self.device = case, device
        generator = torch.Generator(device=device).manual_seed(seed)

        def rand(*shape, scale=0.1):
            return (
                torch.randn(shape, generator=generator, device=device) * scale
            ).bfloat16()

        self.x = rand(case.tokens, case.in_features, scale=1.0)
        self.weight = rand(sum(case.slices), case.in_features, scale=0.02)
        self.a = rand(case.slots, len(case.slices) * case.pool_rank, case.in_features)
        self.b = rand(case.slots, sum(case.slices), case.pool_rank)
        # Real pools pack A slices by active rank, while B retains pool-rank stride.
        self.b[:, :, case.rank :] = 0
        self.slots_cpu = [
            s
            for s, length in zip(case.request_slots, case.request_lengths)
            for _ in range(length)
        ]
        self.slots = torch.tensor(self.slots_cpu, dtype=torch.int32, device=device)
        self.ranks = torch.full(
            (case.slots,), case.rank, dtype=torch.int32, device=device
        )
        self.scales = torch.ones(case.slots, device=device)
        self.offsets = (0, *itertools.accumulate(case.slices))
        self.reference = self.oracle()

    def oracle(self):
        case = self.case
        reference = self.x.float() @ self.weight.float().T
        for slot in set(self.slots_cpu) - {-1}:
            mask = self.slots == slot
            for i, (lo, hi) in enumerate(zip(self.offsets, self.offsets[1:])):
                shrink = (
                    self.x[mask].float()
                    @ self.a[slot, i * case.rank : (i + 1) * case.rank].float().T
                )
                reference[mask, lo:hi] += (
                    shrink @ self.b[slot, lo:hi, : case.rank].float().T
                )
        return reference

    def runner(self, spec):
        import torch

        from sglang.srt.lora.dense.plan import DensePlan, _PlanSpecModel
        from sglang.srt.lora.dense.runner import DenseLoraRunner
        from sglang.srt.lora.utils import Phase
        from sglang.srt.lora.workspace import LoraWorkspace

        case = self.case
        plan = DensePlan(**_PlanSpecModel.model_validate(spec).model_dump())
        engine = DenseLoraRunner(
            LoraWorkspace(), max_loras=case.slots, device=self.device
        )
        base = torch.empty(
            (case.tokens, sum(case.slices)), device=self.device, dtype=self.x.dtype
        )

        def forward():
            # Include routing once per measured site; never reuse yesterday's route.
            engine.begin_batch(
                token_slots=self.slots,
                lora_ranks=self.ranks,
                scalings=self.scales,
                num_tokens=case.tokens,
                phase=Phase(case.phase),
                graph_mode=case.mode == "graph",
                is_prefill_graph=case.phase == "prefill",
            )
            return engine.apply(
                self.x,
                lambda: torch.mm(self.x, self.weight.T, out=base),
                plan,
                a=self.a,
                b=self.b,
                offsets=self.offsets,
            )

        return forward

    def check(self, forward):
        import torch

        base = (self.x @ self.weight.T).float()
        if any(slot >= 0 for slot in self.slots_cpu) and torch.allclose(
            base, self.reference, atol=2e-2, rtol=2e-2
        ):
            raise ValueError(
                "fixture cannot distinguish missing LoRA from the reference"
            )
        result = forward()
        graph = None
        if self.case.mode == "graph":
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                forward()
            torch.cuda.current_stream(self.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                result = forward()
            graph.replay()
        torch.cuda.synchronize(self.device)
        error = float((result.float() - self.reference).abs().max())
        if not torch.isfinite(result).all() or not torch.allclose(
            result.float(), self.reference, atol=2e-2, rtol=2e-2
        ):
            raise ValueError(f"numerical check failed: max_abs={error}")
        if graph is not None:
            saved = self.x.clone()
            try:
                self.x.mul_(-0.5).add_(0.125)
                changed_reference = self.oracle()
                graph.replay()
                torch.cuda.synchronize(self.device)
                if not torch.allclose(
                    result.float(), changed_reference, atol=2e-2, rtol=2e-2
                ):
                    raise ValueError("graph replay failed with changed inputs")
            finally:
                self.x.copy_(saved)
        return error


def paired(workload, specs, args):
    runs = {name: workload.runner(spec) for name, spec in specs.items()}
    errors = {name: workload.check(fn) for name, fn in runs.items()}
    samples = {name: [] for name in runs}
    for repeat in range(args.repeats):
        order = list(runs) if repeat % 2 == 0 else list(reversed(runs))
        for name in order:
            samples[name].extend(
                measure(
                    runs[name],
                    mode=workload.case.mode,
                    warmup=3,
                    repeats=1,
                    iterations=args.iterations,
                )
            )
    return samples, errors


def tune_case(case, args, device, architecture):
    import torch

    from sglang.srt.lora.dense.plan import (
        DenseLoraKind,
        DensePlan,
        DensePlanTable,
        _PlanSpecModel,
    )
    from sglang.srt.lora.utils import Phase

    incumbent = plan_dict(
        DensePlanTable(architecture, case.pool_rank).plan_for(
            DenseLoraKind(case.kind),
            Phase(case.phase),
            case.tokens,
            case.in_features,
            sum(case.slices),
        )
    )
    specs = (
        json.loads(args.candidates.read_text())
        if args.candidates
        else candidate_specs(incumbent)
    )
    if not isinstance(specs, list) or not specs:
        raise ValueError("candidates must be a nonempty list of DensePlan specs")
    random.Random(args.seed).shuffle(specs)
    workload = Workload(case, args.seed, device)
    baseline_error = workload.check(workload.runner(incumbent))
    records = []
    seen = {execution_key(incumbent, case.pool_rank)}

    def preserve_partial(exc, stage, spec):
        exc.tuning_partial = {
            "case": asdict(case),
            "incumbent": incumbent,
            "baseline_max_abs": baseline_error,
            "search": records,
            "validation": {"stage": stage, "spec": spec},
        }

    for spec in specs:
        if spec == incumbent:
            continue
        record = {"spec": spec}
        try:
            normalized = plan_dict(
                DensePlan(**_PlanSpecModel.model_validate(spec).model_dump())
            )
            key = execution_key(normalized, case.pool_rank)
            if key in seen:
                record.update(
                    status="NO_OP",
                    reason="same effective execution as incumbent or earlier candidate",
                )
                records.append(record)
                continue
            seen.add(key)
            samples, errors = paired(
                workload, {"baseline": incumbent, "candidate": spec}, args
            )
            record.update(
                samples_us=samples,
                max_abs=errors,
                comparison=compare(
                    samples["baseline"],
                    samples["candidate"],
                    min_gain=args.min_gain,
                    max_regression=args.max_regression,
                ),
            )
        except (RuntimeError, ValueError, KeyError) as exc:
            record.update(status="INVALID", error=f"{type(exc).__name__}: {exc}")
            if (
                "illegal memory access" in str(exc).lower()
                or "device-side assert" in str(exc).lower()
            ):
                records.append(record)
                preserve_partial(exc, "search", spec)
                raise
        records.append(record)
        torch.cuda.synchronize(device)
    if any(r.get("status") == "INVALID" for r in records) and not any(
        "comparison" in r for r in records
    ):
        exc = ValueError(
            "no valid candidate comparisons; all attempted candidates were invalid"
        )
        preserve_partial(exc, "search", None)
        raise exc
    wins = [r for r in records if r.get("comparison", {}).get("status") == "WIN"]
    selected, validation = incumbent, {"status": "RETAIN"}
    if wins:
        finalist = max(wins, key=lambda r: r["comparison"]["median_gain"])["spec"]
        # One locked finalist on fresh inputs. A failed gate does not try runners-up.
        try:
            heldout = Workload(case, args.seed + 1, device)
            samples, errors = paired(
                heldout, {"baseline": incumbent, "candidate": finalist}, args
            )
            validation = {
                **compare(
                    samples["baseline"],
                    samples["candidate"],
                    min_gain=args.min_gain,
                    max_regression=args.max_regression,
                ),
                "samples_us": samples,
                "max_abs": errors,
                "spec": finalist,
            }
            if validation["status"] == "WIN":
                selected = finalist
        except (RuntimeError, ValueError, KeyError) as exc:
            if (
                "illegal memory access" in str(exc).lower()
                or "device-side assert" in str(exc).lower()
            ):
                preserve_partial(exc, "validation", finalist)
                raise
            validation = {"status": "INVALID", "spec": finalist, "error": str(exc)}
    return {
        "case": asdict(case),
        "incumbent": incumbent,
        "baseline_max_abs": baseline_error,
        "search": records,
        "validation": validation,
        "selected": selected,
    }


def compatible_selections(results):
    """Refuse to merge workloads the production selector cannot distinguish."""
    groups = {}
    for row in results:
        c = row["case"]
        key = (
            c["kind"],
            c["phase"],
            c["tokens"],
            c["pool_rank"],
            c["in_features"],
            sum(c["slices"]),
        )
        groups.setdefault(key, []).append(row)
    return [
        {
            "selector": list(key),
            "cases": [r["case"]["name"] for r in rows],
            "status": "CONSISTENT"
            if all(r["selected"] == rows[0]["selected"] for r in rows)
            else "CONFLICT",
            "spec": rows[0]["selected"]
            if all(r["selected"] == rows[0]["selected"] for r in rows)
            else None,
        }
        for key, rows in groups.items()
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument(
        "--candidates", type=Path, help="optional list of production DensePlan specs"
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="new result file; never overwrites"
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--min-gain", type=float, default=0.02)
    parser.add_argument("--max-regression", type=float, default=0.02)
    parser.add_argument(
        "--check", action="store_true", help="validate workloads without CUDA"
    )
    args = parser.parse_args(argv)
    if (
        args.repeats < 3
        or args.iterations < 1
        or any(
            not math.isfinite(x) or not 0 < x < 1
            for x in (args.min_gain, args.max_regression)
        )
    ):
        parser.error(
            "need >=3 repeats, positive iterations and finite fractional gates in (0,1)"
        )
    cases = [Case.from_dict(c) for c in json.loads(args.workloads.read_text())]
    if not cases or len({c.name for c in cases}) != len(cases):
        parser.error("workloads must have distinct names")
    if args.check:
        print(f"{len(cases)} valid workloads; no GPU execution")
        return
    import torch

    from sglang.srt.lora import dense
    from sglang.srt.lora.utils import architecture_for_capability

    if args.out.exists():
        parser.error("output already exists; use a new filename")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    architecture = architecture_for_capability(
        torch.cuda.get_device_capability(device)[0]
    )
    properties = torch.cuda.get_device_properties(device)
    root = Path(__file__).resolve().parents[3]
    if Path(dense.__file__).resolve().parent != root / "python/sglang/srt/lora/dense":
        parser.error("PYTHONPATH must load this checkout's python/ tree")
    if os.environ.get("SGLANG_LORA_DENSE_CONFIG_DIR"):
        parser.error(
            "use the checkout's incumbent tables; unset SGLANG_LORA_DENSE_CONFIG_DIR"
        )
    sources = sorted((root / "python/sglang/srt/lora").rglob("*.py")) + sorted(
        (root / "python/sglang/srt/lora/dense/configs").glob("*.json")
    )
    sources += [Path(__file__), Path(__file__).resolve().parents[1] / "lora_tuning.py"]
    digest = hashlib.sha256()
    for path in sources:
        digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    report = {
        "device": str(properties),
        "uuid": str(getattr(properties, "uuid", "unavailable")),
        "architecture": architecture,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "source_sha256": digest.hexdigest(),
        "seed": args.seed,
        "objective": "local BF16 base GEMM + dense runner + fresh routing; no collectives or model TPS",
        "gates": {"min_gain": args.min_gain, "max_regression": args.max_regression},
        "results": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as output:
        try:
            for case in cases:
                try:
                    result = tune_case(case, args, device, architecture)
                except Exception as exc:
                    partial = getattr(exc, "tuning_partial", {"case": asdict(case)})
                    partial["validation"] = {
                        **partial.get("validation", {}),
                        "status": "INCOMPLETE",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    report["results"].append(partial)
                    raise
                report["results"].append(result)
                print(
                    case.name, report["results"][-1]["validation"]["status"], flush=True
                )
            report["selections"] = compatible_selections(report["results"])
        finally:
            json.dump(report, output, indent=2, allow_nan=False)
            output.write("\n")


if __name__ == "__main__":
    main()
