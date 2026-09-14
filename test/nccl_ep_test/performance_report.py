"""Validate matched full-model evidence before publishing a comparison."""

import json
import math
from pathlib import Path

from .followup_server import MODEL, REVISION
from .full_model_benchmark import Workload


def statistics(values):
    from .benchmark import statistics_ms

    return statistics_ms(values)


def validate_pair(reports):
    if len(reports) != 2 or [r.get("rank") for r in reports] != [0, 1]:
        raise ValueError("Both rank reports are required")
    first = reports[0]
    if first["configuration"] not in ("serial", "sbo", "tbo", "sbo-tbo"):
        raise ValueError("Unknown configuration")
    workload = Workload(**first["workload"])
    fingerprint = workload.fingerprint()
    expected = {(b, r) for b in workload.buckets for r in range(workload.rounds)}
    indexed = []
    for report in reports:
        if (
            not report.get("passed")
            or not report.get("native_ep_tested")
            or not report.get("cleanup_completed")
            or report.get("profiled")
            or report.get("implementation") != "nccl_ep_full_model_decode_v2"
            or report.get("model") != MODEL
            or report.get("revision") != REVISION
            or report.get("model_shape", {}).get("moe_layers") != 26
            or report.get("model_shape", {}).get("layers") != 27
        ):
            raise ValueError("Complete unprofiled native full-model evidence required")
        for key in (
            "source_head",
            "configuration",
            "workload",
            "model_shape",
            "resolved_args",
        ):
            if report[key] != first[key]:
                raise ValueError(f"Rank evidence differs: {key}")
        if report["workload_fingerprint"] != fingerprint:
            raise ValueError("Workload fingerprint mismatch")
        resolved = report["resolved_args"]
        partition = report.get("attention_partition", {})
        expected_backends = 3 if "tbo" in report["configuration"] else 1
        if (
            partition.get("tile") != workload.attention_split_tile
            or partition.get("max_kv_splits") != [1] * expected_backends
            or resolved.get("triton_attention_split_tile_size")
            != workload.attention_split_tile
        ):
            raise ValueError("Matched fixed attention partitions are required")
        if (
            resolved.get("enable_eplb")
            or resolved.get("ep_size") != 2
            or resolved.get("enable_two_batch_overlap")
            != ("tbo" in report["configuration"])
            or resolved.get("enable_single_batch_overlap")
            != ("sbo" in report["configuration"])
        ):
            raise ValueError("Resolved overlap/EP configuration differs")
        mapping = {(r["bucket"], r["round"]): r for r in report["records"]}
        if len(mapping) != len(report["records"]) or set(mapping) != expected:
            raise ValueError("Missing or duplicate measurement blocks")
        for record in mapping.values():
            n = workload.samples
            tbo = "tbo" in report["configuration"]
            if record["graph_passes"] != n or record["tbo_passes"] != n * tbo:
                raise ValueError("Graph/TBO fallback in measured steps")
            if set(record["samples"]) != {"cuda_step_ms", "host_step_ms"}:
                raise ValueError("Missing timing metric")
            for samples in record["samples"].values():
                if len(samples) != n:
                    raise ValueError("Incomplete samples")
                statistics(samples)
        indexed.append(mapping)
    rows = []
    for bucket in workload.buckets:
        metrics = {}
        for metric in ("cuda_step_ms", "host_step_ms"):
            values = [
                max(a, b)
                for rnd in range(workload.rounds)
                for a, b in zip(
                    indexed[0][bucket, rnd]["samples"][metric],
                    indexed[1][bucket, rnd]["samples"][metric],
                )
            ]
            metrics[metric] = statistics(values)
            # Two independent attention DP ranks each decode B tokens per step.
            metrics[metric]["aggregate_tokens_per_second"] = (
                2 * bucket * 1000 / metrics[metric]["median"]
            )
        rows.append(
            dict(bucket_per_rank=bucket, global_batch=2 * bucket, metrics=metrics)
        )
    return dict(
        source_head=first["source_head"],
        configuration=first["configuration"],
        workload_fingerprint=fingerprint,
        workload=first["workload"],
        rows=rows,
    )


def compare_logits(reference, candidate, *, rtol=0.02, atol=0.02):
    import torch

    if not reference or reference.keys() != candidate.keys():
        raise ValueError("Missing or mismatched logit checkpoints")
    results = []
    for key, expected in reference.items():
        actual = candidate[key]
        if actual.shape != expected.shape or actual.ndim != 2:
            raise ValueError("Logit shapes differ")
        if not torch.isfinite(expected).all() or not torch.isfinite(actual).all():
            raise ValueError("Nonfinite logits")
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        results.append(
            dict(
                checkpoint=key,
                max_abs=(actual - expected).abs().max().item(),
                rms=(actual - expected).square().mean().sqrt().item(),
                argmax_agreement=(actual.argmax(-1) == expected.argmax(-1))
                .float()
                .mean()
                .item(),
            )
        )
    return dict(rtol=rtol, atol=atol, checkpoints=results)


def compare_runs(reference_dir, candidate_dir):
    import torch

    def read(root):
        return [
            json.loads((Path(root) / f"model-rank{r}.json").read_text()) for r in (0, 1)
        ]

    reference, candidate = read(reference_dir), read(candidate_dir)
    summaries = [validate_pair(pair) for pair in (reference, candidate)]
    if reference[0]["configuration"] != "serial":
        raise ValueError("Reference must be serial")
    for key in ("source_head", "workload_fingerprint"):
        if summaries[0][key] != summaries[1][key]:
            raise ValueError(f"Cannot compare different {key}")
    checks = []
    for rank in (0, 1):
        # Same source, machine, native libraries, capacities and full-model shape;
        # only the requested overlap switches may change.
        for key in ("model_shape", "bindings", "environment"):
            if key == "environment":
                for part in ("devices", "torch", "torch_cuda", "topology", "toolkit"):
                    if reference[rank][key][part] != candidate[rank][key][part]:
                        raise ValueError(f"Environment changed: {part}")
            elif reference[rank][key] != candidate[rank][key]:
                raise ValueError(f"Comparison mismatch: {key}")
        a, b = (dict(pair[rank]["resolved_args"]) for pair in (reference, candidate))
        for flag in ("enable_two_batch_overlap", "enable_single_batch_overlap"):
            a.pop(flag)
            b.pop(flag)
        if a != b:
            raise ValueError("Resolved non-overlap arguments differ")
        checkpoints = [
            torch.load(
                Path(root) / f"logits-rank{rank}.pt",
                map_location="cpu",
                weights_only=True,
            )
            for root in (reference_dir, candidate_dir)
        ]
        work = Workload(**reference[rank]["workload"])
        expected = {
            f"B{b}/round{r}/sample{s}"
            for b in work.buckets
            for r in range(work.rounds)
            for s in (0, work.samples - 1)
        }
        if any(set(values) != expected for values in checkpoints):
            raise ValueError("Incomplete checkpoint files")
        checks.append(compare_logits(*checkpoints))
    rows = []
    for a, b in zip(summaries[0]["rows"], summaries[1]["rows"]):
        ratios = {
            metric: a["metrics"][metric]["median"] / b["metrics"][metric]["median"]
            for metric in ("cuda_step_ms", "host_step_ms")
        }
        if not all(math.isfinite(v) for v in ratios.values()):
            raise ValueError("Invalid speedup")
        rows.append(
            dict(
                bucket_per_rank=a["bucket_per_rank"],
                serial=a["metrics"],
                candidate=b["metrics"],
                speedup=ratios,
            )
        )
    return dict(
        passed=True,
        configuration=candidate[0]["configuration"],
        source_head=summaries[0]["source_head"],
        workload_fingerprint=summaries[0]["workload_fingerprint"],
        rows=rows,
        logit_comparison=checks,
        scope="Full-model teacher-forced decode; no sampling/HTTP or generation-quality claim",
    )
