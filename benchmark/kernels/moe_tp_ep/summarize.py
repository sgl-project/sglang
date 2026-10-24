"""Validate paired global workloads and summarize independent process medians."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from environment import hardware_identity


def validate_parallel_layout(backend, tp_size, ep_size):
    if not isinstance(tp_size, int) or tp_size < 1:
        raise ValueError("tp_size must be a positive device count")
    if backend == "none" and ep_size == 1:
        return
    if backend == "deepep" and ep_size == tp_size and tp_size >= 2:
        return
    raise ValueError("use EP=1 for none, or EP=TP>=2 for deepep")


def require_same(records, keys, description):
    for key in keys:
        if any(record[key] != records[0][key] for record in records):
            raise ValueError(f"mismatched {key}: {description}")


def software_identity(metadata, historical=False):
    identity = {
        key: metadata[key]
        for key in (
            "environment",
            "packages",
            "sglang_commit",
            "sglang_diff_sha256",
            "scripts_sha256",
            "execution_mode",
            "image_digest",
        )
    }
    # Older archived records did not include the interpreter version.
    identity["python"] = metadata.get("python")
    identity["source_archive_sha256"] = metadata.get("source_archive_sha256")
    if not historical:
        identity["hardware"] = hardware_identity(metadata)
    if not identity["sglang_commit"]:
        raise ValueError("missing executed SGLang source identity")
    for package in ("torch", "torch-npu", "deep-ep", "sglang"):
        if package not in identity["packages"]:
            raise ValueError(f"missing package identity: {package}")
    for script in ("bench_moe_tp_ep.py", "workload.py"):
        if not identity["scripts_sha256"].get(script):
            raise ValueError(f"missing script identity: {script}")
    return identity


def checked_archive(root):
    """Legacy mode accepts only bytes covered by an explicitly checked archive."""
    files = {}
    for entry in json.loads((root / "manifest.json").read_text()):
        path = root / entry["path"]
        data = path.read_bytes()
        if (
            len(data) != entry["bytes"]
            or hashlib.sha256(data).hexdigest() != entry["sha256"]
        ):
            raise ValueError(f"historical evidence checksum mismatch: {path}")
        files[path.resolve()] = data
    return files


def mode_identity(run):
    mode = run.get("deepep_mode")
    if not isinstance(mode, dict) or set(mode) != {"requested", "resolved"}:
        raise ValueError("missing requested/resolved DeepEP mode")
    if run["backend"] == "none":
        if mode != dict(requested=None, resolved=None):
            raise ValueError("TP baseline must record DeepEP as unused")
    else:
        requested, resolved = mode["requested"], mode["resolved"]
        expected = requested
        if requested == "auto":
            expected = "normal" if run["phase"] == "prefill" else "low_latency"
        if requested not in ("auto", "normal", "low_latency") or resolved != expected:
            raise ValueError("invalid requested/resolved DeepEP mode")
    return mode


def summarize(directory, historical_provenance=None):
    archive = checked_archive(historical_provenance) if historical_provenance else {}
    groups = {}
    for path in sorted(directory.glob("*.json")):
        run = json.loads(path.read_text())
        if "results" not in run:
            continue
        historical = run.get("schema_version") == 2
        if historical:
            if path.read_bytes() != archive.get(path.resolve()):
                raise ValueError("legacy results require checked historical provenance")
        elif run.get("schema_version") != 3:
            raise ValueError("unsupported benchmark schema")
        validate_parallel_layout(run["backend"], run["tp_size"], run["ep_size"])
        metadata = software_identity(run["metadata"], historical=historical)
        mode = None if historical else mode_identity(run)
        # Retain the complete software and helper identity, not just the timing
        # entrypoint. Historical archive runs explicitly record an unknown diff
        # or image as null; equality does not turn these into known identities.
        identity = dict(
            tp_size=run["tp_size"],
            warmup=run["warmup"],
            iters=run["iters"],
            real_weights=run["real_weights"],
            num_experts=run["num_experts"],
            topk=run["topk"],
            hidden=run["hidden"],
            environment=metadata["environment"],
            packages=metadata["packages"],
            source_commit=metadata["sglang_commit"],
            source_diff=metadata["sglang_diff_sha256"],
            scripts=metadata["scripts_sha256"],
            execution_mode=metadata["execution_mode"],
            image_digest=metadata["image_digest"],
            python=metadata["python"],
            source_archive_sha256=metadata["source_archive_sha256"],
            hardware=metadata.get("hardware"),
        )
        if not historical:
            placement = identity["hardware"].get("rank_mapping", [])
            devices = identity["hardware"]["devices"]
            if (
                len(placement) != run["tp_size"]
                or [p["rank"] for p in placement] != list(range(run["tp_size"]))
                or any(
                    p["local_rank"] < 0
                    or p["local_rank"] >= len(devices)
                    or p["device"] != devices[p["local_rank"]]
                    for p in placement
                )
                or len({p["device"]["uuid"] for p in placement}) != run["tp_size"]
            ):
                raise ValueError("invalid rank-to-device mapping for tp_size")
        for point in run["results"]:
            key = run["phase"], point["global_tokens"]
            samples = point["samples_per_rank_ms"]
            if (
                len(samples) != run["tp_size"]
                or run["iters"] < 1
                or any(len(rank) != run["iters"] for rank in samples)
            ):
                raise ValueError(f"invalid per-rank samples: {path}")
            maxima = [max(rank[i][0] for rank in samples) for i in range(run["iters"])]
            if not abs(statistics.median(maxima) - point["median_ms"]) < 1e-6:
                raise ValueError(f"invalid median: {path}")
            expected_rows = point["global_tokens"]
            if run["backend"] == "deepep":
                expected_counts = [
                    expected_rows // run["tp_size"]
                    + (rank < expected_rows % run["tp_size"])
                    for rank in range(run["tp_size"])
                ]
            else:
                expected_counts = [expected_rows] * run["tp_size"]
            if point["tokens_per_rank"] != expected_counts:
                raise ValueError(f"invalid token partition: {path}")
            groups.setdefault(key, {}).setdefault(run["backend"], []).append(
                dict(
                    **identity,
                    deepep_mode=mode,
                    repeat_id=run["repeat_id"],
                    median_ms=point["median_ms"],
                    input_hash=point["input_sha256"],
                    weights_hash=run["weights_sha256"],
                    input_seed=run["input_seed"],
                    file=path.name,
                )
            )
    results = []
    for (phase, n), configs in sorted(groups.items()):
        if set(configs) != {"none", "deepep"}:
            raise ValueError(f"missing paired configuration: {phase} {n}")
        tp, ep = configs["none"], configs["deepep"]
        ids_tp = [x["repeat_id"] for x in tp]
        ids_ep = [x["repeat_id"] for x in ep]
        if len(set(ids_tp)) != len(ids_tp) or sorted(ids_tp) != sorted(ids_ep):
            raise ValueError("unpaired or duplicate independent runs")
        require_same(
            tp + ep,
            (*identity, "input_hash", "weights_hash", "input_seed"),
            f"{phase} {n}",
        )
        values = {}
        for name, runs in configs.items():
            require_same(runs, ("deepep_mode",), f"{phase} {n} {name}")
            medians = [x["median_ms"] for x in runs]
            mid = statistics.median(medians)
            values[name] = dict(
                median_ms=mid,
                run_medians_ms=medians,
                relative_range=(max(medians) - min(medians)) / mid,
            )
            if runs[0]["deepep_mode"] is not None:
                values[name]["deepep_mode"] = runs[0]["deepep_mode"]
        results.append(
            dict(
                phase=phase,
                global_tokens=n,
                independent_runs=len(tp),
                configurations=values,
                ep_over_tp=values["deepep"]["median_ms"] / values["none"]["median_ms"],
            )
        )
    if not results:
        raise ValueError("no benchmark results found")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--historical-provenance", type=Path)
    args = parser.parse_args()
    args.out.write_text(
        json.dumps(summarize(args.directory, args.historical_provenance), indent=2)
        + "\n"
    )
