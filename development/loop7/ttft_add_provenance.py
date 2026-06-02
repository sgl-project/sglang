"""Backfill `run_provenance` into the R20 TTFT artifacts (R21 evidence-hygiene repair).

The R20 `ttft_*.json` were generated before `perf_closed_batch.py` recorded provenance, so
they carried only metric fields. The R20-review required each artifact to carry exact run
provenance (server-code commit, tool commit, tree-dirty flag, GPU, launch command, config,
mem, graph/radix flags, served count, graph evidence, artifact path). The exact run state
is reconstructable from the R20 boot logs + launch commands + `nvidia-smi` + git history, so
this REPAIRS the artifacts in place (the reviewer's instruction: reconstruct, do not rerun).
The metric fields are unchanged; only a `run_provenance` block is added, marked
`reconstructed`. It reuses `build_run_provenance` from the probe so the schema is identical
to a live `--stream` run.

Commit facts (verified via `git log`): f9f6ec056 = R18, 68969deb0 = R19, 30173f08b = R20.
The R20 TTFT servers were launched from the R19 tree (HEAD = 68969deb0) with the `--stream`
probe modification uncommitted; R19 and R20 commits touched only `development/loop7/`, so the
DS/DSA production serving code was unchanged across R18 -> R19 -> R20.

Usage: python development/loop7/ttft_add_provenance.py   # rewrites development/loop7/ttft_*.json
"""

from __future__ import annotations

import glob
import importlib.util
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

# Import build_run_provenance from the probe (single schema source).
_spec = importlib.util.spec_from_file_location(
    "_perf_closed_batch", os.path.join(_HERE, "perf_closed_batch.py")
)
_probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_probe)
build_run_provenance = _probe.build_run_provenance

# Facts common to every R20 TTFT run.
COMMON = dict(
    reconstructed=True,
    reconstructed_in_round=21,
    note=(
        "Provenance reconstructed in R21 from the R20 boot logs + launch commands + "
        "nvidia-smi + git history; the metric fields in this file are UNCHANGED from the "
        "R20 streaming run (no rerun)."
    ),
    server_code_commit="68969deb0",
    server_code_note=(
        "Servers launched from the R19 tree HEAD=68969deb0. DS/DSA production serving code "
        "is unchanged across R18 (f9f6ec056) -> R19 (68969deb0) -> R20 (30173f08b): the R19 "
        "and R20 commits touched only development/loop7/ (no python/sglang/srt change)."
    ),
    measurement_tool_commit="30173f08b",
    tree_dirty_during_run=True,  # the --stream probe edit was uncommitted during the runs
    gpu="NVIDIA H200",
    gpu_count=8,
    tp_size=8,
    graph=True,
    radix_cache="off (--disable-radix-cache)",
    overlap_schedule="off (--disable-overlap-schedule)",
)

# Per-variant facts (launch command, effective config, memory, a representative graph line).
VARIANTS = {
    "ds_default": dict(
        launch_cmd="LOOP7_MEASUREMENT=1 bash development/serve_double_sparsity.sh",
        op_point=(
            "DS int8 / mem_fraction_static=0.7 / fp8-KV / TP=8 / page 64 / flashmla_kv "
            "prefill+decode / radix-off / overlap-off / piecewise-cuda-graph-off; CUDA graph ON"
        ),
        server_config={
            "enable_double_sparsity": True,
            "top_k": 2048,
            "signature_dtype": "int8",
            "page_size": 64,
            "scorer_norm": "off",
            "head_agg": "max",
            "anchor_mode": "off",
            "enable_lifted_budget_decode": False,
            "mem_fraction_static": 0.7,
            "kv_cache_dtype": "fp8_e4m3",
            "attention_backend": "dsa",
            "dsa_decode_backend": "flashmla_kv",
        },
        mem_fraction_static=0.7,
        gpu_mem_per_gpu="125 GB",
        mem_source="nvidia-smi during run: 125116 MiB/GPU",
        graph_evidence=(
            "[TP0] Decode batch, #running-req: 16, #token: 14336, cuda graph: True, "
            "gen throughput (token/s): 442.58, #queue-req: 0 (this boot)"
        ),
    ),
    "ds_hybrid": dict(
        launch_cmd=(
            "LOOP7_MEASUREMENT=1 SCORER_NORM=hybrid HEAD_AGG=mean "
            "bash development/serve_double_sparsity.sh"
        ),
        op_point=(
            "DS int8 / mem_fraction_static=0.7 / fp8-KV / TP=8 / page 64 / flashmla_kv "
            "prefill+decode / radix-off / overlap-off / piecewise-cuda-graph-off; CUDA graph ON"
        ),
        server_config={
            "enable_double_sparsity": True,
            "top_k": 2048,
            "signature_dtype": "int8",
            "page_size": 64,
            "scorer_norm": "hybrid",
            "scorer_norm_hybrid_threshold": 8192,
            "head_agg": "mean",
            "anchor_mode": "off",
            "enable_lifted_budget_decode": False,
            "mem_fraction_static": 0.7,
            "kv_cache_dtype": "fp8_e4m3",
            "attention_backend": "dsa",
            "dsa_decode_backend": "flashmla_kv",
        },
        mem_fraction_static=0.7,
        gpu_mem_per_gpu="125 GB",
        mem_source="nvidia-smi during run: 125116 MiB/GPU",
        graph_evidence=(
            "[TP0] Decode batch, #running-req: 16, #token: 14336, cuda graph: True, "
            "gen throughput (token/s): 466.20, #queue-req: 0 (this boot)"
        ),
    ),
    "dsa": dict(
        launch_cmd="DISABLE_RADIX_CACHE=1 MEM_FRACTION_STATIC=0.85 bash development/serve_native_nsa.sh",
        op_point=(
            "native-NSA (no double-sparsity) / mem_fraction_static=0.85 / fp8-KV / TP=8 / "
            "page 64 / dsa backend / radix-off; CUDA graph ON"
        ),
        server_config={
            "enable_double_sparsity": False,
            "attention_backend": "dsa",
            "page_size": 64,
            "mem_fraction_static": 0.85,
            "kv_cache_dtype": "fp8_e4m3",
            "dsa_decode_backend": "flashmla_kv",
        },
        mem_fraction_static=0.85,
        gpu_mem_per_gpu="133 GB",
        mem_source="nvidia-smi during run: 133066 MiB/GPU",
        graph_evidence=(
            "[TP0] Decode batch, #running-req: 16, #token: 14336, cuda graph: True, "
            "gen throughput (token/s): 907.81, #queue-req: 0 (this boot)"
        ),
    ),
}


def _variant_of(basename: str) -> str:
    # ttft_<variant>_c<conc>[_p770].json -- match longest variant key first.
    stem = basename[len("ttft_") :]
    for key in ("ds_default", "ds_hybrid", "dsa"):
        if stem.startswith(key + "_"):
            return key
    raise ValueError(f"cannot determine variant from {basename!r}")


def main():
    files = sorted(glob.glob(os.path.join(_HERE, "ttft_*.json")))
    if not files:
        raise SystemExit("no ttft_*.json found to backfill")
    for path in files:
        base = os.path.basename(path)
        variant = _variant_of(base)
        with open(path) as f:
            data = json.load(f)
        prov = build_run_provenance(
            served=int(data.get("completed", 0)),
            artifact_path="development/loop7/" + base,
            **COMMON,
            **VARIANTS[variant],
        )
        data["run_provenance"] = prov
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"backfilled {base}: variant={variant} served={prov['served']}")


if __name__ == "__main__":
    main()
