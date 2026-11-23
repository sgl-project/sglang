"""Shared Qwen3.5 LM-head scoring configuration and bounded H100 checks.

Extra A runs separate MIS-off/on files, reusing one server per file for
pointwise and matched two-set/five-candidate setwise requests. Each mode checks
its fixed load and capacity floor with 1,000 requests over at least 30 seconds,
zero errors and a 500 ms HTTP p99 ceiling. Capacity search is local
qualification work, not part of CI.
"""

import asyncio
import json

import requests
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from sglang.test.kits.perf_bench_kit import at_least, at_most, check_perf, reported
from sglang.test.kits.score_bench_kit import build_requests, make_workload, run_load
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
ANCHOR = "<|object_ref_start|>"
# Fixed loads and conservative capacity floors, not peak-capacity estimates.
QPS_BOUNDS = {
    "pointwise": (11, 18),
    "pointwise-mis": (16, 28),
    "setwise": (16, 28),
    "setwise-mis": (15, 27),
}


def resolve_model():
    return snapshot_download(MODEL, revision=REVISION)


def engine_args(enable_mis):
    return dict(
        dtype="bfloat16",
        enable_mis=enable_mis,
        attention_backend="flashinfer",
        linear_attn_prefill_backend="triton",
        disable_radix_cache=True,
        chunked_prefill_size=-1,
        disable_cuda_graph=True,
        language_only=True,
        mem_fraction_static=0.8,
        context_length=4096,
        max_running_requests=64,
        max_prefill_tokens=30000,
        log_level="warning",
    )


def server_args(enable_mis):
    result = []
    for key, value in engine_args(enable_mis).items():
        if value is False:
            continue
        result.append("--" + key.replace("_", "-"))
        if value is not True:
            result.append(str(value))
    return result


def _sentinel(workload, warmup):
    # Fixtures cycle, so this reference is also checked during steady-state load.
    payload = build_requests(workload, warmup + 1)[-1]
    response = requests.post(
        DEFAULT_URL_FOR_TEST + "/v1/score", json=payload, timeout=30
    )
    response.raise_for_status()
    expected = torch.tensor(response.json()["scores"], dtype=torch.float64)

    def check(request, body):
        if request == payload:
            actual = torch.tensor(body["scores"], dtype=torch.float64)
            if actual.shape != expected.shape or not torch.allclose(
                actual, expected, atol=0.06, rtol=0
            ):
                raise ValueError("Scoring sentinel changed under concurrent load")

    return check


def _check_load(test_case, workload, qps, kind, output_dir, metadata, validator):
    result = asyncio.run(
        run_load(
            DEFAULT_URL_FOR_TEST,
            workload,
            qps=qps,
            num_requests=1000,
            min_samples=1000,
            min_duration_s=30,
            warmup=20,
            timeout_s=1,
            metadata={**metadata, "check": kind},
            response_validator=validator,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{workload.mode}_{kind}.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    test_case.assertTrue(
        result["passed"],
        f"{workload.mode}/{kind}: {result['reasons']}; "
        f"p99={result['latency_ms']['p99']}; artifacts: {output_dir}",
    )
    check_perf(
        test_case,
        at_most("p99_latency_ms", result["latency_ms"]["p99"], 500, unit="ms"),
        at_least("achieved_qps", result["achieved_qps"], qps * 0.95, unit="req/s"),
        reported("decisions_per_s", result["decisions_per_s"], unit="scores/s"),
        suffix=f"_{workload.mode}_{kind}",
    )
    response = requests.get(
        DEFAULT_URL_FOR_TEST + "/flush_cache", params={"timeout": 30}, timeout=35
    )
    response.raise_for_status()


def check_qwen35_score_perf(test_case, *, enable_mis, output_dir):
    """Reuse one server for pointwise/setwise fixed-load and capacity-floor checks."""
    gpu = torch.cuda.get_device_name(0)
    test_case.assertIn("H100", gpu, "These QPS bounds target the H100 CI runner")
    model = resolve_model()
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
    metadata = dict(
        model=MODEL,
        requested_revision=REVISION,
        resolved_model=model,
        gpu=gpu,
        torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda,
        server_args=engine_args(enable_mis),
    )
    process = popen_launch_server(
        model,
        DEFAULT_URL_FOR_TEST,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=server_args(enable_mis),
    )
    try:
        response = requests.post(DEFAULT_URL_FOR_TEST + "/freeze_gc", timeout=30)
        response.raise_for_status()
        for prefix in ("pointwise", "setwise"):
            mode = prefix + ("-mis" if enable_mis else "")
            workload = make_workload(mode=mode, tokenizer=tokenizer)
            validator = _sentinel(workload, warmup=20)
            for kind, qps in zip(("fixed_load", "capacity_floor"), QPS_BOUNDS[mode]):
                _check_load(
                    test_case, workload, qps, kind, output_dir, metadata, validator
                )
    finally:
        terminate_and_kill_process_tree(process)
