"""Server launch, GSM8K and prefill-perf helpers the Intel XPU DWDP served-model tests share; test/registered is not an importable package."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from sglang.benchmark.serving import run_benchmark
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.common import get_device_count, get_device_module
from sglang.test.sgl_eval_utils import run_sgl_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    auto_config_device,
    get_benchmark_args,
    popen_launch_server,
    write_github_step_summary,
)

# The XPU MoE path lives in the intel_xpu attention backend.
XPU_SERVER_ARGS = [
    "--device",
    "xpu",
    "--attention-backend",
    "intel_xpu",
    "--dtype",
    "bfloat16",
    "--trust-remote-code",
]
# Shared by the DWDP arm and its TP control: TP mean 0.960 - 3 SE at n=400 on 2x B60
# (sgl-eval 0.1.3, greedy; DWDP scored 0.965).
GPT_OSS_20B_GSM8K_FLOOR = 0.93
GSM8K_NUM_EXAMPLES = 400
# The XPU default is single-stream because intel_xpu attention wedged the driver on
# concurrent prefill; 8 threads run clean at TP=2.
GSM8K_NUM_THREADS = 8
# DWDP's buffer slots cost two to three layers of expert bytes on top of the owned
# shard, so every DWDP test and its TP control get the same fraction.
MEM_FRACTION_STATIC = "0.85"


def launch_dwdp_server(
    model: str,
    base_url: str,
    *,
    extra_args: List[str],
    timeout: float,
    tp_size: Optional[int] = None,
):
    other_args = XPU_SERVER_ARGS + list(extra_args)
    if tp_size is not None:
        other_args += ["--tp-size", str(tp_size)]
    try:
        return popen_launch_server(
            model, base_url, timeout=timeout, other_args=other_args
        )
    except Exception as error:
        write_github_step_summary(f"Failed to launch server for {model}: {error}")
        raise


def assert_gsm8k_accuracy(
    test_case,
    *,
    model: str,
    base_url: str,
    accuracy: float,
    num_examples: int,
    num_threads: int,
    max_tokens: int = 4096,
) -> dict:
    """Score a launched server on GSM8K with sgl-eval and assert the floor.

    Greedy sampling, as in test_disaggregation_dwdp_gpt_oss.py, so the DWDP arm
    and its TP control differ only by the parallelism under test.
    """
    metrics = run_sgl_eval(
        SimpleNamespace(
            base_url=base_url,
            model=model,
            eval_name="gsm8k",
            num_examples=num_examples,
            num_threads=num_threads,
            max_tokens=max_tokens,
            repeat=1,
            temperature=0.0,
            top_p=1.0,
        )
    )
    write_github_step_summary(
        f"### {model} gsm8k\nscore={metrics['score']:.4f} (floor {accuracy})\n"
    )
    test_case.assertGreaterEqual(
        metrics["score"],
        accuracy,
        f"GSM8K score of {model} is {metrics['score']}, below {accuracy}",
    )
    return metrics


def run_prefill_load(
    model: str,
    base_url: str,
    *,
    input_len: int,
    num_prompts: int,
    max_concurrency: int,
) -> Dict[str, Any]:
    """Measure prefill-only serving throughput against a running server.

    Every prompt is exactly input_len random tokens with one output token, so
    the run is all prefill. One unmeasured pass at max_concurrency prompts
    first takes JIT compilation and allocator growth out of the measurement.
    """
    args = get_benchmark_args(
        base_url=base_url,
        dataset_name="random",
        tokenizer=model,
        num_prompts=max_concurrency,
        random_input_len=input_len,
        random_output_len=1,
        device=auto_config_device(),
        max_concurrency=max_concurrency,
    )
    args.random_range_ratio = 1.0
    run_benchmark(args)
    args.num_prompts = num_prompts
    return run_benchmark(args)


GPT_OSS_20B = "openai/gpt-oss-20b"
# Pinned so both platforms load the same weights and tokenize the same prompts.
GPT_OSS_20B_REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
PERF_LAUNCH_TIMEOUT_S = 1800
PERF_INPUT_LEN = 4096
# The load scales with the GPU count so every rank sees the same work.
PERF_NUM_PROMPTS_PER_GPU = 32
PERF_MAX_CONCURRENCY_PER_GPU = 4
# DWDP/TP is at least 1.03 on every measured platform and GPU count; the floor
# leaves a margin under that.
MIN_DWDP_TO_TP_INPUT_THROUGHPUT = 0.9

# Every flag whose default depends on the platform, so the arms differ only in
# --dwdp-size and the platforms only in the device flags launch_dwdp_server adds.
PERF_FIXED_SERVER_ARGS = [
    "--revision",
    GPT_OSS_20B_REVISION,
    "--dtype",
    "bfloat16",
    "--mem-fraction-static",
    MEM_FRACTION_STATIC,
    "--max-prefill-tokens",
    "16384",
    # below the smallest memory-sized pool, so no arm gets a larger KV pool
    "--max-total-tokens",
    "196608",
    "--max-running-requests",
    "64",
    # intel_xpu only takes 64 or 128
    "--page-size",
    "128",
    "--sampling-backend",
    "pytorch",
    "--random-seed",
    "42",
    # auto picks triton_kernel for the TP arm only on CUDA, which would measure the
    # kernel instead of DWDP
    "--moe-runner-backend",
    "triton",
    # XPU has no custom all-reduce
    "--disable-custom-all-reduce",
    # kept off in both arms to match the measured perf baseline
    "--disable-overlap-schedule",
    # DWDP forces CUDA graphs off, so the TP arm turns them off too
    "--disable-cuda-graph",
    "--disable-radix-cache",
]


def run_prefill_perf_vs_tp(*, num_gpus: int) -> float:
    """Bench TP then DWDP on the same num_gpus cards with the same prompts; returns DWDP/TP input throughput."""
    server_args = PERF_FIXED_SERVER_ARGS + [
        "--chunked-prefill-size",
        str(num_gpus * PERF_INPUT_LEN),
    ]
    results: Dict[str, Dict[str, float]] = {}
    for arm, arm_args in (("tp", []), ("dwdp", ["--dwdp-size", str(num_gpus)])):
        process = launch_dwdp_server(
            GPT_OSS_20B,
            DEFAULT_URL_FOR_TEST,
            extra_args=server_args + arm_args,
            timeout=PERF_LAUNCH_TIMEOUT_S,
            tp_size=num_gpus,
        )
        try:
            results[arm] = run_prefill_load(
                GPT_OSS_20B,
                DEFAULT_URL_FOR_TEST,
                input_len=PERF_INPUT_LEN,
                num_prompts=PERF_NUM_PROMPTS_PER_GPU * num_gpus,
                max_concurrency=PERF_MAX_CONCURRENCY_PER_GPU * num_gpus,
            )
        finally:
            kill_process_tree(process.pid)

    ratio = results["dwdp"]["input_throughput"] / results["tp"]["input_throughput"]
    device = f"{num_gpus}x {get_device_module().get_device_name(0)}"
    rows = "".join(
        f"| {arm.upper()} | {result['input_throughput']:.0f} "
        f"| {result['median_ttft_ms']:.1f} | {result['p99_ttft_ms']:.1f} |\n"
        for arm, result in results.items()
    )
    write_github_step_summary(
        f"### {GPT_OSS_20B} DWDP prefill perf ({device}, input {PERF_INPUT_LEN})\n"
        f"| arm | input tok/s | median TTFT (ms) | P99 TTFT (ms) |\n"
        f"| --- | --- | --- | --- |\n{rows}"
        f"DWDP/TP = {ratio:.3f} (floor {MIN_DWDP_TO_TP_INPUT_THROUGHPUT})\n"
    )
    return ratio


def assert_prefill_perf_vs_tp(test_case, num_gpus: int) -> None:
    """Skip unless num_gpus cards are visible, then assert the DWDP/TP floor."""
    if get_device_count() < num_gpus:
        test_case.skipTest(f"needs {num_gpus} visible GPUs, found {get_device_count()}")
    ratio = run_prefill_perf_vs_tp(num_gpus=num_gpus)
    test_case.assertGreaterEqual(
        ratio,
        MIN_DWDP_TO_TP_INPUT_THROUGHPUT,
        f"DWDP prefill throughput is {ratio:.3f}x TP on {num_gpus} GPUs",
    )
