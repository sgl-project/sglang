"""One VLM serving benchmark, run against a named attention backend.

Pinned rather than inherited from `get_default_attn_backend`, so the file name
keeps naming the right kernel after the default moves.
"""

import os
from typing import Optional

from sglang.test.kits.perf_bench_kit import at_least, at_most, check_perf, reported
from sglang.test.test_utils import (
    DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST,
    DEFAULT_URL_FOR_TEST,
    auto_config_device,
    get_benchmark_args,
    run_bench_serving_multi,
)


def _local_tokenizer_path():
    # The HF Hub API call can stall for minutes in CI; prefer a local snapshot.
    try:
        from sglang.srt.utils import find_local_repo_dir

        local_dir = find_local_repo_dir(
            DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST, revision=None
        )
        if local_dir and os.path.isdir(local_dir):
            return local_dir
    except Exception:
        pass
    return None


def check_vlm_serving_perf(
    test_case,
    attention_backend: str,
    *,
    e2e_ms: float,
    ttft_ms: float,
    itl_ms: float,
    output_throughput: Optional[float] = None,
):
    """Offline then online against one server; bound both phases.

    `output_throughput` unset means the offline number is reported, not bounded.
    """
    common = dict(
        base_url=DEFAULT_URL_FOR_TEST,
        dataset_name="mmmu",
        dataset_path="",
        tokenizer=_local_tokenizer_path(),
        random_input_len=4096,
        random_output_len=2048,
        sharegpt_context_len=None,
        disable_stream=False,
        disable_ignore_eos=False,
        seed=0,
        device=auto_config_device(),
        lora_name=None,
    )
    offline = get_benchmark_args(num_prompts=200, request_rate=float("inf"), **common)
    # 50 is enough for a stable median against these loose ceilings.
    online = get_benchmark_args(num_prompts=50, request_rate=1, **common)

    (_, res_offline), (_, res_online) = run_bench_serving_multi(
        DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST,
        DEFAULT_URL_FOR_TEST,
        other_server_args=[
            "--mem-fraction-static",
            "0.7",
            "--attention-backend",
            attention_backend,
        ],
        benchmark_args=[offline, online],
    )

    throughput = res_offline["output_throughput"]
    check_perf(
        test_case,
        (
            at_least("output_throughput", throughput, output_throughput, unit="token/s")
            if output_throughput is not None
            else reported("output_throughput", throughput, unit="token/s")
        ),
        at_most(
            "median_e2e_latency_ms",
            res_online["median_e2e_latency_ms"],
            e2e_ms,
            unit="ms",
        ),
        at_most("median_ttft_ms", res_online["median_ttft_ms"], ttft_ms, unit="ms"),
        at_most("median_itl_ms", res_online["median_itl_ms"], itl_ms, unit="ms"),
    )
