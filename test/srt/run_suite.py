import argparse
import glob
from pathlib import Path

import tabulate

from sglang.test.ci.ci_utils import TestFile, run_unittest_files

# NOTE: please sort the test cases alphabetically by the test file name
suites = {
    "per-commit-4-gpu": [
        TestFile("models/test_qwen3_next_models.py", 350),
        TestFile("models/test_qwen3_next_models_mtp.py", 500),
        TestFile("test_gpt_oss_4gpu.py", 300),
        TestFile("test_multi_instance_release_memory_occupation.py", 64),
        TestFile("test_pp_single_node.py", 500),
        TestFile("test_epd_disaggregation.py", 150),
    ],
    "per-commit-8-gpu-h200": [
        TestFile("test_deepseek_v3_basic.py", 275),
        TestFile("test_deepseek_v3_mtp.py", 275),
        TestFile("test_disaggregation_hybrid_attention.py", 400),
        TestFile("models/test_kimi_k2_models.py", 200),
        TestFile("test_deepseek_v32_basic.py", 360),
        TestFile("test_deepseek_v32_mtp.py", 360),
        TestFile("models/test_mimo_models.py", 200),
    ],
    "per-commit-8-gpu-h20": [
        TestFile("quant/test_w4a8_deepseek_v3.py", 520),
        TestFile("test_disaggregation_different_tp.py", 600),
        TestFile("test_disaggregation_pp.py", 180),
        TestFile("test_disaggregation_dp_attention.py", 155),
    ],
    "per-commit-4-gpu-b200": [
        TestFile("test_deepseek_v3_fp4_4gpu.py", 1500),
        TestFile("test_fp8_blockwise_gemm.py", 280),
        TestFile("test_gpt_oss_4gpu.py", 700),
        TestFile("test_nvfp4_gemm.py", 360),
    ],
    # "per-commit-8-gpu-b200": [
    #     TestFile("test_mistral_large3_basic.py", 275),  # Moved to nightly - large model
    # ],
    "per-commit-4-gpu-gb200": [
        TestFile("test_deepseek_v3_cutedsl_4gpu.py", 1800),
        TestFile("test_disaggregation_aarch64.py", 300),
    ],
    "per-commit-4-gpu-deepep": [
        TestFile("ep/test_deepep_small.py", 531),
        TestFile("ep/test_mooncake_ep_small.py", 660),
    ],
    # Disabled: IBGDA/cudaHostRegister environment issues on 8-GPU runner, see #17175
    # 4-GPU DeepEP tests provide sufficient coverage
    # "per-commit-8-gpu-h200-deepep": [
    #     TestFile("ep/test_deepep_large.py", 563),
    # ],
    # quantization_test suite migrated to test/registered/quant/
    "__not_in_ci__": [
        TestFile("test_release_memory_occupation.py", 200),  # Temporarily disabled
        TestFile("models/test_dummy_grok_models.py"),
        TestFile("test_profile_v2.py"),
        TestFile("models/test_ministral3_models.py"),
        TestFile("test_mistral_large3_basic.py"),
        TestFile("test_prefill_delayer.py"),
        TestFile("test_fla_layernorm_guard.py"),
        TestFile(
            "models/test_qwen3_next_models_pcg.py"
        ),  # Disabled: intermittent failures, see #17039
        TestFile("ep/test_deepep_large.py", 563),  # Disabled: see #17175
    ],
}

# Add AMD tests
# NOTE: please sort the test cases alphabetically by the test file name
suite_amd = {
    "per-commit-amd": [
        # TestFile("hicache/test_hicache.py", 116), # Disabled temporarily, see https://github.com/sgl-project/sglang/issues/12575
        # TestFile("hicache/test_hicache_mla.py", 127), # Disabled temporarily,  # Temporarily disabled, see https://github.com/sgl-project/sglang/issues/12574
        # TestFile("hicache/test_hicache_storage.py", 127), # Disabled temporarily, see https://github.com/sgl-project/sglang/issues/12575
        # LoRA tests moved to test/registered/lora/ - AMD entries need to be re-added there
        # TestFile("lora/test_lora_backend.py", 99), # Disabled temporarily, see https://github.com/sgl-project/sglang/issues/13107
        # TestFile("lora/test_lora_cuda_graph.py", 250), # Disabled temporarily, see https://github.com/sgl-project/sglang/issues/13107
        # TestFile("lora/test_lora_qwen3.py", 97), # Disabled temporarily, see https://github.com/sgl-project/sglang/issues/13107
        # TestFile("test_torch_compile_moe.py", 210), # Disabled temporarily, see https://github.com/sgl-project/sglang/issues/13107
        # Disabled temporarily
        # TestFile("test_vlm_input_format.py", 300),
        # TestFile("openai_server/features/test_openai_server_hidden_states.py", 240),
        # TestFile("rl/test_update_weights_from_tensor.py", 48),
        # TestFile("test_no_overlap_scheduler.py", 234), # Disabled temporarily and track in #7703
        # TestFile("test_vision_chunked_prefill.py", 175), # Disabled temporarily and track in #7701
        # TestFile("test_wave_attention_backend.py", 150), # Disabled temporarily, see https://github.com/sgl-project/sglang/issues/11127
        # The time estimation for `test_int4fp8_moe.py` assumes `mistralai/Mixtral-8x7B-Instruct-v0.1` is already cached (running on 1xMI300X).
    ],
    "per-commit-4-gpu-amd": [
        TestFile("test_pp_single_node.py", 150),
    ],
    # NOTE: AMD nightly suites (nightly-amd, nightly-amd-vlm, nightly-amd-8-gpu)
    # have been migrated to test/registered/amd/nightly/ and are now managed
    # by test/run_suite.py using the registry system.
}

# Add Intel Xeon tests
suite_xeon = {
    "per-commit-cpu": [
        TestFile("cpu/test_activation.py"),
        TestFile("cpu/test_binding.py"),
        TestFile("cpu/test_causal_conv1d.py"),
        TestFile("cpu/test_cpu_graph.py"),
        TestFile("cpu/test_decode.py"),
        TestFile("cpu/test_extend.py"),
        TestFile("cpu/test_gemm.py"),
        TestFile("cpu/test_intel_amx_attention_backend_a.py"),
        TestFile("cpu/test_intel_amx_attention_backend_b.py"),
        TestFile("cpu/test_intel_amx_attention_backend_c.py"),
        TestFile("cpu/test_mamba.py"),
        TestFile("cpu/test_mla.py"),
        TestFile("cpu/test_moe.py"),
        TestFile("cpu/test_norm.py"),
        TestFile("cpu/test_qkv_proj_with_rope.py"),
        TestFile("cpu/test_qwen3.py"),
        TestFile("cpu/test_rope.py"),
        TestFile("cpu/test_shared_expert.py"),
        TestFile("cpu/test_topk.py"),
    ],
}

# Add Intel XPU tests
suite_xpu = {
    "per-commit-xpu": [
        TestFile("xpu/test_intel_xpu_backend.py"),
    ],
}

# Add Ascend NPU tests
# TODO: Set accurate estimate time
# NOTE: please sort the test cases alphabetically by the test file name
suite_ascend = {
    "per-commit-1-npu-a2": [
        TestFile("ascend/test_ascend_graph_tp1_bf16.py", 400),
        TestFile("ascend/test_ascend_piecewise_graph_prefill.py", 400),
        TestFile("ascend/test_ascend_hicache_mha.py", 400),
        TestFile("ascend/test_ascend_sampling_backend.py", 400),
        TestFile("ascend/test_ascend_tp1_bf16.py", 400),
        TestFile("ascend/test_ascend_compile_graph_tp1_bf16.py", 400),
        TestFile("ascend/test_ascend_w8a8_quantization.py", 400),
        TestFile("test_embed_interpolate_unittest.py", 400),
    ],
    "per-commit-2-npu-a2": [
        TestFile("ascend/test_ascend_graph_tp2_bf16.py", 400),
        TestFile("ascend/test_ascend_mla_fia_w8a8int8.py", 400),
        TestFile("ascend/test_ascend_tp2_bf16.py", 400),
        TestFile("ascend/test_ascend_tp2_fia_bf16.py", 400),
    ],
    "per-commit-4-npu-a2": [
        TestFile("ascend/test_ascend_mla_w8a8int8.py", 400),
        TestFile("ascend/test_ascend_hicache_mla.py", 400),
        TestFile("ascend/test_ascend_tp4_bf16.py", 400),
    ],
    "per-commit-16-npu-a3": [
        TestFile("ascend/test_ascend_deepep.py", 3600),
        TestFile("ascend/test_ascend_deepseek_mtp.py", 2800),
        TestFile("ascend/test_ascend_w4a4_quantization.py", 600),
    ],
}

suites.update(suite_amd)
suites.update(suite_xeon)
suites.update(suite_ascend)
suites.update(suite_xpu)


def auto_partition(files, rank, size):
    """
    Partition files into size sublists with approximately equal sums of estimated times
    using stable sorting, and return the partition for the specified rank.

    Args:
        files (list): List of file objects with estimated_time attribute
        rank (int): Index of the partition to return (0 to size-1)
        size (int): Number of partitions

    Returns:
        list: List of file objects in the specified rank's partition
    """
    weights = [f.estimated_time for f in files]

    if not weights or size <= 0 or size > len(weights):
        return []

    # Create list of (weight, original_index) tuples
    # Using negative index as secondary key to maintain original order for equal weights
    indexed_weights = [(w, -i) for i, w in enumerate(weights)]
    # Stable sort in descending order by weight
    # If weights are equal, larger (negative) index comes first (i.e., earlier original position)
    indexed_weights = sorted(indexed_weights, reverse=True)

    # Extract original indices (negate back to positive)
    indexed_weights = [(w, -i) for w, i in indexed_weights]

    # Initialize partitions and their sums
    partitions = [[] for _ in range(size)]
    sums = [0.0] * size

    # Greedy approach: assign each weight to partition with smallest current sum
    for weight, idx in indexed_weights:
        # Find partition with minimum sum
        min_sum_idx = sums.index(min(sums))
        partitions[min_sum_idx].append(idx)
        sums[min_sum_idx] += weight

    # Return the files corresponding to the indices in the specified rank's partition
    indices = partitions[rank]
    return [files[i] for i in indices]


def _sanity_check_suites(suites):
    dir_base = Path(__file__).parent
    disk_files = set(
        [
            str(x.relative_to(dir_base))
            for x in dir_base.glob("**/*.py")
            if x.name.startswith("test_")
        ]
    )

    suite_files = set(
        [test_file.name for _, suite in suites.items() for test_file in suite]
    )

    missing_files = sorted(list(disk_files - suite_files))
    missing_text = "\n".join(f'TestFile("{x}"),' for x in missing_files)
    assert len(missing_files) == 0, (
        f"Some test files are not in test suite. "
        f"If this is intentional, please add the following to `not_in_ci` section:\n"
        f"{missing_text}"
    )

    nonexistent_files = sorted(list(suite_files - disk_files))
    nonexistent_text = "\n".join(f'TestFile("{x}"),' for x in nonexistent_files)
    assert (
        len(nonexistent_files) == 0
    ), f"Some test files in test suite do not exist on disk:\n{nonexistent_text}"

    not_in_ci_files = set(
        [test_file.name for test_file in suites.get("__not_in_ci__", [])]
    )
    in_ci_files = set(
        [
            test_file.name
            for suite_name, suite in suites.items()
            if suite_name != "__not_in_ci__"
            for test_file in suite
        ]
    )
    intersection = not_in_ci_files & in_ci_files
    intersection_text = "\n".join(f'TestFile("{x}"),' for x in intersection)
    assert len(intersection) == 0, (
        f"Some test files are in both `not_in_ci` section and other suites:\n"
        f"{intersection_text}"
    )


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--timeout-per-file",
        type=int,
        default=1200,
        help="The time limit for running one file in seconds.",
    )
    arg_parser.add_argument(
        "--suite",
        type=str,
        default=list(suites.keys())[0],
        choices=list(suites.keys()) + ["all"],
        help="The suite to run",
    )
    arg_parser.add_argument(
        "--auto-partition-id",
        type=int,
        help="Use auto load balancing. The part id.",
    )
    arg_parser.add_argument(
        "--auto-partition-size",
        type=int,
        help="Use auto load balancing. The number of parts.",
    )
    arg_parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=False,
        help="Continue running remaining tests even if one fails (useful for nightly tests)",
    )
    arg_parser.add_argument(
        "--enable-retry",
        action="store_true",
        default=False,
        help="Enable smart retry for accuracy/performance assertion failures (not code errors)",
    )
    arg_parser.add_argument(
        "--max-attempts",
        type=int,
        default=2,
        help="Maximum number of attempts per file including initial run (default: 2)",
    )
    arg_parser.add_argument(
        "--retry-wait-seconds",
        type=int,
        default=60,
        help="Seconds to wait between retries (default: 60)",
    )
    arg_parser.add_argument(
        "--retry-timeout-increase",
        type=int,
        default=600,
        help="Additional timeout in seconds when retry is enabled (default: 600)",
    )
    args = arg_parser.parse_args()
    print(f"{args=}")

    _sanity_check_suites(suites)

    if args.suite == "all":
        files = glob.glob("**/test_*.py", recursive=True)
    else:
        files = suites[args.suite]

    if args.auto_partition_size:
        files = auto_partition(files, args.auto_partition_id, args.auto_partition_size)

    # Print test info at beginning (similar to test/run_suite.py pretty_print_tests)
    if args.auto_partition_size:
        partition_info = (
            f"{args.auto_partition_id + 1}/{args.auto_partition_size} "
            f"(0-based id={args.auto_partition_id})"
        )
    else:
        partition_info = "full"

    headers = ["Suite", "Partition"]
    rows = [[args.suite, partition_info]]
    msg = tabulate.tabulate(rows, headers=headers, tablefmt="psql") + "\n"

    total_est_time = sum(f.estimated_time for f in files)
    msg += f"✅ Enabled {len(files)} test(s) (est total {total_est_time:.1f}s):\n"
    for f in files:
        msg += f"  - {f.name} (est_time={f.estimated_time})\n"

    print(msg, flush=True)

    # Add extra timeout when retry is enabled
    timeout = args.timeout_per_file
    if args.enable_retry:
        timeout += args.retry_timeout_increase

    exit_code = run_unittest_files(
        files,
        timeout,
        args.continue_on_error,
        args.enable_retry,
        args.max_attempts,
        args.retry_wait_seconds,
    )

    # Print tests again at the end for visibility
    msg = "\n" + tabulate.tabulate(rows, headers=headers, tablefmt="psql") + "\n"
    msg += f"✅ Executed {len(files)} test(s) (est total {total_est_time:.1f}s):\n"
    for f in files:
        msg += f"  - {f.name} (est_time={f.estimated_time})\n"
    print(msg, flush=True)

    exit(exit_code)


if __name__ == "__main__":
    main()
