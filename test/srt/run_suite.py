import argparse
import glob
from dataclasses import dataclass

from sglang.test.test_utils import run_unittest_files


@dataclass
class TestFile:
    name: str
    estimated_time: float = 60


suites = {
    "per-commit": [
        TestFile("models/lora/test_lora.py", 76),
        TestFile("models/lora/test_lora_backend.py", 420),
        TestFile("models/lora/test_multi_lora_backend.py", 60),
        TestFile("models/test_embedding_models.py", 35),
        TestFile("models/test_generation_models.py", 103),
        TestFile("models/test_grok_models.py", 60),
        TestFile("models/test_qwen_models.py", 82),
        TestFile("models/test_reward_models.py", 83),
        TestFile("models/test_gme_qwen_models.py", 45),
        TestFile("models/test_clip_models.py", 100),
        TestFile("test_abort.py", 51),
        TestFile("test_block_int8.py", 22),
        TestFile("test_chunked_prefill.py", 336),
        TestFile("test_eagle_infer.py", 500),
        TestFile("test_ebnf_constrained.py"),
        TestFile("test_fa3.py", 5),
        TestFile("test_fp8_kernel.py", 8),
        TestFile("test_embedding_openai_server.py", 36),
        TestFile("test_hidden_states.py", 55),
        TestFile("test_int8_kernel.py", 8),
        TestFile("test_input_embeddings.py", 38),
        TestFile("test_json_constrained.py", 98),
        TestFile("test_large_max_new_tokens.py", 41),
        TestFile("test_metrics.py", 32),
        TestFile("test_mla.py", 162),
        TestFile("test_mla_deepseek_v3.py", 221),
        TestFile("test_mla_int8_deepseek_v3.py", 522),
        TestFile("test_mla_flashinfer.py", 395),
        TestFile("test_mla_fp8.py", 93),
        TestFile("test_no_chunked_prefill.py", 126),
        TestFile("test_no_overlap_scheduler.py", 262),
        TestFile("test_openai_server.py", 124),
        TestFile("test_penalty.py", 41),
        TestFile("test_page_size.py", 60),
        TestFile("test_pytorch_sampling_backend.py", 66),
        TestFile("test_radix_attention.py", 167),
        TestFile("test_reasoning_content.py", 89),
        TestFile("test_regex_constrained.py", 64),
        TestFile("test_release_memory_occupation.py", 44),
        TestFile("test_request_length_validation.py", 31),
        TestFile("test_retract_decode.py", 54),
        TestFile("test_server_args.py", 1),
        TestFile("test_skip_tokenizer_init.py", 72),
        TestFile("test_srt_engine.py", 237),
        TestFile("test_srt_endpoint.py", 94),
        TestFile("test_torch_compile.py", 76),
        TestFile("test_torch_compile_moe.py", 85),
        TestFile("test_torch_native_attention_backend.py", 123),
        TestFile("test_torchao.py", 70),
        TestFile("test_triton_attention_kernels.py", 4),
        TestFile("test_triton_attention_backend.py", 134),
        TestFile("test_update_weights_from_disk.py", 114),
        TestFile("test_update_weights_from_tensor.py", 48),
        TestFile("test_vertex_endpoint.py", 31),
        TestFile("test_vision_chunked_prefill.py", 223),
        TestFile("test_vlm_accuracy.py", 60),
        TestFile("test_vision_openai_server.py", 537),
        TestFile("test_fim_completion.py", 40),
        TestFile("test_w8a8_quantization.py", 46),
        TestFile("test_eval_fp8_accuracy.py", 303),
        TestFile("test_create_kvindices.py", 2),
        TestFile("test_hicache.py", 60),
        TestFile("test_hicache_mla.py", 90),
        TestFile("test_fused_moe.py", 30),
        TestFile("test_triton_moe_channel_fp8_kernel.py", 25),
    ],
    "per-commit-2-gpu": [
        TestFile("models/lora/test_lora_tp.py", 300),
        TestFile("test_data_parallelism.py", 90),
        TestFile("test_dp_attention.py", 90),
        TestFile("test_mla_tp.py", 420),
        TestFile("test_moe_ep.py", 220),
        TestFile("test_patch_torch.py", 30),
        TestFile("test_update_weights_from_distributed.py", 100),
        TestFile("test_verl_engine.py", 100),
    ],
    "nightly": [
        TestFile("test_nightly_gsm8k_eval.py"),
    ],
    "vllm_dependency_test": [
        TestFile("test_vllm_dependency.py"),
        TestFile("test_awq.py"),
        TestFile("test_gguf.py", 78),
        TestFile("test_gptqmodel_dynamic.py", 72),
        TestFile("test_bnb.py"),
    ],
}


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


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--timeout-per-file",
        type=int,
        default=1800,
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
        "--range-begin",
        type=int,
        default=0,
        help="The begin index of the range of the files to run.",
    )
    arg_parser.add_argument(
        "--range-end",
        type=int,
        default=None,
        help="The end index of the range of the files to run.",
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
    args = arg_parser.parse_args()
    print(f"{args=}")

    if args.suite == "all":
        files = glob.glob("**/test_*.py", recursive=True)
    else:
        files = suites[args.suite]

    if args.auto_partition_size:
        files = auto_partition(files, args.auto_partition_id, args.auto_partition_size)
    else:
        files = files[args.range_begin : args.range_end]

    print("The running tests are ", [f.name for f in files])

    exit_code = run_unittest_files(files, args.timeout_per_file)
    exit(exit_code)
