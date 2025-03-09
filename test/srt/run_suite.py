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
        "models/lora/test_lora.py",
        "models/lora/test_lora_backend.py",
        "models/lora/test_multi_lora_backend.py",
        "models/test_embedding_models.py",
        "models/test_generation_models.py",
        "models/test_qwen_models.py",
        "models/test_reward_models.py",
        "test_gptqmodel_dynamic.py",
        "models/test_gme_qwen_models.py",
        "test_abort.py",
        "test_chunked_prefill.py",
        "test_custom_allreduce.py",
        "test_double_sparsity.py",
        "test_eagle_infer.py",
        "test_embedding_openai_server.py",
        "test_eval_accuracy_mini.py",
        "test_gguf.py",
        "test_input_embeddings.py",
        "test_mla.py",
        "test_mla_deepseek_v3.py",
        "test_mla_flashinfer.py",
        "test_mla_fp8.py",
        "test_json_constrained.py",
        "test_large_max_new_tokens.py",
        "test_metrics.py",
        "test_no_chunked_prefill.py",
        "test_no_overlap_scheduler.py",
        "test_openai_server.py",
        "test_penalty.py",
        "test_pytorch_sampling_backend.py",
        "test_radix_attention.py",
        "test_regex_constrained.py",
        "test_release_memory_occupation.py",
        "test_request_length_validation.py",
        "test_retract_decode.py",
        "test_server_args.py",
        "test_skip_tokenizer_init.py",
        "test_srt_engine.py",
        "test_srt_endpoint.py",
        "test_torch_compile.py",
        "test_torch_compile_moe.py",
        "test_torch_native_attention_backend.py",
        "test_torchao.py",
        "test_triton_attention_kernels.py",
        "test_triton_attention_backend.py",
        "test_hidden_states.py",
        "test_update_weights_from_disk.py",
        "test_update_weights_from_tensor.py",
        "test_vertex_endpoint.py",
        "test_vision_chunked_prefill.py",
        "test_vision_llm.py",
        "test_vision_openai_server.py",
        "test_w8a8_quantization.py",
        "test_fp8_kernel.py",
        "test_block_int8.py",
        "test_int8_kernel.py",
        "test_reasoning_content.py",
    ],
    "nightly": [
        "test_nightly_gsm8k_eval.py",
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
