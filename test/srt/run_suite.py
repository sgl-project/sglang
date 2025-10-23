import argparse
import glob
from dataclasses import dataclass

from sglang.test.test_utils import run_unittest_files


@dataclass
class TestFile:
    name: str
    estimated_time: float = 60


# NOTE: please sort the test cases alphabetically by the test file name
suites = {
    "per-commit-1-gpu": [
        TestFile("function_call/test_json_schema_constraint.py", 30),
        TestFile("hicache/test_hicache.py", 116),
        TestFile("hicache/test_hicache_eagle.py", 150),
        TestFile("hicache/test_hicache_mla.py", 127),
        TestFile("hicache/test_hicache_storage.py", 127),
        TestFile("layers/attention/mamba/test_causal_conv1d.py", 25),
        TestFile("layers/attention/mamba/test_mamba_ssm.py", 50),
        TestFile("layers/attention/mamba/test_mamba_ssm_ssd.py", 70),
        TestFile("lora/test_lora.py", 200),
        TestFile("lora/test_lora_eviction.py", 200),
        TestFile("lora/test_lora_eviction_policy.py", 200),
        TestFile("lora/test_lora_backend.py", 99),
        TestFile("lora/test_lora_eviction.py", 200),
        TestFile("lora/test_lora_qwen3.py", 97),
        TestFile("lora/test_lora_radix_cache.py", 100),
        TestFile("lora/test_lora_openai_api.py", 30),
        TestFile("lora/test_lora_update.py", 400),
        TestFile("lora/test_multi_lora_backend.py", 60),
        TestFile("models/test_compressed_tensors_models.py", 42),
        TestFile("models/test_cross_encoder_models.py", 100),
        TestFile("models/test_embedding_models.py", 73),
        TestFile("models/test_encoder_embedding_models.py", 460),
        TestFile("models/test_generation_models.py", 103),
        TestFile("models/test_nvidia_nemotron_nano_v2.py", 300),
        TestFile("models/test_qwen_models.py", 82),
        TestFile("batch_invariant/test_batch_invariant_ops.py", 10),
        TestFile("models/test_reward_models.py", 132),
        TestFile("models/test_transformers_models.py", 320),
        TestFile("models/test_vlm_models.py", 741),
        TestFile("openai_server/basic/test_openai_embedding.py", 141),
        TestFile("openai_server/basic/test_openai_server.py", 149),
        TestFile("openai_server/basic/test_protocol.py", 10),
        TestFile("openai_server/basic/test_serving_chat.py", 10),
        TestFile("openai_server/basic/test_serving_completions.py", 10),
        TestFile("openai_server/basic/test_serving_embedding.py", 10),
        TestFile("openai_server/features/test_enable_thinking.py", 70),
        TestFile("openai_server/features/test_json_constrained.py", 98),
        TestFile("openai_server/features/test_json_mode.py", 90),
        TestFile("openai_server/features/test_openai_server_ebnf.py", 95),
        TestFile("openai_server/features/test_openai_server_hidden_states.py", 240),
        TestFile("openai_server/features/test_reasoning_content.py", 89),
        TestFile("openai_server/features/test_lora_openai_compatible.py", 120),
        TestFile("openai_server/function_call/test_openai_function_calling.py", 60),
        TestFile("openai_server/function_call/test_tool_choice.py", 226),
        TestFile("openai_server/validation/test_large_max_new_tokens.py", 41),
        TestFile("openai_server/validation/test_matched_stop.py", 60),
        TestFile("openai_server/validation/test_openai_server_ignore_eos.py", 85),
        TestFile("openai_server/validation/test_request_length_validation.py", 31),
        TestFile("quant/test_block_int8.py", 22),
        TestFile("quant/test_fp8_kernel.py", 8),
        TestFile("quant/test_int8_kernel.py", 8),
        TestFile("quant/test_triton_scaled_mm.py", 8),
        TestFile("quant/test_w8a8_quantization.py", 46),
        TestFile("rl/test_fp32_lm_head.py", 30),
        TestFile("rl/test_update_weights_from_disk.py", 114),
        TestFile("rl/test_update_weights_from_tensor.py", 48),
        TestFile("test_abort.py", 51),
        TestFile("test_build_eagle_tree.py", 8),
        TestFile("test_chunked_prefill.py", 313),
        TestFile("test_create_kvindices.py", 2),
        TestFile("test_deterministic.py", 320),
        TestFile("test_eagle_infer_a.py", 370),
        TestFile("test_eagle_infer_b.py", 700),
        TestFile("test_eagle_infer_beta.py", 300),
        TestFile("test_ebnf_constrained.py", 108),
        TestFile("test_eval_fp8_accuracy.py", 303),
        TestFile("test_fa3.py", 376),
        # TestFile("test_flashmla.py", 352),
        TestFile("rotary_embedding/test_mrope.py", 300),
        TestFile("test_function_call_parser.py", 10),
        TestFile("test_fused_moe.py", 30),
        TestFile("test_gpt_oss_1gpu.py", 600),
        TestFile("test_harmony_parser.py", 20),
        TestFile("test_hidden_states.py", 55),
        TestFile("test_hybrid_attn_backend.py", 379),
        TestFile("test_input_embeddings.py", 38),
        TestFile("test_io_struct.py", 8),
        TestFile("test_jinja_template_utils.py", 1),
        TestFile("test_mamba_unittest.py", 4),
        TestFile("test_metrics.py", 32),
        TestFile("test_metrics_utils.py", 1),
        TestFile("test_mla.py", 167),
        TestFile("test_mla_deepseek_v3.py", 500),
        TestFile("test_mla_flashinfer.py", 302),
        TestFile("test_mla_fp8.py", 93),
        TestFile("test_mla_int8_deepseek_v3.py", 429),
        TestFile("test_modelopt_loader.py", 30),
        TestFile("test_multi_tokenizer.py", 230),
        TestFile("test_ngram_speculative_decoding.py", 250),
        TestFile("test_no_chunked_prefill.py", 108),
        TestFile("test_no_overlap_scheduler.py", 234),
        TestFile("test_original_logprobs.py", 41),
        TestFile("test_page_size.py", 60),
        TestFile("test_penalty.py", 41),
        TestFile("test_priority_scheduling.py", 100),
        TestFile("test_pytorch_sampling_backend.py", 66),
        TestFile("test_radix_attention.py", 105),
        TestFile("test_radix_cache_unit.py", 5),
        TestFile("test_reasoning_parser.py", 5),
        TestFile("test_regex_constrained.py", 64),
        TestFile("test_request_queue_validation.py", 30),
        TestFile("test_retract_decode.py", 90),
        TestFile("test_score_api.py", 310),
        TestFile("test_server_args.py", 1),
        TestFile("test_skip_tokenizer_init.py", 117),
        TestFile("test_srt_endpoint.py", 130),
        TestFile("test_srt_engine.py", 261),
        TestFile("test_standalone_speculative_decoding.py", 250),
        TestFile("test_start_profile.py", 60),
        TestFile("test_profile_merger.py", 60),
        TestFile("test_profile_merger_http_api.py", 15),
        TestFile("test_swa_unittest.py", 1),
        TestFile("test_torch_compile.py", 76),
        TestFile("test_torch_compile_moe.py", 172),
        TestFile("test_torch_native_attention_backend.py", 123),
        TestFile("test_torchao.py", 70),
        TestFile("test_triton_attention_backend.py", 150),
        TestFile("test_triton_attention_kernels.py", 4),
        TestFile("test_triton_moe_channel_fp8_kernel.py", 25),
        TestFile("test_triton_sliding_window.py", 250),
        TestFile("test_utils_update_weights.py", 48),
        TestFile("test_vision_chunked_prefill.py", 175),
        TestFile("test_vision_openai_server_a.py", 918),
        TestFile("test_vlm_input_format.py", 300),
        TestFile("test_modelopt_loader.py", 30),
        TestFile("test_modelopt_export.py", 30),
    ],
    "per-commit-2-gpu": [
        TestFile("ep/test_moe_ep.py", 140),
        TestFile("hicache/test_hicache_storage_3fs_backend.py", 200),
        TestFile("hicache/test_hicache_storage_file_backend.py", 200),
        TestFile("hicache/test_hicache_storage_mooncake_backend.py", 400),
        TestFile("layers/attention/mamba/test_mamba2_mixer.py", 50),
        TestFile("lora/test_lora_tp.py", 116),
        TestFile("models/test_glm4_moe_models.py", 100),
        TestFile("rl/test_update_weights_from_distributed.py", 103),
        TestFile("test_data_parallelism.py", 73),
        TestFile("test_disaggregation_basic.py", 400),
        TestFile("test_dp_attention.py", 594),
        TestFile("test_load_weights_from_remote_instance.py", 72),
        TestFile("test_patch_torch.py", 19),
        TestFile("test_release_memory_occupation.py", 257),
    ],
    "per-commit-4-gpu": [
        TestFile("models/test_qwen3_next_models.py", 291),
        TestFile("test_disaggregation_dp_attention.py", 155),
        TestFile("test_gpt_oss_4gpu.py", 300),
        TestFile("test_local_attn.py", 411),
        TestFile("test_multi_instance_release_memory_occupation.py", 64),
        TestFile("test_pp_single_node.py", 481),
    ],
    "per-commit-8-gpu-h200": [
        TestFile("lora/test_lora_llama4.py", 400),
        TestFile("test_deepseek_v3_basic.py", 275),
        TestFile("test_deepseek_v3_mtp.py", 275),
        TestFile("test_disaggregation_hybrid_attention.py", 200),
    ],
    "per-commit-8-gpu-h20": [
        TestFile("quant/test_w4a8_deepseek_v3.py", 371),
        TestFile("test_disaggregation_different_tp.py", 600),
        TestFile("test_disaggregation_pp.py", 140),
    ],
    "per-commit-4-gpu-b200": [
        # TestFile("test_flash_attention_4.py"),
        # TestFile("test_gpt_oss_4gpu.py", 600),
        # TestFile("test_deepseek_v3_fp4_4gpu.py", 3600),
    ],
    "per-commit-4-gpu-deepep": [
        TestFile("ep/test_deepep_small.py", 531),
        TestFile("ep/test_mooncake_ep_small.py", 450),
    ],
    "per-commit-8-gpu-h200-deepep": [
        TestFile("ep/test_deepep_large.py", 338),
    ],
    "per-commit-8-gpu-h200-deepseek-v32": [
        TestFile("test_deepseek_v32_basic.py", 275),
        TestFile("test_deepseek_v32_mtp.py", 275),
    ],
    "vllm_dependency_test": [
        TestFile("quant/test_awq.py", 163),
        TestFile("test_bnb.py", 5),
        TestFile("test_gptqmodel_dynamic.py", 102),
        TestFile("test_vllm_dependency.py", 185),
        # TestFile("test_gguf.py", 96),
    ],
    # If the test cases take too long, considering adding them to nightly tests instead of per-commit tests
    "nightly-1-gpu": [],
    "nightly-8-gpu": [],
}

# Add AMD tests
# NOTE: please sort the test cases alphabetically by the test file name
suite_amd = {
    "per-commit-amd": [
        TestFile("function_call/test_json_schema_constraint.py", 30),
        TestFile("hicache/test_hicache.py", 116),
        TestFile("hicache/test_hicache_mla.py", 127),
        TestFile("hicache/test_hicache_storage.py", 127),
        TestFile("lora/test_lora.py", 200),
        TestFile("lora/test_lora_backend.py", 99),
        TestFile("lora/test_lora_cuda_graph.py", 250),
        TestFile("lora/test_lora_eviction.py", 200),
        TestFile("lora/test_lora_qwen3.py", 97),
        TestFile("lora/test_multi_lora_backend.py", 60),
        TestFile("models/test_compressed_tensors_models.py", 42),
        TestFile("models/test_qwen_models.py", 82),
        TestFile("models/test_reward_models.py", 132),
        TestFile("models/test_transformers_models.py", 320),
        TestFile("openai_server/basic/test_openai_embedding.py", 141),
        TestFile("openai_server/basic/test_openai_server.py", 149),
        TestFile("openai_server/basic/test_protocol.py", 10),
        TestFile("openai_server/basic/test_serving_chat.py", 10),
        TestFile("openai_server/basic/test_serving_completions.py", 10),
        TestFile("openai_server/basic/test_serving_embedding.py", 10),
        TestFile("openai_server/features/test_enable_thinking.py", 70),
        TestFile("openai_server/features/test_json_constrained.py", 98),
        TestFile("openai_server/features/test_json_mode.py", 90),
        TestFile("openai_server/features/test_openai_server_ebnf.py", 95),
        TestFile("openai_server/features/test_reasoning_content.py", 89),
        TestFile("openai_server/function_call/test_openai_function_calling.py", 60),
        TestFile("openai_server/function_call/test_tool_choice.py", 226),
        TestFile("openai_server/validation/test_large_max_new_tokens.py", 41),
        TestFile("openai_server/validation/test_matched_stop.py", 60),
        TestFile("openai_server/validation/test_openai_server_ignore_eos.py", 85),
        TestFile("openai_server/validation/test_request_length_validation.py", 31),
        TestFile("quant/test_awq_dequant.py", 2),
        TestFile("quant/test_block_int8.py", 22),
        TestFile("rl/test_update_weights_from_disk.py", 114),
        TestFile("test_abort.py", 51),
        TestFile("test_chunked_prefill.py", 313),
        TestFile("test_create_kvindices.py", 2),
        TestFile("test_ebnf_constrained.py", 108),
        TestFile("test_eval_fp8_accuracy.py", 303),
        TestFile("test_function_call_parser.py", 10),
        TestFile("test_fused_moe.py", 30),
        TestFile("test_input_embeddings.py", 38),
        TestFile("test_io_struct.py", 8),
        TestFile("test_jinja_template_utils.py", 1),
        TestFile("test_metrics.py", 32),
        TestFile("test_metrics_utils.py", 1),
        TestFile("test_mla.py", 242),
        TestFile("test_mla_deepseek_v3.py", 221),
        TestFile("test_no_chunked_prefill.py", 108),
        TestFile("test_page_size.py", 60),
        TestFile("test_penalty.py", 41),
        TestFile("test_pytorch_sampling_backend.py", 66),
        TestFile("test_radix_attention.py", 105),
        TestFile("test_reasoning_parser.py", 5),
        TestFile("test_regex_constrained.py", 64),
        TestFile("test_retract_decode.py", 54),
        TestFile("test_rope_rocm.py", 3),
        TestFile("test_server_args.py", 1),
        TestFile("test_skip_tokenizer_init.py", 117),
        TestFile("test_srt_endpoint.py", 130),
        TestFile("test_srt_engine.py", 261),
        TestFile("test_torch_compile.py", 169),
        TestFile("test_torch_compile_moe.py", 172),
        TestFile("test_torch_native_attention_backend.py", 123),
        TestFile("test_triton_attention_backend.py", 150),
        TestFile("test_wave_attention_kernels.py", 2),
        # Disabled temporarily
        # TestFile("models/test_embedding_models.py", 73), # Disabled temporarily, see https://github.com/sgl-project/sglang/issues/11127
        # TestFile("openai_server/features/test_openai_server_hidden_states.py", 240),
        # TestFile("rl/test_update_weights_from_tensor.py", 48),
        # TestFile("test_no_overlap_scheduler.py", 234), # Disabled temporarily and track in #7703
        # TestFile("test_vision_chunked_prefill.py", 175), # Disabled temporarily and track in #7701
        # TestFile("test_wave_attention_backend.py", 150), # Disabled temporarily, see https://github.com/sgl-project/sglang/issues/11127
    ],
    "per-commit-amd-mi35x": [
        TestFile("test_gpt_oss_1gpu.py", 600),
        TestFile("test_mla.py", 242),
    ],
    "per-commit-2-gpu-amd": [
        TestFile("lora/test_lora_tp.py", 116),
        TestFile("rl/test_update_weights_from_distributed.py", 103),
        TestFile("test_data_parallelism.py", 73),
        TestFile("test_load_weights_from_remote_instance.py", 72),
        # TestFile("test_patch_torch.py", 19), # Disabled temporarily, see https://github.com/sgl-project/sglang/issues/11127
    ],
    "per-commit-4-gpu-amd": [
        TestFile("test_pp_single_node.py", 150),
    ],
    "per-commit-8-gpu-amd": [
        TestFile("test_deepseek_v3_basic.py", 275),
        TestFile("test_deepseek_v3_mtp.py", 275),
    ],
    "nightly-amd": [
        TestFile("test_nightly_gsm8k_eval_amd.py"),
    ],
}

# Add Intel Xeon tests
# NOTE: please sort the test cases alphabetically by the test file name
suite_xeon = {
    "per-commit-cpu": [
        TestFile("cpu/test_activation.py"),
        TestFile("cpu/test_binding.py"),
        TestFile("cpu/test_decode.py"),
        TestFile("cpu/test_extend.py"),
        TestFile("cpu/test_gemm.py"),
        TestFile("cpu/test_mla.py"),
        TestFile("cpu/test_moe.py"),
        TestFile("cpu/test_norm.py"),
        TestFile("cpu/test_qkv_proj_with_rope.py"),
        TestFile("cpu/test_rope.py"),
        TestFile("cpu/test_shared_expert.py"),
        TestFile("cpu/test_topk.py"),
        TestFile("cpu/test_cpu_graph.py"),
        TestFile("cpu/test_intel_amx_attention_backend_a.py"),
        TestFile("cpu/test_intel_amx_attention_backend_b.py"),
        TestFile("cpu/test_intel_amx_attention_backend_c.py"),
    ],
}

# Add Intel XPU tests
suite_xpu = {
    "per-commit-xpu": [
        TestFile("xpu/test_intel_xpu_backend.py"),
    ],
}

# Add Ascend NPU tests
# NOTE: please sort the test cases alphabetically by the test file name
suite_ascend = {
    "per-commit-1-ascend-npu": [
        TestFile("ascend/test_ascend_graph_tp1_bf16.py", 400),
        TestFile("ascend/test_ascend_tp1_bf16.py", 400),
    ],
    "per-commit-2-ascend-npu": [
        TestFile("ascend/test_ascend_graph_tp2_bf16.py", 400),
        TestFile("ascend/test_ascend_mla_fia_w8a8int8.py", 400),
        TestFile("ascend/test_ascend_tp2_bf16.py", 400),
        TestFile("ascend/test_ascend_tp2_fia_bf16.py", 400),
    ],
    "per-commit-4-ascend-npu": [
        TestFile("ascend/test_ascend_mla_w8a8int8.py", 400),
        TestFile("ascend/test_ascend_tp4_bf16.py", 400),
    ],
    "per-commit-16-ascend-a3": [
        TestFile("ascend/test_ascend_deepep.py", 400),
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


if __name__ == "__main__":
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
