import argparse
import glob

from sglang.test.test_utils import run_unittest_files

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
        # Disabled temporarily
        # "test_session_control.py",
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
    ],
    "nightly": [
        "test_nightly_gsm8k_eval.py",
        # Disable temporarily
        # "test_nightly_math_eval.py",
    ],
}

# Expand suite
for target_suite_name, target_tests in suites.items():
    for suite_name, tests in suites.items():
        if suite_name == target_suite_name:
            continue
        if target_suite_name in tests:
            tests.remove(target_suite_name)
            tests.extend(target_tests)

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
    args = arg_parser.parse_args()

    if args.suite == "all":
        files = glob.glob("**/test_*.py", recursive=True)
    else:
        files = suites[args.suite]

    files = files[args.range_begin : args.range_end]

    print(f"{args=}")
    print("The running tests are ", files)

    exit_code = run_unittest_files(files, args.timeout_per_file)
    exit(exit_code)
