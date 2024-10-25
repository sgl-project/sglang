import argparse
import glob

from sglang.test.test_utils import run_unittest_files

suites = {
    "minimal": [
        "models/test_embedding_models.py",
        "models/test_generation_models.py",
        "models/test_lora.py",
        "models/test_reward_models.py",
        "sampling/penaltylib",
        "test_chunked_prefill.py",
        "test_double_sparsity.py",
        "test_embedding_openai_server.py",
        "test_eval_accuracy_mini.py",
        "test_json_constrained.py",
        "test_large_max_new_tokens.py",
        "test_openai_server.py",
        "test_overlap_schedule.py",
        "test_pytorch_sampling_backend.py",
        "test_retract_decode.py",
        "test_server_args.py",
        "test_skip_tokenizer_init.py",
        "test_srt_engine.py",
        "test_srt_endpoint.py",
        "test_torch_compile.py",
        "test_torchao.py",
        "test_triton_attn_backend.py",
        "test_update_weights.py",
        "test_vision_openai_server.py",
    ],
    "sampling/penaltylib": glob.glob(
        "sampling/penaltylib/**/test_*.py", recursive=True
    ),
}

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
        default=2000,
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

    print("The running tests are ", files)

    exit_code = run_unittest_files(files, args.timeout_per_file)
    exit(exit_code)
