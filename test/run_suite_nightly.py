import argparse
import os
import sys
from pathlib import Path

from sglang.test.ci.ci_utils import TestFile, run_unittest_files

# Nightly test suites
suites = {
    "nightly-1-gpu": [
        TestFile("test_nsa_indexer.py", 2),
        TestFile("test_lora_qwen3.py", 97),
        TestFile("test_lora_radix_cache.py", 200),
        TestFile("test_lora_eviction_policy.py", 200),
        TestFile("test_lora_openai_api.py", 30),
        TestFile("test_lora_openai_compatible.py", 150),
        TestFile("test_lora_hf_sgl_logprob_diff.py", 300),
        TestFile("test_batch_invariant_ops.py", 10),
        TestFile("test_cpp_radix_cache.py", 60),
        TestFile("test_deepseek_v3_deterministic.py", 240),
    ],
    "nightly-4-gpu-b200": [
        TestFile("test_flashinfer_trtllm_gen_moe_backend.py", 300),
        TestFile("test_gpt_oss_4gpu_perf.py", 600),
        TestFile("test_flashinfer_trtllm_gen_attn_backend.py", 300),
        TestFile("test_deepseek_v3_fp4_cutlass_moe.py", 900),
        TestFile("test_fp4_moe.py", 300),
        TestFile("test_qwen3_fp4_trtllm_gen_moe.py", 300),
        TestFile("test_eagle_infer_beta_dp_attention_large.py", 600),
    ],
    "nightly-8-gpu-b200": [
        TestFile("test_deepseek_r1_fp8_trtllm_backend.py", 3600),
    ],
    "nightly-4-gpu": [
        TestFile("test_encoder_dp.py", 500),
        TestFile("test_qwen3_next_deterministic.py", 200),
    ],
    "nightly-8-gpu": [],
    "nightly-8-gpu-h200": [
        TestFile("test_deepseek_v32_nsabackend.py", 600),
    ],
    "nightly-8-gpu-h20": [],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        type=str,
        required=True,
        help="Test suite to run (e.g., nightly-1-gpu, nightly-4-gpu, etc.).",
    )
    parser.add_argument(
        "--timeout-per-file",
        type=int,
        default=1200,
        help="The time limit for running one file in seconds (default: 1200).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=False,
        help="Continue running remaining tests even if one fails (default: False, useful for nightly tests).",
    )
    args = parser.parse_args()

    if args.suite not in suites:
        print(f"Error: Suite '{args.suite}' not found in available suites")
        print(f"Available suites: {list(suites.keys())}")
        exit(1)

    files = suites[args.suite]

    # Change directory to test/nightly where the test files are located
    nightly_dir = Path(__file__).parent / "nightly"
    os.chdir(nightly_dir)

    # Add test/ to PYTHONPATH so tests can import shared utils
    test_dir = str(Path(__file__).parent)
    pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{test_dir}:{pythonpath}" if pythonpath else test_dir

    print(f"Running {len(files)} tests from suite: {args.suite}")
    print(f"Test files: {[f.name for f in files]}")

    exit_code = run_unittest_files(
        files,
        timeout_per_file=args.timeout_per_file,
        continue_on_error=args.continue_on_error,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
