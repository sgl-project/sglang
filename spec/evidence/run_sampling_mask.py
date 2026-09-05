"""Run sampling-mask server tests across backends and overlap modes."""

import argparse
import importlib.util
import sys
import unittest
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--deterministic-only",
        action="store_true",
        help="Run only seeded sampling parity.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "sampling_mask_tests", root / "test/registered/sampling/test_sampling_mask.py"
    )
    tests = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tests)
    tests.DEFAULT_SMALL_MODEL_NAME_FOR_TEST = args.model
    # Compile sampling kernels before the endpoint tests' request timeouts start.
    from flashinfer.sampling import get_sampling_module

    get_sampling_module()
    # Exercise prefill/decode graphs and overlap with a small cache footprint.
    server_args = tests._SERVER_ARGS + (
        "--cuda-graph-max-bs-prefill",
        "32",
        "--cuda-graph-max-bs-decode",
        "8",
        "--max-total-tokens",
        "4096",
    )
    passed = True
    if not args.deterministic_only:
        for overlap in (True, False):
            tests._SERVER_ARGS = server_args + (
                () if overlap else ("--disable-overlap-schedule",)
            )
            for case in (tests.TestSamplingMask, tests.TestSamplingMaskPytorch):
                print(
                    f"backend={case._sampling_backend}, overlap={overlap}", flush=True
                )
                suite = unittest.defaultTestLoader.loadTestsFromTestCase(case)
                result = unittest.TextTestRunner(verbosity=2).run(suite)
                passed &= result.wasSuccessful()
    tests._SERVER_ARGS = server_args
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(
            tests.TestSamplingMaskDeterministic
        )
    )
    return 0 if passed and result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
