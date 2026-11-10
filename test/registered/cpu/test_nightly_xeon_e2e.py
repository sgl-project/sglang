"""Xeon nightly model evaluations and full-model performance benchmarks.

Reuse main-branch model tests without pulling their accompanying kernel/config
unit tests into nightly. Accuracy tests launch real serving processes; performance
tests retain the existing offline bench_one_batch measurement and thresholds.
"""

import os
import sys
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2400, suite="nightly-xeon-e2e", nightly=True)

NIGHTLY_TEST_GROUPS = (
    # Existing file estimates: 685 + 47 + 183 = 915 seconds.
    (
        "registered.cpu.test_intel_amx_attention_backend_a.TestIntelAMXAttnBackend",
        "registered.cpu.test_intel_amx_attention_backend_a.TestDPAttention",
        "registered.cpu.test_intel_amx_attention_backend_b.TestIntelAMXAttnBackendQuant",
        "registered.cpu.quant.test_autoround.TestAutoRoundCPU",
    ),
    # Existing file estimates: 477 + 315 + 206 = 998 seconds.
    (
        "registered.cpu.test_intel_amx_attention_backend_c.TestIntelAMXAttnBackendQuant",
        "registered.cpu.test_cpu_graph.TestCPUGraph.test_latency_torch_compile_cpu",
        "registered.cpu.test_cpu_graph.TestCPUGraph.test_mmlu_torch_compile_cpu",
        "registered.e2e.models.test_transformers_backend_eval.TestTransformersBackendEval",
    ),
)


def selected_tests():
    """Select one full-machine shard, or all tests for a standalone run."""
    shard = os.environ.get("SGLANG_XEON_NIGHTLY_SHARD")
    if shard is None:
        return tuple(name for group in NIGHTLY_TEST_GROUPS for name in group)
    if shard not in ("0", "1"):
        raise ValueError("SGLANG_XEON_NIGHTLY_SHARD must be 0 or 1")
    return NIGHTLY_TEST_GROUPS[int(shard)]


def load_tests(loader, tests, pattern):
    # run_suite executes this file directly; make test/ importable for the
    # explicit test identifiers, including namespace-package directories.
    test_root = str(Path(__file__).resolve().parents[2])
    if test_root not in sys.path:
        sys.path.insert(0, test_root)
    names = selected_tests()
    print(f"Xeon nightly shard: {os.environ.get('SGLANG_XEON_NIGHTLY_SHARD', 'all')}")
    return loader.loadTestsFromNames(names)


if __name__ == "__main__":
    # run_suite normally supplies -f. Nightly should collect outcomes for every
    # model, while unittest still exits nonzero if any evaluation fails.
    unittest.main(argv=[arg for arg in sys.argv if arg not in ("-f", "--failfast")])
