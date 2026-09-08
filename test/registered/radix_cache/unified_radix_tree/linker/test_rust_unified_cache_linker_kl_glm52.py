"""Run the GLM-5.2 external-linker E2E with the Rust TreeCore."""

import unittest

import test_unified_cache_linker_kl_glm52 as glm52_linker

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=383, stage="extra-b", runner_config="8-gpu-h200")


class TestRustGLM52UnifiedCacheLinkerKL(glm52_linker.TestGLM52UnifiedCacheLinkerKL):
    tree_core_backend = "rust"


if __name__ == "__main__":
    unittest.main()
