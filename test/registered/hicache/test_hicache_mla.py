"""MLA class of test_hicache_variants.py, kept on H100: DeepSeek-Coder-V2-Lite
(~31 GB bf16) does not fit a 32 GB card."""

import unittest

import test_hicache_variants as base

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import MGSMEnMixin, MMLUMixin
from sglang.test.test_utils import DEFAULT_MLA_MODEL_NAME_FOR_TEST

register_cuda_ci(est_time=150, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=150, suite="stage-b-test-1-gpu-small-amd")


class TestHiCacheMLA(base.HiCacheBaseServer, MMLUMixin, MGSMEnMixin):
    """HiCache with MLA model tests"""

    model_name = DEFAULT_MLA_MODEL_NAME_FOR_TEST
    server_env = {"SGLANG_ENABLE_RANK_CONSENSUS_CHECKER": "1"}
    hicache_args = [
        "--trust-remote-code",
        "--enable-hierarchical-cache",
    ] + (["--hicache-size", 200] if base._is_hip else ["--hicache-ratio", 2])
    mmlu_score_threshold = 0.54
    mmlu_num_examples = 256
    mmlu_num_threads = 32
    mgsm_en_score_threshold = 0.8
    if base._is_hip:
        mgsm_en_num_threads = 32


if __name__ == "__main__":
    unittest.main()
