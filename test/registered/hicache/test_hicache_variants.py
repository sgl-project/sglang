from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=524, suite="stage-b-test-large-1-gpu")
register_amd_ci(est_time=524, suite="stage-b-test-small-1-gpu-amd")
"""
Consolidated HiCache variant tests.
Tests HiCache with different configurations: standard, MLA, EAGLE, and page size variants.
"""

import unittest
from types import SimpleNamespace

import requests

from sglang.bench_serving import get_tokenizer
from sglang.srt.utils import is_hip, kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE3,
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TARGET_MODEL_EAGLE3,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

_is_hip = is_hip()


class HiCacheEvalMixin:
    """Mixin class containing common HiCache evaluation test methods"""

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )

        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], self.expected_mmlu_score)


class HiCacheMGSMEvalMixin:
    """Mixin for tests that also run MGSM evaluation"""

    def test_mgsm_en(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mgsm_en",
            num_examples=None,
            num_threads=1024,
        )

        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.8)


class HiCacheBaseServer(CustomTestCase):
    """Base class for HiCache tests with configurable server setup"""

    model_name = DEFAULT_MODEL_NAME_FOR_TEST
    hicache_args = []
    expected_mmlu_score = 0.65

    @classmethod
    def setUpClass(cls):
        cls.model = cls.model_name
        cls.base_url = DEFAULT_URL_FOR_TEST

        # Setup tokenizer if needed by subclass
        if hasattr(cls, "needs_tokenizer") and cls.needs_tokenizer:
            cls.tokenizer = get_tokenizer(cls.model)

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.hicache_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestHiCacheStandard(HiCacheBaseServer, HiCacheEvalMixin):
    """Standard HiCache configuration tests"""

    model_name = DEFAULT_MODEL_NAME_FOR_TEST
    hicache_args = [
        "--enable-hierarchical-cache",
        "--mem-fraction-static",
        0.7,
        "--hicache-size",
        100 if not _is_hip else 200,
    ]
    expected_mmlu_score = 0.65


class TestHiCacheMLA(HiCacheBaseServer, HiCacheEvalMixin, HiCacheMGSMEvalMixin):
    """HiCache with MLA model tests"""

    model_name = DEFAULT_MLA_MODEL_NAME_FOR_TEST
    hicache_args = [
        "--trust-remote-code",
        "--enable-hierarchical-cache",
    ] + (["--hicache-size", 200] if _is_hip else ["--hicache-ratio", 2])
    expected_mmlu_score = 0.5


@unittest.skipIf(is_hip(), "Disabled for AMD-aiter")
class TestHiCacheEagle(HiCacheBaseServer, HiCacheEvalMixin):
    """HiCache with EAGLE speculative decoding tests"""

    model_name = DEFAULT_TARGET_MODEL_EAGLE3
    needs_tokenizer = True
    hicache_args = [
        "--enable-hierarchical-cache",
        "--hicache-ratio",
        1.2,
        "--mem-fraction-static",
        0.7,
        "--speculative-algorithm",
        "EAGLE3",
        "--speculative-draft-model-path",
        DEFAULT_DRAFT_MODEL_EAGLE3,
        "--speculative-num-steps",
        2,
        "--speculative-eagle-topk",
        1,
        "--speculative-num-draft-tokens",
        3,
        "--dtype",
        "float16",
        "--chunked-prefill-size",
        1024,
    ]
    expected_mmlu_score = 0.72

    def test_mmlu(self):
        """Override to add EAGLE-specific assertions"""
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )

        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], self.expected_mmlu_score)

        # EAGLE-specific check
        server_info = requests.get(self.base_url + "/get_server_info").json()
        avg_spec_accept_length = server_info["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(f"{avg_spec_accept_length=}")
        self.assertGreater(avg_spec_accept_length, 2.26)


class TestHiCachePage(HiCacheBaseServer, HiCacheEvalMixin):
    """HiCache with custom page size tests"""

    model_name = DEFAULT_MODEL_NAME_FOR_TEST
    hicache_args = [
        "--enable-hierarchical-cache",
        "--page-size",
        32,
        "--hicache-write-policy",
        "write_back",
    ]
    expected_mmlu_score = 0.65


if __name__ == "__main__":
    unittest.main()
