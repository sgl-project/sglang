"""
Tests for suffix decoding speculative method.

Adapted from PR #13553 and aligned with the existing ngram test pattern.

Includes:
- Integration tests (server launch + GSM8K accuracy with different attention backends)
- Configuration parameter validation
- SuffixVerifyInput and algorithm registration unit tests
"""

import os
import unittest
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.gsm8k_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_TARGET_MODEL_NGRAM,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=230, suite="stage-b-test-large-1-gpu")
register_amd_ci(est_time=230, suite="stage-b-test-large-1-gpu-amd")

GSM_DATASET_PATH = None

# Default server arguments shared across all tests
DEFAULT_SERVER_ARGS = [
    "--trust-remote-code",
    "--cuda-graph-max-bs",
    "8",
    "--speculative-algorithm",
    "SUFFIX",
    "--speculative-num-draft-tokens",
    "4",
    "--speculative-suffix-max-spec-factor",
    "2.0",
    "--speculative-suffix-min-token-prob",
    "0.2",
    "--mem-fraction-static",
    0.8,
]


# ---------------------------------------------------------------------------
# Integration tests — launch server and run GSM8K
# ---------------------------------------------------------------------------


class TestSuffixDecodingBase(GSM8KMixin, CustomTestCase):
    """Base integration test for suffix decoding."""

    model = DEFAULT_TARGET_MODEL_NGRAM
    base_url = DEFAULT_URL_FOR_TEST
    gsm8k_accuracy_thres = 0.79
    gsm8k_accept_length_thres = 1.3

    @classmethod
    def get_server_args(cls):
        return DEFAULT_SERVER_ARGS

    @classmethod
    def setUpClass(cls):
        envs.SGLANG_JIT_DEEPGEMM_PRECOMPILE.set(False)
        envs.SGLANG_ENABLE_JIT_DEEPGEMM.set(False)
        model = cls.model
        cls.process = popen_launch_server(
            model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.get_server_args(),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestSuffixDecodingTriton(TestSuffixDecodingBase):
    @classmethod
    def get_server_args(cls):
        return DEFAULT_SERVER_ARGS + ["--attention-backend", "triton"]


@unittest.skipIf(
    os.environ.get("SGLANG_IS_IN_CI_AMD", "0") == "1",
    "flashinfer attention backend is not supported on ROCm",
)
class TestSuffixDecodingFlashinfer(TestSuffixDecodingBase):
    @classmethod
    def get_server_args(cls):
        return DEFAULT_SERVER_ARGS + ["--attention-backend", "flashinfer"]


# ---------------------------------------------------------------------------
# Configuration validation tests — no server needed
# ---------------------------------------------------------------------------


class TestSuffixConfiguration(unittest.TestCase):
    """Test configuration validation for suffix decoding."""

    def test_missing_arctic_inference(self):
        with patch(
            "sglang.srt.utils.common.is_arctic_inference_available", return_value=False
        ):
            from sglang.srt.server_args import ServerArgs

            with self.assertRaises(ImportError) as ctx:
                ServerArgs(
                    model_path=DEFAULT_TARGET_MODEL_NGRAM,
                    speculative_algorithm="SUFFIX",
                )
            self.assertIn("Arctic Inference", str(ctx.exception))

    def test_invalid_max_tree_depth(self):
        with patch(
            "sglang.srt.utils.common.is_arctic_inference_available", return_value=True
        ):
            from sglang.srt.server_args import ServerArgs

            with self.assertRaises(ValueError) as ctx:
                ServerArgs(
                    model_path=DEFAULT_TARGET_MODEL_NGRAM,
                    speculative_algorithm="SUFFIX",
                    speculative_suffix_max_tree_depth=0,
                )
            self.assertIn("max_tree_depth", str(ctx.exception).lower())

    def test_invalid_max_cached_requests(self):
        with patch(
            "sglang.srt.utils.common.is_arctic_inference_available", return_value=True
        ):
            from sglang.srt.server_args import ServerArgs

            with self.assertRaises(ValueError) as ctx:
                ServerArgs(
                    model_path=DEFAULT_TARGET_MODEL_NGRAM,
                    speculative_algorithm="SUFFIX",
                    speculative_suffix_max_cached_requests=-2,
                )
            self.assertIn("max_cached_requests", str(ctx.exception).lower())

    def test_invalid_max_spec_factor(self):
        with patch(
            "sglang.srt.utils.common.is_arctic_inference_available", return_value=True
        ):
            from sglang.srt.server_args import ServerArgs

            with self.assertRaises(ValueError) as ctx:
                ServerArgs(
                    model_path=DEFAULT_TARGET_MODEL_NGRAM,
                    speculative_algorithm="SUFFIX",
                    speculative_suffix_max_spec_factor=-0.5,
                )
            self.assertIn("max_spec_factor", str(ctx.exception).lower())

    def test_invalid_min_token_prob(self):
        with patch(
            "sglang.srt.utils.common.is_arctic_inference_available", return_value=True
        ):
            from sglang.srt.server_args import ServerArgs

            with self.assertRaises(ValueError) as ctx:
                ServerArgs(
                    model_path=DEFAULT_TARGET_MODEL_NGRAM,
                    speculative_algorithm="SUFFIX",
                    speculative_suffix_min_token_prob=1.5,
                )
            self.assertIn("min_token_prob", str(ctx.exception).lower())

    def test_valid_configuration(self):
        with patch(
            "sglang.srt.utils.common.is_arctic_inference_available", return_value=True
        ):
            from sglang.srt.server_args import ServerArgs

            args = ServerArgs(
                model_path=DEFAULT_TARGET_MODEL_NGRAM,
                speculative_algorithm="SUFFIX",
                speculative_num_draft_tokens=24,
                speculative_suffix_max_tree_depth=24,
                speculative_suffix_max_cached_requests=10000,
                speculative_suffix_max_spec_factor=1.0,
                speculative_suffix_min_token_prob=0.1,
            )
            self.assertEqual(args.speculative_num_draft_tokens, 24)
            self.assertEqual(args.speculative_suffix_max_tree_depth, 24)
            self.assertEqual(args.speculative_suffix_max_cached_requests, 10000)
            self.assertEqual(args.speculative_suffix_max_spec_factor, 1.0)
            self.assertEqual(args.speculative_suffix_min_token_prob, 0.1)

    def test_default_draft_tokens(self):
        with patch(
            "sglang.srt.utils.common.is_arctic_inference_available", return_value=True
        ):
            from sglang.srt.server_args import ServerArgs

            args = ServerArgs(
                model_path=DEFAULT_TARGET_MODEL_NGRAM,
                speculative_algorithm="SUFFIX",
                speculative_suffix_max_tree_depth=32,
            )
            self.assertEqual(args.speculative_num_draft_tokens, 32)

    def test_cuda_device_requirement(self):
        with patch(
            "sglang.srt.utils.common.is_arctic_inference_available", return_value=True
        ):
            from sglang.srt.server_args import ServerArgs

            with self.assertRaises(ValueError) as ctx:
                ServerArgs(
                    model_path=DEFAULT_TARGET_MODEL_NGRAM,
                    speculative_algorithm="SUFFIX",
                    device="cpu",
                )
            self.assertIn("cuda", str(ctx.exception).lower())

    def test_dp_attention_not_supported(self):
        with patch(
            "sglang.srt.utils.common.is_arctic_inference_available", return_value=True
        ):
            from sglang.srt.server_args import ServerArgs

            with self.assertRaises(ValueError) as ctx:
                ServerArgs(
                    model_path=DEFAULT_TARGET_MODEL_NGRAM,
                    speculative_algorithm="SUFFIX",
                    tp_size=2,
                    dp_size=2,
                    enable_dp_attention=True,
                )
            self.assertIn("dp attention", str(ctx.exception).lower())


# ---------------------------------------------------------------------------
# Unit tests — no server or GPU needed
# ---------------------------------------------------------------------------


class TestSuffixVerifyInput(unittest.TestCase):
    """Test SuffixVerifyInput data structure."""

    def test_verify_input_creation(self):
        import torch

        from sglang.srt.speculative.suffix_info import SuffixVerifyInput

        verify_input = SuffixVerifyInput(
            draft_token=torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64),
            tree_mask=torch.ones((25,), dtype=torch.bool),
            positions=torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64),
            retrive_index=torch.zeros((1, 5), dtype=torch.int64),
            retrive_next_token=torch.zeros((1, 5), dtype=torch.int64),
            retrive_next_sibling=torch.zeros((1, 5), dtype=torch.int64),
            draft_token_num=5,
        )
        self.assertFalse(verify_input.is_draft_input())
        self.assertTrue(verify_input.is_verify_input())
        self.assertEqual(verify_input.draft_token_num, 5)
        self.assertEqual(verify_input.get_spec_adjust_token_coefficient(), (5, 5))


class TestSuffixAlgorithmRegistration(unittest.TestCase):
    """Test that SUFFIX algorithm is properly registered."""

    def test_algorithm_from_string(self):
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        suffix_algo = SpeculativeAlgorithm.from_string("SUFFIX")
        self.assertIsNotNone(suffix_algo)
        self.assertTrue(suffix_algo.is_suffix())
        self.assertFalse(suffix_algo.is_ngram())
        self.assertFalse(suffix_algo.is_eagle())
        self.assertFalse(suffix_algo.is_none())

    def test_algorithm_case_insensitive(self):
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        self.assertTrue(SpeculativeAlgorithm.from_string("suffix").is_suffix())
        self.assertTrue(SpeculativeAlgorithm.from_string("Suffix").is_suffix())

    def test_unknown_algorithm_raises(self):
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        with self.assertRaises(ValueError):
            SpeculativeAlgorithm.from_string("NONEXISTENT")


if __name__ == "__main__":
    unittest.main()
