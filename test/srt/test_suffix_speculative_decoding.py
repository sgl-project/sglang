"""
Unit tests for suffix decoding speculative method.

This test suite validates the suffix decoding implementation including:
- Server launch and basic generation
- Configuration parameter validation
- Worker initialization
- Integration with different attention backends
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests

from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_NGRAM_SPECULATIVE_TARGET_MODEL_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

GSM_DATASET_PATH = None

# Allow overriding the speculative algorithm (e.g., NONE or NGRAM) via env var so
# we can reuse this suite for baseline runs. Example:
#     SUFFIX_TEST_SPEC_ALGO=NONE pytest test/srt/test_suffix_speculative_decoding.py::TestSuffixDecodingBase::test_gsm8k -v -s
SPEC_ALGO_FOR_TEST = os.environ.get("SUFFIX_TEST_SPEC_ALGO", "SUFFIX").upper()
IS_SUFFIX_MODE = SPEC_ALGO_FOR_TEST == "SUFFIX"

# Default server arguments shared across all tests.
DEFAULT_SERVER_ARGS = [
    "--trust-remote-code",
    "--cuda-graph-max-bs",
    "8",
    "--mem-fraction-static",
    "0.8",
]

if SPEC_ALGO_FOR_TEST != "NONE":
    DEFAULT_SERVER_ARGS.extend(
        [
            "--speculative-algorithm",
            SPEC_ALGO_FOR_TEST,
            "--speculative-num-draft-tokens",
            "16",
        ]
    )


class TestSuffixDecodingBase(CustomTestCase):
    """Base test class for suffix decoding functionality."""

    model = DEFAULT_NGRAM_SPECULATIVE_TARGET_MODEL_FOR_TEST
    base_url = DEFAULT_URL_FOR_TEST
    accuracy_threshold = 0.79
    spec_decode_threshold = 1.5  # Suffix decoding threshold (may vary from ngram)
    is_suffix_mode = IS_SUFFIX_MODE

    @classmethod
    def get_server_args(cls):
        """Return the arguments for the server launch. Override in subclasses."""
        return DEFAULT_SERVER_ARGS + ["--attention-backend", "fa3"]

    @classmethod
    def setUpClass(cls):
        # Disable deep gemm precompile to make launch server faster
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

    def test_gsm8k(self):
        """Test suffix decoding with GSM8K few-shot evaluation."""
        requests.get(self.base_url + "/flush_cache")

        args = SimpleNamespace(
            num_shots=4,
            num_questions=100,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
            data_path=GSM_DATASET_PATH,
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")

        # Validate accuracy
        metric_key = "accuracy"
        self.assertGreater(metrics[metric_key], self.accuracy_threshold)

        # Validate speculative decoding performance when running in suffix mode.
        if self.is_suffix_mode:
            server_info = requests.get(self.base_url + "/get_server_info")
            avg_spec_accept_length = server_info.json()["internal_states"][0][
                "avg_spec_accept_length"
            ]
            print(f"{avg_spec_accept_length=}")
            self.assertGreater(avg_spec_accept_length, self.spec_decode_threshold)


class TestSuffixDecodingTriton(TestSuffixDecodingBase):
    """Test suffix decoding with Triton attention backend."""

    @classmethod
    def get_server_args(cls):
        return DEFAULT_SERVER_ARGS + ["--attention-backend", "triton"]


class TestSuffixDecodingFlashinfer(TestSuffixDecodingBase):
    """Test suffix decoding with FlashInfer attention backend."""

    @classmethod
    def get_server_args(cls):
        return DEFAULT_SERVER_ARGS + ["--attention-backend", "flashinfer"]


class TestSuffixConfiguration(unittest.TestCase):
    """Test configuration validation for suffix decoding."""

    def test_missing_arctic_inference(self):
        """Test error when arctic-inference is not installed."""
        with patch(
            "sglang.srt.utils.common.is_arctic_inference_available", return_value=False
        ):
            with self.assertRaises(ImportError) as context:
                server_args = ServerArgs(
                    model_path=DEFAULT_NGRAM_SPECULATIVE_TARGET_MODEL_FOR_TEST,
                    speculative_algorithm="SUFFIX",
                )

            self.assertIn("Arctic Inference", str(context.exception))

    def test_invalid_max_tree_depth(self):
        """Test validation of max_tree_depth parameter."""
        with patch(
            "sglang.srt.utils.common.is_arctic_inference_available", return_value=True
        ):
            with self.assertRaises(ValueError) as context:
                server_args = ServerArgs(
                    model_path=DEFAULT_NGRAM_SPECULATIVE_TARGET_MODEL_FOR_TEST,
                    speculative_algorithm="SUFFIX",
                    speculative_suffix_max_tree_depth=0,  # Invalid
                )

            self.assertIn("max_tree_depth", str(context.exception).lower())

    def test_invalid_max_cached_requests(self):
        """Test validation of max_cached_requests parameter."""
        with patch(
            "sglang.srt.utils.common.is_arctic_inference_available", return_value=True
        ):
            with self.assertRaises(ValueError) as context:
                server_args = ServerArgs(
                    model_path=DEFAULT_NGRAM_SPECULATIVE_TARGET_MODEL_FOR_TEST,
                    speculative_algorithm="SUFFIX",
                    speculative_suffix_max_cached_requests=-2,  # Invalid (< -1 or 0)
                )

            self.assertIn("max_cached_requests", str(context.exception).lower())

    def test_invalid_max_spec_factor(self):
        """Test validation of max_spec_factor parameter."""
        with patch(
            "sglang.srt.utils.common.is_arctic_inference_available", return_value=True
        ):
            with self.assertRaises(ValueError) as context:
                server_args = ServerArgs(
                    model_path=DEFAULT_NGRAM_SPECULATIVE_TARGET_MODEL_FOR_TEST,
                    speculative_algorithm="SUFFIX",
                    speculative_suffix_max_spec_factor=-0.5,  # Invalid
                )

            self.assertIn("max_spec_factor", str(context.exception).lower())

    def test_invalid_min_token_prob(self):
        """Test validation of min_token_prob parameter."""
        with patch(
            "sglang.srt.utils.common.is_arctic_inference_available", return_value=True
        ):
            with self.assertRaises(ValueError) as context:
                server_args = ServerArgs(
                    model_path=DEFAULT_NGRAM_SPECULATIVE_TARGET_MODEL_FOR_TEST,
                    speculative_algorithm="SUFFIX",
                    speculative_suffix_min_token_prob=1.5,  # Invalid (>1.0)
                )

            self.assertIn("min_token_prob", str(context.exception).lower())

    def test_valid_configuration(self):
        """Test valid suffix decoding configuration."""
        with patch(
            "sglang.srt.utils.common.is_arctic_inference_available", return_value=True
        ):
            server_args = ServerArgs(
                model_path=DEFAULT_NGRAM_SPECULATIVE_TARGET_MODEL_FOR_TEST,
                speculative_algorithm="SUFFIX",
                speculative_num_draft_tokens=24,
                speculative_suffix_max_tree_depth=24,
                speculative_suffix_max_cached_requests=10000,
                speculative_suffix_max_spec_factor=1.0,
                speculative_suffix_min_token_prob=0.1,
            )

            # Check defaults were set correctly
            self.assertEqual(server_args.speculative_num_draft_tokens, 24)
            self.assertEqual(server_args.speculative_suffix_max_tree_depth, 24)
            self.assertEqual(server_args.speculative_suffix_max_cached_requests, 10000)
            self.assertEqual(server_args.speculative_suffix_max_spec_factor, 1.0)
            self.assertEqual(server_args.speculative_suffix_min_token_prob, 0.1)

    def test_default_draft_tokens(self):
        """Test that speculative_num_draft_tokens defaults to max_tree_depth."""
        with patch(
            "sglang.srt.utils.common.is_arctic_inference_available", return_value=True
        ):
            server_args = ServerArgs(
                model_path=DEFAULT_NGRAM_SPECULATIVE_TARGET_MODEL_FOR_TEST,
                speculative_algorithm="SUFFIX",
                speculative_suffix_max_tree_depth=32,
            )

            # Should default to max_tree_depth
            self.assertEqual(server_args.speculative_num_draft_tokens, 32)

    def test_cuda_device_requirement(self):
        """Test that suffix decoding requires CUDA device."""
        with patch(
            "sglang.srt.utils.common.is_arctic_inference_available", return_value=True
        ):
            with self.assertRaises(ValueError) as context:
                server_args = ServerArgs(
                    model_path=DEFAULT_NGRAM_SPECULATIVE_TARGET_MODEL_FOR_TEST,
                    speculative_algorithm="SUFFIX",
                    device="cpu",  # Invalid - requires CUDA
                )

            self.assertIn("cuda", str(context.exception).lower())

    def test_dp_attention_not_supported(self):
        """Test that dp_attention is not supported with suffix decoding."""
        with patch(
            "sglang.srt.utils.common.is_arctic_inference_available", return_value=True
        ):
            with self.assertRaises(ValueError) as context:
                server_args = ServerArgs(
                    model_path=DEFAULT_NGRAM_SPECULATIVE_TARGET_MODEL_FOR_TEST,
                    speculative_algorithm="SUFFIX",
                    tp_size=2,
                    dp_size=2,
                    enable_dp_attention=True,
                )

            self.assertIn("dp attention", str(context.exception).lower())


class TestSuffixWorker(unittest.TestCase):
    """Test SuffixWorker implementation."""

    @patch("sglang.srt.utils.common.is_arctic_inference_available", return_value=True)
    @patch("arctic_inference.suffix_decoding.SuffixDecodingCache")
    def test_worker_initialization(self, mock_cache_class, mock_has_arctic):
        """Test SuffixWorker initialization."""
        # Set up the mock
        mock_cache_instance = Mock()
        mock_cache_class.return_value = mock_cache_instance

        from sglang.srt.speculative.suffix_worker import SuffixWorker

        server_args = ServerArgs(
            model_path=DEFAULT_NGRAM_SPECULATIVE_TARGET_MODEL_FOR_TEST,
            speculative_algorithm="SUFFIX",
            speculative_num_draft_tokens=16,
            page_size=1,
        )

        # Create mock target worker
        mock_target_worker = Mock()
        mock_target_worker.max_running_requests = 48
        mock_target_worker.model_runner = Mock()

        worker = SuffixWorker(
            server_args=server_args,
            gpu_id=0,
            tp_rank=0,
            dp_rank=None,
            moe_ep_rank=0,
            nccl_port=12345,
            target_worker=mock_target_worker,
        )

        self.assertEqual(worker.draft_token_num, 16)
        self.assertIsNotNone(worker.ngram_cache)
        self.assertEqual(worker.ngram_cache.max_tree_depth, 24)
        self.assertEqual(worker.ngram_cache.max_spec_factor, 1.0)
        self.assertEqual(worker.ngram_cache.min_token_prob, 0.1)

    @patch("sglang.srt.utils.common.is_arctic_inference_available", return_value=True)
    @patch("arctic_inference.suffix_decoding.SuffixDecodingCache")
    def test_worker_with_custom_parameters(self, mock_cache_class, mock_has_arctic):
        """Test SuffixWorker initialization with custom parameters."""
        # Set up the mock
        mock_cache_instance = Mock()
        mock_cache_class.return_value = mock_cache_instance

        from sglang.srt.speculative.suffix_worker import SuffixWorker

        server_args = ServerArgs(
            model_path=DEFAULT_NGRAM_SPECULATIVE_TARGET_MODEL_FOR_TEST,
            speculative_algorithm="SUFFIX",
            speculative_num_draft_tokens=32,
            speculative_suffix_max_tree_depth=48,
            speculative_suffix_max_spec_factor=2.0,
            speculative_suffix_min_token_prob=0.05,
            page_size=1,
        )

        # Create mock target worker
        mock_target_worker = Mock()
        mock_target_worker.max_running_requests = 48
        mock_target_worker.model_runner = Mock()

        worker = SuffixWorker(
            server_args=server_args,
            gpu_id=0,
            tp_rank=0,
            dp_rank=None,
            moe_ep_rank=0,
            nccl_port=12345,
            target_worker=mock_target_worker,
        )

        self.assertEqual(worker.draft_token_num, 32)
        self.assertEqual(worker.ngram_cache.max_tree_depth, 48)
        self.assertEqual(worker.ngram_cache.max_spec_factor, 2.0)
        self.assertEqual(worker.ngram_cache.min_token_prob, 0.05)


class TestSuffixVerifyInput(unittest.TestCase):
    """Test SuffixVerifyInput data structure."""

    def test_verify_input_creation(self):
        """Test creating SuffixVerifyInput."""
        import torch

        from sglang.srt.speculative.suffix_info import SuffixVerifyInput

        draft_token = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)
        tree_mask = torch.ones((25,), dtype=torch.bool)
        positions = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
        retrive_index = torch.zeros((1, 5), dtype=torch.int64)
        retrive_next_token = torch.zeros((1, 5), dtype=torch.int64)
        retrive_next_sibling = torch.zeros((1, 5), dtype=torch.int64)
        draft_token_num = 5

        verify_input = SuffixVerifyInput(
            draft_token=draft_token,
            tree_mask=tree_mask,
            positions=positions,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            draft_token_num=draft_token_num,
        )

        self.assertFalse(verify_input.is_draft_input())
        self.assertTrue(verify_input.is_verify_input())
        self.assertEqual(verify_input.draft_token_num, 5)
        self.assertEqual(verify_input.get_spec_adjust_token_coefficient(), (5, 5))


class TestSuffixAlgorithmRegistration(unittest.TestCase):
    """Test that SUFFIX algorithm is properly registered."""

    def test_algorithm_registered(self):
        """Test that SUFFIX is registered in SpeculativeAlgorithm."""
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        # Test from_string method
        suffix_algo = SpeculativeAlgorithm.from_string("SUFFIX")
        self.assertIsNotNone(suffix_algo)
        self.assertEqual(suffix_algo.name, "SUFFIX")

        # Test is_suffix method
        self.assertTrue(suffix_algo.is_suffix())
        self.assertFalse(suffix_algo.is_ngram())
        self.assertFalse(suffix_algo.is_eagle())

    def test_suffix_in_registry(self):
        """Test that SUFFIX is in the algorithm registry."""
        from sglang.srt.speculative.spec_info import list_registered_workers

        workers = list_registered_workers()
        self.assertIn("SUFFIX", workers)


if __name__ == "__main__":
    unittest.main()
