"""
Unit tests for TokenizerManager helper methods.

This tests the refactored tokenization functionality including input format detection,
tokenizer input preparation, and result extraction logic.

Usage:
python3 -m unittest test_tokenizer_manager.TestInputFormatDetection
python3 -m unittest test_tokenizer_manager.TestTokenizerInputPreparation
python3 -m unittest test_tokenizer_manager.TestTokenizerResultExtraction
python3 -m unittest test_tokenizer_manager.TestTokenizerManagerIntegration
"""

import unittest
from typing import List, Optional, Union
from unittest.mock import Mock, patch

from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST


class TestInputFormatDetection(unittest.TestCase):
    """Test cases for _detect_input_format method."""

    def setUp(self):
        """Set up test fixtures."""
        with patch("sglang.srt.utils.get_device", return_value="cpu"):
            self.server_args = ServerArgs(model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST)
            self.port_args = PortArgs.init_new(self.server_args)

        with patch("zmq.asyncio.Context"), patch(
            "sglang.srt.utils.get_zmq_socket"
        ), patch(
            "sglang.srt.utils.hf_transformers_utils.get_tokenizer"
        ) as mock_tokenizer:
            mock_tokenizer.return_value = Mock(vocab_size=32000)
            self.tokenizer_manager = TokenizerManager(self.server_args, self.port_args)

    def test_detect_single_string(self):
        """Test detection of single string input."""
        text = "Hello world"
        result = self.tokenizer_manager._detect_input_format(
            text, is_cross_encoder=False
        )
        self.assertEqual(result, "single_string")

    def test_detect_single_string_cross_encoder_disabled(self):
        """Test single string with cross_encoder disabled still returns single_string."""
        text = "Hello world"
        result = self.tokenizer_manager._detect_input_format(
            text, is_cross_encoder=True
        )
        self.assertEqual(result, "single_string")

    def test_detect_batch_strings(self):
        """Test detection of batch string inputs."""
        texts = ["Hello", "World", "How are you?"]
        result = self.tokenizer_manager._detect_input_format(
            texts, is_cross_encoder=False
        )
        self.assertEqual(result, "batch_strings")

    def test_detect_batch_strings_cross_encoder_disabled(self):
        """Test batch strings with cross_encoder disabled."""
        texts = ["Hello", "World"]
        result = self.tokenizer_manager._detect_input_format(
            texts, is_cross_encoder=True
        )
        self.assertEqual(result, "batch_strings")

    def test_detect_cross_encoder_single_pair(self):
        """Test detection of cross-encoder single pair."""
        texts = [["query text", "document text"]]
        result = self.tokenizer_manager._detect_input_format(
            texts, is_cross_encoder=True
        )
        self.assertEqual(result, "cross_encoder_pairs")

    def test_detect_cross_encoder_multiple_pairs(self):
        """Test detection of cross-encoder multiple pairs."""
        texts = [["q1", "d1"], ["q2", "d2"], ["q3", "d3"]]
        result = self.tokenizer_manager._detect_input_format(
            texts, is_cross_encoder=True
        )
        self.assertEqual(result, "cross_encoder_pairs")

    def test_detect_cross_encoder_disabled_with_pairs(self):
        """Test pairs with cross_encoder disabled should return batch_strings."""
        texts = [["query", "document"]]
        result = self.tokenizer_manager._detect_input_format(
            texts, is_cross_encoder=False
        )
        self.assertEqual(result, "batch_strings")

    def test_detect_empty_list(self):
        """Test detection with empty list."""
        texts = []
        result = self.tokenizer_manager._detect_input_format(
            texts, is_cross_encoder=True
        )
        self.assertEqual(result, "batch_strings")

    def test_detect_malformed_cross_encoder_pairs(self):
        """Test malformed cross-encoder pairs (not length 2)."""
        texts = [["query only"]]  # Single element, not a pair
        result = self.tokenizer_manager._detect_input_format(
            texts, is_cross_encoder=True
        )
        self.assertEqual(result, "batch_strings")

        texts = [["query", "doc", "extra"]]  # Three elements, not a pair
        result = self.tokenizer_manager._detect_input_format(
            texts, is_cross_encoder=True
        )
        self.assertEqual(result, "batch_strings")


class TestTokenizerInputPreparation(unittest.TestCase):
    """Test cases for _prepare_tokenizer_input method."""

    def setUp(self):
        """Set up test fixtures."""
        with patch("sglang.srt.utils.get_device", return_value="cpu"):
            self.server_args = ServerArgs(model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST)
            self.port_args = PortArgs.init_new(self.server_args)

        with patch("zmq.asyncio.Context"), patch(
            "sglang.srt.utils.get_zmq_socket"
        ), patch(
            "sglang.srt.utils.hf_transformers_utils.get_tokenizer"
        ) as mock_tokenizer:
            mock_tokenizer.return_value = Mock(vocab_size=32000)
            self.tokenizer_manager = TokenizerManager(self.server_args, self.port_args)

    def test_prepare_single_string_input(self):
        """Test preparation of single string input."""
        text = "Hello world"
        result = self.tokenizer_manager._prepare_tokenizer_input(text, "single_string")
        self.assertEqual(result, ["Hello world"])

    def test_prepare_batch_strings_input(self):
        """Test preparation of batch strings input."""
        texts = ["Hello", "World", "Test"]
        result = self.tokenizer_manager._prepare_tokenizer_input(texts, "batch_strings")
        self.assertEqual(result, ["Hello", "World", "Test"])

    def test_prepare_cross_encoder_pairs_input(self):
        """Test preparation of cross-encoder pairs input."""
        texts = [["query1", "doc1"], ["query2", "doc2"]]
        result = self.tokenizer_manager._prepare_tokenizer_input(
            texts, "cross_encoder_pairs"
        )
        self.assertEqual(result, [["query1", "doc1"], ["query2", "doc2"]])

    def test_prepare_cross_encoder_single_pair_input(self):
        """Test preparation of single cross-encoder pair."""
        texts = [["query text", "document text"]]
        result = self.tokenizer_manager._prepare_tokenizer_input(
            texts, "cross_encoder_pairs"
        )
        self.assertEqual(result, [["query text", "document text"]])

    def test_prepare_unknown_input_format(self):
        """Test preparation with unknown input format falls back to returning as-is."""
        texts = ["test"]
        result = self.tokenizer_manager._prepare_tokenizer_input(
            texts, "unknown_format"
        )
        self.assertEqual(result, ["test"])


class TestTokenizerResultExtraction(unittest.TestCase):
    """Test cases for _extract_tokenizer_results method."""

    def setUp(self):
        """Set up test fixtures."""
        with patch("sglang.srt.utils.get_device", return_value="cpu"):
            self.server_args = ServerArgs(model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST)
            self.port_args = PortArgs.init_new(self.server_args)

        with patch("zmq.asyncio.Context"), patch(
            "sglang.srt.utils.get_zmq_socket"
        ), patch(
            "sglang.srt.utils.hf_transformers_utils.get_tokenizer"
        ) as mock_tokenizer:
            mock_tokenizer.return_value = Mock(vocab_size=32000)
            self.tokenizer_manager = TokenizerManager(self.server_args, self.port_args)

    def test_extract_single_string_results(self):
        """Test extraction for single string input."""
        input_ids = [[101, 2129, 102]]
        token_type_ids = [[0, 0, 0]]

        result_input_ids, result_token_type_ids = (
            self.tokenizer_manager._extract_tokenizer_results(
                input_ids, token_type_ids, "single_string", original_batch_size=1
            )
        )

        self.assertEqual(result_input_ids, [101, 2129, 102])
        self.assertEqual(result_token_type_ids, [0, 0, 0])

    def test_extract_single_cross_encoder_results(self):
        """Test extraction for single cross-encoder pair."""
        input_ids = [[101, 2129, 102, 4068, 102]]
        token_type_ids = [[0, 0, 0, 1, 1]]

        result_input_ids, result_token_type_ids = (
            self.tokenizer_manager._extract_tokenizer_results(
                input_ids, token_type_ids, "cross_encoder_pairs", original_batch_size=1
            )
        )

        self.assertEqual(result_input_ids, [101, 2129, 102, 4068, 102])
        self.assertEqual(result_token_type_ids, [0, 0, 0, 1, 1])

    def test_extract_batch_results(self):
        """Test extraction for batch inputs."""
        input_ids = [[101, 2129, 102], [101, 4068, 102]]
        token_type_ids = [[0, 0, 0], [0, 0, 0]]

        result_input_ids, result_token_type_ids = (
            self.tokenizer_manager._extract_tokenizer_results(
                input_ids, token_type_ids, "batch_strings", original_batch_size=2
            )
        )

        self.assertEqual(result_input_ids, [[101, 2129, 102], [101, 4068, 102]])
        self.assertEqual(result_token_type_ids, [[0, 0, 0], [0, 0, 0]])

    def test_extract_multiple_cross_encoder_results(self):
        """Test extraction for multiple cross-encoder pairs."""
        input_ids = [[101, 2129, 102, 4068, 102], [101, 7592, 102, 2088, 102]]
        token_type_ids = [[0, 0, 0, 1, 1], [0, 0, 0, 1, 1]]

        result_input_ids, result_token_type_ids = (
            self.tokenizer_manager._extract_tokenizer_results(
                input_ids, token_type_ids, "cross_encoder_pairs", original_batch_size=2
            )
        )

        self.assertEqual(
            result_input_ids, [[101, 2129, 102, 4068, 102], [101, 7592, 102, 2088, 102]]
        )
        self.assertEqual(result_token_type_ids, [[0, 0, 0, 1, 1], [0, 0, 0, 1, 1]])

    def test_extract_empty_results(self):
        """Test extraction with empty results."""
        input_ids = []
        token_type_ids = None

        result_input_ids, result_token_type_ids = (
            self.tokenizer_manager._extract_tokenizer_results(
                input_ids, token_type_ids, "single_string", original_batch_size=1
            )
        )

        self.assertEqual(result_input_ids, [])
        self.assertIsNone(result_token_type_ids)

    def test_extract_with_none_token_type_ids(self):
        """Test extraction when token_type_ids is None."""
        input_ids = [[101, 2129, 102]]
        token_type_ids = None

        result_input_ids, result_token_type_ids = (
            self.tokenizer_manager._extract_tokenizer_results(
                input_ids, token_type_ids, "single_string", original_batch_size=1
            )
        )

        self.assertEqual(result_input_ids, [101, 2129, 102])
        self.assertIsNone(result_token_type_ids)


class TestTokenizerManagerIntegration(unittest.TestCase):
    """Integration tests combining multiple helper methods."""

    def setUp(self):
        """Set up test fixtures."""
        with patch("sglang.srt.utils.get_device", return_value="cpu"):
            self.server_args = ServerArgs(model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST)
            self.port_args = PortArgs.init_new(self.server_args)

        with patch("zmq.asyncio.Context"), patch(
            "sglang.srt.utils.get_zmq_socket"
        ), patch(
            "sglang.srt.utils.hf_transformers_utils.get_tokenizer"
        ) as mock_tokenizer:
            mock_tokenizer.return_value = Mock(vocab_size=32000)
            self.tokenizer_manager = TokenizerManager(self.server_args, self.port_args)

    def test_full_workflow_single_string(self):
        """Test complete workflow for single string input."""
        text = "Hello world"

        # Step 1: Detect format
        input_format = self.tokenizer_manager._detect_input_format(
            text, is_cross_encoder=False
        )
        self.assertEqual(input_format, "single_string")

        # Step 2: Prepare input
        tokenizer_input = self.tokenizer_manager._prepare_tokenizer_input(
            text, input_format
        )
        self.assertEqual(tokenizer_input, ["Hello world"])

        # Step 3: Extract results (simulated tokenizer output)
        mock_input_ids = [[101, 2129, 4248, 102]]
        mock_token_type_ids = None

        result_input_ids, result_token_type_ids = (
            self.tokenizer_manager._extract_tokenizer_results(
                mock_input_ids, mock_token_type_ids, input_format, original_batch_size=1
            )
        )

        self.assertEqual(result_input_ids, [101, 2129, 4248, 102])
        self.assertIsNone(result_token_type_ids)

    def test_full_workflow_cross_encoder_pairs(self):
        """Test complete workflow for cross-encoder pairs."""
        texts = [
            ["How many people live in Berlin?", "Berlin is well known for its museums."]
        ]

        # Step 1: Detect format
        input_format = self.tokenizer_manager._detect_input_format(
            texts, is_cross_encoder=True
        )
        self.assertEqual(input_format, "cross_encoder_pairs")

        # Step 2: Prepare input
        tokenizer_input = self.tokenizer_manager._prepare_tokenizer_input(
            texts, input_format
        )
        self.assertEqual(tokenizer_input, texts)

        # Step 3: Extract results (simulated tokenizer output for cross-encoder)
        mock_input_ids = [[101, 2129, 2116, 102, 4068, 2003, 102]]
        mock_token_type_ids = [[0, 0, 0, 0, 1, 1, 1]]

        result_input_ids, result_token_type_ids = (
            self.tokenizer_manager._extract_tokenizer_results(
                mock_input_ids, mock_token_type_ids, input_format, original_batch_size=1
            )
        )

        self.assertEqual(result_input_ids, [101, 2129, 2116, 102, 4068, 2003, 102])
        self.assertEqual(result_token_type_ids, [0, 0, 0, 0, 1, 1, 1])

    def test_full_workflow_batch_strings(self):
        """Test complete workflow for batch strings."""
        texts = ["Hello", "World", "Test"]

        # Step 1: Detect format
        input_format = self.tokenizer_manager._detect_input_format(
            texts, is_cross_encoder=False
        )
        self.assertEqual(input_format, "batch_strings")

        # Step 2: Prepare input
        tokenizer_input = self.tokenizer_manager._prepare_tokenizer_input(
            texts, input_format
        )
        self.assertEqual(tokenizer_input, ["Hello", "World", "Test"])

        # Step 3: Extract results (simulated tokenizer output)
        mock_input_ids = [[101, 7592, 102], [101, 2088, 102], [101, 2774, 102]]
        mock_token_type_ids = None

        result_input_ids, result_token_type_ids = (
            self.tokenizer_manager._extract_tokenizer_results(
                mock_input_ids, mock_token_type_ids, input_format, original_batch_size=3
            )
        )

        self.assertEqual(
            result_input_ids, [[101, 7592, 102], [101, 2088, 102], [101, 2774, 102]]
        )
        self.assertIsNone(result_token_type_ids)


if __name__ == "__main__":
    unittest.main(verbosity=2)
