"""
Unit tests for enable_tokenizer_batch_encode feature.

This tests the batch tokenization functionality which allows processing
multiple text inputs in a single batch for improved performance.

Usage:
python3 -m unittest test_tokenizer_batch_encode.TestTokenizerBatchEncode.test_batch_validation_constraints
python3 -m unittest test_tokenizer_batch_encode.TestTokenizerBatchEncodeUnit.test_batch_tokenize_and_process_logic
python3 -m unittest test_tokenizer_batch_encode.TestTokenizerBatchEncodeLogic.test_batch_processing_path
"""

import asyncio
import unittest
from typing import List
from unittest.mock import AsyncMock, Mock, call, patch

from sglang.srt.managers.io_struct import GenerateReqInput, TokenizedGenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST


class TestTokenizerBatchEncode(unittest.TestCase):
    """Test cases for tokenizer batch encoding validation and setup."""

    def setUp(self):
        """Set up test fixtures."""
        self.server_args = ServerArgs(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            enable_tokenizer_batch_encode=True,
        )
        self.port_args = PortArgs.init_new(self.server_args)

        with patch("zmq.asyncio.Context"), patch(
            "sglang.srt.utils.get_zmq_socket"
        ), patch(
            "sglang.srt.utils.hf_transformers_utils.get_tokenizer"
        ) as mock_tokenizer:

            mock_tokenizer.return_value = Mock(vocab_size=32000)
            self.tokenizer_manager = TokenizerManager(self.server_args, self.port_args)

    def test_batch_encode_enabled(self):
        """Test that batch encoding is enabled when configured."""
        self.assertTrue(self.server_args.enable_tokenizer_batch_encode)

    def test_batch_encode_disabled(self):
        """Test that batch encoding can be disabled."""
        server_args_disabled = ServerArgs(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            enable_tokenizer_batch_encode=False,
        )
        self.assertFalse(server_args_disabled.enable_tokenizer_batch_encode)

    def test_multimodal_input_validation(self):
        """Test that multimodal inputs are rejected in batch mode."""
        req = GenerateReqInput(text="test", image_data=["dummy"])
        req.contains_mm_input = Mock(return_value=True)

        batch_obj = Mock()
        batch_obj.__getitem__ = lambda self, i: req

        self.tokenizer_manager.is_generation = True

        with self.assertRaises(ValueError) as cm:
            self.tokenizer_manager._validate_batch_tokenization_constraints(
                1, batch_obj
            )

        self.assertIn("multimodal", str(cm.exception))

    def test_pretokenized_input_validation(self):
        """Test that pre-tokenized inputs are rejected in batch mode."""
        req = GenerateReqInput(input_ids=[1, 2, 3])

        batch_obj = Mock()
        batch_obj.__getitem__ = lambda self, i: req

        with self.assertRaises(ValueError) as cm:
            self.tokenizer_manager._validate_batch_tokenization_constraints(
                1, batch_obj
            )

        self.assertIn("pre-tokenized", str(cm.exception))

    def test_input_embeds_validation(self):
        """Test that input embeds are rejected in batch mode."""
        req = GenerateReqInput(input_embeds=[0.1, 0.2])

        batch_obj = Mock()
        batch_obj.__getitem__ = lambda self, i: req

        with self.assertRaises(ValueError) as cm:
            self.tokenizer_manager._validate_batch_tokenization_constraints(
                1, batch_obj
            )

        self.assertIn("input_embeds", str(cm.exception))

    def test_valid_text_only_requests_pass_validation(self):
        """Test that valid text-only requests pass validation."""
        # Create valid requests (text-only)
        requests = []
        for i in range(3):
            req = GenerateReqInput(text=f"test text {i}")
            req.contains_mm_input = Mock(return_value=False)
            requests.append(req)

        batch_obj = Mock()
        batch_obj.__getitem__ = Mock(side_effect=lambda i: requests[i])

        # Should not raise any exception
        try:
            self.tokenizer_manager._validate_batch_tokenization_constraints(
                3, batch_obj
            )
        except Exception as e:
            self.fail(f"Validation failed for valid text-only requests: {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
