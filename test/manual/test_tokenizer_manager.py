"""
Unit tests for TokenizerManager helper methods.

This tests the refactored tokenization functionality including input format detection,
tokenizer input preparation, result extraction logic, and ReqState text buffering.

Usage:
python3 -m unittest test_tokenizer_manager.TestInputFormatDetection
python3 -m unittest test_tokenizer_manager.TestTokenizerInputPreparation
python3 -m unittest test_tokenizer_manager.TestTokenizerResultExtraction
python3 -m unittest test_tokenizer_manager.TestTokenizerManagerIntegration
python3 -m unittest test_tokenizer_manager.TestReqStateTextBuffering
python3 -m unittest test_tokenizer_manager.TestReqStateCrashDump
"""

import asyncio
import unittest
from unittest.mock import Mock, patch

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import (
    InputFormat,
    ReqState,
    TokenizerManager,
)
from sglang.srt.observability.req_time_stats import APIServerReqTimeStats
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST


class TestInputFormatDetection(unittest.TestCase):
    """Test cases for _detect_input_format method."""

    def setUp(self):
        """Set up test fixtures."""
        with patch("sglang.srt.utils.get_device", return_value="cpu"):
            self.server_args = ServerArgs(model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST)
            self.port_args = PortArgs.init_new(self.server_args)

        with (
            patch("zmq.asyncio.Context"),
            patch("sglang.srt.utils.network.get_zmq_socket"),
            patch(
                "sglang.srt.utils.hf_transformers_utils.get_tokenizer"
            ) as mock_tokenizer,
        ):
            mock_tokenizer.return_value = Mock(vocab_size=32000)
            self.tokenizer_manager = TokenizerManager(self.server_args, self.port_args)

    def test_detect_single_string(self):
        """Test detection of single string input."""
        text = "Hello world"
        result = self.tokenizer_manager._detect_input_format(
            text, is_cross_encoder=False
        )
        self.assertEqual(result, InputFormat.SINGLE_STRING)

    def test_detect_single_string_cross_encoder_disabled(self):
        """Test single string with cross_encoder disabled still returns single_string."""
        text = "Hello world"
        result = self.tokenizer_manager._detect_input_format(
            text, is_cross_encoder=True
        )
        self.assertEqual(result, InputFormat.SINGLE_STRING)

    def test_detect_batch_strings(self):
        """Test detection of batch string inputs."""
        texts = ["Hello", "World", "How are you?"]
        result = self.tokenizer_manager._detect_input_format(
            texts, is_cross_encoder=False
        )
        self.assertEqual(result, InputFormat.BATCH_STRINGS)

    def test_detect_batch_strings_cross_encoder_disabled(self):
        """Test batch strings with cross_encoder disabled."""
        texts = ["Hello", "World"]
        result = self.tokenizer_manager._detect_input_format(
            texts, is_cross_encoder=True
        )
        self.assertEqual(result, InputFormat.BATCH_STRINGS)

    def test_detect_cross_encoder_single_pair(self):
        """Test detection of cross-encoder single pair."""
        texts = [["query text", "document text"]]
        result = self.tokenizer_manager._detect_input_format(
            texts, is_cross_encoder=True
        )
        self.assertEqual(result, InputFormat.CROSS_ENCODER_PAIRS)

    def test_detect_cross_encoder_multiple_pairs(self):
        """Test detection of cross-encoder multiple pairs."""
        texts = [["q1", "d1"], ["q2", "d2"], ["q3", "d3"]]
        result = self.tokenizer_manager._detect_input_format(
            texts, is_cross_encoder=True
        )
        self.assertEqual(result, InputFormat.CROSS_ENCODER_PAIRS)

    def test_detect_cross_encoder_disabled_with_pairs(self):
        """Test pairs with cross_encoder disabled should return batch_strings."""
        texts = [["query", "document"]]
        result = self.tokenizer_manager._detect_input_format(
            texts, is_cross_encoder=False
        )
        self.assertEqual(result, InputFormat.BATCH_STRINGS)

    def test_detect_empty_list(self):
        """Test detection with empty list."""
        texts = []
        result = self.tokenizer_manager._detect_input_format(
            texts, is_cross_encoder=True
        )
        self.assertEqual(result, InputFormat.BATCH_STRINGS)

    def test_detect_malformed_cross_encoder_pairs(self):
        """Test malformed cross-encoder pairs (not length 2)."""
        texts = [["query only"]]  # Single element, not a pair
        result = self.tokenizer_manager._detect_input_format(
            texts, is_cross_encoder=True
        )
        self.assertEqual(result, InputFormat.BATCH_STRINGS)

        texts = [["query", "doc", "extra"]]  # Three elements, not a pair
        result = self.tokenizer_manager._detect_input_format(
            texts, is_cross_encoder=True
        )
        self.assertEqual(result, InputFormat.BATCH_STRINGS)


class TestTokenizerInputPreparation(unittest.TestCase):
    """Test cases for _prepare_tokenizer_input method."""

    def setUp(self):
        """Set up test fixtures."""
        with patch("sglang.srt.utils.get_device", return_value="cpu"):
            self.server_args = ServerArgs(model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST)
            self.port_args = PortArgs.init_new(self.server_args)

        with (
            patch("zmq.asyncio.Context"),
            patch("sglang.srt.utils.network.get_zmq_socket"),
            patch(
                "sglang.srt.utils.hf_transformers_utils.get_tokenizer"
            ) as mock_tokenizer,
        ):
            mock_tokenizer.return_value = Mock(vocab_size=32000)
            self.tokenizer_manager = TokenizerManager(self.server_args, self.port_args)

    def test_prepare_single_string_input(self):
        """Test preparation of single string input."""
        text = "Hello world"
        result = self.tokenizer_manager._prepare_tokenizer_input(
            text, InputFormat.SINGLE_STRING
        )
        self.assertEqual(result, ["Hello world"])

    def test_prepare_batch_strings_input(self):
        """Test preparation of batch strings input."""
        texts = ["Hello", "World", "Test"]
        result = self.tokenizer_manager._prepare_tokenizer_input(
            texts, InputFormat.BATCH_STRINGS
        )
        self.assertEqual(result, ["Hello", "World", "Test"])

    def test_prepare_cross_encoder_pairs_input(self):
        """Test preparation of cross-encoder pairs input."""
        texts = [["query1", "doc1"], ["query2", "doc2"]]
        result = self.tokenizer_manager._prepare_tokenizer_input(
            texts, InputFormat.CROSS_ENCODER_PAIRS
        )
        self.assertEqual(result, [["query1", "doc1"], ["query2", "doc2"]])

    def test_prepare_cross_encoder_single_pair_input(self):
        """Test preparation of single cross-encoder pair."""
        texts = [["query text", "document text"]]
        result = self.tokenizer_manager._prepare_tokenizer_input(
            texts, InputFormat.CROSS_ENCODER_PAIRS
        )
        self.assertEqual(result, [["query text", "document text"]])

    def test_prepare_batch_strings_input_format_passthrough(self):
        """Batch strings should pass through unchanged."""
        texts = ["test"]
        result = self.tokenizer_manager._prepare_tokenizer_input(
            texts, InputFormat.BATCH_STRINGS
        )
        self.assertEqual(result, ["test"])


class TestTokenizerResultExtraction(unittest.TestCase):
    """Test cases for _extract_tokenizer_results method."""

    def setUp(self):
        """Set up test fixtures."""
        with patch("sglang.srt.utils.get_device", return_value="cpu"):
            self.server_args = ServerArgs(model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST)
            self.port_args = PortArgs.init_new(self.server_args)

        with (
            patch("zmq.asyncio.Context"),
            patch("sglang.srt.utils.network.get_zmq_socket"),
            patch(
                "sglang.srt.utils.hf_transformers_utils.get_tokenizer"
            ) as mock_tokenizer,
        ):
            mock_tokenizer.return_value = Mock(vocab_size=32000)
            self.tokenizer_manager = TokenizerManager(self.server_args, self.port_args)

    def test_extract_single_string_results(self):
        """Test extraction for single string input."""
        input_ids = [[101, 2129, 102]]
        token_type_ids = [[0, 0, 0]]

        result_input_ids, result_token_type_ids = (
            self.tokenizer_manager._extract_tokenizer_results(
                input_ids,
                token_type_ids,
                InputFormat.SINGLE_STRING,
                original_batch_size=1,
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
                input_ids,
                token_type_ids,
                InputFormat.CROSS_ENCODER_PAIRS,
                original_batch_size=1,
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
                input_ids,
                token_type_ids,
                InputFormat.BATCH_STRINGS,
                original_batch_size=2,
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
                input_ids,
                token_type_ids,
                InputFormat.CROSS_ENCODER_PAIRS,
                original_batch_size=2,
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
                input_ids,
                token_type_ids,
                InputFormat.SINGLE_STRING,
                original_batch_size=1,
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
                input_ids,
                token_type_ids,
                InputFormat.SINGLE_STRING,
                original_batch_size=1,
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

        with (
            patch("zmq.asyncio.Context"),
            patch("sglang.srt.utils.network.get_zmq_socket"),
            patch(
                "sglang.srt.utils.hf_transformers_utils.get_tokenizer"
            ) as mock_tokenizer,
        ):
            mock_tokenizer.return_value = Mock(vocab_size=32000)
            self.tokenizer_manager = TokenizerManager(self.server_args, self.port_args)

    def test_full_workflow_single_string(self):
        """Test complete workflow for single string input."""
        text = "Hello world"

        # Step 1: Detect format
        input_format = self.tokenizer_manager._detect_input_format(
            text, is_cross_encoder=False
        )
        self.assertEqual(input_format, InputFormat.SINGLE_STRING)

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
        self.assertEqual(input_format, InputFormat.CROSS_ENCODER_PAIRS)

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
        self.assertEqual(input_format, InputFormat.BATCH_STRINGS)

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


class TestDetokenizeTopLogprobsTokens(unittest.TestCase):
    """Test cases for detokenize_top_logprobs_tokens batched decoding.

    We avoid constructing a real TokenizerManager (which requires GPU-dependent
    setup) and instead invoke the method as an unbound function against a bare
    stand-in object that only exposes the attributes the method actually uses.
    """

    def setUp(self):
        self.fn = TokenizerManager.detokenize_top_logprobs_tokens
        self.stub = Mock(
            spec=["tokenizer", "detokenize_logprob_tokens", "_batch_decode_token_ids"]
        )
        self.stub.tokenizer = Mock(spec=["batch_decode", "backend_tokenizer"])
        self.stub.tokenizer.backend_tokenizer = None
        self.stub.tokenizer.batch_decode = Mock(
            side_effect=lambda ids: [f"tok_{i[0]}" for i in ids]
        )

        def batch_decode_token_ids(token_ids):
            return TokenizerManager._batch_decode_token_ids(self.stub, token_ids)

        def detokenize_logprob_tokens(vals, idxs, decode_to_text):
            return TokenizerManager.detokenize_logprob_tokens(
                self.stub, vals, idxs, decode_to_text
            )

        self.stub._batch_decode_token_ids = batch_decode_token_ids
        # Delegate to the real helper so we exercise the production path.
        self.stub.detokenize_logprob_tokens = detokenize_logprob_tokens

    def _call(self, vals, idxs, decode_to_text):
        return self.fn(self.stub, vals, idxs, decode_to_text)

    def _reference_impl(self, vals, idxs, decode_to_text):
        """Per-position reference for what the old implementation produced."""
        ret = []
        for i in range(len(vals)):
            if vals[i]:
                if not decode_to_text:
                    ret.append([(lp, tid, None) for lp, tid in zip(vals[i], idxs[i])])
                else:
                    texts = [f"tok_{tid}" for tid in idxs[i]]
                    ret.append(list(zip(vals[i], idxs[i], texts)))
            else:
                ret.append(None)
        return ret

    def test_decode_to_text_false_skips_tokenizer(self):
        """When decode_to_text=False, batch_decode should not be called."""
        vals = [[-0.1, -0.2], [-0.3]]
        idxs = [[10, 20], [30]]

        result = self._call(vals, idxs, decode_to_text=False)

        self.stub.tokenizer.batch_decode.assert_not_called()
        self.assertEqual(
            result,
            [[(-0.1, 10, None), (-0.2, 20, None)], [(-0.3, 30, None)]],
        )

    def test_all_empty_positions_returns_nones(self):
        """All-empty input returns [None, ...] and never calls batch_decode."""
        vals = [[], [], None]
        idxs = [[], [], None]

        result = self._call(vals, idxs, decode_to_text=True)

        self.assertEqual(result, [None, None, None])
        self.stub.tokenizer.batch_decode.assert_not_called()

    def test_empty_input_list(self):
        """Zero-length input returns an empty list."""
        result = self._call([], [], decode_to_text=True)
        self.assertEqual(result, [])
        self.stub.tokenizer.batch_decode.assert_not_called()

    def test_mixed_empty_and_nonempty_positions(self):
        """Texts are sliced back to the correct positions; empties stay None."""
        vals = [[-0.1, -0.2], [], [-0.5, -0.6, -0.7], None, [-0.9]]
        idxs = [[10, 20], [], [30, 40, 50], None, [60]]

        result = self._call(vals, idxs, decode_to_text=True)

        expected = [
            [(-0.1, 10, "tok_10"), (-0.2, 20, "tok_20")],
            None,
            [(-0.5, 30, "tok_30"), (-0.6, 40, "tok_40"), (-0.7, 50, "tok_50")],
            None,
            [(-0.9, 60, "tok_60")],
        ]
        self.assertEqual(result, expected)

    def test_single_batch_decode_call_with_flattened_ids(self):
        """Efficiency guarantee: batch_decode is called once with flattened token ids."""
        vals = [[-0.1, -0.2], [], [-0.5, -0.6, -0.7], [-0.9]]
        idxs = [[10, 20], [], [30, 40, 50], [60]]

        self._call(vals, idxs, decode_to_text=True)

        self.assertEqual(self.stub.tokenizer.batch_decode.call_count, 1)
        (called_ids,), _ = self.stub.tokenizer.batch_decode.call_args
        self.assertEqual(list(called_ids), [[10], [20], [30], [40], [50], [60]])

    def test_prefers_backend_decode_batch_when_available(self):
        """Use the tokenizer backend to avoid per-sequence Python decode loops."""
        vals = [[-0.1, -0.2], [], [-0.5, -0.6, -0.7], [-0.9]]
        idxs = [[10, 20], [], [30, 40, 50], [60]]
        backend_tokenizer = Mock(spec=["decode_batch"])
        backend_tokenizer.decode_batch = Mock(
            side_effect=lambda ids, skip_special_tokens: [
                f"tok_{i[0]}_{skip_special_tokens}" for i in ids
            ]
        )
        self.stub.tokenizer.backend_tokenizer = backend_tokenizer

        result = self._call(vals, idxs, decode_to_text=True)

        self.stub.tokenizer.batch_decode.assert_not_called()
        backend_tokenizer.decode_batch.assert_called_once_with(
            [[10], [20], [30], [40], [50], [60]], skip_special_tokens=False
        )
        self.assertEqual(result[0][0], (-0.1, 10, "tok_10_False"))
        self.assertEqual(result[2][2], (-0.7, 50, "tok_50_False"))

    def test_matches_reference_per_position_implementation(self):
        """Batched result is equivalent to per-position decoding."""
        vals = [[-0.1, -0.2, -0.3], [], [-0.5], None, [-0.7, -0.8]]
        idxs = [[10, 20, 30], [], [40], None, [50, 60]]

        for decode_to_text in (True, False):
            self.stub.tokenizer.batch_decode.reset_mock()
            batched = self._call(vals, idxs, decode_to_text=decode_to_text)
            expected = self._reference_impl(vals, idxs, decode_to_text)
            self.assertEqual(batched, expected)


def _make_state() -> ReqState:
    """Create a minimal ReqState for testing."""
    obj = Mock(spec=GenerateReqInput)
    return ReqState(
        out_list=[],
        finished=False,
        event=asyncio.Event(),
        obj=obj,
        time_stats=APIServerReqTimeStats(),
    )


class TestReqStateTextBuffering(unittest.TestCase):
    """Test ReqState.append_text / get_text in both buffering modes."""

    def test_collects_chunks_lazily(self):
        state = _make_state()
        state.append_text("hello ")
        state.append_text("world")
        self.assertEqual(state.text, "")
        self.assertEqual(state.text_chunks, ["hello ", "world"])
        self.assertEqual(state.get_text(), "hello world")
        self.assertEqual(state.text_chunks, [])

    def test_get_text_preserves_materialized_prefix(self):
        state = _make_state()
        state.append_text("hello ")
        self.assertEqual(state.get_text(), "hello ")
        state.append_text("world")
        self.assertEqual(state.get_text(), "hello world")


class TestReqStateCrashDump(unittest.TestCase):
    """Test ReqState.get_crash_dump_output."""

    def test_empty_state(self):
        state = _make_state()
        self.assertEqual(state.get_crash_dump_output(), {})

    def test_with_text_only(self):
        state = _make_state()
        state.append_text("partial output")
        self.assertEqual(state.get_crash_dump_output(), {"text": "partial output"})

    def test_with_output_ids_only(self):
        state = _make_state()
        state.output_ids = [1, 2, 3]
        self.assertEqual(state.get_crash_dump_output(), {"output_ids": [1, 2, 3]})

    def test_with_text_and_output_ids(self):
        state = _make_state()
        state.append_text("hello")
        state.output_ids = [10, 20]
        self.assertEqual(
            state.get_crash_dump_output(),
            {"text": "hello", "output_ids": [10, 20]},
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
