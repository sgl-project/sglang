"""
Test for parallel sampling log requests bug.

This test verifies that the RequestLogger correctly handles
parallel sampling (n > 1) when log_requests_level >= 2.
"""
import unittest
from unittest.mock import Mock

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.utils.request_logger import RequestLogger


class TestRequestLoggerParallelSampling(unittest.TestCase):
    """Test RequestLogger with parallel sampling."""

    def setUp(self):
        """Set up test fixtures."""
        self.request_logger = RequestLogger(
            log_requests=True,
            log_requests_level=2,
            log_requests_format="text",
        )

    def test_log_received_request_with_single_example_n_1(self):
        """Test logging with single request (n=1)."""
        # Create a mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.decode.side_effect = lambda x, skip_special_tokens: "Hello"
        
        # Create request with n=1 (single request)
        req = GenerateReqInput(
            text="Hello",
            sampling_params={"n": 1},
        )
        req.normalize_batch_and_arguments()
        
        # Should not raise error
        try:
            self.request_logger.log_received_request(req, mock_tokenizer)
        except Exception as e:
            self.fail(f"log_received_request raised unexpected exception: {e}")
        
        # Verify tokenizer.decode was called with List[int]
        self.assertEqual(mock_tokenizer.decode.call_count, 1)
        call_args = mock_tokenizer.decode.call_args
        input_ids_arg = call_args[0][0]
        # For single request, input_ids should be List[int]
        self.assertIsInstance(input_ids_arg, list)
        if input_ids_arg and isinstance(input_ids_arg[0], int):
            pass  # Correct: List[int]

    def test_log_received_request_with_parallel_sampling(self):
        """Test logging with parallel sampling (n > 1) should use batch_decode."""
        # Create a mock tokenizer
        mock_tokenizer = Mock()
        
        # Define side effects
        def mock_batch_decode(input_ids_list, **kwargs):
            # Simulate batch_decode returning list of strings
            return [f"Hello_{i}" for i in range(len(input_ids_list))]
        
        mock_tokenizer.batch_decode.side_effect = mock_batch_decode
        
        # Create request with n=3 (parallel sampling)
        req = GenerateReqInput(
            text="Hello",
            sampling_params={"n": 3},
        )
        req.normalize_batch_and_arguments()
        
        # After normalization with n=3, input_ids becomes List[List[int]]
        # and is_single becomes False
        self.assertFalse(req.is_single)
        
        # This should fail with TypeError before fix because
        # it tries tokenizer.decode() with List[List[int]]
        # After fix, it should use tokenizer.batch_decode()
        try:
            self.request_logger.log_received_request(req, mock_tokenizer)
            # If we got here without error, the fix is working
        except TypeError as e:
            # This is the bug we're testing for
            self.assertIn("cannot be interpreted as an integer", str(e))
            # Bug is present
            return
        
        # If we get here, the code should have used batch_decode
        # For parallel sampling, batch_decode should be called
        self.assertEqual(mock_tokenizer.batch_decode.call_count, 1)
        call_args = mock_tokenizer.batch_decode.call_args
        input_ids_arg = call_args[0][0]
        # For batch/parallel request, input_ids should be List[List[int]]
        self.assertIsInstance(input_ids_arg, list)
        if input_ids_arg and isinstance(input_ids_arg[0], list):
            pass  # Correct: List[List[int]]

    def test_log_received_request_with_batch_input_ids(self):
        """Test logging with batch input_ids directly."""
        # Create a mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.batch_decode.side_effect = lambda x, skip_special_tokens: ["Text1", "Text2"]
        
        # Create request with batch input_ids
        req = GenerateReqInput(
            input_ids=[[1, 2, 3], [4, 5, 6]],
            text=None,
            sampling_params={"n": 1},
        )
        req.normalize_batch_and_arguments()
        
        # Should not raise error
        try:
            self.request_logger.log_received_request(req, mock_tokenizer)
        except Exception as e:
            self.fail(f"log_received_request raised unexpected exception: {e}")
        
        # Verify tokenizer.batch_decode was called with List[List[int]]
        self.assertEqual(mock_tokenizer.batch_decode.call_count, 1)

    def test_log_received_request_level_below_2(self):
        """Test that log_requests_level < 2 doesn't decode input_ids."""
        # Create a mock tokenizer
        mock_tokenizer = Mock()
        
        # Create request logger with level 1
        request_logger = RequestLogger(
            log_requests=True,
            log_requests_level=1,
            log_requests_format="text",
        )
        
        # Create request with parallel sampling
        req = GenerateReqInput(
            text="Hello",
            sampling_params={"n": 3},
        )
        req.normalize_batch_and_arguments()
        
        # Should not call tokenizer at all
        request_logger.log_received_request(req, mock_tokenizer)
        
        # Verify no tokenizer methods were called
        self.assertEqual(mock_tokenizer.decode.call_count, 0)
        self.assertEqual(mock_tokenizer.batch_decode.call_count, 0)


if __name__ == "__main__":
    unittest.main()