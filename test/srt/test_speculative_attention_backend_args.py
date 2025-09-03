import unittest
from unittest.mock import patch
import sys

from sglang.srt.server_args import ServerArgs


class TestSpeculativeAttentionBackendArgs(unittest.TestCase):
    """Test the speculative attention backend argument parsing."""

    def test_default_speculative_attention_backend(self):
        """Test that the default value is 'prefill'."""
        args = ServerArgs()
        self.assertEqual(args.speculative_attention_backend, "prefill")

    def test_prefill_backend_argument(self):
        """Test parsing --speculative-attention-backend prefill."""
        test_args = [
            "test_script.py",
            "--model-path", "dummy_model",
            "--speculative-attention-backend", "prefill"
        ]
        
        with patch.object(sys, 'argv', test_args):
            args = ServerArgs.add_cli_args(ServerArgs.add_parser())
            parsed_args = args.parse_args(test_args[1:])  # Skip script name
            
            # Create ServerArgs from parsed args
            server_args = ServerArgs()
            for key, value in vars(parsed_args).items():
                if hasattr(server_args, key):
                    setattr(server_args, key, value)
            
            self.assertEqual(server_args.speculative_attention_backend, "prefill")

    def test_decode_backend_argument(self):
        """Test parsing --speculative-attention-backend decode."""
        test_args = [
            "test_script.py",
            "--model-path", "dummy_model", 
            "--speculative-attention-backend", "decode"
        ]
        
        with patch.object(sys, 'argv', test_args):
            args = ServerArgs.add_cli_args(ServerArgs.add_parser())
            parsed_args = args.parse_args(test_args[1:])  # Skip script name
            
            # Create ServerArgs from parsed args
            server_args = ServerArgs()
            for key, value in vars(parsed_args).items():
                if hasattr(server_args, key):
                    setattr(server_args, key, value)
            
            self.assertEqual(server_args.speculative_attention_backend, "decode")

    def test_invalid_backend_argument(self):
        """Test that invalid backend values are rejected."""
        test_args = [
            "test_script.py",
            "--model-path", "dummy_model",
            "--speculative-attention-backend", "invalid"
        ]
        
        with patch.object(sys, 'argv', test_args):
            args = ServerArgs.add_cli_args(ServerArgs.add_parser())
            with self.assertRaises(SystemExit):
                args.parse_args(test_args[1:])


if __name__ == "__main__":
    unittest.main()