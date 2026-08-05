import sys
import unittest
from unittest.mock import patch


class TestAWQNoVllm(unittest.TestCase):
    def test_import_awq_without_vllm(self):
        """
        Test that AWQConfig can be imported and instantiated even if vllm is not available.
        This ensures that we don't have a hard dependency on vllm for AWQ.
        Ref: Issue #6580
        """
        # Patch sys.modules to simulate vllm missing
        with patch.dict(sys.modules, {"vllm": None}):
            # Force reload of the module if it was already imported
            if "sglang.srt.layers.quantization.awq" in sys.modules:
                del sys.modules["sglang.srt.layers.quantization.awq"]

            try:
                from sglang.srt.layers.quantization.awq import AWQConfig

                config = AWQConfig(weight_bits=4, group_size=128, zero_point=True)
                self.assertIsInstance(config, AWQConfig)
                self.assertEqual(config.weight_bits, 4)
            except ImportError as e:
                self.fail(f"Failed to import AWQConfig without vllm: {e}")
            except ValueError as e:
                # If the error message mentions vllm, it's a regression
                if "vllm" in str(e).lower():
                    self.fail(f"AWQConfig raised ValueError about vllm: {e}")
                else:
                    raise e


if __name__ == "__main__":
    unittest.main()
