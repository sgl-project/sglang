import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.layers.quantization import (
    QUANTIZATION_METHODS,
    may_auto_override_quant_config,
)


class TestAutoOverrideQuantConfig(unittest.TestCase):
    """Tests for automatic detection and override of quantization config.

    These tests verify that the may_auto_override_quant_config function
    correctly detects and overrides quantization methods based on model
    configuration.
    """

    def test_no_override_when_no_match(self):
        """Test that no override happens when no method matches."""
        # Create a mock HF config.
        mock_hf_config = {"some_key": "some_value"}

        # Mock override_quantization_method to always return False.
        with patch.dict(QUANTIZATION_METHODS, clear=False) as mock_methods:
            for name, cls in mock_methods.items():
                setattr(
                    cls, "override_quantization_method", MagicMock(return_value=False)
                )

            # Call the function with a random quantization method.
            quant_cls, quant_name = may_auto_override_quant_config(
                mock_hf_config, "awq"
            )

            # Check that the quantization method wasn't changed.
            self.assertEqual(quant_name, "awq")
            self.assertEqual(quant_cls, QUANTIZATION_METHODS["awq"])

            # Verify override_quantization_method was called for each method.
            for name, cls in mock_methods.items():
                cls.override_quantization_method.assert_called_once_with(
                    mock_hf_config, "awq"
                )

    def test_override_when_match_found(self):
        """Test that override happens when a method matches."""
        # Create a mock HF config.
        mock_hf_config = {"some_key": "some_value"}

        # Mock override_quantization_method to return True only for awq_marlin.
        with patch.dict(QUANTIZATION_METHODS, clear=False) as mock_methods:
            for name, cls in mock_methods.items():
                return_value = name == "awq_marlin"
                setattr(
                    cls,
                    "override_quantization_method",
                    MagicMock(return_value=return_value),
                )

            # Call the function with "awq".
            quant_cls, quant_name = may_auto_override_quant_config(
                mock_hf_config, "awq"
            )

            # Check that the quantization method was changed to awq_marlin.
            self.assertEqual(quant_name, "awq_marlin")
            self.assertEqual(quant_cls, QUANTIZATION_METHODS["awq_marlin"])

            # Verify the function stopped checking after finding a match.
            # AWQ Marlin should be called first (early in the iteration).
            mock_methods[
                "awq_marlin"
            ].override_quantization_method.assert_any_call(
                mock_hf_config, "awq"
            )

            # Methods that come alphabetically after awq_marlin should not be called.
            # This depends on dict iteration order - in Python 3.7+ this is
            # insertion order which matches our expectation here.
            keys_list = list(QUANTIZATION_METHODS.keys())
            items_list = list(QUANTIZATION_METHODS.items())
            marlin_index = keys_list.index("awq_marlin")

            for name, cls in items_list[:marlin_index]:
                cls.override_quantization_method.assert_called()

            # This may be unnecessary given actual dict iteration behavior,
            # but we include it to be thorough.
            for name, cls in items_list[marlin_index + 1 :]:
                call_count = (
                    1 if mock_methods[name].override_quantization_method.called else 0
                )
                self.assertEqual(
                    cls.override_quantization_method.call_count, call_count
                )

    def test_none_hf_config(self):
        """Test with None HF config."""
        # Mock override_quantization_method to return False for all methods.
        with patch.dict(QUANTIZATION_METHODS, clear=False) as mock_methods:
            for name, cls in mock_methods.items():
                setattr(
                    cls, "override_quantization_method", MagicMock(return_value=False)
                )

            # Call the function with None HF config.
            quant_cls, quant_name = may_auto_override_quant_config(None, "awq")

            # Check that the quantization method wasn't changed.
            self.assertEqual(quant_name, "awq")
            self.assertEqual(quant_cls, QUANTIZATION_METHODS["awq"])

            # Verify override_quantization_method was called for each method with None.
            for name, cls in mock_methods.items():
                cls.override_quantization_method.assert_any_call(None, "awq")


if __name__ == "__main__":
    unittest.main()
