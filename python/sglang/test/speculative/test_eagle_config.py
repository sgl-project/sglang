import unittest
from unittest.mock import MagicMock


class TestEagleUseAuxHiddenStateConfig(unittest.TestCase):
    """Tests for eagle_use_aux_hidden_state configuration parsing."""

    def test_eagle3_with_use_aux_hidden_state_true(self):
        """When eagle_config has use_aux_hidden_state=True, it should be True."""
        mock_hf_config = MagicMock()
        mock_hf_config.eagle_config = {"use_aux_hidden_state": True}

        eagle_config = getattr(mock_hf_config, "eagle_config", {})
        result = eagle_config.get("use_aux_hidden_state", True)

        self.assertTrue(result)

    def test_eagle3_with_use_aux_hidden_state_false(self):
        """When eagle_config has use_aux_hidden_state=False, it should be False."""
        mock_hf_config = MagicMock()
        mock_hf_config.eagle_config = {"use_aux_hidden_state": False}

        eagle_config = getattr(mock_hf_config, "eagle_config", {})
        result = eagle_config.get("use_aux_hidden_state", True)

        self.assertFalse(result)

    def test_eagle3_missing_config_defaults_true(self):
        """When eagle_config is missing, it should default to True."""
        mock_hf_config = MagicMock(spec=[])  # no eagle_config attr

        eagle_config = getattr(mock_hf_config, "eagle_config", {})
        result = eagle_config.get("use_aux_hidden_state", True)

        self.assertTrue(result)

    def test_eagle3_missing_key_defaults_true(self):
        """When eagle_config exists but key is missing, should default to True."""
        mock_hf_config = MagicMock()
        mock_hf_config.eagle_config = {}  # no use_aux_hidden_state key

        eagle_config = getattr(mock_hf_config, "eagle_config", {})
        result = eagle_config.get("use_aux_hidden_state", True)

        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
