"""Unit test for DummyModelLoader operation ordering.

Verifies that _post_load_weights is called before process_weights_after_loading,
matching the contract established by DefaultModelLoader (where post_load_weights
runs inside model.load_weights() before the outer process_weights_after_loading loop).
"""

import unittest
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDummyModelLoaderOrder(unittest.TestCase):
    @patch("sglang.srt.model_loader.loader._post_load_weights")
    @patch("sglang.srt.model_loader.loader.initialize_dummy_weights")
    @patch("sglang.srt.model_loader.loader._initialize_model")
    @patch("sglang.srt.model_loader.loader._get_quantization_config", return_value=None)
    @patch("sglang.srt.model_loader.loader.get_bool_env_var", return_value=False)
    def test_post_load_weights_before_process_weights(
        self,
        _mock_env,
        _mock_quant,
        mock_initialize_model,
        mock_initialize_dummy_weights,
        mock_post_load_weights,
    ):
        from sglang.srt.model_loader.loader import DummyModelLoader, LoadConfig

        call_log = []

        # Build a fake model with one module that has a quant_method
        fake_model = MagicMock()
        fake_model.eval.return_value = fake_model

        quant_method = MagicMock()
        quant_method.process_weights_after_loading.side_effect = (
            lambda *args, **kwargs: call_log.append("process_weights_after_loading")
        )

        fake_module = MagicMock()
        fake_module.quant_method = quant_method
        fake_model.named_modules.return_value = [("layer", fake_module)]

        mock_initialize_model.return_value = fake_model
        mock_initialize_dummy_weights.side_effect = (
            lambda *args, **kwargs: call_log.append("initialize_dummy_weights")
        )
        mock_post_load_weights.side_effect = (
            lambda *args, **kwargs: call_log.append("post_load_weights")
        )

        load_config = LoadConfig(load_format="dummy")
        loader = DummyModelLoader(load_config)

        model_config = MagicMock()
        model_config.dtype = None
        device_config = MagicMock()
        device_config.device = "cpu"

        with patch("sglang.srt.model_loader.loader.set_default_torch_dtype"):
            loader.load_model(model_config=model_config, device_config=device_config)

        self.assertLess(
            call_log.index("post_load_weights"),
            call_log.index("process_weights_after_loading"),
            "DummyModelLoader must call _post_load_weights before "
            "process_weights_after_loading",
        )


if __name__ == "__main__":
    unittest.main()
