"""Unit test for DummyModelLoader operation ordering.

Verifies that _post_load_weights is called before process_weights_after_loading,
matching the contract established by DefaultModelLoader (where post_load_weights
runs inside model.load_weights() before the outer process_weights_after_loading loop).
"""

import unittest
from unittest.mock import MagicMock, call, patch

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
        mock_init_model,
        mock_init_dummy,
        mock_post_load,
    ):
        from sglang.srt.model_loader.loader import DummyModelLoader, LoadConfig

        # Build a fake model with one module that has a quant_method
        fake_model = MagicMock()
        fake_model.eval.return_value = fake_model

        quant_method = MagicMock()
        quant_method.process_weights_after_loading = MagicMock()

        fake_module = MagicMock(spec=[])  # spec=[] → hasattr returns False for everything
        fake_module.quant_method = quant_method

        fake_model.named_modules.return_value = [("layer", fake_module)]
        mock_init_model.return_value = fake_model

        manager = MagicMock()
        manager.mock_calls  # reset

        # Track global call order across the three operations
        call_log = []
        mock_post_load.side_effect = lambda *a, **kw: call_log.append("post_load_weights")
        mock_init_dummy.side_effect = lambda *a, **kw: call_log.append(
            "initialize_dummy_weights"
        )
        quant_method.process_weights_after_loading.side_effect = lambda *a, **kw: call_log.append(
            "process_weights_after_loading"
        )

        load_config = LoadConfig(load_format="dummy")
        loader = DummyModelLoader(load_config)

        model_config = MagicMock()
        model_config.dtype = None
        device_config = MagicMock()
        device_config.device = "cpu"

        with patch("sglang.srt.model_loader.loader.set_default_torch_dtype"):
            loader.load_model(model_config=model_config, device_config=device_config)

        self.assertEqual(
            call_log,
            [
                "initialize_dummy_weights",
                "post_load_weights",
                "process_weights_after_loading",
            ],
            "DummyModelLoader must call post_load_weights before process_weights_after_loading",
        )


if __name__ == "__main__":
    unittest.main()
