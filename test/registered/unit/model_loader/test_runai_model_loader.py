import unittest
from types import SimpleNamespace
from unittest.mock import patch, sentinel

import torch
import torch.nn as nn

from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.model_loader.loader import RunaiModelStreamerLoader
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestRunaiModelStreamerLoader(CustomTestCase):
    def test_load_model_passes_quant_config_to_initialize_model(self):
        load_config = LoadConfig(load_format=LoadFormat.RUNAI_STREAMER)
        model_config = SimpleNamespace(
            dtype=torch.float32,
            model_path="/tmp/model",
            revision=None,
            modelopt_quant=False,
        )
        device_config = DeviceConfig(device="cpu")
        model = nn.Module()

        with (
            patch("sglang.srt.model_loader.loader.set_runai_streamer_env"),
            patch(
                "sglang.srt.model_loader.loader._get_quantization_config",
                return_value=sentinel.quant_config,
            ) as get_quantization_config,
            patch(
                "sglang.srt.model_loader.loader._initialize_model",
                return_value=model,
            ) as initialize_model,
            patch(
                "sglang.srt.model_loader.loader.DefaultModelLoader."
                "load_weights_and_postprocess"
            ) as load_weights_and_postprocess,
        ):
            loader = RunaiModelStreamerLoader(load_config)
            loaded_model = loader.load_model(
                model_config=model_config,
                device_config=device_config,
            )

        self.assertIs(loaded_model, model)
        get_quantization_config.assert_called_once_with(model_config, load_config)
        initialize_model.assert_called_once_with(
            model_config,
            load_config,
            sentinel.quant_config,
        )
        load_weights_and_postprocess.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=3)
