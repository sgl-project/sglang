"""Unit tests for RunAI streamer loading (quant_config + distributed tensor clone).

No server, no real model weights — mocks only.
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.model_loader.loader import RunaiModelStreamerLoader
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestRunaiModelStreamerLoaderQuantConfig(CustomTestCase):
    def test_load_model_passes_quantization_config_to_initialize(self):
        load_config = LoadConfig(load_format=LoadFormat.RUNAI_STREAMER)
        loader = RunaiModelStreamerLoader(load_config)
        mock_model = MagicMock()
        qcfg = object()
        model_config = MagicMock()
        model_config.modelopt_quant = False
        model_config.dtype = torch.float16
        device_config = DeviceConfig(device="cpu", gpu_id=-1)

        with (
            patch(
                "sglang.srt.model_loader.loader._get_quantization_config",
                return_value=qcfg,
            ) as mock_get_q,
            patch(
                "sglang.srt.model_loader.loader._initialize_model",
                return_value=mock_model,
            ) as mock_init,
            patch(
                "sglang.srt.model_loader.loader.DefaultModelLoader.load_weights_and_postprocess"
            ),
            patch.object(
                loader,
                "_get_all_weights",
                return_value=iter(()),
            ),
        ):
            out = loader.load_model(
                model_config=model_config, device_config=device_config
            )

        mock_get_q.assert_called_once_with(model_config, load_config)
        mock_init.assert_called_once_with(model_config, load_config, qcfg)
        self.assertIs(out, mock_model)


class TestRunaiSafetensorsWeightsIterator(CustomTestCase):
    """Stub ``runai_model_streamer`` so CI does not need the optional wheel."""

    def setUp(self):
        super().setUp()
        self._saved_runai = sys.modules.pop("runai_model_streamer", None)
        stub = types.ModuleType("runai_model_streamer")

        def _mock_streamer_factory():
            mock_streamer = MagicMock()
            mock_streamer.files_to_tensors_metadata = {"dummy.safetensors": [{}]}

            def _set_tensors(tensors):
                mock_streamer.get_tensors.return_value = iter(tensors)
                mock_streamer.__enter__ = MagicMock(return_value=mock_streamer)
                mock_streamer.__exit__ = MagicMock(return_value=False)
                return mock_streamer

            mock_streamer._set_tensors = _set_tensors
            return mock_streamer

        self._streamer_instance = _mock_streamer_factory()
        stub.SafetensorsStreamer = MagicMock(return_value=self._streamer_instance)
        sys.modules["runai_model_streamer"] = stub

    def tearDown(self):
        if self._saved_runai is not None:
            sys.modules["runai_model_streamer"] = self._saved_runai
        else:
            sys.modules.pop("runai_model_streamer", None)
        super().tearDown()

    @patch("sglang.srt.model_loader.weight_utils.tqdm", side_effect=lambda x, **k: x)
    def test_distributed_always_clones_tensors(self, _tqdm):
        from sglang.srt.model_loader.weight_utils import (
            runai_safetensors_weights_iterator,
        )

        t = torch.tensor([1.0, 2.0])
        self._streamer_instance._set_tensors([("w", t)])

        name, got = next(
            iter(runai_safetensors_weights_iterator(["p"], is_distributed=True))
        )
        self.assertEqual(name, "w")
        self.assertTrue(torch.equal(got, t))
        self.assertIsNot(got.untyped_storage(), t.untyped_storage())

    @patch("sglang.srt.model_loader.weight_utils.tqdm", side_effect=lambda x, **k: x)
    def test_non_distributed_yields_same_tensor(self, _tqdm):
        from sglang.srt.model_loader.weight_utils import (
            runai_safetensors_weights_iterator,
        )

        t = torch.tensor([3.0])
        self._streamer_instance._set_tensors([("w", t)])

        name, got = next(
            iter(runai_safetensors_weights_iterator(["p"], is_distributed=False))
        )
        self.assertIs(got, t)


if __name__ == "__main__":
    unittest.main()
