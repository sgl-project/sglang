# SPDX-License-Identifier: Apache-2.0
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.layers.moe.topk import TopKConfig
from sglang.srt.weight_cache.ipc_loader import IpcModelLoader
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestWeightCacheMoERoutingBias(CustomTestCase):
    def setUp(self):
        self.owner_bias = torch.arange(16, dtype=torch.float32)[4:12]
        self.other_bias = self.owner_bias.clone()
        self.independent_bias = torch.full((8,), -1.0, dtype=torch.float32)

    def _attach(self):
        def make_model(*_):
            model = torch.nn.Module()
            model.gate = torch.nn.Module()
            model.gate.bias = torch.nn.Parameter(torch.empty(8))
            model.register_buffer("other_bias", torch.empty(8))
            for name, bias in (
                ("direct", model.gate.bias),
                ("data", model.gate.bias.data),
                ("slice", model.gate.bias[1::2]),
                ("other", model.other_bias),
                ("independent", self.independent_bias),
                ("unbiased", None),
            ):
                router = torch.nn.Module()
                router.topk_config = TopKConfig(top_k=1, correction_bias=bias)
                model.add_module(name + "_router", router)
            model.correction_bias = model.gate.bias.data[2:6]
            self.old_cast = model.direct_router.topk_config.correction_bias_for_dtype(
                torch.float64
            )
            self.independent_cast = (
                model.independent_router.topk_config.correction_bias_for_dtype(
                    torch.float64
                )
            )
            return model

        loader = IpcModelLoader(LoadConfig(load_format=LoadFormat.IPC_CACHE))
        entries = {
            "gate.bias": {"tensor": self.owner_bias},
            "other_bias": {"tensor": self.other_bias, "is_param": False},
        }
        with (
            patch.object(loader, "_resolve_engine_quant", return_value=("", None)),
            patch.object(
                loader, "_fetch_from_cache", return_value={"entries": entries}
            ),
            patch.object(
                loader._transport_backend,
                "import_tensor",
                side_effect=lambda entry: entry["tensor"],
            ),
            patch(
                "sglang.srt.model_loader.loader._get_quantization_config",
                return_value=None,
            ),
            patch(
                "sglang.srt.weight_cache.ipc_loader._initialize_model",
                side_effect=make_model,
            ),
        ):
            return loader.load_model(
                model_config=SimpleNamespace(dtype=torch.float32), device_config=None
            )

    def test_restores_only_biases_referencing_attached_tensors(self):
        client = self._attach()
        self.assertIs(
            client.direct_router.topk_config.correction_bias, client.gate.bias
        )
        self.assertIs(
            client.other_router.topk_config.correction_bias, client.other_bias
        )
        self.assertNotEqual(
            client.gate.bias.untyped_storage()._cdata,
            client.other_bias.untyped_storage()._cdata,
        )
        self.assertIs(
            client.independent_router.topk_config.correction_bias, self.independent_bias
        )
        self.assertIsNone(client.unbiased_router.topk_config.correction_bias)

        for actual, expected in (
            (client.data_router.topk_config.correction_bias, client.gate.bias.data),
            (client.slice_router.topk_config.correction_bias, client.gate.bias[1::2]),
            (client.correction_bias, client.gate.bias[2:6]),
        ):
            with self.subTest(
                stride=expected.stride(), offset=expected.storage_offset()
            ):
                self.assertNotIsInstance(actual, torch.nn.Parameter)
                self.assertEqual(
                    actual.untyped_storage()._cdata,
                    client.gate.bias.untyped_storage()._cdata,
                )
                self.assertEqual(actual.stride(), expected.stride())
                self.assertEqual(actual.storage_offset(), expected.storage_offset())
                torch.testing.assert_close(actual, expected)

    def test_invalidates_only_rebound_dtype_cache(self):
        client = self._attach()
        config = client.direct_router.topk_config
        self.assertTrue(self.old_cast.is_meta)
        self.assertIsNone(config._correction_bias_dtype_cache)
        self.assertIsNone(config._correction_bias_cache_key)
        converted = config.correction_bias_for_dtype(torch.float64)
        torch.testing.assert_close(converted, self.owner_bias.to(torch.float64))
        self.assertIs(config.correction_bias_for_dtype(torch.float64), converted)
        independent = client.independent_router.topk_config
        self.assertIs(
            independent.correction_bias_for_dtype(torch.float64), self.independent_cast
        )
        torch.testing.assert_close(
            self.owner_bias, torch.arange(4, 12, dtype=torch.float32)
        )


if __name__ == "__main__":
    unittest.main()
