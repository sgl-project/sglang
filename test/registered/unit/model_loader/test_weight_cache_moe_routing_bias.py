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
        self.owner_bias = torch.arange(8, dtype=torch.float32)
        self.other_bias = self.owner_bias.clone()
        self.independent_bias = torch.full((8,), -1.0, dtype=torch.float32)

    @staticmethod
    def _router(bias):
        router = torch.nn.Module()
        router.topk_config = TopKConfig(top_k=1, correction_bias=bias)
        return router

    def _make_model(self, *_):
        model = torch.nn.Module()
        model.gate = torch.nn.Module()
        model.gate.bias = torch.nn.Parameter(torch.empty(8))
        model.gate.other_bias = torch.nn.Parameter(torch.empty(8))
        for name, bias in (
            ("direct", model.gate.bias),
            ("shared", model.gate.bias),
            ("other", model.gate.other_bias),
            ("independent", self.independent_bias),
            ("unbiased", None),
        ):
            model.add_module(name + "_router", self._router(bias))
        self.old_cast = model.direct_router.topk_config.correction_bias_for_dtype(
            torch.float64
        )
        self.independent_cast = (
            model.independent_router.topk_config.correction_bias_for_dtype(
                torch.float64
            )
        )
        return model

    def _attach(self, factory=None, hook=None):
        loader = IpcModelLoader(LoadConfig(load_format=LoadFormat.IPC_CACHE))
        entries = {
            "gate.bias": {"tensor": self.owner_bias},
            "gate.other_bias": {"tensor": self.other_bias},
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
                side_effect=factory or self._make_model,
            ),
            patch.object(loader, "_rebuild_stale_views", side_effect=hook) as rebuild,
        ):
            self.rebuild = rebuild
            return loader.load_model(
                model_config=SimpleNamespace(dtype=torch.float32), device_config=None
            )

    def test_restores_shared_parameter_without_confusing_equal_parameters(self):
        client = self._attach()
        for router in (client.direct_router, client.shared_router):
            self.assertIs(
                router.topk_config.correction_bias, client.get_parameter("gate.bias")
            )
        self.assertIs(
            client.other_router.topk_config.correction_bias,
            client.get_parameter("gate.other_bias"),
        )
        self.assertIsNot(client.gate.bias, client.gate.other_bias)
        self.assertNotEqual(
            client.gate.bias.untyped_storage()._cdata,
            client.gate.other_bias.untyped_storage()._cdata,
        )
        torch.testing.assert_close(client.gate.bias, client.gate.other_bias)
        self.assertEqual(
            client.gate.bias.untyped_storage()._cdata,
            self.owner_bias.untyped_storage()._cdata,
        )
        torch.testing.assert_close(
            self.owner_bias, torch.arange(8, dtype=torch.float32)
        )
        self.assertIs(
            client.independent_router.topk_config.correction_bias, self.independent_bias
        )
        self.assertIsNone(client.unbiased_router.topk_config.correction_bias)

    def test_invalidates_only_rebound_dtype_cache(self):
        client = self._attach()
        config = client.direct_router.topk_config
        self.assertTrue(self.old_cast.is_meta)
        self.assertIsNone(config._correction_bias_dtype_cache)
        self.assertIsNone(config._correction_bias_cache_key)
        converted = config.correction_bias_for_dtype(torch.float64)
        torch.testing.assert_close(converted, self.owner_bias.to(torch.float64))
        self.assertIs(config.correction_bias_for_dtype(torch.float64), converted)
        self.assertIs(
            client.independent_router.topk_config.correction_bias_for_dtype(
                torch.float64
            ),
            self.independent_cast,
        )

    def test_reports_unrestored_meta_bias_after_model_hooks(self):
        for kind in ("clone", "data", "slice", "unregistered_tensor"):
            with self.subTest(kind=kind):

                def make_model(*args):
                    model = self._make_model(*args)
                    bias = {
                        "clone": lambda: model.gate.bias.clone(),
                        "data": lambda: model.gate.bias.data,
                        "slice": lambda: model.gate.bias[::2],
                        "unregistered_tensor": lambda: torch.empty(8),
                    }[kind]()
                    model.orphan = self._router(bias)
                    return model

                with self.assertRaisesRegex(
                    RuntimeError,
                    r"meta.*orphan\.topk_config\.correction_bias",
                ):
                    self._attach(make_model)
                self.rebuild.assert_called_once()

    def test_allows_existing_model_hook_to_restore_an_extra_reference(self):
        def make_model(*args):
            model = self._make_model(*args)
            model.extra = self._router(model.gate.bias.data)
            return model

        def rebuild(model):
            model.extra.topk_config.set_correction_bias(
                model.get_parameter("gate.bias")
            )

        client = self._attach(make_model, rebuild)
        self.assertIs(client.extra.topk_config.correction_bias, client.gate.bias)


if __name__ == "__main__":
    unittest.main()
