import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import sglang.srt.model_loader.loader as loader_mod
from sglang.srt.model_loader.loader import DefaultModelLoader, ModelOptModelLoader
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDefaultModelLoader(CustomTestCase):
    def test_weight_loaded_hook_precedes_postprocessing_on_both_boot_paths(self):
        self.assertIsNone(DefaultModelLoader.weight_loaded_hook)
        events = []
        loader = object.__new__(DefaultModelLoader)
        loader.load_config = object()
        model = Mock()
        model.quant_config = None
        model.eval.return_value = model
        module = Mock()
        module.quant_method.process_weights_after_loading.side_effect = lambda _: (
            events.append("postprocess")
        )
        model.load_weights.side_effect = lambda _: events.append("load")
        model.named_modules.return_value = [("layer", module)]
        model_config = SimpleNamespace(modelopt_quant=None, dtype=torch.float32)
        device_config = SimpleNamespace(device="cpu")
        hook = Mock(side_effect=lambda *_: events.append("hook"))

        with (
            patch.object(loader_mod, "_get_quantization_config", return_value=None),
            patch.object(loader_mod, "_initialize_model", return_value=model),
            patch.object(loader, "_get_all_weights", return_value=iter(())),
            patch.object(loader_mod, "is_cuda_alike", return_value=False),
            patch.object(
                loader_mod,
                "device_loading_context",
                side_effect=lambda *_: nullcontext(),
            ),
            patch.object(DefaultModelLoader, "weight_loaded_hook", hook),
        ):
            DefaultModelLoader.load_weights_and_postprocess(
                model, iter(()), torch.device("cpu")
            )
            self.assertEqual(events, ["load", "postprocess"])
            hook.assert_not_called()

            events.clear()
            loader.load_model(
                model_config=model_config,
                device_config=device_config,
            )
            self.assertEqual(events, ["load", "hook", "postprocess"])
            hook.assert_called_once_with(model, model_config)

            events.clear()
            hook.reset_mock()
            loader.commit_model_weights(
                model=model,
                model_config=model_config,
                resolved_sources=(),
                target_device=torch.device("cpu"),
                startup_prefetch_active=True,
            )
            self.assertEqual(events, ["load", "hook", "postprocess"])
            hook.assert_called_once_with(model, model_config)

    def test_custom_loader_override_requires_an_inactive_hook(self):
        custom_load = Mock()

        class CustomModelLoader(DefaultModelLoader):
            def load_weights_and_postprocess(self, model, weights, target_device):
                custom_load(model, target_device)

        loader = object.__new__(CustomModelLoader)
        model = object()
        with patch.object(DefaultModelLoader, "weight_loaded_hook", None):
            loader._load_boot_weights_and_postprocess(
                model,
                iter(()),
                "cpu",
                model_config=object(),
            )

        custom_load.assert_called_once_with(model, "cpu")
        custom_load.reset_mock()

        hook = Mock()
        with (
            patch.object(DefaultModelLoader, "weight_loaded_hook", hook),
            self.assertRaisesRegex(
                RuntimeError,
                "CustomModelLoader.load_weights_and_postprocess bypasses "
                "DefaultModelLoader.weight_loaded_hook",
            ),
        ):
            loader._load_boot_weights_and_postprocess(
                model,
                iter(()),
                "cpu",
                model_config=object(),
            )

        custom_load.assert_not_called()
        hook.assert_not_called()

    def test_modelopt_bypasses_reject_an_active_hook(self):
        hook = Mock()
        device_config = object()

        bypasses = (
            (
                "legacy DefaultModelLoader path",
                object.__new__(DefaultModelLoader),
                SimpleNamespace(modelopt_quant="fp8"),
            ),
            (
                "non-prequantized ModelOptModelLoader path",
                object.__new__(ModelOptModelLoader),
                SimpleNamespace(
                    model_path="unused",
                    _is_already_quantized=lambda: False,
                ),
            ),
        )
        with patch.object(DefaultModelLoader, "weight_loaded_hook", hook):
            for name, model_loader, model_config in bypasses:
                with (
                    self.subTest(name=name),
                    self.assertRaisesRegex(
                        RuntimeError,
                        "ModelOpt loading bypasses "
                        "DefaultModelLoader.weight_loaded_hook",
                    ),
                ):
                    model_loader.load_model(
                        model_config=model_config,
                        device_config=device_config,
                    )
        hook.assert_not_called()


if __name__ == "__main__":
    unittest.main()
