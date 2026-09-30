import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import sglang.srt.model_loader.loader as loader_mod
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDefaultModelLoader(CustomTestCase):
    def test_debug_memory_probes_preserve_online_quantization_cleanup(self):
        for debug in (False, True):
            for online in (False, True):
                with self.subTest(debug=debug, online=online):
                    model = Mock()
                    model.quant_config = (
                        SimpleNamespace(is_nvfp4_online=True, get_name=lambda: "nvfp4")
                        if online
                        else None
                    )
                    weights = iter(())
                    with (
                        patch.object(loader_mod, "is_cuda_alike", return_value=True),
                        patch.object(
                            loader_mod.logger, "isEnabledFor", return_value=debug
                        ),
                        patch.object(
                            loader_mod, "get_available_gpu_memory", return_value=1.0
                        ) as memory_probe,
                        patch.object(torch.cuda, "current_device", return_value=0),
                        patch.object(
                            torch.cuda, "max_memory_allocated", return_value=0
                        ) as peak_memory,
                        patch.object(torch.cuda, "synchronize") as synchronize,
                        patch.object(torch.cuda, "empty_cache") as empty_cache,
                    ):
                        DefaultModelLoader.load_weights_only(
                            model, weights, torch.device("cuda")
                        )

                    model.load_weights.assert_called_once_with(weights)
                    self.assertEqual(memory_probe.call_count, 2 if debug else 0)
                    self.assertEqual(peak_memory.call_count, 1 if debug else 0)
                    self.assertEqual(synchronize.call_count, 1 if online else 0)
                    self.assertEqual(empty_cache.call_count, 1 if online else 0)

    def test_load_weights_only_precedes_postprocessing(self):
        events = []
        model = Mock()
        model.quant_config = None
        module = Mock()
        module.quant_method.process_weights_after_loading.side_effect = lambda _: (
            events.append("postprocess")
        )
        model.load_weights.side_effect = lambda _: events.append("load")
        model.named_modules.return_value = [("layer", module)]

        with (
            patch.object(loader_mod, "is_cuda_alike", return_value=False),
            patch.object(
                loader_mod,
                "device_loading_context",
                side_effect=lambda *_: nullcontext(),
            ),
        ):
            DefaultModelLoader.load_weights_and_postprocess(
                model,
                iter(()),
                torch.device("cpu"),
            )

        self.assertEqual(events, ["load", "postprocess"])

    def test_boot_paths_preserve_load_order_and_custom_override(self):
        class CustomModelLoader(DefaultModelLoader):
            def load_weights_and_postprocess(self, model, weights, target_device):
                events.append("override")
                DefaultModelLoader.load_weights_and_postprocess(
                    model, weights, target_device
                )

        for loader_class in (DefaultModelLoader, CustomModelLoader):
            with self.subTest(loader=loader_class.__name__):
                events = []
                loader = object.__new__(loader_class)
                loader.load_config = object()
                model = Mock(quant_config=None)
                model.eval.return_value = model
                model.load_weights.side_effect = lambda weights: events.append(
                    ("load", list(weights))
                )
                module = Mock()
                module.quant_method.process_weights_after_loading.side_effect = (
                    lambda _: events.append("postprocess")
                )
                model.named_modules.return_value = [("layer", module)]
                model_config = SimpleNamespace(modelopt_quant=None, dtype=torch.float32)
                resolved_source = SimpleNamespace(source=object())

                with (
                    patch.object(loader_mod, "is_cuda_alike", return_value=False),
                    patch.object(
                        loader_mod, "_get_quantization_config", return_value=None
                    ),
                    patch.object(loader_mod, "_initialize_model", return_value=model),
                    patch.object(
                        loader_mod,
                        "device_loading_context",
                        side_effect=lambda *_: nullcontext(),
                    ),
                    patch.object(
                        loader, "_get_all_weights", return_value=iter([("direct", 1)])
                    ),
                    patch.object(
                        loader,
                        "_get_weights_iterator",
                        return_value=iter([("deferred", 2)]),
                    ) as get_weights,
                ):
                    self.assertIs(
                        loader.load_model(
                            model_config=model_config,
                            device_config=SimpleNamespace(device="cpu"),
                        ),
                        model,
                    )
                    loader.commit_model_weights(
                        model=model,
                        model_config=model_config,
                        resolved_sources=(resolved_source,),
                        target_device=torch.device("cpu"),
                        startup_prefetch_active=True,
                    )

                expected = []
                for weights in ([("direct", 1)], [("deferred", 2)]):
                    if loader_class is CustomModelLoader:
                        expected.append("override")
                    expected.extend([("load", weights), "postprocess"])
                self.assertEqual(events, expected)
                get_weights.assert_called_once_with(
                    resolved_source.source,
                    resolved_source=resolved_source,
                    startup_prefetch_started=True,
                    startup_prefetch_active=True,
                )


if __name__ == "__main__":
    unittest.main()
