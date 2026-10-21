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

    def test_reload_requires_explicit_initial_load_marker(self):
        self.assertFalse(loader_mod.QuantizedRLModelLoader.is_reload_scenario(Mock()))
        model = torch.nn.Module()
        model.original_weights_rebuild_keys = set()
        model.recorded_loader = {}
        for marker in (None, False, 1, Mock()):
            with self.subTest(marker=marker):
                model.flash_rl_initial_load_complete = marker
                self.assertFalse(
                    loader_mod.QuantizedRLModelLoader.is_reload_scenario(model)
                )
        model.flash_rl_initial_load_complete = True
        self.assertTrue(loader_mod.QuantizedRLModelLoader.is_reload_scenario(model))

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

    def test_kv_cache_scale_dropped_on_mismatch(self):
        cases = [
            # (checkpoint_bits, runtime_kv_dtype, expect_scales_dropped, kv_cache_quant_algo)
            (8, "nvfp4", True, None),
            (8, "bfloat16", True, None),
            (8, "fp8", False, None),
            (8, "auto", False, "FP8"),
            (4, "nvfp4", False, None),
            (4, "fp8", True, None),
        ]
        for checkpoint_bits, runtime_dtype, expect_dropped, *extra in cases:
            quant_algo = extra[0] if extra else None
            with self.subTest(ckpt=checkpoint_bits, runtime=runtime_dtype):
                model = Mock()
                model.quant_config = SimpleNamespace(
                    kv_cache_scheme={"num_bits": checkpoint_bits},
                    kv_cache_quant_algo=quant_algo,
                    is_nvfp4_online=False,
                    get_name=lambda: "dummy",
                )
                loaded = []
                model.load_weights.side_effect = lambda ws: loaded.extend(list(ws))

                dummy_weights = [
                    ("model.layers.0.self_attn.q_proj.weight", torch.zeros(1)),
                    ("model.layers.0.self_attn.k_scale", torch.ones(1)),
                    ("model.layers.0.self_attn.v_scale", torch.ones(1)),
                ]

                model_bag = SimpleNamespace(kv_cache_dtype=runtime_dtype)
                with (
                    patch.object(loader_mod, "is_cuda_alike", return_value=False),
                    patch(
                        "sglang.srt.runtime_context.get_model",
                        return_value=model_bag,
                    ),
                    patch.object(
                        loader_mod,
                        "monokernel_weights_iterator",
                        side_effect=lambda model, weights: weights,
                    ),
                ):
                    DefaultModelLoader.load_weights_only(
                        model, iter(dummy_weights), torch.device("cpu")
                    )

                loaded_names = [name for name, _ in loaded]
                if expect_dropped:
                    self.assertEqual(
                        loaded_names, ["model.layers.0.self_attn.q_proj.weight"]
                    )
                else:
                    self.assertEqual(
                        loaded_names,
                        [
                            "model.layers.0.self_attn.q_proj.weight",
                            "model.layers.0.self_attn.k_scale",
                            "model.layers.0.self_attn.v_scale",
                        ],
                    )


if __name__ == "__main__":
    unittest.main()
