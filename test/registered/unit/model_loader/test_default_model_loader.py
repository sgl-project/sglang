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


class TestGGUFConfigIsolation(CustomTestCase):
    """Config isolation for nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB."""

    def test_gemma4_name_mapping_does_not_select_a_transformers_attention_backend(self):
        from types import SimpleNamespace

        from transformers import Gemma4TextConfig

        from sglang.srt.model_loader.loader import GGUFModelLoader

        config = Gemma4TextConfig(
            vocab_size=32,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=6,
            num_kv_shared_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            global_head_dim=16,
            hidden_size_per_layer_input=0,
            per_layer_config=None,
        )
        before = config._attn_implementation
        loader = GGUFModelLoader.__new__(GGUFModelLoader)
        names = loader._get_gguf_weights_map(SimpleNamespace(hf_config=config))
        self.assertEqual(names["token_embd.weight"], "model.embed_tokens.weight")
        self.assertEqual(config._attn_implementation, before)
        self.assertEqual(config.num_kv_shared_layers, 2)
        self.assertEqual(
            names["blk.5.layer_output_scale.weight"], "model.layers.5.layer_scalar"
        )
        for projection in ("k", "v"):
            self.assertEqual(
                names[f"blk.5.attn_{projection}.weight"],
                f"model.layers.5.self_attn.{projection}_proj.weight",
            )


if __name__ == "__main__":
    unittest.main()
