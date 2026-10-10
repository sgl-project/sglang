import os
import tempfile
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import safetensors.torch
import torch

import sglang.srt.model_loader.loader as loader_mod
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDefaultCheckpointSelection(CustomTestCase):
    def setUp(self):
        server_args = patch.object(loader_mod, "get_server_args", return_value=None)
        server_args.start()
        self.addCleanup(server_args.stop)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.folder = tmp.name
        self.files = []
        for role, value in (("main", 1.0), ("draft", 2.0)):
            path = os.path.join(self.folder, f"{role}.safetensors")
            safetensors.torch.save_file({role + ".weight": torch.tensor([value])}, path)
            self.files.append(path)
        self.config = SimpleNamespace(
            model_path=self.folder, revision=None, hf_config=SimpleNamespace()
        )
        self.model = SimpleNamespace(
            is_unused_checkpoint_weight=lambda name: not name.startswith("draft.")
        )
        self.loader = DefaultModelLoader(
            LoadConfig(
                load_format="safetensors",
                model_loader_extra_config={"enable_multithread_load": False},
            )
        )

    def test_ordinary_load_does_not_read_unused_shards(self):
        with patch.object(
            loader_mod,
            "get_model",
            return_value=SimpleNamespace(
                weight_loader_disable_mmap=False,
                weight_loader_prefetch_checkpoints=False,
                weight_loader_prefetch_num_threads=1,
                weight_loader_drop_cache_after_load=False,
            ),
        ):
            loaded = dict(self.loader._get_all_weights(self.config, self.model))
        self.assertEqual(set(loaded), {"draft.weight"})
        torch.testing.assert_close(loaded["draft.weight"], torch.tensor([2.0]))

    def test_resolution_filters_before_startup_prefetch(self):
        resolved = self.loader.resolve_model_weights(self.config, self.model)
        self.assertEqual(resolved[0].weight_files, (self.files[1],))
        # The external I/O boundary receives only the selected checkpoint files.
        with patch.object(loader_mod, "_prefetch_all_checkpoints") as prefetch:
            self.loader.start_checkpoint_prefetch(resolved, num_threads=1)
        prefetch.assert_called_once_with([self.files[1]], num_threads=1)

    def test_unadapted_model_keeps_all_files(self):
        resolved = self.loader.resolve_model_weights(self.config, SimpleNamespace())
        self.assertEqual(set(resolved[0].weight_files), set(self.files))

    def test_prefixed_secondary_source_is_not_filtered_in_wrong_name_domain(self):
        source = DefaultModelLoader.Source.init_new(self.config, self.model)
        source.prefix = "nested."
        model = SimpleNamespace(secondary_weights=[source])
        resolved = self.loader.resolve_model_weights(self.config, model)
        self.assertEqual(set(resolved[1].weight_files), set(self.files))

    def test_layer_remapping_preserves_original_file_list(self):
        self.loader.load_config.draft_model_idx = 1
        resolved = self.loader.resolve_model_weights(self.config, self.model)
        self.assertEqual(set(resolved[0].weight_files), set(self.files))


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


if __name__ == "__main__":
    unittest.main()
