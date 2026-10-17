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

    def test_layer_aware_model_selects_files_before_read(self):
        self.loader.load_config.draft_model_idx = 1
        self.model.is_unused_checkpoint_weight_with_draft_model_idx = lambda name: (
            not name.startswith("draft.")
        )
        resolved = self.loader.resolve_model_weights(self.config, self.model)
        self.assertEqual(resolved[0].weight_files, (self.files[1],))

    def test_indexed_draft_reads_selected_shard_and_remaps_layer(self):
        paths = []
        for idx in (0, 1):
            path = os.path.join(self.folder, f"mtp-{idx}.safetensors")
            safetensors.torch.save_file(
                {f"model.mtp.layers.{idx}.weight": torch.tensor([float(idx)])},
                path,
            )
            paths.append(path)
        self.loader.load_config.draft_model_idx = 1
        self.model.is_unused_checkpoint_weight_with_draft_model_idx = lambda name: (
            name != "model.mtp.layers.1.weight"
        )
        resolved = self.loader.resolve_model_weights(self.config, self.model)
        self.assertEqual(resolved[0].weight_files, (paths[1],))
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
        self.assertEqual(set(loaded), {"model.mtp.layers.0.weight"})
        torch.testing.assert_close(
            loaded["model.mtp.layers.0.weight"], torch.tensor([1.0])
        )

    def test_indexed_remap_leaves_step_layer_names_unchanged(self):
        weights = iter(
            (
                ("model.layers.62.eh_proj.weight", torch.tensor([1.0])),
                ("model.mtp.layers.1.eh_proj.weight", torch.tensor([2.0])),
            )
        )
        selected = list(DefaultModelLoader._filter_mtp_weights(weights, "", 1))
        self.assertEqual(
            [name for name, _ in selected],
            ["model.layers.62.eh_proj.weight", "model.mtp.layers.0.eh_proj.weight"],
        )

    def test_checkpoint_inventory_covers_primary_and_secondary_sources(self):
        observed = set()
        resets = []

        def begin():
            resets.append(True)
            observed.clear()

        def observe(names):
            observed.update(names)

        with tempfile.TemporaryDirectory() as secondary_dir:
            secondary_file = os.path.join(secondary_dir, "target.safetensors")
            safetensors.torch.save_file(
                {"target.layer.weight": torch.ones(1)}, secondary_file
            )
            secondary = DefaultModelLoader.Source(
                secondary_dir,
                None,
                is_unused_weight=lambda name: name.startswith("target."),
            )
            model = SimpleNamespace(
                is_unused_checkpoint_weight=lambda name: name.startswith("main."),
                begin_checkpoint_weight_selection=begin,
                observe_checkpoint_weight_names=observe,
                secondary_weights=[secondary],
            )
            resolved = self.loader.resolve_model_weights(self.config, model)
            self.assertEqual(len(resolved), 2)
            self.assertEqual(
                observed,
                {"main.weight", "draft.weight", "target.layer.weight"},
            )
            self.loader.resolve_model_weights(self.config, model)
            self.assertEqual(len(resets), 2)
            self.assertEqual(
                observed,
                {"main.weight", "draft.weight", "target.layer.weight"},
            )

    def test_checkpoint_inventory_includes_unfiltered_secondary_sources(self):
        observed = []
        with tempfile.TemporaryDirectory() as secondary_dir:
            path = os.path.join(secondary_dir, "target.safetensors")
            safetensors.torch.save_file({"target.weight": torch.ones(1)}, path)
            model = SimpleNamespace(
                observe_checkpoint_weight_names=lambda names: observed.extend(names),
                secondary_weights=[
                    DefaultModelLoader.Source(
                        secondary_dir,
                        None,
                        prefix="nested.",
                        is_unused_weight=lambda _: True,
                    ),
                    DefaultModelLoader.Source(secondary_dir, None),
                ],
            )
            resolved = self.loader.resolve_model_weights(self.config, model)
        self.assertEqual(len(resolved), 3)
        self.assertEqual(observed.count("target.weight"), 1)
        self.assertEqual(observed.count("nested.target.weight"), 1)

    def test_serial_load_observes_unfiltered_secondary_source(self):
        observed = []
        with tempfile.TemporaryDirectory() as secondary_dir:
            path = os.path.join(secondary_dir, "target.safetensors")
            safetensors.torch.save_file({"target.weight": torch.ones(1)}, path)
            model = SimpleNamespace(
                observe_checkpoint_weight_names=lambda names: observed.extend(names),
                secondary_weights=[DefaultModelLoader.Source(secondary_dir, None)],
            )
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
                loaded = dict(self.loader._get_all_weights(self.config, model))
        self.assertIn("target.weight", loaded)
        self.assertIn("target.weight", observed)

    def test_pt_checkpoint_does_not_apply_safetensors_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "pytorch_model.pt")
            torch.save({"target.weight": torch.ones(1)}, path)
            events = []
            model = SimpleNamespace(
                begin_checkpoint_weight_selection=lambda: events.append("begin"),
                observe_checkpoint_weight_names=lambda names: events.append("observe"),
                is_unused_checkpoint_weight=lambda _: True,
            )
            loader = DefaultModelLoader(LoadConfig(load_format="pt"))
            config = SimpleNamespace(
                model_path=folder, revision=None, hf_config=SimpleNamespace()
            )
            resolved = loader.resolve_model_weights(config, model)
            self.assertFalse(resolved[0].use_safetensors)
            self.assertEqual(resolved[0].weight_files, (path,))
            self.assertEqual(events, ["begin"])


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
