import json
import os
import tempfile
import unittest
from contextlib import nullcontext
from itertools import product
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import sglang.srt.model_loader.loader as loader_mod
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.model_loader.loader import DefaultModelLoader, ModelOptModelLoader
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


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

    def test_mistral_honors_index_opt_out(self):
        """safetensors_index_file=None must skip the consolidated index too."""
        with tempfile.TemporaryDirectory() as model_path:
            files = [
                os.path.join(model_path, f"consolidated-0000{i}.safetensors")
                for i in (1, 2)
            ]
            for path in files:
                open(path, "w").close()
            with open(
                os.path.join(model_path, "consolidated.safetensors.index.json"), "w"
            ) as f:
                json.dump({"weight_map": {"w": os.path.basename(files[0])}}, f)
            loader = DefaultModelLoader(LoadConfig(load_format=LoadFormat.MISTRAL))
            with patch.object(loader_mod, "get_server_args", return_value=None):
                for index_file, expected in (
                    ("model.safetensors.index.json", files[:1]),
                    (None, files),
                ):
                    with self.subTest(index_file=index_file):
                        _, weight_files, _ = loader._prepare_weights(
                            model_path,
                            None,
                            fall_back_to_pt=False,
                            safetensors_index_file=index_file,
                        )
                        self.assertEqual(sorted(weight_files), expected)

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


class TestModelOptLoaderSelection(CustomTestCase):
    def test_explicit_shard_formats_override_modelopt_loader_selection(self):
        """ModelOpt auto-selection must not intercept processed-shard formats."""
        for quantization in (
            "modelopt",
            "modelopt_fp8",
            "modelopt_fp4",
            "modelopt_mixed",
        ):
            for load_format, expected_loader in (
                (LoadFormat.SHARDED_STATE, loader_mod.ShardedStateLoader),
                (LoadFormat.PRESHARDED, loader_mod.PreshardedModelLoader),
            ):
                with self.subTest(quantization=quantization, load_format=load_format):
                    model_config = SimpleNamespace(
                        quantization=quantization, _is_already_quantized=lambda: True
                    )
                    loader = loader_mod.get_model_loader(
                        LoadConfig(load_format=load_format), model_config
                    )
                    self.assertIsInstance(loader, expected_loader)

    def test_shard_selection_preserves_modelopt_workflows(self):
        """Raw sources need calibration; processed shards do not, even with export."""
        from transformers import PretrainedConfig

        from sglang.srt.configs.model_config import ModelConfig

        with tempfile.TemporaryDirectory() as directory:
            config = ModelConfig.__new__(ModelConfig)
            config.model_path = directory
            config.revision = None
            config.quantization = "modelopt_fp8"
            for serialized, workflow, (load_format, shard_loader) in product(
                (False, True),
                (
                    None,
                    "modelopt_export_path",
                    "modelopt_checkpoint_restore_path",
                    "modelopt_checkpoint_save_path",
                ),
                (
                    (LoadFormat.SHARDED_STATE, loader_mod.ShardedStateLoader),
                    (LoadFormat.PRESHARDED, loader_mod.PreshardedModelLoader),
                ),
            ):
                with self.subTest(
                    serialized=serialized, workflow=workflow, format=load_format
                ):
                    # Use the real checkpoint-metadata detector, not a boolean stub.
                    config.hf_config = PretrainedConfig()
                    if serialized:
                        config.hf_config.quantization_config = {
                            "quant_method": "modelopt",
                            "quant_algo": "FP8",
                        }
                    kwargs = (
                        {workflow: os.path.join(directory, "out")} if workflow else {}
                    )
                    loader = loader_mod.get_model_loader(
                        LoadConfig(load_format=load_format, **kwargs), config
                    )
                    self.assertIsInstance(
                        loader, shard_loader if serialized else ModelOptModelLoader
                    )

    def test_nested_metadata_matches_quant_config_loading(self):
        """Text-only quant metadata must not send processed shards to calibration."""
        from transformers import PretrainedConfig

        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.model_loader.weight_utils import get_quant_config

        with tempfile.TemporaryDirectory() as directory:
            config = ModelConfig.__new__(ModelConfig)
            config.model_path = directory
            config.revision = None
            config.quantization = "modelopt_fp8"
            config.is_draft_model = False
            config.hf_config = PretrainedConfig(
                text_config=PretrainedConfig(
                    quantization_config={
                        "quant_method": "modelopt",
                        "quant_algo": "FP8",
                    }
                )
            )
            quant_config = get_quant_config(config, LoadConfig(), {})
            self.assertTrue(quant_config.is_checkpoint_fp8_serialized)
            for load_format, expected in (
                (LoadFormat.SHARDED_STATE, loader_mod.ShardedStateLoader),
                (LoadFormat.PRESHARDED, loader_mod.PreshardedModelLoader),
            ):
                with self.subTest(load_format=load_format):
                    loader = loader_mod.get_model_loader(
                        LoadConfig(load_format=load_format), config
                    )
                    self.assertIsInstance(loader, expected)

    def test_shard_selection_uses_checkpoint_revision(self):
        """A pinned cached export must agree with config parsing, not query main."""
        from transformers import PretrainedConfig

        from sglang.srt.configs.model_config import ModelConfig

        with (
            tempfile.TemporaryDirectory() as cache,
            patch("huggingface_hub.constants.HF_HUB_CACHE", cache),
            patch("huggingface_hub.constants.HF_HUB_OFFLINE", True),
        ):
            commit = "a" * 40
            repo_cache = os.path.join(cache, "models--test-org--fp8-model")
            snapshot = os.path.join(repo_cache, "snapshots", commit)
            os.makedirs(snapshot)
            with open(os.path.join(snapshot, "hf_quant_config.json"), "w") as f:
                json.dump({"quantization": {"quant_algo": "FP8"}}, f)
            os.makedirs(os.path.join(repo_cache, "refs"))
            with open(os.path.join(repo_cache, "refs", "fp8-export"), "w") as f:
                f.write(commit)

            config = ModelConfig.__new__(ModelConfig)
            config.model_path = "test-org/fp8-model"
            config.quantization = "modelopt_fp8"
            config.hf_config = PretrainedConfig()
            for revision, serialized in (
                (commit, True),
                ("fp8-export", True),
                (None, False),
            ):
                config.revision = revision
                parsed = config._parse_quant_hf_config()
                self.assertEqual(parsed is not None, serialized)
                for load_format, shard_loader in (
                    (LoadFormat.SHARDED_STATE, loader_mod.ShardedStateLoader),
                    (LoadFormat.PRESHARDED, loader_mod.PreshardedModelLoader),
                ):
                    with self.subTest(revision=revision, load_format=load_format):
                        loader = loader_mod.get_model_loader(
                            LoadConfig(load_format=load_format), config
                        )
                        self.assertIsInstance(
                            loader, shard_loader if serialized else ModelOptModelLoader
                        )

            # A cache miss online must ask the Hub about the same revision too.
            config.revision = "remote-export"
            with (
                self.subTest(online=True),
                patch("huggingface_hub.constants.HF_HUB_OFFLINE", False),
                patch(
                    "huggingface_hub.HfApi.file_exists",
                    side_effect=lambda *args, revision=None: (
                        revision == "remote-export"
                    ),
                ),
            ):
                self.assertIsInstance(
                    loader_mod.get_model_loader(
                        LoadConfig(load_format=LoadFormat.SHARDED_STATE), config
                    ),
                    loader_mod.ShardedStateLoader,
                )

    def test_every_modelopt_quantization_selects_the_modelopt_loader(self):
        """Without an explicit shard format, retain the ModelOpt loader."""
        for quantization in (
            "modelopt",
            "modelopt_fp8",
            "modelopt_fp4",
            "modelopt_mixed",
        ):
            with self.subTest(quantization=quantization):
                model_config = SimpleNamespace(
                    quantization=quantization, _is_already_quantized=lambda: True
                )
                self.assertIsInstance(
                    loader_mod.get_model_loader(LoadConfig(), model_config),
                    ModelOptModelLoader,
                )


class TestModelOptFp8ProcessedShards(CustomTestCase):
    @staticmethod
    def _model(is_gated, intermediate):
        from sglang.srt.layers.moe import MoeRunnerBackend
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp8MoEMethod

        model = torch.nn.Module()
        layer = model.experts = torch.nn.Module()
        layer.moe_runner_config = MoeRunnerConfig(is_gated=is_gated)
        layer.intermediate_size_per_partition = intermediate
        num_parts = 2 if is_gated else 1
        shapes = {
            "w13_weight": (2, num_parts * intermediate, 16),
            "w2_weight": (2, 16, intermediate),
        }
        for name, shape in shapes.items():
            values = (torch.arange(torch.Size(shape).numel()) % 31 + 1).reshape(shape)
            layer.register_parameter(
                name,
                torch.nn.Parameter(values.to(torch.float8_e4m3fn), requires_grad=False),
            )
        # Processed shards already have per-expert weight scales. These also
        # let the real post-load method run without GPU requantization kernels.
        for name in ("w13_weight_scale", "w2_weight_scale"):
            layer.register_parameter(
                name, torch.nn.Parameter(torch.tensor([0.25, 0.5]), requires_grad=False)
            )
        for name in ("w13_input_scale", "w2_input_scale"):
            layer.register_parameter(
                name, torch.nn.Parameter(torch.tensor(0.125), requires_grad=False)
            )
        layer.quant_method = ModelOptFp8MoEMethod.__new__(ModelOptFp8MoEMethod)
        layer.quant_method._moe_runner_backend = MoeRunnerBackend.TRITON
        model.load_weights = lambda weights: model.load_state_dict(dict(weights))
        return model

    @staticmethod
    def _load(model, state):
        from safetensors.torch import save_file

        from sglang.srt.runtime_context import get_parallel

        with tempfile.TemporaryDirectory() as directory:
            save_file(state, os.path.join(directory, "model-rank-0-part-0.safetensors"))
            model_config = SimpleNamespace(
                model_path=directory,
                revision=None,
                dtype=torch.float32,
                quantization="modelopt_fp8",
                _is_already_quantized=lambda: True,
            )
            loader = loader_mod.get_model_loader(
                LoadConfig(load_format=LoadFormat.SHARDED_STATE), model_config
            )
            with (
                get_parallel().override(tp_rank=0, tp_size=1, pp_rank=0),
                patch.object(loader_mod, "_get_quantization_config", return_value=None),
                patch.object(loader_mod, "_initialize_model", return_value=model),
            ):
                return loader.load_model(
                    model_config=model_config,
                    device_config=SimpleNamespace(device="cpu"),
                )

    def test_legacy_shards_restore_each_projection_and_zero_tails(self):
        """A contiguous old W13 copy shifts the gate and contaminates the up tail."""
        for is_gated in (False, True):
            with self.subTest(is_gated=is_gated):
                model = self._model(is_gated, intermediate=232)
                state = {
                    name: value.clone() for name, value in model.state_dict().items()
                }
                for param in model.parameters():
                    param.data.fill_(float("nan"))
                expected = dict(state)
                w13 = torch.zeros(2, (2 if is_gated else 1) * 240, 16)
                w13[:, :232] = state["experts.w13_weight"][:, :232].float()
                if is_gated:
                    w13[:, 240:472] = state["experts.w13_weight"][:, 232:].float()
                w2 = torch.zeros(2, 16, 240)
                w2[:, :, :232] = state["experts.w2_weight"].float()
                expected.update({"experts.w13_weight": w13, "experts.w2_weight": w2})
                loaded = self._load(model, state)
                for name, value in loaded.state_dict().items():
                    torch.testing.assert_close(
                        value.float(), expected[name].float(), rtol=0, atol=0
                    )
                self.assertEqual(loaded.experts.intermediate_size_per_partition, 232)

    def test_padded_and_aligned_shards_round_trip(self):
        for is_gated in (False, True):
            for intermediate in (232, 240):
                with self.subTest(is_gated=is_gated, intermediate=intermediate):
                    source = self._model(is_gated, intermediate)
                    source.experts.quant_method.process_weights_after_loading(
                        source.experts
                    )
                    state = source.state_dict()
                    destination = self._model(is_gated, intermediate)
                    for param in destination.parameters():
                        param.data.fill_(float("nan"))
                    loaded = self._load(destination, state)
                    for name, value in loaded.state_dict().items():
                        torch.testing.assert_close(
                            value.float(), state[name].float(), rtol=0, atol=0
                        )

    def test_unrelated_w13_keeps_existing_copy_behavior(self):
        # The same names and shapes on a non-ModelOpt layer must not trigger
        # projection-aware copying or change the existing padding contents.
        model = torch.nn.Module()
        model.experts = torch.nn.Module()
        model.experts.w13_weight = torch.nn.Parameter(
            torch.full((2, 480, 16), -1.0), requires_grad=False
        )
        source = torch.arange(2 * 464 * 16).reshape(2, 464, 16).float()
        loaded = self._load(model, {"experts.w13_weight": source})
        expected = torch.full((2, 480, 16), -1.0)
        expected[:, :464] = source
        torch.testing.assert_close(loaded.experts.w13_weight, expected, rtol=0, atol=0)

    def test_presharded_factory_loads_checkpoint_then_reuses_processed_cache(self):
        """Both cold and cached startup must use the explicit presharded loader."""
        from safetensors.torch import save_file

        from sglang.srt.layers.moe import MoeRunnerBackend
        from sglang.srt.runtime_context import get_context, get_flags, get_parallel

        def initialize_model(*_args):
            model = self._model(is_gated=True, intermediate=232)
            for param in model.parameters():
                param.data.fill_(float("nan"))
            return model

        with (
            tempfile.TemporaryDirectory() as directory,
            get_context().override_server_args(model_path=directory, device="cpu"),
            get_flags().moe.override(runner_backend=MoeRunnerBackend.TRITON),
            get_parallel().override(
                tp_rank=0, tp_size=1, pp_rank=0, pp_size=1, world_group=None
            ),
            patch.object(loader_mod, "is_cuda_alike", return_value=False),
            patch.object(loader_mod, "_get_quantization_config", return_value=None),
            patch.object(loader_mod, "_initialize_model", side_effect=initialize_model),
        ):
            model_config = SimpleNamespace(
                model_path=directory,
                revision=None,
                dtype=torch.float32,
                quantization="modelopt_fp8",
                _is_already_quantized=lambda: True,
                is_draft_model=False,
                hf_config=SimpleNamespace(),
            )
            checkpoint = os.path.join(directory, "model.safetensors")
            source = self._model(is_gated=True, intermediate=232)
            save_file(source.state_dict(), checkpoint)
            source.experts.quant_method.process_weights_after_loading(source.experts)
            expected = source.state_dict()

            for cached in (False, True):
                with self.subTest(cached=cached):
                    loader = loader_mod.get_model_loader(
                        LoadConfig(load_format=LoadFormat.PRESHARDED), model_config
                    )
                    loaded = loader.load_model(
                        model_config=model_config,
                        device_config=SimpleNamespace(device="cpu"),
                    )
                    for name, value in loaded.state_dict().items():
                        torch.testing.assert_close(
                            value.float(), expected[name].float(), rtol=0, atol=0
                        )
                    # A second startup must succeed without the original checkpoint.
                    if not cached:
                        os.unlink(checkpoint)

    def test_copy_helper_clears_preexisting_padding(self):
        # Do not rely on post-load allocation having initialized the tails.
        for split_dim, num_parts in ((1, 1), (1, 2), (2, 1)):
            with self.subTest(split_dim=split_dim, num_parts=num_parts):
                shape = (2, num_parts * 232, 16) if split_dim == 1 else (2, 16, 232)
                padded_shape = (
                    (2, num_parts * 240, 16) if split_dim == 1 else (2, 16, 240)
                )
                source = torch.ones(shape, dtype=torch.float8_e4m3fn)
                destination = torch.full(padded_shape, float("nan"), dtype=source.dtype)
                loader_mod.ShardedStateLoader._copy_padded_shard(
                    destination=destination,
                    source=source,
                    original_shape=source.shape,
                    split_dim=split_dim,
                    num_parts=num_parts,
                )
                expected = torch.zeros(padded_shape)
                if split_dim == 1:
                    expected[:, :232] = 1
                    if num_parts == 2:
                        expected[:, 240:472] = 1
                else:
                    expected[:, :, :232] = 1
                torch.testing.assert_close(
                    destination.float(), expected, rtol=0, atol=0
                )

    def test_unexpected_legacy_shape_is_rejected(self):
        for name in ("w13_weight", "w2_weight"):
            with self.subTest(name=name):
                model = self._model(is_gated=True, intermediate=232)
                state = {
                    name: value.clone() for name, value in model.state_dict().items()
                }
                key = f"experts.{name}"
                state[key] = state[key][:, :, :-1].contiguous()
                with self.assertRaisesRegex(ValueError, "processed shard shape"):
                    self._load(model, state)


if __name__ == "__main__":
    unittest.main()
