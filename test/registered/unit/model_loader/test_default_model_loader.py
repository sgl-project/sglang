import json
import os
import tempfile
import unittest
from contextlib import nullcontext
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
    def test_every_modelopt_quantization_selects_the_modelopt_loader(self):
        """A pre-quantized checkpoint under any ModelOpt quantization value must
        load through ModelOptModelLoader, not fall through to the default loader."""
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


class TestShardedStateModelOptFp8(CustomTestCase):
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
        return model

    @staticmethod
    def _load(model, state):
        from safetensors.torch import save_file

        from sglang.srt.runtime_context import get_parallel

        with tempfile.TemporaryDirectory() as directory:
            save_file(state, os.path.join(directory, "model-rank-0-part-0.safetensors"))
            loader = loader_mod.ShardedStateLoader(
                LoadConfig(load_format=LoadFormat.SHARDED_STATE)
            )
            with (
                get_parallel().override(tp_rank=0, tp_size=1, pp_rank=0),
                patch.object(loader_mod, "_get_quantization_config", return_value=None),
                patch.object(loader_mod, "_initialize_model", return_value=model),
            ):
                return loader.load_model(
                    model_config=SimpleNamespace(
                        model_path=directory, revision=None, dtype=torch.float32
                    ),
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
