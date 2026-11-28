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


class TestCpuQuantizationLoader(CustomTestCase):
    def test_default_loader_dispatches_to_cpu_quantization(self):
        """DefaultModelLoader must honor SGLANG_CPU_QUANTIZATION=1 and never bounce
        quantized weights back through device_loading_context."""
        events = []

        class _QuantLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(4, 4))
                self.quant_method = Mock()
                self.quant_method.process_weights_after_loading.side_effect = lambda m: (
                    events.append(("quantize", m.weight.device.type))
                )

        class _ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = _QuantLayer()
                self.quant_config = None

            def load_weights(self, weights):
                events.append(("load", self.layer.weight.device.type, list(weights)))

        model = _ToyModel()
        loader = DefaultModelLoader(LoadConfig())
        model_config = SimpleNamespace(dtype=torch.bfloat16)

        with (
            patch.object(
                loader_mod.envs.SGLANG_CPU_QUANTIZATION, "get", return_value=True
            ),
            patch.object(loader_mod, "_get_quantization_config", return_value=None),
            patch.object(loader_mod, "_initialize_model", return_value=model),
            patch.object(loader_mod, "is_cuda_alike", return_value=False),
            patch.object(
                loader,
                "_get_all_weights",
                return_value=iter([("layer.weight", torch.ones(4, 4))]),
            ),
            patch.object(
                loader_mod,
                "device_loading_context",
                side_effect=AssertionError("must not bounce weights back to CPU"),
            ),
        ):
            out = loader.load_model(
                model_config=model_config,
                device_config=SimpleNamespace(device="cpu"),
            )

        self.assertIs(out, model)
        self.assertGreater(loader.counter_after_loading_weights, 0.0)
        self.assertEqual(events[0][0], "load")
        self.assertEqual(events[0][1], "cpu")
        self.assertEqual(events[1], ("quantize", "cpu"))

    def test_dummy_loader_initializes_weights_and_runs_post_load(self):
        """DummyModelLoader under SGLANG_CPU_QUANTIZATION=1 must call both
        initialize_dummy_weights and post_load_weights before quantization."""
        events = []

        class _ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(4, 4, bias=False)
                self.lin.quant_method = Mock()
                self.lin.quant_method.process_weights_after_loading.side_effect = (
                    lambda _: events.append("quantize")
                )

            def post_load_weights(self):
                events.append("post_load")

        model = _ToyModel()
        loader = loader_mod.DummyModelLoader(LoadConfig(load_format=LoadFormat.DUMMY))
        model_config = SimpleNamespace(dtype=torch.bfloat16)

        with (
            patch.object(
                loader_mod.envs.SGLANG_CPU_QUANTIZATION, "get", return_value=True
            ),
            patch.object(loader_mod, "_get_quantization_config", return_value=None),
            patch.object(loader_mod, "_initialize_model", return_value=model),
            patch.object(
                loader_mod,
                "initialize_dummy_weights",
                side_effect=lambda _: events.append("dummy_init"),
            ),
        ):
            loader.load_model(
                model_config=model_config,
                device_config=SimpleNamespace(device="cpu"),
            )

        self.assertEqual(events, ["dummy_init", "post_load", "quantize"])

    def test_move_module_to_device_skips_meta_and_preserves_weight_tying(self):
        """_move_module_to_device must leave meta tensors untouched and preserve
        parameter object identity for tied weights."""
        shared = torch.nn.Parameter(torch.ones(4, 4))
        model = torch.nn.Module()
        model.embed = torch.nn.Module()
        model.embed.weight = shared
        model.lm_head = torch.nn.Module()
        model.lm_head.weight = shared
        model.offloaded = torch.nn.Parameter(torch.empty(4, 4, device="meta"))

        target = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        loader_mod._move_module_to_device(model, target)

        self.assertIs(model.embed.weight, model.lm_head.weight)
        self.assertEqual(model.embed.weight.device, target)
        self.assertTrue(model.offloaded.is_meta)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for GPU memory test")
    def test_cpu_quantization_halves_peak_gpu_memory_and_matches_default(self):
        """Per-module in-place GPU quantization keeps quantized FP8 weights on GPU
        without bouncing back to CPU, cutting peak GPU memory well below BF16 size."""

        class _MockFp8Method:
            def __init__(self, seen_devices):
                self._seen_devices = seen_devices

            def process_weights_after_loading(self, layer):
                w = layer.weight.data
                self._seen_devices.append(w.device.type)
                assert w.is_cuda, f"expected CUDA weight, got {w.device}"
                scale = w.abs().amax(dim=-1, keepdim=True).float() / 448.0
                q = (
                    (w.float() / scale.clamp(min=1e-12))
                    .clamp(-448.0, 448.0)
                    .to(torch.float8_e4m3fn)
                )
                layer.weight = torch.nn.Parameter(q, requires_grad=False)
                layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)

        class _Stack(torch.nn.Module):
            def __init__(self, n_layers, dim, seen_devices):
                super().__init__()
                self.quant_config = None
                self.layers = torch.nn.ModuleList()
                for _ in range(n_layers):
                    lin = torch.nn.Linear(dim, dim, bias=False)
                    lin.quant_method = _MockFp8Method(seen_devices)
                    self.layers.append(lin)

            def load_weights(self, weights):
                params = dict(self.named_parameters())
                for name, tensor in weights:
                    params[name].data.copy_(tensor)

        n_layers, dim = 32, 4096  # 32 * 32 MiB = 1024 MiB BF16 -> 512 MiB FP8
        torch.manual_seed(0)
        raw_weights = [
            (f"layers.{i}.weight", torch.randn(dim, dim, dtype=torch.bfloat16))
            for i in range(n_layers)
        ]

        # Pre-import BaseLayerWithLoRA so lazy import frames do not pin the first model.
        from sglang.srt.lora.layers import BaseLayerWithLoRA  # noqa: F401

        # 1) SGLANG_CPU_QUANTIZATION=1 peak memory (in-place per-module GPU quant)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        seen_cpuq = []
        loader_cpuq = DefaultModelLoader(LoadConfig())
        with (
            patch.object(
                loader_mod.envs.SGLANG_CPU_QUANTIZATION, "get", return_value=True
            ),
            patch.object(loader_mod, "_get_quantization_config", return_value=None),
            patch.object(
                loader_mod,
                "_initialize_model",
                side_effect=lambda *a, **kw: _Stack(n_layers, dim, seen_cpuq),
            ),
            patch.object(
                loader_cpuq, "_get_all_weights", return_value=iter(raw_weights)
            ),
        ):
            m_cpuq = loader_cpuq.load_model(
                model_config=SimpleNamespace(dtype=torch.bfloat16),
                device_config=SimpleNamespace(device="cuda"),
            )
        peak_cpuq = torch.cuda.max_memory_allocated()
        self.assertEqual(seen_cpuq, ["cuda"] * n_layers)
        cpuq_q = [layer.weight.data.cpu() for layer in m_cpuq.layers]
        cpuq_s = [layer.weight_scale.data.cpu() for layer in m_cpuq.layers]
        for layer in m_cpuq.layers:
            self.assertEqual(layer.weight.device.type, "cuda")
        del m_cpuq, loader_cpuq
        torch.cuda.empty_cache()

        # 2) Default GPU load peak memory (holds full 1024 MiB BF16 on GPU)
        torch.cuda.reset_peak_memory_stats()
        seen_default = []
        loader_def = DefaultModelLoader(LoadConfig())
        with (
            patch.object(
                loader_mod.envs.SGLANG_CPU_QUANTIZATION, "get", return_value=False
            ),
            patch.object(loader_mod, "_get_quantization_config", return_value=None),
            patch.object(
                loader_mod,
                "_initialize_model",
                side_effect=lambda *a, **kw: _Stack(n_layers, dim, seen_default),
            ),
            patch.object(
                loader_def, "_get_all_weights", return_value=iter(raw_weights)
            ),
        ):
            m_def = loader_def.load_model(
                model_config=SimpleNamespace(dtype=torch.bfloat16),
                device_config=SimpleNamespace(device="cuda"),
            )
        peak_default = torch.cuda.max_memory_allocated()
        for i, layer in enumerate(m_def.layers):
            self.assertTrue(
                torch.equal(
                    layer.weight.data.cpu().view(torch.uint8),
                    cpuq_q[i].view(torch.uint8),
                )
            )
            self.assertTrue(torch.equal(layer.weight_scale.data.cpu(), cpuq_s[i]))
        del m_def, loader_def
        torch.cuda.empty_cache()

        # Default holds all 1024 MiB BF16 on GPU; CPU quant peaks near 512 MiB FP8 + 1 layer.
        self.assertLess(peak_cpuq, peak_default * 0.70)


if __name__ == "__main__":
    unittest.main()
