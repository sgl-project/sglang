import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import sglang.srt.model_loader.loader as loader_mod
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.model_loader.loader import DefaultModelLoader, post_load_weights
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


class _ShardedParameter(torch.nn.Parameter):
    pass


class _TiedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(8, 4)
        self.lm_head = torch.nn.Linear(4, 8, bias=False)
        self.lm_head.weight = self.embed.weight
        self.proj = torch.nn.Module()
        self.proj.weight = _ShardedParameter(torch.empty(3, 4), requires_grad=False)
        self.proj.weight.weight_loader = "sharded loader"
        self.register_buffer("cos_sin_cache", torch.empty(5))


class TestInitializeModelWithoutStorage(CustomTestCase):
    def _initialize(self):
        built_on = []

        def initialize_model(model_config, load_config, quant_config):
            built_on.append(torch.empty(0).device.type)
            return _TiedModel()

        loader = object.__new__(DefaultModelLoader)
        loader.load_config = object()
        with (
            patch.object(loader_mod, "_get_quantization_config", return_value=None),
            patch.object(loader_mod, "_initialize_model", side_effect=initialize_model),
        ):
            model, built_shapes_by_name = loader.initialize_model_without_storage(
                model_config=SimpleNamespace(dtype=torch.float32),
                device=torch.device("cpu"),
            )
        return model, built_shapes_by_name, built_on

    def test_the_model_is_built_on_meta_and_keeps_no_parameter_storage(self):
        """A trainer holding a replica per engine rank cannot afford each one's startup allocation."""
        model, built_shapes_by_name, built_on = self._initialize()

        self.assertEqual(built_on, ["meta"])
        self.assertEqual(
            {name: param.numel() for name, param in model.named_parameters()},
            {"embed.weight": 0, "proj.weight": 0},
        )
        self.assertEqual(
            built_shapes_by_name,
            {"embed.weight": torch.Size([8, 4]), "proj.weight": torch.Size([3, 4])},
        )
        self.assertEqual(model.cos_sin_cache.device.type, "meta")

    def test_parameters_keep_their_subclass_attributes_and_ties(self):
        """Loaders dispatch on the parameter class and its attributes, and a tied weight must stay one parameter."""
        model, _, _ = self._initialize()

        self.assertIsInstance(model.proj.weight, _ShardedParameter)
        self.assertEqual(model.proj.weight.weight_loader, "sharded loader")
        self.assertFalse(model.proj.weight.requires_grad)
        self.assertIs(model.lm_head.weight, model.embed.weight)


class TestPostLoadWeights(CustomTestCase):
    def test_gemma_weight_follows_a_weight_written_without_its_loader(self):
        """p2p weight updates and the remote-instance loader write params by address; fused allreduce kernels read
        `gemma_weight`, so a stale one would run the old norm."""
        model = torch.nn.Module()
        model.norm = GemmaRMSNorm(4)
        gemma_weight_address = model.norm.gemma_weight.data_ptr()
        new_weight = torch.tensor([0.5, -0.25, 0.0, 2.0])

        model.norm.weight.data.copy_(new_weight)
        post_load_weights(model)

        torch.testing.assert_close(model.norm.gemma_weight, new_weight + 1)
        self.assertEqual(model.norm.gemma_weight.data_ptr(), gemma_weight_address)


if __name__ == "__main__":
    unittest.main()
