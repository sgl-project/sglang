"""Packed base weights must survive dynamic LoRA activation and replacement."""

import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import patch

import gguf
import torch
from torch import nn

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MiniMaxH3DiTArchConfig,
)
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.lora.linear import wrap_with_lora_layer
from sglang.multimodal_gen.runtime.layers.quantization.gguf import GGUFConfig
from sglang.multimodal_gen.runtime.loader.gguf_weights import GGUFTensorMeta
from sglang.multimodal_gen.runtime.loader.transformer_load_utils import (
    _validate_gguf_runtime_support,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
    LayerwiseOffloadManager,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import MiniMaxH3DiTModel
from sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline import LoRAPipeline
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# The diffusion runner installs the optional pipeline dependencies and exercises
# the real CUDA GGUF kernel. CPU execution substitutes only that kernel.
register_cuda_ci(est_time=60, stage="base-b", runner_config="diffusion-unit-1-gpu-h100")

_LINEAR = "sglang.multimodal_gen.runtime.layers.lora.linear"
_PIPELINE = "sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline"
_LOADER = "sglang.multimodal_gen.runtime.loader.transformer_load_utils"
_QUANT = "sglang.multimodal_gen.runtime.layers.quantization.gguf"


class _Pipeline(LoRAPipeline):
    def create_pipeline_stages(self, server_args):
        pass


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cpu_dequantize(packed, weight_type, dtype):
    # Substitute only the CUDA dequantization kernel with gguf-py's decoder.
    # The packed parameter, quant method, wrappers and adapter lifecycle are real.
    # The Q8_0 CUDA decoder multiplies scale and integer in half2, even for
    # float32 output. Preserve that rounding in this independent reference.
    return (
        torch.from_numpy(gguf.dequantize(packed.cpu().numpy(), weight_type))
        .half()
        .to(device=packed.device, dtype=dtype)
    )


def _layer(layer_type=ReplicatedLinear, skip_bias_add=False):
    generator = torch.Generator().manual_seed(7)
    dense = torch.randn(12, 32, generator=generator)
    packed = torch.from_numpy(
        gguf.quantize(dense.numpy(), gguf.GGMLQuantizationType.Q8_0)
    )
    meta = GGUFTensorMeta(
        int(gguf.GGMLQuantizationType.Q8_0),
        tuple(dense.shape),
        tuple(packed.shape),
        torch.uint8,
        "proj.qweight",
    )
    kwargs = dict(
        bias=True,
        skip_bias_add=skip_bias_add,
        quant_config=GGUFConfig("unused.gguf", {"proj.weight": meta}),
        prefix="proj",
    )
    if layer_type is not ReplicatedLinear:
        kwargs["tp_group"] = SimpleNamespace(world_size=1, rank_in_group=0)
    if layer_type is MergedColumnParallelLinear:
        layer = layer_type(32, [4, 8], **kwargs)
    else:
        layer = layer_type(32, 12, **kwargs)
    layer.qweight.data.copy_(packed)
    layer.bias.data.copy_(torch.randn(12, generator=generator))
    return layer.to(_device())


def _pipeline(layer):
    pipeline = object.__new__(_Pipeline)
    pipeline._stages = []
    pipeline.modules = {"transformer": nn.Module()}
    pipeline.modules["transformer"].add_module("proj", layer)
    pipeline.server_args = SimpleNamespace(lora_alpha=None, lora_merge_mode="auto")
    pipeline.lora_initialized = False
    pipeline.lora_rank = pipeline.lora_alpha = None
    pipeline.exclude_lora_layers = []
    pipeline.lora_adapters = defaultdict(dict)
    for key in (
        "loaded_adapter_paths",
        "loaded_adapter_files",
        "loaded_adapter_configs",
        "loaded_adapter_fingerprints",
        "loaded_adapter_alphas",
        "cur_adapter_name",
        "cur_adapter_path",
        "cur_adapter_strength",
        "cur_adapter_config",
        "lora_layers",
        "lora_layers_transformer_2",
        "lora_layers_critic",
        "is_lora_merged",
        "auto_merge_rounding_loss",
    ):
        setattr(pipeline, key, {})
    for name, rank in (("first", 2), ("second", 3)):
        g = torch.Generator().manual_seed(rank)
        pipeline.loaded_adapter_paths[name] = f"/{name}"
        pipeline.loaded_adapter_alphas[name] = rank * 2
        pipeline.lora_adapters[name] = {
            "proj.lora_A": torch.randn(rank, 32, generator=g).to(_device()),
            "proj.lora_B": torch.randn(12, rank, generator=g).to(_device()),
        }
    return pipeline


class TestGGUFLoRA(CustomTestCase):
    def setUp(self):
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        self.addCleanup(
            setattr, torch.backends.cuda.matmul, "allow_tf32", previous_tf32
        )
        torch.backends.cuda.matmul.allow_tf32 = False
        if not torch.cuda.is_available():
            kernel = patch(
                f"{_QUANT}.dequantize_gguf_weight", side_effect=_cpu_dequantize
            )
            kernel.start()
            self.addCleanup(kernel.stop)
        for context in (
            patch(f"{_LINEAR}.get_tp_rank", return_value=0),
            patch(f"{_PIPELINE}.dist.get_rank", return_value=0),
            patch.object(torch._dynamo.config, "disable", True),
        ):
            context.start()
            self.addCleanup(context.stop)

    def test_delta_matches_dense_reference_without_changing_packed_storage(self):
        """Catch packed-shape slicing, alpha scaling and duplicated/skipped bias."""
        for cls in (
            ReplicatedLinear,
            ColumnParallelLinear,
            MergedColumnParallelLinear,
            RowParallelLinear,
        ):
            for skip_bias in (False, True):
                with self.subTest(layer=cls.__name__, skip_bias=skip_bias):
                    base = _layer(cls, skip_bias)
                    before = base.qweight.clone()
                    wrapper = wrap_with_lora_layer(base, lora_rank=2, lora_alpha=4)
                    self.assertIsNone(wrapper.cpu_weight)
                    self.assertFalse(wrapper.can_merge_base_weight)
                    self.assertIs(wrapper.base_layer.qweight, base.qweight)
                    g = torch.Generator().manual_seed(10)
                    x = torch.randn(2, 3, 32, generator=g).to(_device())
                    a, b = (
                        torch.randn(2, 32, generator=g).to(_device()),
                        torch.randn(12, 2, generator=g).to(_device()),
                    )
                    wrapper.set_lora_weights(a, b, strength=0.25, merge_weights=False)
                    out, bias = wrapper(x)
                    weight = _cpu_dequantize(
                        before, gguf.GGMLQuantizationType.Q8_0, x.dtype
                    )
                    expected = nn.functional.linear(
                        x, weight + 0.5 * (b @ a), None if skip_bias else base.bias
                    )
                    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)
                    self.assertIs(bias, base.bias if skip_bias else None)
                    torch.testing.assert_close(base.qweight, before, rtol=0, atol=0)
                    self.assertFalse(hasattr(base, "weight"))

    def test_pipeline_switch_disable_and_reactivate(self):
        """An adapter with a different rank must not leave the previous delta active."""
        base = _layer()
        before = base.qweight.clone()
        x = torch.randn(2, 32, generator=torch.Generator().manual_seed(11)).to(
            _device()
        )
        baseline = base(x)[0]
        pipeline = _pipeline(base)
        for name, strength in (("first", 0.4), ("second", -0.3), ("first", 0.4)):
            pipeline.set_lora(
                name,
                target="transformer",
                strength=strength,
                merge_mode="auto",
                cache_merged=True,
            )
            wrapper = pipeline.modules["transformer"].proj
            a = pipeline.lora_adapters[name]["proj.lora_A"]
            b = pipeline.lora_adapters[name]["proj.lora_B"]
            expected = baseline + 2 * strength * (x @ a.T @ b.T)
            torch.testing.assert_close(wrapper(x)[0], expected)
            self.assertFalse(pipeline.is_lora_merged["transformer"])
            pipeline.deactivate_lora_weights()
            torch.testing.assert_close(wrapper(x)[0], baseline, atol=0, rtol=0)
            pipeline.set_lora(name, target="transformer", strength=strength)
            torch.testing.assert_close(wrapper(x)[0], expected)
        torch.testing.assert_close(base.qweight, before, atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, "dynamic"):
            pipeline.set_lora("second", merge_mode="merge")
        torch.testing.assert_close(wrapper(x)[0], expected)
        with self.assertRaisesRegex(ValueError, "one adapter"):
            pipeline.set_lora(
                ["first", "second"], lora_path=[None, None], merge_mode="dynamic"
            )
        torch.testing.assert_close(wrapper(x)[0], expected)

    def test_merge_rejected_before_adapter_or_storage_mutation(self):
        base = _layer()
        wrapper = wrap_with_lora_layer(base, lora_rank=2, lora_alpha=2)
        before = base.qweight.clone()
        with self.assertRaisesRegex(ValueError, "dynamic"):
            wrapper.set_lora_weights(torch.ones(2, 32), torch.ones(12, 2))
        self.assertIsNone(wrapper.lora_A)
        self.assertTrue(wrapper.disable_lora)
        with self.assertRaisesRegex(ValueError, "dynamic"):
            wrapper.merge_lora_weights()
        torch.testing.assert_close(base.qweight, before, atol=0, rtol=0)

    def test_placeholder_wrapping_retains_parameter_for_offload(self):
        base = _layer()
        parameter = base.qweight
        packed = parameter.detach().clone()
        parameter.data = torch.empty(1, dtype=torch.uint8)
        pipeline = _pipeline(base)
        pipeline.convert_to_lora_layers(snapshot_base=False)
        wrapper = pipeline.modules["transformer"].proj
        self.assertIs(wrapper.base_layer.qweight, parameter)
        self.assertIsNone(wrapper.cpu_weight)
        # The offload manager rebinds the same Parameter when the block is loaded.
        parameter.data = packed
        pipeline.set_lora("first", merge_mode="dynamic")
        x = torch.ones(1, 32, device=_device())
        a = pipeline.lora_adapters["first"]["proj.lora_A"]
        b = pipeline.lora_adapters["first"]["proj.lora_B"]
        expected = base(x)[0] + 2 * (x @ a.T @ b.T)
        torch.testing.assert_close(wrapper(x)[0], expected)
        torch.testing.assert_close(base.qweight, packed, atol=0, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA offload streams")
    def test_dynamic_lora_with_layerwise_offload(self):
        """Wrapping must retain the manager's qweight targets across release/prefetch."""

        class Block(nn.Module):
            def __init__(self, proj):
                super().__init__()
                self.proj = proj

            def forward(self, x):
                return self.proj(x)[0]

        class Transformer(LayerwiseOffloadableModuleMixin, nn.Module):
            def __init__(self, proj):
                super().__init__()
                self.blocks = nn.ModuleList([Block(proj)])

            def forward(self, x):
                return self.blocks[0](x)

        for startup_adapter in (False, True):
            with self.subTest(startup_adapter=startup_adapter):
                base = _layer()
                pipeline = _pipeline(base)
                model = Transformer(base)
                pipeline.modules["transformer"] = model
                for adapter in pipeline.lora_adapters.values():
                    for name in list(adapter):
                        adapter[f"blocks.0.{name}"] = adapter.pop(name)
                x = torch.ones(2, 32, device="cuda")
                baseline = model(x)
                if startup_adapter:
                    pipeline.set_lora("first", strength=0.25, merge_mode="dynamic")
                manager = LayerwiseOffloadManager(
                    model,
                    layers_attr_str="blocks",
                    num_layers=1,
                    enabled=True,
                    pin_cpu_memory=False,
                    prefetch_size=1,
                )
                model.layerwise_offload_managers = [manager]
                self.addCleanup(manager.remove_forward_hooks)
                for nickname, strength in (
                    ("first", 0.25),
                    ("second", -0.5),
                    ("first", 0.25),
                ):
                    pipeline.set_lora(nickname, strength=strength, merge_mode="dynamic")
                    a = pipeline.lora_adapters[nickname]["blocks.0.proj.lora_A"]
                    b = pipeline.lora_adapters[nickname]["blocks.0.proj.lora_B"]
                    torch.testing.assert_close(
                        model(x), baseline + 2 * strength * (x @ a.T @ b.T)
                    )
                    self.assertEqual(base.qweight.numel(), 1)
                    pipeline.deactivate_lora_weights()
                    torch.testing.assert_close(model(x), baseline, atol=0, rtol=0)
                    self.assertEqual(base.qweight.numel(), 1)

    def test_tp_and_merge_rejected_before_conversion(self):
        for tp, mode, error in ((2, "dynamic", "TP=1"), (1, "merge", "dynamic")):
            with self.subTest(tp=tp, mode=mode):
                base = _layer(ColumnParallelLinear)
                base.tp_size = tp
                pipeline = _pipeline(base)
                with self.assertRaisesRegex(ValueError, error):
                    pipeline.set_lora("first", merge_mode=mode)
                self.assertFalse(pipeline.lora_initialized)
                self.assertIs(pipeline.modules["transformer"].proj, base)

    def test_startup_support_matches_dynamic_api(self):
        args = SimpleNamespace(
            quantization=None,
            nunchaku_config=None,
            use_fsdp_inference=False,
            lora_path="some/adapter",
            lora_merge_mode="auto",
            tp_size=1,
            minimax_h3_adaln_online=False,
            minimax_h3_adaln_cache_path=None,
        )
        with patch(f"{_LOADER}.current_platform") as platform:
            platform.is_cuda.return_value = True
            for mode in ("dynamic", "auto"):
                args.lora_merge_mode = mode
                _validate_gguf_runtime_support(args)
            args.lora_merge_mode = "merge"
            with self.assertRaisesRegex(ValueError, "dynamic"):
                _validate_gguf_runtime_support(args)
            args.lora_merge_mode, args.tp_size = "dynamic", 2
            with self.assertRaisesRegex(ValueError, "TP=1"):
                _validate_gguf_runtime_support(args)

    def test_pruned_adaln_without_projection_rejects_full_width_adapter(self):
        """GGUF curve metadata used to let incompatible AdaLN deltas be installed."""
        model = SimpleNamespace(
            arch=MiniMaxH3DiTArchConfig(time_embed_dim=2, adaln_curve_grid=9),
            _adaln_precomputed=False,
        )
        prefix = "blocks.0.adaln_proj.linear."
        adapter = {
            prefix + "lora_A": torch.ones(1, 3),
            prefix + "lora_B": torch.ones(4, 1),
        }
        with self.assertRaisesRegex(ValueError, "adaln_basis.*adaln_mean"):
            MiniMaxH3DiTModel.prepare_lora_adapter(model, adapter)

        full_model = SimpleNamespace(
            arch=MiniMaxH3DiTArchConfig(time_embed_dim=3),
            _adaln_precomputed=False,
        )
        self.assertIs(
            MiniMaxH3DiTModel.prepare_lora_adapter(full_model, adapter), adapter
        )
        adapter[prefix + "lora_A"] = torch.ones(1, 2)
        self.assertIs(MiniMaxH3DiTModel.prepare_lora_adapter(model, adapter), adapter)
        attention = {"blocks.0.attn.qkv_proj.lora_A": torch.ones(1, 3)}
        self.assertIs(
            MiniMaxH3DiTModel.prepare_lora_adapter(model, attention), attention
        )


if __name__ == "__main__":
    unittest.main()
