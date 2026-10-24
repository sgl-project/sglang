# SPDX-License-Identifier: Apache-2.0
"""CPU-safe FLUX.3 tests for the rowwise-FP8 and ModelOpt NVFP4 DiT paths."""

import contextlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.runtime.layers import linear
from sglang.multimodal_gen.runtime.layers.linear import UnquantizedLinearMethod
from sglang.multimodal_gen.runtime.layers.quantization import flux3_prims_ts
from sglang.multimodal_gen.runtime.layers.quantization.flux3_prims_ts import (
    Flux3Fp8LinearMethod,
    Flux3ModelOptFp4Config,
    Flux3PrimsTsRole,
    Flux3RowwiseFp8Config,
)
from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    _linear_nvfp4_scales_to_tma,
    _rebase_partition_weight_scales,
    _swizzled_nvfp4_scales_to_linear,
)
from sglang.multimodal_gen.runtime.layers.quantization.rowwise_fp8 import (
    RowwiseFp8Config,
)
from sglang.multimodal_gen.runtime.pipelines.flux3_image_weights import (
    build_flux3_image_fp8r_quant_config,
    build_flux3_image_nvfp4_quant_config,
    select_flux3_image_quant_config,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.platforms.interface import DeviceCapability

# Building the DiT constructs attention layers, which read the SP group size.
pytestmark = pytest.mark.usefixtures("single_process_model_parallel")

_requires_flashinfer = unittest.skipUnless(
    importlib.util.find_spec("flashinfer"),
    "the PrimsTS weight pack ships with flashinfer (CUDA-only)",
)

_LEAVES = ("to_qkv", "mlp_in", "attn_out", "mlp_out")
_ALL_ROLES = {
    "to_qkv": Flux3PrimsTsRole.QKV_QKNORM_ROPE,
    "mlp_in": Flux3PrimsTsRole.SWIGLU,
    "attn_out": Flux3PrimsTsRole.LINEAR,
    "mlp_out": Flux3PrimsTsRole.LINEAR,
}
_SPLIT_FP8 = tuple(
    f"single_blocks.0.{leaf}"
    for leaf in ("q_proj", "k_proj", "v_proj", "mlp_in", "attn_out", "mlp_out")
)
_QUANTIZED = ("blocks.0.linear", "blocks.1.linear")


def _bfl_checkpoint(directory, *, layer_tensors, metadata, quantization):
    """Two quantized layers, one BF16-excluded layer, and one 1-D tensor."""
    tensors = {
        f"{prefix}.{name}": tensor.clone()
        for prefix in _QUANTIZED
        for name, tensor in layer_tensors.items()
    }
    tensors["excluded.weight"] = torch.zeros(4, 16, dtype=torch.bfloat16)
    tensors["norm.scale"] = torch.ones(4, dtype=torch.bfloat16)
    weights = Path(directory) / "model.safetensors"
    save_file(tensors, str(weights), metadata=metadata)
    sidecar = Path(directory) / "hf_quant_config.json"
    sidecar.write_text(
        json.dumps({"producer": {"name": "bfl"}, "quantization": quantization})
    )
    return weights, sidecar


def _fp8r_checkpoint(directory, *, excluded=("excluded",)):
    return _bfl_checkpoint(
        directory,
        layer_tensors={
            "weight": torch.zeros(4, 16, dtype=torch.float8_e4m3fn),
            "weight_scale": torch.ones(4),
        },
        metadata={"quantization": "fp8r", "scale": "per_row_fp32"},
        quantization={
            "quant_algo": "FP8_PER_CHANNEL_PER_TOKEN",
            "exclude_modules": list(excluded),
        },
    )


class TestBflCheckpointContract(unittest.TestCase):
    def test_builds_fp8r_rowwise_config(self):
        with tempfile.TemporaryDirectory() as directory:
            config = build_flux3_image_fp8r_quant_config(*_fp8r_checkpoint(directory))
        self.assertIsInstance(config, RowwiseFp8Config)
        self.assertEqual(config.fp8_layers, frozenset(_QUANTIZED))

    def test_fp8r_rejects_sidecar_exclusions_that_disagree_with_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            weights, sidecar = _fp8r_checkpoint(directory, excluded=())
            with self.assertRaisesRegex(ValueError, "exclude_modules disagrees"):
                build_flux3_image_fp8r_quant_config(weights, sidecar)

    def test_builds_nvfp4_modelopt_config(self):
        with tempfile.TemporaryDirectory() as directory:
            weights, sidecar = _bfl_checkpoint(
                directory,
                layer_tensors={
                    "weight": torch.zeros(4, 8, dtype=torch.uint8),
                    "weight_scale": torch.ones(4, 1).to(torch.float8_e4m3fn),
                    "weight_scale_2": torch.tensor(1.0),
                    "input_scale": torch.tensor(1.0),
                },
                metadata={"quantization": "nvfp4", "format": "pt"},
                quantization={
                    "quant_algo": "NVFP4",
                    "group_size": 16,
                    "exclude_modules": ["excluded"],
                },
            )
            config = build_flux3_image_nvfp4_quant_config(weights, sidecar)
        self.assertIsInstance(config, ModelOptFp4Config)


def _tiny_fp8_config(fp8_layers=None, input_scale_layers=()):
    if fp8_layers is None:
        fp8_layers = [f"single_blocks.0.{leaf}" for leaf in _LEAVES]
    return RowwiseFp8Config(
        format_version="1.1",
        fp8_layers=frozenset(fp8_layers),
        input_scale_layers=frozenset(input_scale_layers),
    )


def _tiny_nvfp4_config(*, dynamic_input_scale=False):
    return ModelOptFp4Config(
        is_checkpoint_nvfp4_serialized=True,
        group_size=16,
        exclude_modules=[],
        checkpoint_weight_scale_layout="swizzled",
        dynamic_input_scale=dynamic_input_scale,
    )


def _flux3_fp8_config(**kwargs):
    return Flux3RowwiseFp8Config.from_base_config(
        _tiny_fp8_config(**kwargs), head_dim=128
    )


def _flux3_nvfp4_config(**kwargs):
    return Flux3ModelOptFp4Config.from_base_config(
        _tiny_nvfp4_config(**kwargs), head_dim=128
    )


def _prims_ts_device(supported=True):
    return mock.patch.object(
        flux3_prims_ts, "flux3_prims_ts_available", return_value=supported
    )


def _tiny_native_model(
    quant_config, *, tp_size=1, tp_rank=0, depth=0, prims_ts_supported=True
):
    from sglang.multimodal_gen.configs.pipeline_configs.flux3_image import (
        flux3_image_dit_config,
    )
    from sglang.multimodal_gen.runtime.models.dits import flux3_image
    from sglang.multimodal_gen.runtime.models.dits.flux3_image import Flux3ImageDiT

    # These dimensions satisfy the production PrimsTS kernel's K/N tiling
    # requirements at TP=1; TP=2 shards some projections below the tiles.
    config = flux3_image_dit_config(
        hidden_size=512,
        num_attention_heads=4,
        depth=depth,
        depth_single_blocks=1,
        context_in_dim=64,
        axes_dim=(32, 32, 32, 32),
    )
    # NVFP4 linear methods refuse pre-Blackwell devices at construction.
    blackwell = (
        mock.patch.object(
            current_platform,
            "get_device_capability",
            return_value=DeviceCapability(10, 0),
        )
        if isinstance(quant_config, ModelOptFp4Config)
        else contextlib.nullcontext()
    )
    group = object()
    with (
        mock.patch.object(linear, "get_tp_group", return_value=group),
        mock.patch.object(linear, "get_group_size", return_value=tp_size),
        mock.patch.object(linear, "get_group_rank", return_value=tp_rank),
        mock.patch.object(flux3_image, "get_tp_world_size", return_value=tp_size),
        _prims_ts_device(prims_ts_supported),
        mock.patch.object(torch, "get_default_dtype", return_value=torch.bfloat16),
        blackwell,
    ):
        return Flux3ImageDiT(config, quant_config=quant_config)


def _roles(block):
    return {
        name: getattr(getattr(block, name).quant_method, "prims_ts_role", None)
        for name in _LEAVES
    }


class TestFlux3Fp8Construction(unittest.TestCase):
    @_requires_flashinfer
    def test_process_weights_prepares_prims_ts_weights(self):
        from sglang.kernels.ops.diffusion import pack_flux3_mlp_weight

        block = _tiny_native_model(_flux3_fp8_config()).single_blocks[0]
        self.assertEqual(_roles(block), _ALL_ROLES)
        sources = {}
        for name in _LEAVES:
            layer = getattr(block, name)
            layer.weight.data.copy_(
                torch.randn(layer.weight.shape).to(layer.weight.dtype)
            )
            layer.weight_scale.data.copy_(torch.rand(layer.weight_scale.shape))
            sources[name] = (
                layer.weight.data_ptr(),
                layer.weight.detach().clone(),
                layer.weight_scale.detach().clone(),
            )
            layer.quant_method.process_weights_after_loading(layer)

        # Every role hands PrimsTS the [N, K] storage itself; no second copy.
        for name in _LEAVES:
            layer = getattr(block, name)
            storage, weight_nk, scale = sources[name]
            if name == "mlp_in":
                weight_nk, scale = pack_flux3_mlp_weight(weight_nk.t(), scale)
            self.assertEqual(layer.weight.data_ptr(), storage)
            self.assertTrue(layer.weight.t().is_contiguous())
            self.assertTrue(
                torch.equal(
                    layer.weight.t().view(torch.uint8), weight_nk.view(torch.uint8)
                )
            )
            self.assertTrue(torch.equal(layer.weight_scale, scale))

    def test_prims_ts_falls_back_on_unsupported_gpu(self):
        """An unplanned mlp_in keeps the checkpoint's [gate | value] rows."""
        model = _tiny_native_model(_flux3_fp8_config(), prims_ts_supported=False)
        layer = model.single_blocks[0].mlp_in
        layer.weight.data.copy_(torch.randn(layer.weight.shape).to(layer.weight.dtype))
        source = layer.weight.detach().clone()
        layer.quant_method.process_weights_after_loading(layer)
        self.assertTrue(
            torch.equal(layer.weight.t().view(torch.uint8), source.view(torch.uint8))
        )

    def test_tp_local_shard_shapes_gate_prims_ts(self):
        """At TP=2 the QKV shard's N=768 no longer fits the 512-wide tiles."""
        block = _tiny_native_model(_flux3_fp8_config(), tp_size=2).single_blocks[0]
        self.assertEqual(block.to_qkv.output_size_per_partition, 768)
        self.assertEqual(_roles(block), {**_ALL_ROLES, "to_qkv": None})

    @_requires_flashinfer
    def test_swiglu_layout_rejects_generic_apply(self):
        layer = _tiny_native_model(_flux3_fp8_config()).single_blocks[0].mlp_in
        layer.quant_method.process_weights_after_loading(layer)
        with self.assertRaisesRegex(RuntimeError, "interleaved for the PrimsTS"):
            layer.quant_method.apply(layer, torch.zeros(1, 512, dtype=torch.bfloat16))

    def test_metadata_declared_bf16_path_stays_unquantized(self):
        quant_config = _flux3_fp8_config(fp8_layers=("single_blocks.0.to_qkv",))
        block = _tiny_native_model(quant_config).single_blocks[0]
        self.assertIsInstance(block.to_qkv.quant_method, Flux3Fp8LinearMethod)
        self.assertIsInstance(block.mlp_in.quant_method, UnquantizedLinearMethod)

    @_requires_flashinfer
    def test_fused_mlp_weight_pack_reverses_each_gate_value_pair(self):
        from sglang.kernels.ops.diffusion import pack_flux3_mlp_weight

        # Stored live weight is [K, gate|value].
        weight = torch.tensor([[1, 2, 11, 12], [3, 4, 13, 14]])
        scale = torch.tensor([0.1, 0.2, 1.1, 1.2])
        packed_weight, packed_scale = pack_flux3_mlp_weight(weight, scale)
        self.assertTrue(
            torch.equal(
                packed_weight,
                torch.tensor([[11, 13], [1, 3], [12, 14], [2, 4]]),
            )
        )
        self.assertTrue(torch.equal(packed_scale, torch.tensor([1.1, 0.1, 1.2, 0.2])))

    def test_flux3_config_remaps_split_qkv_metadata_and_input_scales(self):
        base = _tiny_fp8_config(_SPLIT_FP8, input_scale_layers=_SPLIT_FP8[:3])
        packed = _tiny_fp8_config(input_scale_layers=("single_blocks.0.to_qkv",))

        config = Flux3RowwiseFp8Config.from_base_config(base, head_dim=128)

        self.assertEqual(config.fp8_layers, packed.fp8_layers)
        self.assertEqual(config.input_scale_layers, packed.input_scale_layers)
        self.assertEqual(base.fp8_layers, frozenset(_SPLIT_FP8))
        qkv = _tiny_native_model(config).single_blocks[0].to_qkv
        self.assertIsInstance(qkv.quant_method, Flux3Fp8LinearMethod)
        self.assertEqual(tuple(qkv.input_scale.shape), (1,))
        # PrimsTS quantizes activations per token, so a static scale stays generic.
        self.assertIsNone(qkv.quant_method.prims_ts_role)


class TestFlux3Nvfp4Construction(unittest.TestCase):
    def test_dynamic_activations_use_generic_nvfp4_backend(self):
        model = _tiny_native_model(_flux3_nvfp4_config(dynamic_input_scale=True))
        self.assertEqual(set(_roles(model.single_blocks[0]).values()), {None})

    def test_prims_ts_coverage_exempts_unsupported_hardware(self):
        """Static NVFP4 loads on the generic backend where PrimsTS is absent."""
        config = _flux3_nvfp4_config()
        model = _tiny_native_model(config, depth=1, prims_ts_supported=False)
        self.assertEqual(set(_roles(model.single_blocks[0]).values()), {None})
        with _prims_ts_device(False):
            config.verify_prims_ts_coverage(model)

    def test_prims_ts_coverage_fails_closed_on_a_generic_projection(self):
        """The NVFP4 fast path covers QKV, attention-out, MLP-up, and MLP-down."""
        config = _flux3_nvfp4_config()
        model = _tiny_native_model(config, depth=1)
        for block in (model.content_mode_blocks["image"][0], model.single_blocks[0]):
            self.assertEqual(_roles(block), _ALL_ROLES)
        with _prims_ts_device():
            config.verify_prims_ts_coverage(model)

        block = model.single_blocks[0]
        for layer, demoted in (
            (block.attn_out, None),
            (block.to_qkv, Flux3PrimsTsRole.LINEAR),
        ):
            with self.subTest(layer.prefix):
                planned = layer.quant_method.prims_ts_role
                layer.quant_method.prims_ts_role = demoted
                with _prims_ts_device(), self.assertRaises(RuntimeError):
                    config.verify_prims_ts_coverage(model)
                layer.quant_method.prims_ts_role = planned

    def test_prims_ts_loader_preserves_128x4_scales_and_qkv_factors(self):
        block = _tiny_native_model(_flux3_nvfp4_config()).single_blocks[0]
        for layer in (block.to_qkv, block.mlp_in, block.attn_out, block.mlp_out):
            layer.weight_scale.data.fill_(1.0)
            layer.input_scale.data.fill_(1.0)
            layer.weight_scale_2.data.fill_(1.0)
            if layer is block.to_qkv:
                layer.input_scale.data.copy_(torch.tensor([0.5, 1.0, 2.0]))
                layer.weight_scale_2.data.copy_(torch.tensor([1.0, 2.0, 4.0]))
            layer.quant_method.process_weights_after_loading(layer)
            rows, packed_k = layer.weight.shape
            scale_k = ((packed_k * 2 // 16 + 3) // 4) * 4
            expected_numel = ((rows + 127) // 128) * 128 * scale_k
            self.assertEqual(layer.weight_scale_prims_ts_128x4.dtype, torch.uint8)
            self.assertEqual(layer.weight_scale_prims_ts_128x4.ndim, 1)
            self.assertEqual(layer.weight_scale_prims_ts_128x4.numel(), expected_numel)
            if layer is block.to_qkv:
                # Exact per-partition factors bypass the generic rebasing.
                self.assertTrue(
                    torch.equal(
                        layer.weight_scale_prims_ts_128x4,
                        _linear_nvfp4_scales_to_tma(layer.weight_scale)
                        .view(torch.uint8)
                        .reshape(-1),
                    )
                )
                self.assertTrue(
                    torch.equal(layer.prims_ts_qkv_scale, torch.tensor([2.0, 4.0, 8.0]))
                )
                self.assertEqual(layer.prims_ts_input_global_scale, 2.0)
                self.assertEqual(layer.prims_ts_weight_global_scale, 4.0)
            else:
                self.assertFalse(hasattr(layer, "prims_ts_qkv_scale"))

    def test_prims_ts_loader_interleaves_gated_mlp_rows_and_scales(self):
        """SwiGLU PrimsTS weights must be [value_0, gate_0, ...], not split."""
        layer = _tiny_native_model(_flux3_nvfp4_config()).single_blocks[0].mlp_in
        self.assertIs(layer.quant_method.prims_ts_role, Flux3PrimsTsRole.SWIGLU)
        n = layer.weight.shape[0]
        # Gate rows are tagged 1..8 and value rows 9..16, cycling with the row index.
        tags = torch.cat((torch.arange(n // 2) % 8 + 1, torch.arange(n // 2) % 8 + 9))
        layer.weight.data.copy_(tags[:, None].expand_as(layer.weight))
        layer.weight_scale.data.copy_(
            tags[:, None].float().expand_as(layer.weight_scale)
        )
        layer.weight_scale.nvfp4_scale_layout = "linear"
        layer.input_scale.data.fill_(1.0)
        layer.weight_scale_2.data.fill_(1.0)
        layer.quant_method.process_weights_after_loading(layer)

        scales = _swizzled_nvfp4_scales_to_linear(
            layer.weight_scale_prims_ts_128x4.view(torch.float8_e4m3fn).view(n, -1)
        )
        for rows in (layer.weight, scales.float()):
            self.assertEqual(rows[:4, 0].tolist(), [9, 1, 10, 2])
            self.assertEqual(rows[-2:, -1].tolist(), [16, 8])

    def test_prims_ts_prepared_layer_rejects_generic_apply(self):
        layer = _tiny_native_model(_flux3_nvfp4_config()).single_blocks[0].attn_out
        with self.assertRaisesRegex(RuntimeError, "prepared for PrimsTS"):
            layer.quant_method.apply(layer, torch.zeros(1, 512, dtype=torch.bfloat16))

    def test_flux3_method_keeps_the_v2_weight_loader(self):
        """Merged scalars such as ``mlp_in.weight_scale_2`` need the v2 loader."""
        block = _tiny_native_model(_flux3_nvfp4_config()).single_blocks[0]
        for name in _LEAVES:
            layer = getattr(block, name)
            for param in (layer.weight, layer.input_scale, layer.weight_scale_2):
                self.assertEqual(param.weight_loader, layer.weight_loader_v2)

    def test_row_parallel_weight_scale_shard_is_the_logical_shard(self):
        """A rank's row-parallel scales must be its own K range's scales.

        Narrowing the swizzled 128x4 bytes as if dim 1 were the logical K axis
        hands each rank another block's scales under TP>1.
        """
        from sglang.multimodal_gen.runtime.models import parameter as parameter_module

        tp_size, tp_rank = 2, 1
        model = _tiny_native_model(
            _flux3_nvfp4_config(), tp_size=tp_size, tp_rank=tp_rank
        )
        attn_out = model.single_blocks[0].attn_out
        rows = attn_out.weight_scale.shape[0]
        k_blocks = attn_out.weight_scale.shape[1] * tp_size
        # Every logical (row, k_block) entry is distinct before serialization.
        logical_full = (
            (torch.arange(rows * k_blocks, dtype=torch.int64) % 240 + 1)
            .to(torch.float8_e4m3fn)
            .view(rows, k_blocks)
        )
        with mock.patch.object(parameter_module, "get_tp_rank", return_value=tp_rank):
            attn_out.weight_scale.weight_loader(
                attn_out.weight_scale, _linear_nvfp4_scales_to_tma(logical_full)
            )

        shard = k_blocks // tp_size
        expected = logical_full[:, tp_rank * shard : (tp_rank + 1) * shard]
        recovered = attn_out.quant_method._logical_weight_scales(attn_out)
        self.assertEqual(tuple(recovered.shape), tuple(expected.shape))
        self.assertTrue(
            torch.equal(recovered.view(torch.uint8), expected.view(torch.uint8))
        )

    def test_generic_method_rebases_packed_partition_scales(self):
        """Dynamic NVFP4 keeps one global decode scale on the generic backend."""
        scales = torch.ones(6, 2).to(torch.float8_e4m3fn)
        rebased = _rebase_partition_weight_scales(
            scales, [2, 4], torch.tensor([1.0, 4.0]), torch.tensor(4.0)
        )
        self.assertTrue(
            torch.equal(
                rebased.float(),
                torch.tensor([[0.25] * 2] * 2 + [[1.0] * 2] * 4),
            )
        )


class TestFlux3QuantConfigSelection(unittest.TestCase):
    def test_loader_selects_flux3_configs(self):
        dit_config = SimpleNamespace(
            arch_config=SimpleNamespace(hidden_size=3072, num_attention_heads=24)
        )
        nvfp4 = _tiny_nvfp4_config()
        # quantization_utils adjusts this after construction; it must survive.
        nvfp4.swap_weight_nibbles = True
        for base, flux3_cls in (
            (_tiny_fp8_config(), Flux3RowwiseFp8Config),
            (nvfp4, Flux3ModelOptFp4Config),
        ):
            with self.subTest(flux3_cls.__name__):
                self.assertIsInstance(
                    select_flux3_image_quant_config(base, dit_config), flux3_cls
                )
                # Dynamic LoRA needs the generic layout, so it disables PrimsTS.
                fallback = select_flux3_image_quant_config(
                    base, dit_config, enable_prims_ts=False
                )
                self.assertIsInstance(fallback, flux3_cls)
                self.assertFalse(fallback.enable_prims_ts)

        selected = select_flux3_image_quant_config(nvfp4, dit_config)
        self.assertEqual(selected.head_dim, 128)
        self.assertTrue(selected.swap_weight_nibbles)


if __name__ == "__main__":
    unittest.main()
