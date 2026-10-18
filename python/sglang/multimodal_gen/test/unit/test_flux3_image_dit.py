# SPDX-License-Identifier: Apache-2.0
"""Tests for the FLUX.3 image DiT (``Flux3ImageDiT``).

The native DiT is the shared FLUX 3 transformer with native blocks; parity is
checked against :class:`Flux3Transformer` at toy size.
"""

import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.kernels import fused_op
from sglang.kernels.spec import KernelBackend, PlatformInfo
from sglang.multimodal_gen.configs.pipeline_configs.flux3_image import (
    flux3_image_dit_config,
)
from sglang.multimodal_gen.runtime.layers import flux3_fused_ops, linear
from sglang.multimodal_gen.runtime.layers.flux3_fused_ops import (
    flux3_gated_mlp,
    flux3_linear,
    flux3_qkv_qknorm_rope,
    flux3_shared_activation,
)
from sglang.multimodal_gen.runtime.layers.quantization import flux3_prims_ts
from sglang.multimodal_gen.runtime.layers.quantization.configs.convrot_int8_config import (
    ConvRotInt8Config,
)
from sglang.multimodal_gen.runtime.layers.quantization.flux3_prims_ts import (
    Flux3PrimsTsRole,
)
from sglang.multimodal_gen.runtime.loader.utils import (
    get_param_names_mapping,
    hf_to_custom_state_dict,
)
from sglang.multimodal_gen.runtime.managers.forward_context import (
    set_forward_context,
)
from sglang.multimodal_gen.runtime.models.dits import flux3_image
from sglang.multimodal_gen.runtime.models.dits.flux3 import (
    Flux3Transformer,
    timestep_embedding,
)
from sglang.multimodal_gen.runtime.models.dits.flux3_image import (
    TEXT,
    Flux3ImageDiT,
    _JointLayout,
    flux3_content_stack_is_active,
    flux3_cross_group_prefetch_target,
    pack,
)
from sglang.multimodal_gen.runtime.pipelines.flux3_image_weights import (
    resolve_flux3_image_dit_quant_config,
)

_TINY = dict(
    hidden_size=256,
    num_attention_heads=2,
    depth=1,
    depth_single_blocks=2,
    context_in_dim=64,
)


def _tiny_native_dit(device="cuda", **overrides):
    torch.manual_seed(0)
    # A single-rank TP group: construction needs no process group.
    with (
        mock.patch.object(linear, "get_tp_group", return_value=object()),
        mock.patch.object(linear, "get_group_size", return_value=1),
        mock.patch.object(linear, "get_group_rank", return_value=0),
        mock.patch.object(flux3_image, "get_tp_world_size", return_value=1),
        mock.patch(
            "sglang.multimodal_gen.runtime.layers.attention.layer.get_ring_parallel_world_size",
            return_value=1,
        ),
    ):
        model = Flux3ImageDiT(flux3_image_dit_config(**_TINY, **overrides))
    model = model.to(device=device, dtype=torch.bfloat16)
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.05)
    return model.eval()


def _sample(text_len: int, image_cond: bool, device="cuda"):
    def tokens(count, width):
        return (
            torch.randn(count, width).bfloat16().to(device),
            torch.rand(count, 4).to(device) * 8,
        )

    return (
        tokens(4, 128),
        (tokens(4, 128) if image_cond else None),
        tokens(text_len, 64),
    )


def _pack(samples):
    return pack(
        [
            {"image": image, "image_cond": condition, TEXT: text}
            for image, condition, text in samples
        ],
        {"image": 128, "image_cond": 128, TEXT: 64},
    )


def test_native_module_tree_matches_the_image_checkpoint():
    """Image checkpoints hold split q/k/v, a separate ``mlp_in`` and one
    ``final_layer.image`` head; the native tree must load them by name."""
    model = _tiny_native_dit("cpu")
    params = set(model.state_dict())
    assert not any(name.startswith("final_layer.image_cond") for name in params)
    assert model.vector_in is None
    mapping = get_param_names_mapping(model.param_names_mapping)
    for name in (
        "single_blocks.1.q_proj.weight",
        "content_mode_blocks.image_cond.0.v_proj.weight",
        "txt_mode_blocks.0.mlp_in.weight",
        "single_blocks.0.attn_out.weight",
        "early_stream_modulations.txt.lin.weight",
        "single_stream_modulations.image.lin.weight",
        "time_in.in_layer.weight",
        "emb_in.image_cond.weight",
        "txt_in.weight",
        "final_layer.image.adaLN_modulation.1.weight",
    ):
        target, _, _ = mapping(name)
        assert model._remap(target) in params, (name, target)
    modules = dict(model.named_modules())
    assert set(model.layer_names) == {
        "single_blocks",
        "txt_mode_blocks",
        "content_mode_blocks.image",
        "content_mode_blocks.image_cond",
    }
    assert all(
        isinstance(modules[name], torch.nn.ModuleList) for name in model.layer_names
    )


def test_attention_backend_failure_is_not_hidden():
    with mock.patch.object(
        flux3_image, "USPAttention", side_effect=ValueError("unsupported backend")
    ):
        with pytest.raises(ValueError, match="unsupported backend"):
            flux3_image.Flux3ImageAttention(2, 128)


def _reference_weights(native) -> dict[str, torch.Tensor]:
    """The native weights renamed to the FLUX 3 transformer's module tree."""
    state = {}
    native_state = native.state_dict()
    for name, tensor in native_state.items():
        if ".to_qkv." in name:
            prefix = name.split(".to_qkv.")[0]
            mlp_in = native_state[f"{prefix}.mlp_in.weight"]
            state[f"{prefix}.qkv_mlp.weight"] = torch.cat([tensor, mlp_in])
        elif ".mlp_in." in name:
            continue
        else:
            state[name.replace("_norm.weight", "_norm.scale")] = tensor
    return state


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA kernels")
@pytest.mark.parametrize("compact_timesteps", [False, True])
@pytest.mark.parametrize("backend", [None, "torch_cudnn_sdpa"])
def test_packed_cfg_forward_matches_flux3_transformer_per_sample(
    request, compact_timesteps, backend, monkeypatch
):
    """Packing both CFG samples into one varlen sequence must not let them see
    each other: each sample's prediction equals the shared FLUX 3 transformer
    run on that sample alone (bf16 fused-kernel tolerance)."""
    request.getfixturevalue("single_process_model_parallel")
    monkeypatch.setattr(
        flux3_image.get_global_server_args(), "attention_backend", backend
    )
    native = _tiny_native_dit()
    assert native.can_reuse_mode_blocks()
    reference = Flux3Transformer(flux3_image_dit_config(**_TINY)).cuda().bfloat16()
    reference.load_state_dict(_reference_weights(native), strict=True)

    samples = [_sample(3, image_cond=True), _sample(5, image_cond=True)]
    packed = _pack(samples)
    packed.timesteps["image"].fill_(0.7)
    if compact_timesteps:
        packed.timesteps = {mode: time[:1] for mode, time in packed.timesteps.items()}
        sliced = packed.sample(1)
        assert sliced.timesteps["image"].shape == (1,)
        assert torch.equal(sliced.timesteps["image"], packed.timesteps["image"])
    with torch.inference_mode():
        predicted = native(packed)["image"].float()
        packed.rope = {mode: native.rope(ids) for mode, ids in packed.ids.items()}
        cached = native(packed)["image"].float()
        torch.testing.assert_close(cached, predicted, rtol=0, atol=0)
        packed.static = flux3_image.StaticStreams()
        modulation = native.single_stream_modulations["txt"]
        with (
            mock.patch.object(
                native.txt_in, "forward", wraps=native.txt_in.forward
            ) as embed,
            mock.patch.object(modulation, "forward", wraps=modulation.forward) as mod,
        ):
            for _ in range(2):
                reused = native(packed)["image"].float()
                torch.testing.assert_close(reused, predicted, rtol=0, atol=0)
        assert embed.call_count == mod.call_count == 1
        for mode, rows in packed.sample(1).rope.items():
            start, end = packed.mode_indices[mode][1]
            assert torch.equal(rows, packed.rope[mode][start:end])
        with set_forward_context(current_timestep=0, attn_metadata=None):
            expected = []
            for image, condition, text in samples:
                zero = torch.zeros(1, dtype=torch.bfloat16, device="cuda")
                out = reference(
                    ctx=text[0][None],
                    ctx_ids=text[1][None],
                    timesteps_ctx=zero,
                    x_image=image[0][None],
                    x_image_ids=image[1][None],
                    x_image_timesteps=zero + 0.7,
                    x_image_cond=condition[0][None],
                    x_image_cond_ids=condition[1][None],
                    x_image_cond_timesteps=zero,
                )
                expected.append(out["x_image"][0].float())
    expected = torch.cat(expected)
    cosine = torch.nn.functional.cosine_similarity(
        predicted.flatten(), expected.flatten(), dim=0
    )
    assert cosine > 0.9995, cosine
    torch.testing.assert_close(predicted, expected, atol=3e-2, rtol=3e-2)
    with torch.inference_mode():
        packed.timesteps["image"].fill_(0.3)
        packed.data["image"].mul_(0.9)
        reused = native(packed)["image"]
        packed.static = None
        uncached = native(packed)["image"]
        torch.testing.assert_close(reused, uncached, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA kernels")
def test_packed_forward_skips_absent_streams(request):
    """t2i has no image_cond tokens: its mode blocks must not run, and only the
    configured output head is returned."""
    request.getfixturevalue("single_process_model_parallel")
    model = _tiny_native_dit()
    called = []
    for name, block in (
        ("image", model.content_mode_blocks["image"][0]),
        ("image_cond", model.content_mode_blocks["image_cond"][0]),
        ("text", model.txt_mode_blocks[0]),
        ("single", model.single_blocks[0]),
    ):
        block.register_forward_pre_hook(
            lambda _module, _inputs, value=name: called.append(value)
        )
    packed = _pack([_sample(3, image_cond=False), _sample(3, image_cond=False)])
    with torch.inference_mode():
        out = model(packed)
    assert set(out) == {"image"}
    assert out["image"].shape == (8, 128)
    assert called == ["image", "text", "single"]
    assert model._layerwise_active_content_mods == frozenset({"image"})


def test_flow_map_adds_each_samples_target_to_all_its_tokens():
    """As in the reference flow map: every token of a sample (text, condition
    and image) gets ``ref_time_in(emb(r))`` of that sample, r rounded to the
    timestep dtype."""
    model = _tiny_native_dit("cpu", flow_map=True)
    modes = ("image", "image_cond", TEXT)
    packed = _pack([_sample(3, True, "cpu"), _sample(5, False, "cpu")])
    assert packed.mode_indices["image_cond"][1] == (4, 4)
    packed.timesteps_ref = torch.tensor([0.5, 0.3])
    vec = {
        mode: torch.zeros(packed.data[mode].shape[0], 256).bfloat16() for mode in modes
    }
    with torch.inference_mode():
        target = model.ref_time_in(
            timestep_embedding(packed.timesteps_ref.bfloat16()).bfloat16()
        )
        model._add_reference_time(packed, modes, vec)
    for mode in modes:
        for sample, (start, end) in enumerate(packed.mode_indices[mode]):
            torch.testing.assert_close(
                vec[mode][start:end], target[sample].expand(end - start, -1)
            )

    with pytest.raises(ValueError, match="flow-map"):
        _tiny_native_dit("cpu")._add_reference_time(packed, modes, vec)
    packed.timesteps_ref = None
    with pytest.raises(ValueError, match="flow-map"):
        model._add_reference_time(packed, modes, vec)


def test_joint_layout_round_trips_in_sample_major_order():
    """The joint sequence is sample-major with modes in sorted order; unpacking
    must put every row back where the mode-major pack had it."""

    def rows(start, count):
        values = torch.arange(start, start + count, dtype=torch.float32)
        return values[:, None].expand(-1, 2).bfloat16(), torch.zeros(count, 4)

    packed = pack(
        [
            {"image": rows(0, 2), "image_cond": None, TEXT: rows(10, 1)},
            {"image": rows(2, 2), "image_cond": rows(20, 3), TEXT: rows(11, 2)},
        ],
        {"image": 2, "image_cond": 2, TEXT: 2},
    )
    layout = _JointLayout(packed, ("image", "image_cond", TEXT))
    joint = layout.pack(packed.data)
    assert joint[:, 0].tolist() == [0, 1, 10, 2, 3, 20, 21, 22, 11, 12]
    assert layout.sample_bounds == [(0, 3), (3, 10)]
    for mode, data in packed.data.items():
        assert torch.equal(layout.unpack(joint, mode, like=data), data)


def test_layerwise_prefetch_follows_the_active_forward_order():
    """Inactive content stacks are never prefetched; the successor walks each
    layer's active content stacks, then text, then the next layer's content."""
    active = frozenset({"image"})
    assert flux3_content_stack_is_active("content_mode_blocks.image", active)
    assert not flux3_content_stack_is_active("content_mode_blocks.image_cond", active)
    assert flux3_content_stack_is_active("single_blocks", active)
    assert flux3_content_stack_is_active("content_mode_blocks.image_cond", None)

    mods = ["image", "image_cond"]
    for (layers_attr, layer_idx), expected in (
        (("content_mode_blocks.image", 0), ("content_mode_blocks.image_cond", 0)),
        (("content_mode_blocks.image_cond", 2), ("txt_mode_blocks", 2)),
        (("txt_mode_blocks", 2), ("content_mode_blocks.image", 3)),
        (("txt_mode_blocks", 4), ("single_blocks", 0)),
        (("single_blocks", 0), None),
    ):
        target = flux3_cross_group_prefetch_target(layers_attr, layer_idx, mods, 5, 28)
        assert target == expected, (layers_attr, layer_idx)


class TestFlux3ImageDiTContract(unittest.TestCase):
    def test_split_qkv_checkpoint_tensors_concatenate_into_to_qkv(self):
        """Split q/k/v weights and scales, including NVFP4's 0-d tensorwise
        scales, must land in the packed ``to_qkv`` tensor in q, k, v order."""
        model = Flux3ImageDiT.__new__(Flux3ImageDiT)
        scalars = ("weight_scale_2", "input_scale")
        suffixes = ("weight", "weight_scale", *scalars)
        weights = [
            (
                f"single_blocks.0.{projection}.{suffix}",
                torch.tensor(float(shard) if suffix in scalars else [float(shard)]),
            )
            for suffix in suffixes
            for shard, projection in ((2, "v_proj"), (1, "k_proj"), (0, "q_proj"))
        ]
        state, _ = hf_to_custom_state_dict(
            model.preprocess_loaded_state_dict(weights),
            get_param_names_mapping(model.param_names_mapping),
        )
        self.assertEqual(
            {name: tensor.tolist() for name, tensor in state.items()},
            {
                f"single_blocks.0.to_qkv.{suffix}": [0.0, 1.0, 2.0]
                for suffix in suffixes
            },
        )

    def test_checkpoint_preprocessor_ignores_only_known_training_extras(self):
        model = Flux3ImageDiT.__new__(Flux3ImageDiT)
        names = (
            "weight",
            "single_blocks.0.query_norm.scale",
            "repa_projectors.block.weight",
            "repa_projectors_extra.block.weight",
            "unsupported.weight",
        )
        weights = model.preprocess_loaded_state_dict(
            (name, torch.ones(1)) for name in names
        )
        self.assertEqual(
            [name for name, _ in weights],
            [
                "weight",
                "single_blocks.0.query_norm.weight",
                "repa_projectors_extra.block.weight",
                "unsupported.weight",
            ],
        )

    def test_resolve_quant_config_from_metadata_or_cli(self):
        metadata = (
            "sglang.multimodal_gen.runtime.utils.quantization_utils."
            "get_quant_config_from_safetensors_metadata"
        )
        with mock.patch(metadata, return_value=object()):
            with self.assertRaisesRegex(ValueError, "do not also set --quantization"):
                resolve_flux3_image_dit_quant_config("dit.safetensors", "kitchen_int8")

        with mock.patch(metadata, return_value=None):
            config = resolve_flux3_image_dit_quant_config(
                "dit.safetensors",
                "kitchen_int8",
                ignored_layers=["adaln.linear"],
            )
        self.assertIsInstance(config, ConvRotInt8Config)
        self.assertEqual(config.ignored_layers, ["adaln.linear"])


class _StubLinear:
    """A FLUX.3 linear whose quant method may have planned a PrimsTS role."""

    def __init__(self, role=None, output="native", **method_attrs):
        self.quant_method = SimpleNamespace(
            prims_ts_role=role,
            activation_format="fp8",
            packed_activation_format=None,
        )
        for name, value in method_attrs.items():
            setattr(self.quant_method, name, value)
        self.output = output
        self.inputs = []

    def __call__(self, x):
        self.inputs.append(x)
        return self.output, None


class TestFlux3FusedOps(unittest.TestCase):
    _CUDA_X = SimpleNamespace(is_cuda=True)

    @staticmethod
    @contextlib.contextmanager
    def _platform(major, minor=0):
        platform = PlatformInfo(
            device_type="cuda", cuda_arch_major=major, cuda_arch_minor=minor
        )
        with (
            mock.patch.object(fused_op, "_platform", return_value=platform),
            mock.patch.object(PlatformInfo, "detect", return_value=platform),
        ):
            yield

    def test_capability_metadata_allows_sm10x_and_rejects_sm120(self):
        for arch, expected in (
            ((10, 0), True),
            ((10, 3), True),
            ((10, 7), True),
            ((9, 0), False),
            ((11, 0), False),
            ((12, 0), False),
        ):
            with self.subTest(arch=arch), self._platform(*arch):
                self.assertEqual(
                    flux3_fused_ops._FLUX3_LINEAR.auto_selected_backend()
                    is KernelBackend.FLASHINFER,
                    expected,
                )
                self.assertEqual(flux3_prims_ts.flux3_prims_ts_available(), expected)
        # A forced non-FlashInfer backend disables PrimsTS planning on sm10x.
        for forced, expected in (
            (KernelBackend.FLASHINFER, True),
            (KernelBackend.TORCH, False),
        ):
            with (
                self._platform(10, 0),
                mock.patch.object(fused_op, "_forced_backend", forced),
            ):
                self.assertEqual(flux3_prims_ts.flux3_prims_ts_available(), expected)

    def test_linear_runs_prims_ts_only_for_planned_layers_on_sm10x(self):
        x = self._CUDA_X
        prims_ts = mock.Mock(return_value="prims_ts")
        planned = _StubLinear(Flux3PrimsTsRole.LINEAR, apply_prims_ts_linear=prims_ts)
        with self._platform(10, 0):
            self.assertEqual(flux3_linear(planned, x, "shared"), "prims_ts")
            self.assertEqual(flux3_linear(_StubLinear(), x), "native")
            # A different planned role does not satisfy a plain projection.
            swiglu = _StubLinear(Flux3PrimsTsRole.SWIGLU)
            self.assertEqual(flux3_linear(swiglu, x), "native")
            with mock.patch.object(fused_op, "_forced_backend", KernelBackend.TORCH):
                self.assertEqual(flux3_linear(planned, x), "native")
        with self._platform(12, 0):
            self.assertEqual(flux3_linear(planned, x), "native")
        prims_ts.assert_called_once_with(planned, x, "shared")

    def test_qkv_prims_ts_epilogue_receives_norm_weights(self):
        x, heads, head_dim = self._CUDA_X, 2, 4
        epilogue = mock.Mock(return_value=torch.randn(1, 3, 3 * heads * head_dim))
        layer = _StubLinear(
            Flux3PrimsTsRole.QKV_QKNORM_ROPE, apply_prims_ts_qkv_qknorm_rope=epilogue
        )
        norm = SimpleNamespace(
            query_norm=SimpleNamespace(weight="q_w"),
            key_norm=SimpleNamespace(weight="k_w"),
        )
        with self._platform(10, 7):
            q, _, _ = flux3_qkv_qknorm_rope(
                layer, x, norm, "cos", "pos", heads, "shared"
            )
        epilogue.assert_called_once_with(layer, x, "q_w", "k_w", "cos", "pos", "shared")
        self.assertEqual(q.shape, (1, 3, heads, head_dim))

    def test_gated_mlp_hands_the_swiglu_output_to_mlp_down(self):
        """MLP-down takes the BF16 SwiGLU output, or the packed one directly
        when both projections share a packed activation format."""
        hidden = SimpleNamespace(is_cuda=True)
        swiglu = mock.Mock(return_value=hidden)
        down = mock.Mock(return_value="down")
        from_packed = mock.Mock(return_value="down_from_packed")
        mlp_in = _StubLinear(Flux3PrimsTsRole.SWIGLU, apply_prims_ts_swiglu=swiglu)
        mlp_out = _StubLinear(
            Flux3PrimsTsRole.LINEAR,
            apply_prims_ts_linear=down,
            apply_prims_ts_linear_from_packed=from_packed,
        )
        with self._platform(10, 0):
            result = flux3_gated_mlp(mlp_in, mlp_out, self._CUDA_X)
            self.assertEqual(result, "down")
            swiglu.assert_called_once_with(mlp_in, self._CUDA_X, None)
            down.assert_called_once_with(mlp_out, hidden, None)

            for layer in (mlp_in, mlp_out):
                layer.quant_method.packed_activation_format = "nvfp4"
            result = flux3_gated_mlp(mlp_in, mlp_out, self._CUDA_X, "shared")
        self.assertEqual(result, "down_from_packed")
        swiglu.assert_called_with(mlp_in, self._CUDA_X, "shared", packed_output=True)
        from_packed.assert_called_once_with(mlp_out, hidden)

    def test_shared_activation_requires_matching_prims_ts_consumers(self):
        x = self._CUDA_X
        quantize = mock.Mock(return_value="shared")
        mlp_in = _StubLinear(
            Flux3PrimsTsRole.SWIGLU, quantize_prims_ts_activation=quantize
        )
        qkv = _StubLinear(Flux3PrimsTsRole.QKV_QKNORM_ROPE)
        plain_qkv = _StubLinear(Flux3PrimsTsRole.LINEAR)
        nvfp4_qkv = _StubLinear(Flux3PrimsTsRole.LINEAR, activation_format="nvfp4")
        self.assertEqual(flux3_shared_activation(x, qkv, mlp_in), "shared")
        self.assertEqual(flux3_shared_activation(x, plain_qkv, mlp_in), "shared")
        self.assertIsNone(flux3_shared_activation(x, _StubLinear(), mlp_in))
        self.assertIsNone(flux3_shared_activation(x, qkv, _StubLinear()))
        self.assertIsNone(flux3_shared_activation(x, nvfp4_qkv, mlp_in))
        # Off-device inputs never take the PrimsTS operand path.
        cpu_x = SimpleNamespace(is_cuda=False)
        self.assertIsNone(flux3_shared_activation(cpu_x, qkv, mlp_in))
        self.assertEqual(quantize.call_args_list, [mock.call(mlp_in, x)] * 2)


if __name__ == "__main__":
    unittest.main()
