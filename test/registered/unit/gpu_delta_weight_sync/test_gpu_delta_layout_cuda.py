"""Exact-byte mapping checks against the normal SGLang/FlashInfer loader helpers.

Run with CUDA and the serving image's FlashInfer. These checks are outside the
update timing path and deliberately use the normal value-layout implementation
as an independent oracle for the feature-owned byte-mask implementation.
"""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _reference_layer(values, *, independent_mma=False):
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

    from sglang.srt.layers.moe.moe_runner.flashinfer_cutedsl import (
        interleave_w13_halves,
        resolve_cutedsl_standard_scales,
    )
    from sglang.srt.layers.quantization.modelopt_quant import _compute_gemm1_alphas
    from sglang.srt.layers.quantization.utils import swizzle_blockscale
    from sglang.srt.layers.utils import (
        alias_or_bind_derived_param,
        copy_or_rebind_param,
    )

    layer = torch.nn.Module()
    layer.moe_tp_size = 1
    layer.moe_ep_rank = 1
    layer.num_experts, layer.num_local_experts = 4, 2
    layer.quant_method = SimpleNamespace(_is_cutedsl_v2_standard=True)
    layer.moe_runner_config = SimpleNamespace(is_gated=True)
    layer.quant_config = SimpleNamespace(use_per_token_activation=False)
    layer._map_global_expert_id_to_local_expert_id = lambda expert: (
        expert - 2 if 2 <= expert < 4 else -1
    )
    layer._cutedsl_wrapper = SimpleNamespace(quant_mode="w4a16")
    for suffix in ("weight", "weight_scale"):
        # The actual checkpoint loader puts Up before Gate for CuTe DSL.
        fused = torch.cat([values["up", suffix], values["gate", suffix]], dim=1)
        copy_or_rebind_param(layer, "w13_" + suffix, interleave_w13_halves(fused))
        copy_or_rebind_param(layer, "w2_" + suffix, values["down", suffix].clone())
    for stem in ("w13", "w2"):
        scale = getattr(layer, stem + "_weight_scale")
        alias_or_bind_derived_param(
            layer,
            stem + "_weight_scale",
            stem + "_blockscale_swizzled",
            swizzle_blockscale(scale),
        )
        weight = getattr(layer, stem + "_weight")
        mma = convert_sf_to_mma_layout(
            getattr(layer, stem + "_blockscale_swizzled")
            .contiguous()
            .view(torch.uint8)
            .reshape(-1),
            m=weight.shape[1],
            k=2 * weight.shape[2],
            num_groups=2,
            sf_vec_size=16,
        )
        # Both public-loader aliasing and an independent existing MMA consumer
        # image must receive exactly one mask application.
        if independent_mma:
            mma = mma.clone(memory_format=torch.preserve_format)
        copy_or_rebind_param(layer, stem + "_blockscale_mma", mma)
    copy_or_rebind_param(
        layer,
        "w13_weight_scale_2",
        torch.stack(
            [values["gate", "weight_scale_2"], values["up", "weight_scale_2"]], dim=1
        ),
    )
    copy_or_rebind_param(
        layer, "w2_weight_scale_2", values["down", "weight_scale_2"].clone()
    )
    # W4A16 ignores canonical activation scales for its GEMM alphas. Keep their
    # global checkpoint buffers distinct from the neutralized derived values.
    copy_or_rebind_param(layer, "w13_input_scale", torch.ones(4, 2, device="cuda"))
    copy_or_rebind_param(layer, "w2_input_scale", torch.ones(4, device="cuda"))
    neutral = torch.ones(2, device="cuda")
    gate, up = _compute_gemm1_alphas(layer.w13_weight_scale_2, neutral, True)
    copy_or_rebind_param(layer, "g1_alphas", gate)
    copy_or_rebind_param(layer, "g1_alphas_up", up)
    copy_or_rebind_param(layer, "g2_alphas", neutral * layer.w2_weight_scale_2)
    copy_or_rebind_param(layer, "w13_input_scale_quant", neutral.clone())
    copy_or_rebind_param(layer, "w2_input_scale_quant", neutral.clone())
    a, b, c, d = resolve_cutedsl_standard_scales(layer)
    layer._cutedsl_scales = tuple(x.clone() for x in (a, b, c))
    layer._cutedsl_input_scale = d.clone()
    return layer


def _canonical(seed):
    rng = torch.Generator(device="cuda").manual_seed(seed)
    values = {}
    for projection, rows, cols in (
        ("gate", 128, 256),
        ("up", 128, 256),
        ("down", 256, 128),
    ):
        values[projection, "weight"] = torch.randint(
            256, (2, rows, cols // 2), dtype=torch.uint8, device="cuda", generator=rng
        )
        # Legal positive scale values; the resulting XOR masks are never
        # interpreted as FP8 numeric values by the delta mapper.
        values[projection, "weight_scale"] = torch.randint(
            1,
            120,
            (2, rows, cols // 16),
            dtype=torch.uint8,
            device="cuda",
            generator=rng,
        ).view(torch.float8_e4m3fn)
        values[projection, "weight_scale_2"] = (
            torch.rand(2, device="cuda", generator=rng) + 0.5
        )
    return values


@pytest.mark.parametrize("independent_mma", [False, True])
def test_expert_delta_matches_full_loader_layout_and_scale_refresh(
    monkeypatch, independent_mma
):
    from sglang.srt.layers.moe.moe_runner.flashinfer_cutedsl import (
        refresh_cutedsl_standard_scales_for_weight_update,
    )
    from sglang.srt.weight_sync.gpu_delta_layout import GpuDeltaLayout, _moe_binding

    monkeypatch.setenv("SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16", "1")
    old, new = _canonical(13), _canonical(29)
    live, expected = (
        _reference_layer(old, independent_mma=independent_mma),
        _reference_layer(new),
    )
    images = {
        name: parameter
        for name, parameter in live.named_parameters(remove_duplicate=False)
    }
    images.update(
        {f"wrapper.{i}": tensor for i, tensor in enumerate(live._cutedsl_scales)}
    )
    pointers = {name: tensor.data_ptr() for name, tensor in images.items()}
    for (projection, suffix), before in old.items():
        after = new[projection, suffix]
        for local in range(2):
            value = before[local]
            name = f"model.layers.3.mlp.experts.{local + 2}.{projection}_proj.{suffix}"
            dtype = {
                torch.uint8: "U8",
                torch.float8_e4m3fn: "F8_E4M3",
                torch.float32: "F32",
            }[value.dtype]
            binding = _moe_binding(
                name,
                {"dtype": dtype, "shape": list(value.shape)},
                live,
                local + 2,
                projection,
                suffix,
            )
            mask = value.reshape(-1).view(torch.uint8) ^ after[local].reshape(-1).view(
                torch.uint8
            )
            if binding.encoding == "raw_bytes":
                torch._foreach_copy_([binding.storage[0]], [after[local]])
            else:
                binding.xor(mask)
    plan = GpuDeltaLayout.__new__(GpuDeltaLayout)
    plan.derived = []
    plan._add_moe_derived("model.layers.3.mlp.experts", live)
    for image in plan.derived:
        image.destination.copy_(image.compute())
    refresh_cutedsl_standard_scales_for_weight_update(expected)
    for name, target in expected.named_parameters(remove_duplicate=False):
        actual = dict(live.named_parameters(remove_duplicate=False))[name]
        torch.testing.assert_close(
            actual.view(torch.uint8), target.view(torch.uint8), rtol=0, atol=0
        )
    for actual, target in zip(live._cutedsl_scales, expected._cutedsl_scales):
        torch.testing.assert_close(
            actual.view(torch.uint8), target.view(torch.uint8), rtol=0, atol=0
        )
    assert pointers == {name: tensor.data_ptr() for name, tensor in images.items()}
    assert live.w13_weight_scale.data_ptr() == live.w13_blockscale_swizzled.data_ptr()
    assert live.w2_weight_scale.data_ptr() == live.w2_blockscale_swizzled.data_ptr()


def test_feature_scale_permutation_matches_loader_padding():
    from sglang.srt.layers.quantization.utils import swizzle_blockscale
    from sglang.srt.weight_sync.gpu_delta_layout import swizzle_scale_bytes

    for shape in ((17, 3), (128, 64), (2, 256, 19)):
        values = torch.randint(1, 120, shape, dtype=torch.uint8, device="cuda")
        expected = swizzle_blockscale(values.view(torch.float8_e4m3fn)).view(
            torch.uint8
        )
        torch.testing.assert_close(
            swizzle_scale_bytes(values), expected, rtol=0, atol=0
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
