import unittest
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import torch

from sglang.srt.layers.moe.moe_runner.flashinfer_cutedsl import interleave_w13_halves
from sglang.srt.layers.quantization.modelopt_quant import ModelOptNvFp4FusedMoEMethod
from sglang.srt.layers.quantization.utils import (
    prepare_static_weights_for_trtllm_fp4_moe,
    swizzle_blockscale,
)
from sglang.srt.layers.utils.common import alias_or_bind_derived_param
from sglang.srt.utils.weight_checker import (
    _build_check_entries,
    _build_quantized_set,
    _check_tensors,
    _hash_tensor,
)
from sglang.srt.utils.weight_checker_comparator import compare_weights
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

_EXPERTS, _HIDDEN, _INTERMEDIATE = 2, 256, 128
_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _checkpoint(hidden=_HIDDEN):
    generator = torch.Generator().manual_seed(42)
    result = {}
    for op, n, k in (
        ("w13", 2 * _INTERMEDIATE, hidden),
        ("w2", hidden, _INTERMEDIATE),
    ):
        codes = torch.randint(
            0, 16, (_EXPERTS, n, k), generator=generator, dtype=torch.uint8
        )
        result[f"{op}_weight"] = (codes[..., ::2] | (codes[..., 1::2] << 4)).cuda()
        result[f"{op}_weight_scale"] = (
            torch.randint(1, 8, (_EXPERTS, n, k // 16), generator=generator)
            .cuda()
            .to(torch.float8_e4m3fn)
        )
    result["w13_weight_scale_2"] = torch.tensor(
        [[0.25, 0.5], [1.0, 2.0]], device="cuda"
    )
    result["w2_weight_scale_2"] = torch.tensor([0.5, 0.25], device="cuda")
    return result


def _layer(checkpoint, backend):
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

    layer = torch.nn.Module()
    layer.moe_runner_config = SimpleNamespace(is_gated=True, activation="silu")
    method = ModelOptNvFp4FusedMoEMethod.__new__(ModelOptNvFp4FusedMoEMethod)
    method.enable_flashinfer_trtllm_moe = backend == "trtllm"
    layer.quant_method = method

    def put(name, value):
        layer.register_parameter(name, torch.nn.Parameter(value, requires_grad=False))

    for name, value in checkpoint.items():
        put(name, value.clone())
    if backend == "trtllm":
        q13, s13, q2, s2 = prepare_static_weights_for_trtllm_fp4_moe(
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            _HIDDEN,
            _INTERMEDIATE,
            _EXPERTS,
        )
        for name, value in (
            ("w13_weight", q13),
            ("w13_weight_scale", s13),
            ("w2_weight", q2),
            ("w2_weight_scale", s2),
        ):
            put(name, value)
    else:
        for suffix in ("weight", "weight_scale"):
            tensor = getattr(layer, f"w13_{suffix}")
            gate, up = tensor.chunk(2, dim=1)
            put(f"w13_{suffix}", interleave_w13_halves(torch.cat((up, gate), dim=1)))
        for op in ("w13", "w2"):
            alias_or_bind_derived_param(
                layer,
                f"{op}_weight_scale",
                f"{op}_blockscale_swizzled",
                swizzle_blockscale(getattr(layer, f"{op}_weight_scale")),
            )
            q = getattr(layer, f"{op}_weight")
            put(
                f"{op}_blockscale_mma",
                convert_sf_to_mma_layout(
                    getattr(layer, f"{op}_blockscale_swizzled")
                    .view(torch.uint8)
                    .reshape(-1),
                    m=q.shape[1],
                    k=q.shape[2] * 2,
                    num_groups=_EXPERTS,
                ),
            )
    put("g1_alphas", layer.w13_weight_scale_2[:, 0].clone())
    put("g1_alphas_up", layer.w13_weight_scale_2[:, 1].clone())
    put("g2_alphas", layer.w2_weight_scale_2.clone())
    if backend == "trtllm":
        put("g1_scale_c", layer.g1_alphas_up.clone() * 2)
    return layer


def _entries(layer, *, snapshot=False):
    with patch.object(
        ModelOptNvFp4FusedMoEMethod,
        "_is_cutedsl_v2_standard",
        new_callable=PropertyMock,
        return_value=True,
    ):
        plan = _build_quantized_set(layer)
    raw = {
        name: p.detach().cpu().contiguous() if snapshot else p
        for name, p in layer.named_parameters()
    }
    return list(_build_check_entries(raw, set(), plan))


class TestNvfp4WeightChecker(CustomTestCase):
    def test_unaliased_padded_scales_are_checked(self):
        checkpoint = _checkpoint(hidden=192)
        layer = _layer(checkpoint, "cutedsl")
        self.assertIsNot(layer.w2_weight_scale, layer.w2_blockscale_swizzled)
        expected = _entries(layer, snapshot=True)
        _check_tensors(expected, _entries(layer))
        for field in ("w2_weight_scale", "w2_blockscale_swizzled", "w2_blockscale_mma"):
            layer = _layer(checkpoint, "cutedsl")
            getattr(layer, field).view(torch.uint8).fill_(127)
            with self.assertRaisesRegex(Exception, "check tensor equality failed"):
                _check_tensors(expected, _entries(layer), allow_quant_error=True)

    def test_real_backend_layouts_decode_checkpoint_values(self):
        checkpoint = _checkpoint()
        for backend in ("trtllm", "cutedsl"):
            with self.subTest(backend=backend):
                entries = {
                    name: comparable
                    for name, _, comparable in _entries(
                        _layer(checkpoint, backend), snapshot=True
                    )
                }
                for op in ("w13", "w2"):
                    q = checkpoint[f"{op}_weight"]
                    codes = torch.stack((q & 15, q >> 4), dim=-1).reshape(
                        _EXPERTS, q.shape[1], -1
                    )
                    values = torch.tensor(_VALUES, device="cuda")[
                        (codes & 7).long()
                    ] * (1 - 2 * (codes >> 3).float())
                    s = (
                        checkpoint[f"{op}_weight_scale"]
                        .float()
                        .repeat_interleave(16, dim=-1)
                    )
                    g = checkpoint[f"{op}_weight_scale_2"].reshape(_EXPERTS, -1)
                    g = g.repeat_interleave(q.shape[1] // g.shape[1], dim=1).unsqueeze(
                        -1
                    )
                    expected = (values * s * g).to(torch.bfloat16)
                    if backend == "trtllm":
                        from flashinfer.fused_moe.core import (
                            _maybe_get_cached_w3_w1_permute_indices,
                            get_w2_permute_indices_with_cache,
                        )

                        permute = (
                            _maybe_get_cached_w3_w1_permute_indices
                            if op == "w13"
                            else get_w2_permute_indices_with_cache
                        )
                        expected = expected[:, permute({}, q[0], 128)]
                    elif op == "w13":
                        gate, up = expected.chunk(2, dim=1)
                        expected = interleave_w13_halves(torch.cat((up, gate), dim=1))
                    torch.testing.assert_close(
                        entries[f"{op}_weight"].dequantize(),
                        expected.flatten(0, 1),
                        rtol=0,
                        atol=0,
                    )

    def test_equivalent_global_and_block_scales_compare_and_hash_equal(self):
        checkpoint = _checkpoint()
        other = {name: tensor.clone() for name, tensor in checkpoint.items()}
        for op in ("w13", "w2"):
            other[f"{op}_weight_scale"] = (other[f"{op}_weight_scale"].float() / 2).to(
                torch.float8_e4m3fn
            )
            other[f"{op}_weight_scale_2"] *= 2
        for backend in ("trtllm", "cutedsl"):
            with self.subTest(backend=backend):
                expected = _entries(_layer(checkpoint, backend), snapshot=True)
                actual = _entries(_layer(other, backend))
                _check_tensors(expected, actual)
                self.assertEqual(
                    {
                        n: _hash_tensor(c.dequantize())
                        for n, flag, c in expected
                        if flag
                    },
                    {n: _hash_tensor(c.dequantize()) for n, flag, c in actual if flag},
                )

    def test_corruption_is_detected_in_weights_scales_and_execution_alphas(self):
        checkpoint = _checkpoint()
        for backend in ("trtllm", "cutedsl"):
            fields = [
                "w13_weight",
                "w2_weight",
                "w13_weight_scale",
                "w2_weight_scale",
                "w13_weight_scale_2",
                "w2_weight_scale_2",
                "g1_alphas",
                "g1_alphas_up",
                "g2_alphas",
            ]
            fields += (
                ["g1_scale_c"]
                if backend == "trtllm"
                else ["w13_blockscale_mma", "w2_blockscale_mma"]
            )
            expected = _entries(_layer(checkpoint, backend), snapshot=True)
            for field in fields:
                with self.subTest(backend=backend, field=field):
                    layer = _layer(checkpoint, backend)
                    tensor = getattr(layer, field)
                    if field.endswith("weight"):
                        tensor[..., 0] ^= 0x88
                    elif tensor.element_size() == 1:
                        tensor.view(torch.uint8).fill_(127)
                    else:
                        tensor.mul_(16)
                    with self.assertRaisesRegex(
                        Exception, "check tensor equality failed"
                    ):
                        _check_tensors(
                            expected, _entries(layer), allow_quant_error=True
                        )

    def test_chunking_and_snapshot_strides_do_not_change_results(self):
        checkpoint = _checkpoint()
        for backend in ("trtllm", "cutedsl"):
            with self.subTest(backend=backend):
                layer = _layer(checkpoint, backend)
                expected, actual = _entries(layer, snapshot=True), _entries(layer)
                with patch(
                    "sglang.srt.utils.weight_checker_comparator.CHUNK_NUMEL",
                    256 * 4 * 7,
                ):
                    for (_, _, lhs), (_, _, rhs) in zip(expected, actual, strict=True):
                        result = compare_weights(lhs, rhs)
                        self.assertTrue(result.equal, result)


if __name__ == "__main__":
    unittest.main()
