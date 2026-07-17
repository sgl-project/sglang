import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.cp.mimo_v2 import (
    _collect_mimo_qkv_adaptations,
    maybe_adapt_mimo_v2_fused_qkv_for_cp,
    repack_mimo_v2_fused_qkv_block_fp8,
)
from sglang.srt.layers.parameter import BlockQuantScaleParameter
from sglang.srt.layers.quantization.fp8_utils import (
    block_quant_dequant,
    transform_scale_ue8m0,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _block_quantize_fixture(weight, block_size):
    block_n, block_k = block_size
    n, k = weight.shape
    padded = torch.zeros(
        (
            (n + block_n - 1) // block_n * block_n,
            (k + block_k - 1) // block_k * block_k,
        ),
        dtype=torch.float32,
    )
    padded[:n, :k] = weight
    blocks = padded.view(
        padded.shape[0] // block_n,
        block_n,
        padded.shape[1] // block_k,
        block_k,
    )
    scale = (blocks.abs().amax(dim=(1, 3)) / 448.0).clamp(min=1e-12)
    quantized = (blocks / scale[:, None, :, None]).to(torch.float8_e4m3fn)
    return quantized.view_as(padded)[:n, :k].contiguous(), scale.contiguous()


_RELOAD_BLOCK_SIZE = [128, 128]
_RELOAD_Q_ROWS = 512
_RELOAD_K_ROWS = 384
_RELOAD_V_ROWS = 256
_RELOAD_WEIGHT_COLS = 256
_RELOAD_WEIGHT_ROWS = _RELOAD_Q_ROWS + _RELOAD_K_ROWS + _RELOAD_V_ROWS


def _reload_checkpoint_fixture():
    checkpoint_groups = []
    for rank in range(4):
        checkpoint_groups.append(
            torch.cat(
                [
                    torch.full((128, _RELOAD_WEIGHT_COLS), float(rank + 1)),
                    torch.full((96, _RELOAD_WEIGHT_COLS), float(16 * (rank + 1))),
                    torch.full((64, _RELOAD_WEIGHT_COLS), float(-(rank + 1))),
                ]
            )
        )
    quantized_groups = [
        _block_quantize_fixture(group, _RELOAD_BLOCK_SIZE)
        for group in checkpoint_groups
    ]
    return (
        torch.cat([item[0] for item in quantized_groups]),
        torch.cat([item[1] for item in quantized_groups]),
    )


def _packed_runtime_scale(fill_value):
    # DeepGEMM repeats each block-row scale across 128 weight rows, packs four
    # UE8M0 bytes into int32, and leaves this runtime-only shape behind.
    return torch.full(
        (_RELOAD_WEIGHT_ROWS, 1), fill_value, dtype=torch.int32
    ).contiguous()


def _canonical_scale_for_weight(weight_shape):
    shape = (
        *weight_shape[:-2],
        (weight_shape[-2] + 127) // 128,
        (weight_shape[-1] + 127) // 128,
    )
    exponents = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape) % 8
    return torch.pow(2.0, exponents).contiguous()


def _pack_runtime_scale(canonical_scale, weight_rows):
    return transform_scale_ue8m0(
        canonical_scale,
        mn=weight_rows,
        use_torch_impl=True,
    )


class _FakeBlockFP8Linear:
    def __init__(
        self,
        weight_shape,
        *,
        canonical_scale=None,
        scale_data=None,
        format_ue8m0=True,
        use_mxfp8=False,
        serialized=True,
    ):
        self.weight = torch.nn.Parameter(
            torch.empty(weight_shape, dtype=torch.float8_e4m3fn, device="meta"),
            requires_grad=False,
        )
        canonical_scale = (
            _canonical_scale_for_weight(weight_shape)
            if canonical_scale is None
            else canonical_scale
        )
        if scale_data is None:
            if use_mxfp8:
                scale_data = torch.ones_like(canonical_scale, dtype=torch.uint8)
            elif format_ue8m0:
                scale_data = _pack_runtime_scale(canonical_scale, weight_shape[-2])
            else:
                scale_data = canonical_scale.clone()
        self.scale_loader = object()
        self.weight_scale_inv = BlockQuantScaleParameter(
            data=scale_data,
            input_dim=1,
            output_dim=0,
            weight_loader=self.scale_loader,
        )
        self.weight_scale_inv.format_ue8m0 = format_ue8m0
        self.quant_method = SimpleNamespace(
            block_quant=True,
            use_mxfp8=use_mxfp8,
            is_checkpoint_fp8_serialized=serialized,
            quant_config=SimpleNamespace(
                weight_block_size=_RELOAD_BLOCK_SIZE,
                is_checkpoint_fp8_serialized=serialized,
                use_mxfp8=use_mxfp8,
            ),
        )
        self.canonical_scale = canonical_scale


class _FakePackedMoE:
    def __init__(
        self,
        *,
        w13_shape=(2, 256, 512),
        w2_shape=(2, 128, 512),
        w13_format_ue8m0=True,
        w2_format_ue8m0=True,
        use_mxfp8=False,
        serialized=True,
        is_fp4_expert=False,
    ):
        self.w13_weight = torch.nn.Parameter(
            torch.empty(w13_shape, dtype=torch.float8_e4m3fn, device="meta"),
            requires_grad=False,
        )
        self.w2_weight = torch.nn.Parameter(
            torch.empty(w2_shape, dtype=torch.float8_e4m3fn, device="meta"),
            requires_grad=False,
        )
        self.w13_canonical_scale = _canonical_scale_for_weight(w13_shape)
        self.w2_canonical_scale = _canonical_scale_for_weight(w2_shape)
        w13_scale = (
            _pack_runtime_scale(self.w13_canonical_scale, w13_shape[-2])
            if w13_format_ue8m0
            else self.w13_canonical_scale.clone()
        )
        w2_scale = (
            _pack_runtime_scale(self.w2_canonical_scale, w2_shape[-2])
            if w2_format_ue8m0
            else self.w2_canonical_scale.clone()
        )
        self.w13_weight_scale_inv = torch.nn.Parameter(w13_scale, requires_grad=False)
        self.w2_weight_scale_inv = torch.nn.Parameter(w2_scale, requires_grad=False)
        self.w13_scale_loader = object()
        self.w2_scale_loader = object()
        self.w13_weight_scale_inv.weight_loader = self.w13_scale_loader
        self.w2_weight_scale_inv.weight_loader = self.w2_scale_loader
        self.w13_weight_scale_inv.format_ue8m0 = w13_format_ue8m0
        self.w2_weight_scale_inv.format_ue8m0 = w2_format_ue8m0
        self.quant_method = SimpleNamespace(
            block_quant=True,
            use_mxfp8=use_mxfp8,
            is_fp4_expert=is_fp4_expert,
            quant_config=SimpleNamespace(
                weight_block_size=_RELOAD_BLOCK_SIZE,
                is_checkpoint_fp8_serialized=serialized,
                use_mxfp8=use_mxfp8,
                is_fp4_experts=is_fp4_expert,
            ),
        )


def _capture_scale_state(scale):
    return {
        "parameter": scale,
        "data_ptr": scale.data_ptr(),
        "value": scale.detach().clone(),
        "dtype": scale.dtype,
        "shape": tuple(scale.shape),
        "stride": tuple(scale.stride()),
        "format_ue8m0": bool(getattr(scale, "format_ue8m0", False)),
        "weight_loader": getattr(scale, "weight_loader", None),
    }


class _FakeQKVParallelLinear:
    def __init__(self, runtime_scale, *, use_mxfp8=False):
        self.q_proj_shard_size = _RELOAD_Q_ROWS
        self.kv_proj_shard_size = _RELOAD_K_ROWS
        self.v_proj_shard_size = _RELOAD_V_ROWS
        self.weight = torch.nn.Parameter(
            torch.empty(
                (_RELOAD_WEIGHT_ROWS, _RELOAD_WEIGHT_COLS),
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        self.weight_scale_inv = torch.nn.Parameter(runtime_scale, requires_grad=False)
        self.weight_scale_inv.format_ue8m0 = True
        self.weight_scale_inv.weight_loader = object()
        self.quant_method = SimpleNamespace(
            use_mxfp8=use_mxfp8,
            is_checkpoint_fp8_serialized=True,
            quant_config=SimpleNamespace(
                weight_block_size=_RELOAD_BLOCK_SIZE,
                use_mxfp8=use_mxfp8,
            ),
        )


def _reload_model_with_modules(named_modules, architecture="MiMoV2ForCausalLM"):
    return SimpleNamespace(
        config=SimpleNamespace(
            architectures=[architecture],
            attention_projection_layout="fused_qkv",
            num_key_value_heads=4,
        ),
        named_modules=lambda: list(named_modules),
    )


def _reload_model(*qkv_projections):
    return _reload_model_with_modules(
        [
            (f"model.layers.{index}.self_attn.qkv_proj", qkv_proj)
            for index, qkv_proj in enumerate(qkv_projections)
        ]
    )


def _fill_qkv_from_checkpoint(qkv_proj):
    checkpoint_weight, checkpoint_scale = _reload_checkpoint_fixture()
    qkv_proj.weight.data.copy_(checkpoint_weight)
    qkv_proj.weight_scale_inv.data.copy_(checkpoint_scale)


class TestMiMoV2CPWeightAdapter(CustomTestCase):
    def assertScaleState(self, scale, state):
        self.assertIs(scale, state["parameter"])
        self.assertEqual(scale.data_ptr(), state["data_ptr"])
        self.assertTrue(torch.equal(scale, state["value"]))
        self.assertEqual(scale.dtype, state["dtype"])
        self.assertEqual(tuple(scale.shape), state["shape"])
        self.assertEqual(tuple(scale.stride()), state["stride"])
        self.assertEqual(
            bool(getattr(scale, "format_ue8m0", False)),
            state["format_ue8m0"],
        )
        self.assertIs(getattr(scale, "weight_loader", None), state["weight_loader"])

    def test_standard_runtime_packed_scale_is_canonicalized_for_reload(self):
        qkv_proj = _FakeQKVParallelLinear(_packed_runtime_scale(11))
        down_proj = _FakeBlockFP8Linear((4096, 16384))
        scale_param = down_proj.weight_scale_inv
        model = _reload_model_with_modules(
            [
                ("model.layers.0.self_attn.qkv_proj", qkv_proj),
                ("model.layers.0.mlp.down_proj", down_proj),
            ]
        )

        with (
            patch(
                "sglang.srt.layers.cp.mimo_v2.QKVParallelLinear",
                _FakeQKVParallelLinear,
            ),
            patch(
                "sglang.srt.layers.cp.mimo_v2.get_parallel",
                return_value=SimpleNamespace(attn_cp_size=4, attn_tp_size=1),
            ),
        ):
            with maybe_adapt_mimo_v2_fused_qkv_for_cp(model):
                self.assertIs(down_proj.weight_scale_inv, scale_param)
                self.assertIs(scale_param.weight_loader, down_proj.scale_loader)
                self.assertEqual(tuple(scale_param.shape), (32, 128))
                self.assertEqual(scale_param.dtype, torch.float32)
                self.assertFalse(scale_param.format_ue8m0)
                self.assertTrue(torch.equal(scale_param, down_proj.canonical_scale))
                _fill_qkv_from_checkpoint(qkv_proj)

        self.assertIs(down_proj.weight_scale_inv, scale_param)
        self.assertEqual(tuple(scale_param.shape), (32, 128))
        self.assertTrue(torch.equal(scale_param, down_proj.canonical_scale))

    def test_paired_moe_runtime_packed_scales_are_canonicalized(self):
        qkv_proj = _FakeQKVParallelLinear(_packed_runtime_scale(13))
        moe = _FakePackedMoE()
        w13_scale = moe.w13_weight_scale_inv
        w2_scale = moe.w2_weight_scale_inv
        model = _reload_model_with_modules(
            [
                ("model.layers.0.self_attn.qkv_proj", qkv_proj),
                ("model.layers.0.mlp.experts", moe),
            ]
        )

        with (
            patch(
                "sglang.srt.layers.cp.mimo_v2.QKVParallelLinear",
                _FakeQKVParallelLinear,
            ),
            patch(
                "sglang.srt.layers.cp.mimo_v2.get_parallel",
                return_value=SimpleNamespace(attn_cp_size=4, attn_tp_size=1),
            ),
        ):
            with maybe_adapt_mimo_v2_fused_qkv_for_cp(model):
                self.assertIs(moe.w13_weight_scale_inv, w13_scale)
                self.assertIs(moe.w2_weight_scale_inv, w2_scale)
                self.assertIs(w13_scale.weight_loader, moe.w13_scale_loader)
                self.assertIs(w2_scale.weight_loader, moe.w2_scale_loader)
                self.assertEqual(tuple(w13_scale.shape), (2, 2, 4))
                self.assertEqual(tuple(w2_scale.shape), (2, 1, 4))
                self.assertEqual(w13_scale.dtype, torch.float32)
                self.assertEqual(w2_scale.dtype, torch.float32)
                self.assertFalse(w13_scale.format_ue8m0)
                self.assertFalse(w2_scale.format_ue8m0)
                self.assertTrue(torch.equal(w13_scale, moe.w13_canonical_scale))
                self.assertTrue(torch.equal(w2_scale, moe.w2_canonical_scale))
                _fill_qkv_from_checkpoint(qkv_proj)

    def test_mixed_ownership_and_two_consecutive_reload_entries(self):
        qkv_proj = _FakeQKVParallelLinear(_packed_runtime_scale(17))
        down_proj = _FakeBlockFP8Linear((256, 512))
        moe = _FakePackedMoE(w2_format_ue8m0=False)
        scale_params = [
            qkv_proj.weight_scale_inv,
            down_proj.weight_scale_inv,
            moe.w13_weight_scale_inv,
            moe.w2_weight_scale_inv,
        ]
        scale_loaders = [
            getattr(scale, "weight_loader", None) for scale in scale_params
        ]
        model = _reload_model_with_modules(
            [
                ("model.layers.0.self_attn.qkv_proj", qkv_proj),
                ("model.layers.0.mlp.down_proj", down_proj),
                ("model.layers.0.mlp.experts", moe),
            ]
        )

        with (
            patch(
                "sglang.srt.layers.cp.mimo_v2.QKVParallelLinear",
                _FakeQKVParallelLinear,
            ),
            patch(
                "sglang.srt.layers.cp.mimo_v2.get_parallel",
                return_value=SimpleNamespace(attn_cp_size=4, attn_tp_size=1),
            ),
        ):
            with maybe_adapt_mimo_v2_fused_qkv_for_cp(model):
                # QKV is prepared for its TP-interleaved checkpoint shape and
                # is not double-owned by the generic packed-scale adapter.
                self.assertEqual(tuple(qkv_proj.weight_scale_inv.shape), (12, 2))
                self.assertTrue(
                    torch.equal(down_proj.weight_scale_inv, down_proj.canonical_scale)
                )
                self.assertTrue(
                    torch.equal(moe.w13_weight_scale_inv, moe.w13_canonical_scale)
                )
                self.assertTrue(
                    torch.equal(moe.w2_weight_scale_inv, moe.w2_canonical_scale)
                )
                _fill_qkv_from_checkpoint(qkv_proj)

            qkv_proj.weight_scale_inv.data = _packed_runtime_scale(19)
            qkv_proj.weight_scale_inv.format_ue8m0 = True
            down_second = down_proj.canonical_scale * 2
            down_proj.weight_scale_inv.data = _pack_runtime_scale(down_second, 256)
            down_proj.weight_scale_inv.format_ue8m0 = True
            w13_second = moe.w13_canonical_scale * 2
            moe.w13_weight_scale_inv.data = _pack_runtime_scale(w13_second, 256)
            moe.w13_weight_scale_inv.format_ue8m0 = True

            with maybe_adapt_mimo_v2_fused_qkv_for_cp(model):
                self.assertEqual(tuple(qkv_proj.weight_scale_inv.shape), (12, 2))
                self.assertTrue(torch.equal(down_proj.weight_scale_inv, down_second))
                self.assertTrue(torch.equal(moe.w13_weight_scale_inv, w13_second))
                self.assertTrue(
                    torch.equal(moe.w2_weight_scale_inv, moe.w2_canonical_scale)
                )
                _fill_qkv_from_checkpoint(qkv_proj)

        for current, original, loader in zip(
            [
                qkv_proj.weight_scale_inv,
                down_proj.weight_scale_inv,
                moe.w13_weight_scale_inv,
                moe.w2_weight_scale_inv,
            ],
            scale_params,
            scale_loaders,
        ):
            self.assertIs(current, original)
            self.assertIs(getattr(current, "weight_loader", None), loader)

    def test_prepare_failure_restores_every_successful_adaptation(self):
        qkv_proj = _FakeQKVParallelLinear(_packed_runtime_scale(23))
        down_proj = _FakeBlockFP8Linear((256, 512))
        moe = _FakePackedMoE()
        malformed = _FakeBlockFP8Linear(
            (256, 512),
            scale_data=torch.zeros((128, 2), dtype=torch.int32),
        )
        scales = [
            qkv_proj.weight_scale_inv,
            down_proj.weight_scale_inv,
            moe.w13_weight_scale_inv,
            moe.w2_weight_scale_inv,
            malformed.weight_scale_inv,
        ]
        states = [_capture_scale_state(scale) for scale in scales]
        model = _reload_model_with_modules(
            [
                ("model.layers.0.self_attn.qkv_proj", qkv_proj),
                ("model.layers.0.mlp.down_proj", down_proj),
                ("model.layers.0.mlp.experts", moe),
                ("model.layers.1.mlp.down_proj", malformed),
            ]
        )

        with (
            patch(
                "sglang.srt.layers.cp.mimo_v2.QKVParallelLinear",
                _FakeQKVParallelLinear,
            ),
            patch(
                "sglang.srt.layers.cp.mimo_v2.get_parallel",
                return_value=SimpleNamespace(attn_cp_size=4, attn_tp_size=1),
            ),
        ):
            with self.assertRaisesRegex(
                ValueError,
                r"model\.layers\.1\.mlp\.down_proj\.weight_scale_inv.*"
                r"\(128, 2\).*\(2, 4\)",
            ):
                with maybe_adapt_mimo_v2_fused_qkv_for_cp(model):
                    _fill_qkv_from_checkpoint(qkv_proj)

        for scale, state in zip(scales, states):
            self.assertScaleState(scale, state)

    def test_body_failure_restores_qkv_standard_and_moe_scales(self):
        qkv_proj = _FakeQKVParallelLinear(_packed_runtime_scale(29))
        down_proj = _FakeBlockFP8Linear((256, 512))
        moe = _FakePackedMoE()
        scales = [
            qkv_proj.weight_scale_inv,
            down_proj.weight_scale_inv,
            moe.w13_weight_scale_inv,
            moe.w2_weight_scale_inv,
        ]
        states = [_capture_scale_state(scale) for scale in scales]
        model = _reload_model_with_modules(
            [
                ("model.layers.0.self_attn.qkv_proj", qkv_proj),
                ("model.layers.0.mlp.down_proj", down_proj),
                ("model.layers.0.mlp.experts", moe),
            ]
        )

        with (
            patch(
                "sglang.srt.layers.cp.mimo_v2.QKVParallelLinear",
                _FakeQKVParallelLinear,
            ),
            patch(
                "sglang.srt.layers.cp.mimo_v2.get_parallel",
                return_value=SimpleNamespace(attn_cp_size=4, attn_tp_size=1),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "reload body failed"):
                with maybe_adapt_mimo_v2_fused_qkv_for_cp(model):
                    self.assertFalse(down_proj.weight_scale_inv.format_ue8m0)
                    self.assertFalse(moe.w13_weight_scale_inv.format_ue8m0)
                    self.assertFalse(moe.w2_weight_scale_inv.format_ue8m0)
                    raise RuntimeError("reload body failed")

        for scale, state in zip(scales, states):
            self.assertScaleState(scale, state)

    def test_qkv_finish_failure_restores_qkv_standard_and_moe_scales(self):
        qkv_proj = _FakeQKVParallelLinear(_packed_runtime_scale(31))
        down_proj = _FakeBlockFP8Linear((256, 512))
        moe = _FakePackedMoE()
        scales = [
            qkv_proj.weight_scale_inv,
            down_proj.weight_scale_inv,
            moe.w13_weight_scale_inv,
            moe.w2_weight_scale_inv,
        ]
        states = [_capture_scale_state(scale) for scale in scales]
        model = _reload_model_with_modules(
            [
                ("model.layers.0.self_attn.qkv_proj", qkv_proj),
                ("model.layers.0.mlp.down_proj", down_proj),
                ("model.layers.0.mlp.experts", moe),
            ]
        )

        with (
            patch(
                "sglang.srt.layers.cp.mimo_v2.QKVParallelLinear",
                _FakeQKVParallelLinear,
            ),
            patch(
                "sglang.srt.layers.cp.mimo_v2.get_parallel",
                return_value=SimpleNamespace(attn_cp_size=4, attn_tp_size=1),
            ),
            patch(
                "sglang.srt.layers.cp.mimo_v2.repack_mimo_v2_fused_qkv_block_fp8",
                side_effect=RuntimeError("qkv finish failed"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "qkv finish failed"):
                with maybe_adapt_mimo_v2_fused_qkv_for_cp(model):
                    self.assertFalse(down_proj.weight_scale_inv.format_ue8m0)
                    self.assertFalse(moe.w13_weight_scale_inv.format_ue8m0)
                    _fill_qkv_from_checkpoint(qkv_proj)

        for scale, state in zip(scales, states):
            self.assertScaleState(scale, state)

    def test_mixed_packed_and_canonical_retry_is_accepted(self):
        qkv_proj = _FakeQKVParallelLinear(_packed_runtime_scale(37))
        packed = _FakeBlockFP8Linear((256, 512))
        canonical = _FakeBlockFP8Linear((256, 512), format_ue8m0=False)
        moe = _FakePackedMoE(w2_format_ue8m0=False)
        canonical_state = _capture_scale_state(canonical.weight_scale_inv)
        w2_state = _capture_scale_state(moe.w2_weight_scale_inv)
        model = _reload_model_with_modules(
            [
                ("model.layers.0.self_attn.qkv_proj", qkv_proj),
                ("model.layers.0.mlp.down_proj", packed),
                ("model.layers.1.mlp.down_proj", canonical),
                ("model.layers.0.mlp.experts", moe),
            ]
        )

        with (
            patch(
                "sglang.srt.layers.cp.mimo_v2.QKVParallelLinear",
                _FakeQKVParallelLinear,
            ),
            patch(
                "sglang.srt.layers.cp.mimo_v2.get_parallel",
                return_value=SimpleNamespace(attn_cp_size=4, attn_tp_size=1),
            ),
        ):
            with maybe_adapt_mimo_v2_fused_qkv_for_cp(model):
                self.assertTrue(
                    torch.equal(packed.weight_scale_inv, packed.canonical_scale)
                )
                self.assertScaleState(canonical.weight_scale_inv, canonical_state)
                self.assertTrue(
                    torch.equal(moe.w13_weight_scale_inv, moe.w13_canonical_scale)
                )
                self.assertScaleState(moe.w2_weight_scale_inv, w2_state)
                _fill_qkv_from_checkpoint(qkv_proj)

    def test_true_mxfp8_is_untouched_while_standard_scale_adapts(self):
        qkv_proj = _FakeQKVParallelLinear(_packed_runtime_scale(41))
        eligible = _FakeBlockFP8Linear((256, 512))
        mxfp8 = _FakeBlockFP8Linear((256, 512), use_mxfp8=True)
        mxfp8_state = _capture_scale_state(mxfp8.weight_scale_inv)
        model = _reload_model_with_modules(
            [
                ("model.layers.0.self_attn.qkv_proj", qkv_proj),
                ("model.layers.0.mlp.down_proj", eligible),
                ("model.layers.1.mlp.down_proj", mxfp8),
            ]
        )

        with (
            patch(
                "sglang.srt.layers.cp.mimo_v2.QKVParallelLinear",
                _FakeQKVParallelLinear,
            ),
            patch(
                "sglang.srt.layers.cp.mimo_v2.get_parallel",
                return_value=SimpleNamespace(attn_cp_size=4, attn_tp_size=1),
            ),
        ):
            with maybe_adapt_mimo_v2_fused_qkv_for_cp(model):
                self.assertTrue(
                    torch.equal(eligible.weight_scale_inv, eligible.canonical_scale)
                )
                self.assertScaleState(mxfp8.weight_scale_inv, mxfp8_state)
                _fill_qkv_from_checkpoint(qkv_proj)

        self.assertScaleState(mxfp8.weight_scale_inv, mxfp8_state)

    def test_non_mimo_and_non_cp_runtime_scales_are_noops(self):
        with patch(
            "sglang.srt.layers.cp.mimo_v2.QKVParallelLinear",
            _FakeQKVParallelLinear,
        ):
            non_mimo = _FakeBlockFP8Linear((256, 512))
            non_mimo_state = _capture_scale_state(non_mimo.weight_scale_inv)
            non_mimo_model = _reload_model_with_modules(
                [("model.layers.0.mlp.down_proj", non_mimo)],
                architecture="OtherForCausalLM",
            )
            with patch(
                "sglang.srt.layers.cp.mimo_v2.get_parallel",
                return_value=SimpleNamespace(attn_cp_size=4, attn_tp_size=1),
            ):
                with maybe_adapt_mimo_v2_fused_qkv_for_cp(non_mimo_model):
                    self.assertScaleState(non_mimo.weight_scale_inv, non_mimo_state)

            qkv_proj = _FakeQKVParallelLinear(_packed_runtime_scale(43))
            non_cp = _FakeBlockFP8Linear((256, 512))
            qkv_state = _capture_scale_state(qkv_proj.weight_scale_inv)
            non_cp_state = _capture_scale_state(non_cp.weight_scale_inv)
            non_cp_model = _reload_model_with_modules(
                [
                    ("model.layers.0.self_attn.qkv_proj", qkv_proj),
                    ("model.layers.0.mlp.down_proj", non_cp),
                ]
            )
            with patch(
                "sglang.srt.layers.cp.mimo_v2.get_parallel",
                return_value=SimpleNamespace(attn_cp_size=1, attn_tp_size=4),
            ):
                with maybe_adapt_mimo_v2_fused_qkv_for_cp(non_cp_model):
                    self.assertScaleState(qkv_proj.weight_scale_inv, qkv_state)
                    self.assertScaleState(non_cp.weight_scale_inv, non_cp_state)

    def test_malformed_packed_and_canonical_shapes_are_diagnostic(self):
        cases = [
            (
                "packed",
                torch.zeros((128, 2), dtype=torch.int32),
                True,
                r"\(128, 2\).*\(2, 4\)",
            ),
            (
                "canonical",
                torch.zeros((1, 4), dtype=torch.float32),
                False,
                r"\(1, 4\).*\(2, 4\)",
            ),
        ]
        for label, scale_data, format_ue8m0, shapes_regex in cases:
            with self.subTest(label=label):
                qkv_proj = _FakeQKVParallelLinear(_packed_runtime_scale(47))
                malformed = _FakeBlockFP8Linear(
                    (256, 512),
                    scale_data=scale_data,
                    format_ue8m0=format_ue8m0,
                )
                model = _reload_model_with_modules(
                    [
                        ("model.layers.0.self_attn.qkv_proj", qkv_proj),
                        ("model.layers.0.mlp.down_proj", malformed),
                    ]
                )
                with (
                    patch(
                        "sglang.srt.layers.cp.mimo_v2.QKVParallelLinear",
                        _FakeQKVParallelLinear,
                    ),
                    patch(
                        "sglang.srt.layers.cp.mimo_v2.get_parallel",
                        return_value=SimpleNamespace(attn_cp_size=4, attn_tp_size=1),
                    ),
                ):
                    with self.assertRaisesRegex(
                        ValueError,
                        r"model\.layers\.0\.mlp\.down_proj\.weight_scale_inv.*"
                        + shapes_regex,
                    ):
                        with maybe_adapt_mimo_v2_fused_qkv_for_cp(model):
                            _fill_qkv_from_checkpoint(qkv_proj)

    def test_reload_from_packed_ue8m0_is_reentrant_and_body_failure_restores(self):
        checkpoint_weight, checkpoint_scale = _reload_checkpoint_fixture()
        qkv_proj = _FakeQKVParallelLinear(_packed_runtime_scale(17))
        model = _reload_model(qkv_proj)
        scale_param = qkv_proj.weight_scale_inv
        scale_loader = scale_param.weight_loader

        with (
            patch(
                "sglang.srt.layers.cp.mimo_v2.QKVParallelLinear",
                _FakeQKVParallelLinear,
            ),
            patch(
                "sglang.srt.layers.cp.mimo_v2.get_parallel",
                return_value=SimpleNamespace(attn_cp_size=4, attn_tp_size=1),
            ),
        ):
            with maybe_adapt_mimo_v2_fused_qkv_for_cp(model):
                self.assertIs(qkv_proj.weight_scale_inv, scale_param)
                self.assertIs(scale_param.weight_loader, scale_loader)
                self.assertEqual(tuple(scale_param.shape), (12, 2))
                self.assertEqual(scale_param.dtype, torch.float32)
                self.assertFalse(scale_param.format_ue8m0)
                qkv_proj.weight.data.copy_(checkpoint_weight)
                scale_param.data.copy_(checkpoint_scale)

            self.assertIs(qkv_proj.weight_scale_inv, scale_param)
            self.assertIs(scale_param.weight_loader, scale_loader)
            self.assertEqual(tuple(scale_param.shape), (9, 2))
            self.assertEqual(scale_param.dtype, torch.float32)
            self.assertFalse(scale_param.format_ue8m0)

            packed_after_postprocess = _packed_runtime_scale(29)
            packed_after_postprocess_copy = packed_after_postprocess.clone()
            packed_data_ptr = packed_after_postprocess.data_ptr()
            scale_param.data = packed_after_postprocess
            scale_param.format_ue8m0 = True

            with self.assertRaisesRegex(RuntimeError, "reload body failed"):
                with maybe_adapt_mimo_v2_fused_qkv_for_cp(model):
                    self.assertIs(qkv_proj.weight_scale_inv, scale_param)
                    self.assertIs(scale_param.weight_loader, scale_loader)
                    self.assertEqual(tuple(scale_param.shape), (12, 2))
                    self.assertEqual(scale_param.dtype, torch.float32)
                    self.assertFalse(scale_param.format_ue8m0)
                    raise RuntimeError("reload body failed")

        self.assertIs(qkv_proj.weight_scale_inv, scale_param)
        self.assertIs(scale_param.weight_loader, scale_loader)
        self.assertEqual(scale_param.data_ptr(), packed_data_ptr)
        self.assertTrue(torch.equal(scale_param, packed_after_postprocess_copy))
        self.assertEqual(tuple(scale_param.shape), (_RELOAD_WEIGHT_ROWS, 1))
        self.assertEqual(scale_param.dtype, torch.int32)
        self.assertTrue(scale_param.format_ue8m0)

    def test_finish_failure_restores_every_packed_ue8m0_scale(self):
        checkpoint_weight, checkpoint_scale = _reload_checkpoint_fixture()
        qkv_projections = [
            _FakeQKVParallelLinear(_packed_runtime_scale(41)),
            _FakeQKVParallelLinear(_packed_runtime_scale(53)),
        ]
        model = _reload_model(*qkv_projections)
        original_scales = [
            qkv.weight_scale_inv.detach().clone() for qkv in qkv_projections
        ]
        original_data_ptrs = [
            qkv.weight_scale_inv.data_ptr() for qkv in qkv_projections
        ]
        repack_calls = 0

        def fail_second_finish(*args, **kwargs):
            nonlocal repack_calls
            repack_calls += 1
            if repack_calls == 2:
                raise RuntimeError("adapter finish failed")
            return repack_mimo_v2_fused_qkv_block_fp8(*args, **kwargs)

        with (
            patch(
                "sglang.srt.layers.cp.mimo_v2.QKVParallelLinear",
                _FakeQKVParallelLinear,
            ),
            patch(
                "sglang.srt.layers.cp.mimo_v2.get_parallel",
                return_value=SimpleNamespace(attn_cp_size=4, attn_tp_size=1),
            ),
            patch(
                "sglang.srt.layers.cp.mimo_v2.repack_mimo_v2_fused_qkv_block_fp8",
                side_effect=fail_second_finish,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "adapter finish failed"):
                with maybe_adapt_mimo_v2_fused_qkv_for_cp(model):
                    for qkv_proj in qkv_projections:
                        qkv_proj.weight.data.copy_(checkpoint_weight)
                        qkv_proj.weight_scale_inv.data.copy_(checkpoint_scale)

        self.assertEqual(repack_calls, 2)
        for qkv_proj, original_scale, original_data_ptr in zip(
            qkv_projections, original_scales, original_data_ptrs
        ):
            scale = qkv_proj.weight_scale_inv
            self.assertEqual(scale.data_ptr(), original_data_ptr)
            self.assertTrue(torch.equal(scale, original_scale))
            self.assertEqual(tuple(scale.shape), (_RELOAD_WEIGHT_ROWS, 1))
            self.assertEqual(scale.dtype, torch.int32)
            self.assertTrue(scale.format_ue8m0)

    def test_serialized_mxfp8_qkv_is_rejected_by_quant_method_semantics(self):
        qkv_proj = _FakeQKVParallelLinear(
            torch.ones((9, 2), dtype=torch.float32), use_mxfp8=True
        )
        model = _reload_model(qkv_proj)

        with (
            patch(
                "sglang.srt.layers.cp.mimo_v2.QKVParallelLinear",
                _FakeQKVParallelLinear,
            ),
            patch(
                "sglang.srt.layers.cp.mimo_v2.get_parallel",
                return_value=SimpleNamespace(attn_cp_size=4, attn_tp_size=1),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "serialized MXFP8 QKV"):
                _collect_mimo_qkv_adaptations(model)

    def test_mtp_draft_input_embedding_clamps_sentinel_ids(self):
        from sglang.srt.model_executor.runner.eager_runner import (
            _get_cp_v2_input_embeds,
        )

        embedding = torch.nn.Embedding.from_pretrained(
            torch.arange(12, dtype=torch.float32).reshape(4, 3)
        )

        class DraftModel:
            config = SimpleNamespace(
                architectures=["MiMoV2MTP"],
                vocab_size=4,
            )

            def get_input_embedding(self, input_ids):
                raise AssertionError(
                    "MiMoV2MTP inherits an unavailable singular accessor"
                )

            def get_input_embeddings(self):
                return embedding

        input_ids = torch.tensor([-100, 1, 999])
        expected = embedding(torch.tensor([0, 1, 3]))

        self.assertTrue(
            torch.equal(_get_cp_v2_input_embeds(DraftModel(), input_ids), expected)
        )

    def test_mtp_draft_qkv_is_collected_for_cp_repacking(self):
        class FakeQKVParallelLinear:
            def __init__(self):
                self.q_proj_shard_size = 32
                self.kv_proj_shard_size = 24
                self.v_proj_shard_size = 16
                self.weight = torch.empty((72, 4), dtype=torch.float8_e4m3fn)
                self.weight_scale_inv = torch.empty((18, 2), dtype=torch.float32)
                self.quant_method = SimpleNamespace(
                    quant_config=SimpleNamespace(weight_block_size=[4, 2])
                )

        qkv_proj = FakeQKVParallelLinear()
        model = SimpleNamespace(
            config=SimpleNamespace(
                architectures=["MiMoV2MTP"],
                attention_projection_layout="fused_qkv",
                num_key_value_heads=4,
            ),
            named_modules=lambda: [("model.mtp_block.self_attn.qkv_proj", qkv_proj)],
        )

        with (
            patch(
                "sglang.srt.layers.cp.mimo_v2.QKVParallelLinear",
                FakeQKVParallelLinear,
            ),
            patch(
                "sglang.srt.layers.cp.mimo_v2.get_parallel",
                return_value=SimpleNamespace(attn_cp_size=4, attn_tp_size=1),
            ),
        ):
            adaptations = _collect_mimo_qkv_adaptations(model)

        self.assertEqual(len(adaptations), 1)
        self.assertEqual(
            adaptations[0].module_name,
            "model.mtp_block.self_attn.qkv_proj",
        )

    def test_repack_fused_tp4_qkv_requantizes_across_block_boundaries(self):
        block_size = [4, 2]
        q_values = [1.0, 2.0, 4.0, 8.0]
        k_values = [16.0, 32.0, 64.0, 128.0]
        v_values = [-1.0, -2.0, -4.0, -8.0]

        checkpoint_groups = []
        for rank in range(4):
            checkpoint_groups.append(
                torch.cat(
                    [
                        torch.full((8, 4), q_values[rank]),
                        torch.full((6, 4), k_values[rank]),
                        torch.full((4, 4), v_values[rank]),
                    ]
                )
            )

        quantized_groups = [
            _block_quantize_fixture(group, block_size) for group in checkpoint_groups
        ]
        checkpoint_weight = torch.cat([item[0] for item in quantized_groups])
        checkpoint_scale = torch.cat([item[1] for item in quantized_groups])

        repacked_weight, repacked_scale = repack_mimo_v2_fused_qkv_block_fp8(
            checkpoint_weight,
            checkpoint_scale,
            q_rows=32,
            k_rows=24,
            v_rows=16,
            checkpoint_tp_size=4,
            block_size=block_size,
            output_dtype=torch.float32,
        )
        actual = block_quant_dequant(
            repacked_weight,
            repacked_scale,
            block_size,
            torch.float32,
        )
        expected = torch.cat(
            [
                *(torch.full((8, 4), value) for value in q_values),
                *(torch.full((6, 4), value) for value in k_values),
                *(torch.full((4, 4), value) for value in v_values),
            ]
        )

        self.assertEqual(tuple(checkpoint_weight.shape), (72, 4))
        self.assertEqual(tuple(checkpoint_scale.shape), (20, 2))
        self.assertEqual(tuple(repacked_weight.shape), (72, 4))
        self.assertEqual(tuple(repacked_scale.shape), (18, 2))
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
