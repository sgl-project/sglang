"""When the MiniMax-M3.1 low-batch FFN engages, and that every other case keeps
the stock MoE: a layer it cannot serve, a call outside a full decode graph of
1-32 rows, moved weights, the switch off."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsFusedMoEMethod,
)
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.layers.quantization.fp8_utils import Mxfp8DenseGemmBackend
from sglang.srt.layers.quantization.minimax_m3_nvfp4 import MiniMaxM3Nvfp4MoE
from sglang.srt.models import minimax_m3
from sglang.srt.models.minimax_common.nvidia import low_batch_ffn
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HIDDEN, EXPERTS = 6144, 128


def _meta(*shape, dtype=torch.uint8):
    return torch.empty(shape, dtype=dtype, device="meta")


def _linear(n, k, backend=Mxfp8DenseGemmBackend.FLASHINFER_CUTEDSL):
    method = object.__new__(Fp8LinearMethod)
    method.use_mxfp8 = True
    method.convert_mxfp8_to_block = False
    method.mxfp8_dense_backend = backend
    return SimpleNamespace(
        quant_method=method,
        weight=_meta(n, k, dtype=torch.float8_e4m3fn),
        weight_scale_inv_swizzled=_meta(n * k // 32),
    )


def _moe(columns=768):
    """A processed M3.1 MoE layer of one TP rank, on the meta device."""
    scheme = object.__new__(MiniMaxM3Nvfp4MoE)
    scheme.use_flashinfer_trtllm = True
    scheme.use_mega_moe_prefill = False
    bias = _meta(EXPERTS, dtype=torch.float32)
    experts = SimpleNamespace(
        quant_method=object.__new__(CompressedTensorsFusedMoEMethod),
        scheme=scheme,
        intermediate_size_per_partition=columns,
        num_experts=EXPERTS,
        num_local_experts=EXPERTS,
        hidden_size=HIDDEN,
        _minimax_activation="swigluoai",
        moe_runner_config=SimpleNamespace(
            gemm1_alpha=1.702, gemm1_beta=1.0, gemm1_clamp_limit=7.0, swiglu_limit=7.0
        ),
        w13_weight=_meta(EXPERTS, 2 * columns, HIDDEN // 2),
        w13_weight_scale=_meta(EXPERTS, 2 * columns * HIDDEN // 16),
        w2_weight=_meta(EXPERTS, HIDDEN, columns // 2),
        w2_weight_scale=_meta(EXPERTS, HIDDEN * columns // 16),
        g1_alphas=_meta(EXPERTS, dtype=torch.float32),
        g2_alphas=_meta(EXPERTS, dtype=torch.float32),
    )
    topk_config = SimpleNamespace(
        scoring_func="sigmoid",
        top_k=4,
        renormalize=True,
        use_grouped_topk=False,
        custom_routing_function=None,
        sqrtsoftplus_log1p=False,
        apply_routed_scaling_factor_on_output=True,
        routed_scaling_factor=2.0,
        correction_bias=bias,
    )
    return SimpleNamespace(
        layer_id=3,
        supports_deferred_finalize=True,
        trtllm_decode_with_mega_moe=False,
        num_fused_shared_experts=0,
        routed_scaling_factor=2.0,
        e_score_correction_bias=bias,
        topk=SimpleNamespace(
            topk_config=topk_config, is_draft=False, enable_waterfill=False
        ),
        experts=experts,
        shared_experts=SimpleNamespace(
            swiglu_alpha=1.702,
            swiglu_limit=7.0,
            gate_up_proj=_linear(2 * columns, HIDDEN),
            down_proj=_linear(HIDDEN, columns),
        ),
    )


ENGAGING_CALL = dict(
    defer_finalize=True,
    decode_or_verify=True,
    capture_mode=True,
    breakable_graph=False,
    stream_capturing=True,
    can_run_tbo=False,
    sp_active=False,
    rows=8,
)


class TestLayerSupport(CustomTestCase):
    def setUp(self):
        # Platform, server flags and parallel layout are config, checked apart.
        patcher = patch.multiple(
            low_batch_ffn,
            _platform_reason=lambda: None,
            _server_reason=lambda: None,
            _parallel_reason=lambda: None,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_tp4_and_tp8_layers_are_supported(self):
        for columns in (768, 384):
            self.assertIsNone(low_batch_ffn.unsupported_reason(_moe(columns)))

    def test_unsupported_layers_keep_the_stock_moe(self):
        def set_attr(path, value):
            def mutate(moe):
                *owners, name = path.split(".")
                target = moe
                for owner in owners:
                    target = getattr(target, owner)
                setattr(target, name, value)

            return mutate

        cases = {
            "other widths": lambda moe: _moe(512),
            "no deferred finalize": set_attr("supports_deferred_finalize", False),
            "MegaMoE decode": set_attr("trtllm_decode_with_mega_moe", True),
            "MegaMoE prefill": set_attr("experts.scheme.use_mega_moe_prefill", True),
            "draft layer": set_attr("topk.is_draft", True),
            "fused shared expert": set_attr("num_fused_shared_experts", 1),
            "softmax routing": set_attr("topk.topk_config.scoring_func", "softmax"),
            "top-8": set_attr("topk.topk_config.top_k", 8),
            "swiglu limit": set_attr("experts.moe_runner_config.gemm1_clamp_limit", 8),
            "shuffled shared weight": set_attr(
                "shared_experts.gate_up_proj.quant_method.mxfp8_dense_backend",
                Mxfp8DenseGemmBackend.FLASHINFER_TRTLLM,
            ),
            "bf16 bias": set_attr(
                "e_score_correction_bias", _meta(EXPERTS, dtype=torch.bfloat16)
            ),
            "fewer local experts": set_attr("experts.num_local_experts", 32),
        }
        for name, mutate in cases.items():
            with self.subTest(name):
                moe = _moe()
                moe = mutate(moe) or moe
                self.assertIsNotNone(low_batch_ffn.unsupported_reason(moe))


class TestPlatform(unittest.TestCase):
    def test_sm100_family_without_4over6_only(self):
        """The front reproduces the stock per-token NVFP4 at e4m3 max 448;
        4over6 changes that quantization."""
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "get_device_capability", return_value=(10, 3)),
        ):
            with envs.FLASHINFER_NVFP4_4OVER6.override(True):
                self.assertIn("4OVER6", low_batch_ffn._platform_reason())
            with envs.FLASHINFER_NVFP4_4OVER6.override(False):
                self.assertIsNone(low_batch_ffn._platform_reason())
            with patch.object(
                torch.cuda, "get_device_capability", return_value=(12, 0)
            ):
                self.assertIsNotNone(low_batch_ffn._platform_reason())


class TestPerCall(unittest.TestCase):
    def test_full_decode_graph_of_1_to_32_rows_engages(self):
        for rows in (1, 32):
            self.assertTrue(
                low_batch_ffn._per_call_engages(**{**ENGAGING_CALL, "rows": rows})
            )

    def test_every_other_call_runs_stock(self):
        for name, value in (
            ("defer_finalize", False),  # boundary does not take the handoff
            ("decode_or_verify", False),  # prefill graph runner, extend, mixed
            ("capture_mode", False),  # eager forward
            ("stream_capturing", False),  # warm-up iteration inside capture mode
            ("breakable_graph", True),
            ("can_run_tbo", True),
            ("sp_active", True),
            ("rows", 0),
            ("rows", 33),
        ):
            with self.subTest(f"{name}={value}"):
                self.assertFalse(
                    low_batch_ffn._per_call_engages(**{**ENGAGING_CALL, name: value})
                )

    def test_moved_weights_run_stock(self):
        moe = _moe()
        ffn = object.__new__(low_batch_ffn.LowBatchFfn)
        ffn.bound_ptrs = (-1,) * 11  # a reload moved every tensor
        forward = SimpleNamespace(defer_moe_finalize=True, sp_active=False)
        mode = SimpleNamespace(is_decode=lambda: True, is_target_verify=lambda: False)
        batch = SimpleNamespace(forward_mode=mode, can_run_tbo=False)
        x = torch.zeros(8, HIDDEN, dtype=torch.bfloat16)
        with (
            patch.object(low_batch_ffn, "get_forward", return_value=forward),
            patch.object(low_batch_ffn, "get_is_capture_mode", return_value=True),
            patch.object(
                low_batch_ffn, "is_in_breakable_cuda_graph", return_value=False
            ),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=True),
            patch.object(
                low_batch_ffn.LowBatchFfn,
                "_run",
                side_effect=AssertionError("ran the low-batch kernels"),
            ),
        ):
            self.assertIsNone(ffn.maybe_run(moe, x, batch))


class TestModelHookup(unittest.TestCase):
    def test_switch_off_binds_nothing(self):
        layer = SimpleNamespace(is_layer_sparse=True)  # no .mlp: touching it raises
        with envs.SGLANG_MINIMAX_M3_LOW_BATCH_FFN.override(False):
            low_batch_ffn.bind_low_batch_ffns([layer])

    def test_forward_falls_back_to_stock_when_maybe_run_declines(self):
        none_backend = SimpleNamespace(
            is_megamoe=lambda: False,
            is_deepep=lambda: False,
            is_ascend_ops=lambda: False,
        )
        x = torch.zeros(4, HIDDEN)
        for low_batch, expected in (
            (None, "stock"),
            (SimpleNamespace(maybe_run=lambda moe, x, batch: None), "stock"),
            (SimpleNamespace(maybe_run=lambda moe, x, batch: "low"), "low"),
        ):
            moe = SimpleNamespace(
                low_batch_ffn=low_batch, forward_normal=lambda hidden: "stock"
            )
            with patch.object(
                minimax_m3, "get_moe_a2a_backend", return_value=none_backend
            ):
                self.assertEqual(
                    minimax_m3.MiniMaxM3MoE.forward(moe, x, SimpleNamespace()),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
