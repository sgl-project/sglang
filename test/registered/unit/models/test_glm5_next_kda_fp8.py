import sys
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.kernels.ops.gemm import _prefer_torch_rowwise_fp8
from sglang.kernels.spec import KernelBackend
from sglang.srt.environ import envs
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.models import glm5_next
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

INPUT_MODULES = ("qkv_proj", "f_a_proj", "g_a_proj")
ALL_MODULES = ",".join(INPUT_MODULES + ("o_proj",))


def fake_fp8_process(self, layer):
    """Stand-in for the FP8 quantization: keep the values, store [K, N] as the FP8 path does."""
    layer.weight = torch.nn.Parameter(layer.weight.data.t(), requires_grad=False)
    layer.weight_scale = torch.ones(layer.weight.shape[1], 1)


def fake_fp8_apply(self, layer, x, bias=None):
    """Stand-in for the generic FP8 path; doubled so a test can tell it ran."""
    return 2 * F.linear(x, layer.weight.t(), bias)


def fake_per_token_quant(x):
    """Stand-in for the per-token FP8 quantization: values kept, unit scales."""
    return x, torch.ones(x.shape[0], 1)


def fake_gemm(factor):
    """Stand-in for one FP8 GEMM kernel; the factor tells a test which kernel ran."""

    def gemm(x_q, weight, x_scale, weight_scale, out_dtype, bias=None, **kwargs):
        return factor * (x_q @ weight).to(out_dtype)

    return gemm


GENERIC, TRITON, AOT, SHARED = 2, 3, 5, 7
TILE = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "num_warps": 4,
    "num_stages": 3,
}


def kda_fp8_modules(value, fp8_gemm=True):
    with (
        envs.SGLANG_OPT_GLM53_KDA_PTPC_MODULES.override(value),
        patch.object(glm5_next, "cutlass_fp8_supported", return_value=fp8_gemm),
    ):
        return glm5_next._kda_fp8_modules()


class TestKdaFp8Modules(CustomTestCase):
    def test_input_projections_switch_together(self):
        # The fused input projection is one GEMM, so its parts cannot differ in dtype.
        self.assertEqual(kda_fp8_modules(ALL_MODULES), set(INPUT_MODULES) | {"o_proj"})
        self.assertEqual(kda_fp8_modules("o_proj"), {"o_proj"})
        for partial in ("qkv_proj", "qkv_proj,f_a_proj", "g_a_proj,o_proj"):
            with (
                self.subTest(partial=partial),
                self.assertRaisesRegex(ValueError, "together"),
            ):
                kda_fp8_modules(partial)

    def test_rejects_unknown_module_and_missing_fp8_gemm(self):
        with self.assertRaisesRegex(ValueError, "Unknown"):
            kda_fp8_modules("in_proj")
        with self.assertRaisesRegex(ValueError, "SM89"):
            kda_fp8_modules("o_proj", fp8_gemm=False)
        with (
            envs.SGLANG_FORCE_FP8_MARLIN.override(True),
            self.assertRaisesRegex(ValueError, "SGLANG_FORCE_FP8_MARLIN"),
        ):
            kda_fp8_modules("o_proj")
        self.assertEqual(kda_fp8_modules("", fp8_gemm=False), frozenset())


@patch.object(Fp8LinearMethod, "apply", fake_fp8_apply)
@patch.object(Fp8LinearMethod, "process_weights_after_loading", fake_fp8_process)
@patch.object(glm5_next, "sglang_per_token_quant_fp8", fake_per_token_quant)
@patch.object(glm5_next, "triton_scaled_mm", fake_gemm(TRITON))
@patch.object(glm5_next, "fp8_scaled_mm", fake_gemm(SHARED))
@patch.object(glm5_next, "get_w8a8_channelwise_fp8_config", return_value=None)
@patch.dict(sys.modules, {"sgl_kernel": SimpleNamespace(fp8_scaled_mm=fake_gemm(AOT))})
class TestKdaFp8LinearMethod(CustomTestCase):
    def process(self, n, k, input_scale=None, marlin=False):
        method = glm5_next._KdaFp8Config(is_checkpoint_fp8_serialized=False)
        method = method.get_quant_method(None, "")
        method.use_marlin = marlin
        layer = torch.nn.Module()
        method.create_weights(layer, k, [n], k, n, torch.bfloat16)
        weight = torch.randn(n, k, dtype=torch.bfloat16)
        layer.weight.data.copy_(weight)
        layer.input_scale = input_scale
        method.process_weights_after_loading(layer)
        return method, layer, weight

    def test_pads_n_to_16_and_keeps_output_width(self, _):
        # TP8's fused input projection has N = 3336 rows per rank.
        method, layer, weight = self.process(3336, 16)
        self.assertEqual(tuple(layer.weight.shape), (16, 3344))
        self.assertTrue(torch.equal(layer.weight[:, 3336:], torch.zeros(16, 8)))
        for m in (1, 1024):
            with self.subTest(m=m):
                x = torch.randn(m, 16, dtype=torch.bfloat16)
                y = method.apply(layer, x)
                self.assertEqual(tuple(y.shape), (m, 3336))
                torch.testing.assert_close(y, AOT * F.linear(x, weight))

    def test_kernel_choice_matches_apply_fp8_linear(self, get_tile):
        # fp8_scaled_mm takes the AOT CUTLASS kernel below 8192 rows unless a backend
        # is forced, and a tuned tile wins when enabled; any other apply_fp8_linear
        # path (K % 16 != 0 included) keeps the generic code.
        for m, kernel, case in (
            (8191, AOT, {}),
            (8192, SHARED, {}),
            (16, SHARED, {"forced": KernelBackend.TORCH}),
            (16, SHARED, {"oot": True}),
            (384, TRITON, {"tile": TILE}),
            (384, AOT, {"tile": TILE, "tune": False}),
            (16, GENERIC, {"bias": torch.ones(48, dtype=torch.bfloat16)}),
            (16, GENERIC, {"input_scale": torch.ones(1)}),
            (16, GENERIC, {"marlin": True}),
            (16, GENERIC, {"triton_w8a8": True}),
            (16, GENERIC, {"k": 24}),
        ):
            with (
                self.subTest(m=m, **{k: v is not None for k, v in case.items()}),
                envs.SGLANG_ENABLE_FP8_GEMM_CONFIG_TUNE.override(
                    case.get("tune", True)
                ),
                patch.object(
                    glm5_next,
                    "use_triton_w8a8_fp8_kernel",
                    case.get("triton_w8a8", False),
                ),
                patch.object(
                    glm5_next, "get_fused_op_backend", return_value=case.get("forced")
                ),
                patch.object(
                    glm5_next.current_platform,
                    "is_out_of_tree",
                    return_value=case.get("oot", False),
                ),
            ):
                get_tile.reset_mock()
                get_tile.return_value = case.get("tile")
                k = case.get("k", 16)
                method, layer, weight = self.process(
                    48, k, case.get("input_scale"), case.get("marlin", False)
                )
                x = torch.randn(m, k, dtype=torch.bfloat16)
                bias = case.get("bias")
                torch.testing.assert_close(
                    method.apply(layer, x, bias), kernel * F.linear(x, weight, bias)
                )
                if "tile" in case and case.get("tune", True):
                    get_tile.assert_called_once_with(N=48, K=16, M=m)

    def test_rejects_a_second_post_load_pass(self, _):
        # A weight update that reruns post-load processing would quantize FP8 again.
        method, layer, _ = self.process(48, 16)
        with self.assertRaisesRegex(RuntimeError, "quantized at load"):
            method.process_weights_after_loading(layer)

    def test_bf16_weight_lives_in_its_own_pool_until_processed(self, _):
        entered = []

        @contextmanager
        def use_mem_pool(pool):
            entered.append(pool)
            yield

        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "MemPool", object),
            patch.object(torch.cuda, "use_mem_pool", use_mem_pool),
            patch.object(torch.cuda, "empty_cache"),
        ):
            method, _, _ = self.process(40, 16)
        # Weight creation and the padding copy both allocate in the pool, which is
        # dropped so that empty_cache() can release it.
        self.assertEqual(len(entered), 2)
        self.assertIs(entered[0], entered[1])
        self.assertIsNone(method._bf16_pool)


class TestFp8ScaledMMRowThreshold(CustomTestCase):
    def test_aot_kernel_below_8192_rows(self):
        # _apply_dynamic_per_token calls the AOT kernel directly below 8192 rows, where
        # fp8_scaled_mm never prefers torch._scaled_mm.
        cuda = torch.device("cuda", 0)

        def tensor(shape, stride, dtype):
            # The predicate reads only metadata, so CUDA tensors are described, not allocated.
            return SimpleNamespace(
                shape=shape,
                ndim=len(shape),
                stride=lambda dim: stride[dim],
                dtype=dtype,
                device=cuda,
                is_contiguous=lambda: stride[-1] == 1,
                numel=lambda: shape[0] * shape[1],
            )

        def prefers_torch(m, n=24896, k=4096):
            return _prefer_torch_rowwise_fp8(
                tensor((m, k), (k, 1), torch.float8_e4m3fn),
                tensor((k, n), (1, k), torch.float8_e4m3fn),
                tensor((m, 1), (1, 1), torch.float32),
                tensor((n, 1), (1, 1), torch.float32),
                torch.bfloat16,
                None,
            )

        self.assertTrue(prefers_torch(8192))
        for n, k in ((24896, 4096), (6416, 4096), (3344, 4096), (4096, 8192)):
            with self.subTest(n=n, k=k):
                self.assertFalse(prefers_torch(8191, n, k))


class TestKdaFp8Wiring(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args(
            device="cpu", enable_lora=False, lora_paths=None
        )
        override.install()
        self.addCleanup(override.restore)
        parallel = get_parallel().override(
            tp_size=1,
            tp_rank=0,
            attn_tp_size=1,
            attn_tp_rank=0,
            attn_dp_size=1,
            attn_dp_rank=0,
            attn_cp_size=1,
            attn_cp_rank=0,
            moe_tp_size=1,
        )
        parallel.__enter__()
        self.addCleanup(parallel.__exit__, None, None, None)

    def build(self, modules, quant_config=None):
        with (
            envs.SGLANG_OPT_GLM53_KDA_PTPC_MODULES.override(modules),
            patch.object(glm5_next, "cutlass_fp8_supported", return_value=True),
        ):
            return glm5_next.Glm5NextLinearAttention(
                layer_idx=0,
                hidden_size=16,
                config=SimpleNamespace(
                    linear_attn_config={
                        "head_dim": 8,
                        "num_heads": 4,
                        "short_conv_kernel_size": 4,
                    }
                ),
                quant_config=quant_config,
                prefix="model.layers.0.self_attn",
            )

    def test_switch_selects_the_fp8_method_per_projection(self):
        for modules, in_fp8, o_fp8 in (
            ("", False, False),
            ("o_proj", False, True),
            (ALL_MODULES, True, True),
        ):
            with self.subTest(modules=modules):
                attention = self.build(modules)
                for layer, fp8 in (
                    (attention.fused_qkvbfg_a_proj, in_fp8),
                    (attention.o_proj, o_fp8),
                ):
                    expected = (
                        glm5_next._KdaFp8LinearMethod
                        if fp8
                        else UnquantizedLinearMethod
                    )
                    self.assertIs(type(layer.quant_method), expected)

    def test_unfused_input_projection_is_rejected(self):
        # LoRA turns the projection fusion off.
        override = get_context().override_server_args(enable_lora=True)
        override.install()
        self.addCleanup(override.restore)
        self.assertIsInstance(self.build("o_proj").o_proj.quant_method, Fp8LinearMethod)
        with self.assertRaisesRegex(ValueError, "fused"):
            self.build(ALL_MODULES)


if __name__ == "__main__":
    unittest.main()
