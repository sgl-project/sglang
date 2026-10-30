"""SM107 (VR200) routes the GLM-5 TP4 decode BF16 GEMMs to FlashInfer warp
split-K with PDL; every other GPU keeps its existing route."""

import sys
import types
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

import sglang.srt.layers.quantization.unquant as unquant
from sglang.srt.environ import envs
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
from sglang.srt.runtime_context import get_context, override_platform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

# GLM-5 TP4 (hidden 6144) decode GEMMs (m, n, k) under EAGLE 4-1-5,
# where draft M = batch size and verify M = 5 x batch size.
_O_PROJ_BS1_VERIFY = (5, 6144, 4096)
_SHARED_DOWN_BS4_DRAFT = (4, 6144, 512)
_FUSED_QKV_A_BS4_VERIFY = (20, 2624, 6144)
_GLM_KEYS = (_O_PROJ_BS1_VERIFY, _SHARED_DOWN_BS4_DRAFT, _FUSED_QKV_A_BS4_VERIFY)
_FUSED_QKV_A = (2624, 6144)
# Muse Spark TP4 qkvg: on the warp split-K route on every Blackwell GPU already.
_MUSE_SPARK_KEY = (1, 4608, 8192)

_DISPATCH_GLOBALS = (
    "_BF16_GEMM_BACKEND",
    "_cutedsl_bf16_gemm",
    "_use_cutedsl_bf16_gemm",
    "_splitk_tactic",
    "_run_splitk_dense",
    "_direct_default_tactic",
    "_prefer_direct",
    "_run_direct_dense",
    "_warp_splitk_tactic",
    "_run_warp_splitk_dense",
    "_enable_bf16_splitk_gemm",
    "_use_sm107_splitk_tactics",
)


def _module(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


def _fake_kernel_modules(routes):
    """FlashInfer and CuTe DSL entry points that log their route and run on CPU."""

    def run_warp_splitk_dense(x, weight_t, out, pdl, tactic, bias):
        routes.append(("warp_splitk", pdl))
        torch.matmul(x, weight_t, out=out)

    def run_splitk_dense(x, weight_t, bias, out, pdl, tactic):
        routes.append(("splitk", pdl))
        torch.matmul(x, weight_t, out=out)

    def run_direct_dense(x, weight_t, out, pdl, tactic):
        routes.append(("direct", pdl))
        torch.matmul(x, weight_t, out=out)

    def cutedsl_bf16_gemm(x, weight, bias=None):
        routes.append(("tgv", True))
        return F.linear(x, weight, bias)

    kernels = "flashinfer.gemm.kernels"
    return {
        "flashinfer": _module("flashinfer"),
        "flashinfer.gemm": _module("flashinfer.gemm"),
        kernels: _module(kernels),
        f"{kernels}.dense_bf16_gemm_direct": _module(
            f"{kernels}.dense_bf16_gemm_direct",
            default_tactic=lambda m, n, k: None,
            prefer_direct_bf16_gemm_sm100=lambda m, n, k: False,
            run_direct_dense=run_direct_dense,
        ),
        f"{kernels}.dense_bf16_gemm_sm100_splitk": _module(
            f"{kernels}.dense_bf16_gemm_sm100_splitk",
            SplitKTactic=lambda *fields: fields,
            run_splitk_dense=run_splitk_dense,
        ),
        f"{kernels}.dense_bf16_gemm_warp_splitk": _module(
            f"{kernels}.dense_bf16_gemm_warp_splitk",
            WarpSplitKTactic=lambda *fields: fields,
            run_warp_splitk_dense=run_warp_splitk_dense,
        ),
        "sglang.kernels.ops.gemm.cutedsl_bf16_gemm": _module(
            "sglang.kernels.ops.gemm.cutedsl_bf16_gemm",
            cutedsl_bf16_gemm=cutedsl_bf16_gemm,
            use_cutedsl_bf16_gemm=lambda m, n, k: False,
        ),
    }


class TestSm107SplitKDispatch(CustomTestCase):
    def setUp(self):
        self.routes = []
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.dict(sys.modules, _fake_kernel_modules(self.routes)))
        # initialize_bf16_gemm_config rebinds these and grows the warp table.
        for name in _DISPATCH_GLOBALS:
            stack.enter_context(patch.object(unquant, name, getattr(unquant, name)))
        stack.enter_context(patch.dict(unquant._BF16_WARP_SPLITK_TUNED_TACTICS))
        stack.enter_context(
            get_context().override_server_args(bf16_gemm_backend="cutedsl")
        )

    def _initialize(self, device_sm: int, enable_splitk: bool = True):
        with (
            override_platform(is_sm100=True, device_sm=device_sm),
            envs.SGLANG_ENABLE_BF16_SPLITK_GEMM.override(enable_splitk),
        ):
            unquant.initialize_bf16_gemm_config()

    def _route(self, key):
        m, n, k = key
        torch.manual_seed(0)
        x = torch.randn(m, k, dtype=torch.bfloat16)
        weight = torch.randn(n, k, dtype=torch.bfloat16) / k**0.5
        self.routes.clear()
        output = unquant._bf16_gemm_dispatch_impl(x, weight, None)
        torch.testing.assert_close(
            output, F.linear(x, weight), rtol=1.6e-2, atol=1.6e-2
        )
        return self.routes[0] if self.routes else ("cublas", False)

    def test_sm107_routes_glm_decode_gemms_to_warp_splitk_with_pdl(self):
        self._initialize(device_sm=107)
        for key in _GLM_KEYS + (_MUSE_SPARK_KEY,):
            with self.subTest(key=key):
                self.assertEqual(self._route(key), ("warp_splitk", True))

    def test_other_gpus_keep_their_routes(self):
        """The GLM keys are SM107 measurements; B200/B300 and GB200/GB300
        must not pick them up, while their own split-K keys still apply."""
        for device_sm in (100, 103):
            with self.subTest(device_sm=device_sm):
                self._initialize(device_sm=device_sm)
                for key in _GLM_KEYS:
                    self.assertEqual(self._route(key), ("cublas", False))
                self.assertEqual(self._route(_MUSE_SPARK_KEY), ("warp_splitk", True))

    def test_fused_qkv_a_leaves_fused_a_only_for_a_split_k_route(self):
        """Leaving the fused_a JIT for anything but warp split-K (cuBLAS
        when split-K is off, unmeasured M, other GPUs) is a slowdown."""
        weight = torch.empty(_FUSED_QKV_A, device="meta")
        cases = (
            (107, True, 1, True),
            (107, True, 4, True),
            (107, True, 5, True),
            (107, True, 2, False),
            (107, True, 16, False),
            (107, False, 5, False),
            (100, True, 5, False),
            (103, True, 1, False),
        )
        for device_sm, enable_splitk, m, expected in cases:
            with self.subTest(device_sm=device_sm, enable_splitk=enable_splitk, m=m):
                self._initialize(device_sm=device_sm, enable_splitk=enable_splitk)
                x = torch.empty(m, _FUSED_QKV_A[1], device="meta")
                self.assertEqual(
                    unquant.prefer_bf16_splitk_over_fused_a(x=x, weight=weight),
                    expected,
                )
                if expected:
                    self.assertEqual(
                        self._route((m, *_FUSED_QKV_A)), ("warp_splitk", True)
                    )


class _RecordingLinear:
    def __init__(self, routes):
        self.weight = torch.empty(_FUSED_QKV_A, device="meta")
        self.routes = routes

    def __call__(self, x):
        self.routes.append("linear")
        return (x.new_zeros(x.shape[0], _FUSED_QKV_A[0]),)


class TestPrepareQkvLatentFusedARoute(CustomTestCase):
    def _route(self, m: int, sm107_tactics: bool):
        routes = []

        def fused_a(layer, hidden_states, *, backend):
            routes.append("fused_a")
            return hidden_states.new_zeros(hidden_states.shape[0], _FUSED_QKV_A[0])

        attn = SimpleNamespace(
            q_lora_rank=2048,
            _use_min_latency_fused_a_gemm=True,
            fused_qkv_a_proj_with_mqa=_RecordingLinear(routes),
            fused_a_gemm_backend="auto",
        )
        with (
            patch.object(unquant, "_use_sm107_splitk_tactics", sm107_tactics),
            patch("sglang.srt.models.deepseek_v2.linear_with_fused_a_gemm", fused_a),
        ):
            latent = DeepseekV2AttentionMLA.prepare_qkv_latent(
                attn, torch.zeros(m, _FUSED_QKV_A[1]), None
            )
        self.assertEqual(latent.shape, (m, _FUSED_QKV_A[0]))
        return routes

    def test_fused_qkv_a_takes_the_linear_route_for_sm107_split_k_keys(self):
        self.assertEqual(self._route(m=5, sm107_tactics=True), ["linear"])
        self.assertEqual(self._route(m=2, sm107_tactics=True), ["fused_a"])
        self.assertEqual(self._route(m=5, sm107_tactics=False), ["fused_a"])


if __name__ == "__main__":
    unittest.main()
