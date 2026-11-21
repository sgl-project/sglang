"""CPU unit tests for Qwen3.5 blockwise FP8 output quantization layout & gating."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from sglang.srt.layers.quantization.fp8_utils import (
    deepgemm_w8a8_block_fp8_linear_with_fallback,
    get_block_fp8_tuple_input_layout,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _FakeCudaTensor:
    """Minimal tensor stub simulating a contiguous CUDA tensor for CPU unit tests."""

    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.dtype = dtype
        self.is_cuda = True
        self.device = torch.device("cuda", 0)

    def is_contiguous(self):
        return True


def _make_linear(
    *,
    n: int = 2048,
    k: int = 4096,
    block_quant: bool = True,
    use_marlin: bool = False,
    use_mxfp8: bool = False,
    block_fp8_as_mxfp8: bool = False,
    block_fp8_mxfp8_ready: bool = False,
    use_npu_block_fp8: bool = False,
    weight_block_size: list[int] | None = None,
    scale_fmt: str | None = None,
    input_is_parallel: bool = True,
    has_base_layer: bool = False,
    weight_scale_dtype: torch.dtype = torch.float32,
):
    qm = object.__new__(Fp8LinearMethod)
    qm.block_quant = block_quant
    qm.use_marlin = use_marlin
    qm.use_mxfp8 = use_mxfp8
    qm.block_fp8_as_mxfp8 = block_fp8_as_mxfp8
    qm.use_npu_block_fp8 = use_npu_block_fp8
    qm.weight_block_size = (
        [128, 128] if weight_block_size is None else weight_block_size
    )
    qm.w8a8_block_fp8_linear = deepgemm_w8a8_block_fp8_linear_with_fallback
    qm.quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128]
    )
    if scale_fmt is not None:
        qm.quant_config.scale_fmt = scale_fmt
    ns = SimpleNamespace(
        quant_method=qm,
        weight=_FakeCudaTensor((n, k), torch.float8_e4m3fn),
        weight_scale_inv=_FakeCudaTensor(
            ((n + 127) // 128, k // 128), weight_scale_dtype
        ),
        input_is_parallel=input_is_parallel,
        block_fp8_mxfp8_ready=block_fp8_mxfp8_ready,
    )
    if has_base_layer:
        ns.base_layer = object()
    return ns


class TestBlockFp8TupleInputLayout(CustomTestCase):
    def _eval(
        self,
        linear,
        m=8,
        k=4096,
        dtype=torch.bfloat16,
        *,
        dg_ue8m0=False,
        dg_supported=True,
        gemv=False,
        bi=False,
        det=False,
        rl_target=None,
        lora=False,
    ):
        exec_ctx = SimpleNamespace(
            deterministic=SimpleNamespace(
                enable_deterministic_inference=det,
                rl_on_policy_target=rl_target,
            )
        )
        lora_ctx = SimpleNamespace(enable_lora=lora)
        with (
            patch(
                "sglang.srt.batch_invariant_ops.is_batch_invariant_mode_enabled",
                return_value=bi,
            ),
            patch("sglang.srt.runtime_context.get_exec", return_value=exec_ctx),
            patch("sglang.srt.runtime_context.get_lora", return_value=lora_ctx),
            patch(
                "sglang.srt.layers.quantization.fp8_utils.deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0",
                dg_ue8m0,
            ),
            patch(
                "sglang.srt.layers.quantization.fp8_utils._deepgemm_block_fp8_supported",
                return_value=dg_supported,
            ),
            patch(
                "sglang.srt.layers.quantization.fp8_utils._fp8_block_gemv_enabled",
                return_value=gemv,
            ),
            patch(
                "sglang.srt.layers.quantization.fp8_utils.use_hopper_fp8_block_gemv",
                return_value=gemv,
            ),
        ):
            return get_block_fp8_tuple_input_layout(linear, m, k, dtype)

    def test_layout_selection(self):
        linear = _make_linear(n=2048, k=4096)
        # SM90 Hopper block GEMV -> row-major fp32
        self.assertEqual(
            self._eval(linear, dg_ue8m0=False, dg_supported=True, gemv=True),
            (False, False),
        )
        # SM90 DeepGEMM -> col-major TMA fp32
        self.assertEqual(
            self._eval(linear, dg_ue8m0=False, dg_supported=True, gemv=False),
            (False, True),
        )
        # Triton fallback (non-DeepGEMM shape) -> row-major fp32
        self.assertEqual(
            self._eval(linear, dg_ue8m0=False, dg_supported=False, gemv=False),
            (False, False),
        )
        # SM100 UE8M0 -> col-major TMA packed int32 UE8M0 when (k // 128) % 4 == 0
        self.assertEqual(
            self._eval(linear, dg_ue8m0=True, dg_supported=True, gemv=False),
            (True, True),
        )
        # SM100 UE8M0 with non-4-group-aligned K (e.g. 768 = 6 groups) -> None
        linear_768 = _make_linear(n=2048, k=768)
        self.assertIsNone(
            self._eval(linear_768, k=768, dg_ue8m0=True, dg_supported=True, gemv=False)
        )

    def test_rejection_gates(self):
        cases = [
            ("m_zero", _make_linear(), {"m": 0}),
            ("k_unaligned", _make_linear(k=200), {"k": 200}),
            ("fp16", _make_linear(), {"dtype": torch.float16}),
            ("batch_invariant", _make_linear(), {"bi": True}),
            ("deterministic", _make_linear(), {"det": True}),
            ("rl_on_policy", _make_linear(), {"rl_target": "ref"}),
            ("lora_enabled", _make_linear(), {"lora": True}),
            ("base_layer", _make_linear(has_base_layer=True), {}),
            ("not_input_parallel", _make_linear(input_is_parallel=False), {}),
            ("marlin", _make_linear(use_marlin=True), {}),
            ("mxfp8", _make_linear(use_mxfp8=True), {}),
            (
                "block_fp8_mxfp8_ready",
                _make_linear(block_fp8_as_mxfp8=True, block_fp8_mxfp8_ready=True),
                {},
            ),
            ("block32", _make_linear(weight_block_size=[32, 32]), {}),
            ("sm90_ue8m0_ckpt", _make_linear(scale_fmt="ue8m0"), {"dg_ue8m0": False}),
        ]
        for desc, linear, kw in cases:
            with self.subTest(case=desc):
                self.assertIsNone(self._eval(linear, **kw))


if __name__ == "__main__":
    unittest.main()
