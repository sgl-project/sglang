import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.layers import logits_processor
from sglang.srt.layers.logits_processor import (
    LogitsProcessor,
    _supports_mm_fp32_out_dtype,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=9, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=15, suite="stage-b-test-1-gpu-small-amd")


class LMHeadStub(nn.Module):
    def __init__(self, vocab, hidden, dtype, device=get_device()):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(vocab, hidden, dtype=dtype, device=device)
        )


class DummyMeta:
    gathered_buffer = None
    next_token_logits_buffer = None

    def compute_dp_attention_metadata(self): ...


class TestLMHeadFP32(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() and not (
            hasattr(torch, "xpu") and torch.xpu.is_available()
        ):
            raise unittest.SkipTest("needs CUDA GPU or XPU")

    @staticmethod
    def _expected_operation(dtype):
        # The fused mm(out_dtype=fp32) path is only taken where torch
        # implements it (not on pre-2.9.0 ROCm builds, not for BF16 below
        # sm80), so the expectation has to be resolved the same way the
        # kernel selects it.
        supported = torch.cuda.is_available() and _supports_mm_fp32_out_dtype(
            torch.device(get_device()).type, dtype
        )
        return "mm" if supported else "matmul"

    def _make_logprocessor(self, vocab_size, enable_fp32):
        # LogitsProcessor reads get_exec().features.enable_fp32_lm_head
        # from the published config.
        override = get_context().override_server_args(enable_fp32_lm_head=enable_fp32)
        override.install()
        self.addCleanup(override.restore)
        cfg = SimpleNamespace(vocab_size=vocab_size, final_logit_softcapping=None)
        return LogitsProcessor(cfg, skip_all_gather=True, logit_scale=None)

    def _run_case(
        self,
        hidden_state_dtype,
        enable_fp32,
        weights_dtype,
        expected_a_dtype,
        expected_b_dtype,
        expected_operation,
        force_support=None,
    ):
        # force_support pins what the capability check reports, so both
        # branches of the selection are covered on every runner instead of
        # only the one this host happens to support. The intercepted op is
        # then emulated rather than executed, since a host that reports no
        # support would raise on the real call.
        device = get_device()
        BATCH_SIZE, HIDDEN_SIZE, VOCAB_SIZE = 2, 64, 128
        hidden_state = torch.randn(
            BATCH_SIZE, HIDDEN_SIZE, dtype=hidden_state_dtype, device=device
        )
        head = LMHeadStub(VOCAB_SIZE, HIDDEN_SIZE, dtype=weights_dtype, device=device)
        meta = DummyMeta()
        logprocessor = self._make_logprocessor(VOCAB_SIZE, enable_fp32)

        original_matmul = torch.matmul
        original_mm = torch.mm
        original_linear = F.linear

        state = {
            "called": False,  # Whether a matmul/linear call has been intercepted yet
            "operation": None,  # Which operation was captured ("matmul" or "linear")
            "a": None,  # The dtype of the first input tensor to the operation
            "b": None,  # The dtype of the second input tensor to the operation
            "out_dtype": None,
        }

        def probe_matmul(a, b, *args, **kw):
            if not state["called"]:
                state.update(
                    called=True,
                    operation="matmul",
                    a=a.dtype,
                    b=b.dtype,
                    out_dtype=kw.get("out_dtype"),
                )
            return original_matmul(a, b, *args, **kw)

        def probe_mm(a, b, *args, **kw):
            first = not state["called"]
            if first:
                state.update(
                    called=True,
                    operation="mm",
                    a=a.dtype,
                    b=b.dtype,
                    out_dtype=kw.get("out_dtype"),
                )
            if first and force_support is not None:
                # Do not issue the real GEMM: this branch is under test
                # precisely on hosts that cannot run it.
                return torch.zeros(
                    a.shape[0], b.shape[1], dtype=kw["out_dtype"], device=a.device
                )
            return original_mm(a, b, *args, **kw)

        def probe_linear(x, w, bias=None):
            if not state["called"]:
                state.update(called=True, ooperationp="linear", a=x.dtype, b=w.dtype)
            return original_linear(x, w, bias)

        support_patch = (
            patch(
                "sglang.srt.layers.logits_processor._supports_mm_fp32_out_dtype",
                new=lambda *a, **kw: force_support,
            )
            if force_support is not None
            else nullcontext()
        )

        with (
            support_patch,
            patch("torch.matmul", new=probe_matmul),
            patch("torch.mm", new=probe_mm),
            patch("torch.nn.functional.linear", new=probe_linear),
        ):
            logits = logprocessor._get_logits(hidden_state, head, meta)
        self.assertEqual(hidden_state.dtype, hidden_state_dtype)
        self.assertTrue(state["called"], "no call lm head matlmul/linear")
        self.assertEqual(state["operation"], expected_operation)
        self.assertEqual(state["a"], expected_a_dtype)
        self.assertEqual(state["b"], expected_b_dtype)
        self.assertEqual(
            state["out_dtype"],
            torch.float32 if expected_operation == "mm" else None,
        )

    def test_flag_true_fp16_activations(self):
        expected_operation = self._expected_operation(torch.float16)
        expected_dtype = (
            torch.float32 if expected_operation == "matmul" else torch.float16
        )
        self._run_case(
            torch.float16,
            True,
            torch.float16,
            expected_dtype,
            expected_dtype,
            expected_operation,
        )

    def test_flag_true_bf16_activations(self):
        expected_operation = self._expected_operation(torch.bfloat16)
        expected_dtype = (
            torch.float32 if expected_operation == "matmul" else torch.bfloat16
        )
        self._run_case(
            torch.bfloat16,
            True,
            torch.bfloat16,
            expected_dtype,
            expected_dtype,
            expected_operation,
        )

    def test_probe_supported_selects_mm(self):
        # Runs the same on every runner, including hosts whose BLAS backend
        # cannot execute mm(out_dtype=fp32).
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self._run_case(dtype, True, dtype, dtype, dtype, "mm", True)

    def test_probe_unsupported_falls_back_to_explicit_fp32_matmul(self):
        # The regression guard: an unsupported backend must take the explicit
        # FP32 cast instead of raising. Covered on NVIDIA runners too.
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self._run_case(
                    dtype, True, dtype, torch.float32, torch.float32, "matmul", False
                )

    def test_flag_true_fp32_falls_back_to_explicit_fp32_matmul(self):
        self._run_case(
            torch.float32,
            True,
            torch.float32,
            torch.float32,
            torch.float32,
            "matmul",
        )

    def test_flag_false_fp16_path(self):
        self._run_case(
            torch.float16, False, torch.float16, torch.float16, torch.float16, "matmul"
        )

    def test_flag_false_bf16_path(self):
        self._run_case(
            torch.bfloat16,
            False,
            torch.bfloat16,
            torch.bfloat16,
            torch.bfloat16,
            "matmul",
        )


class TestMMFP32OutDtypeGate(unittest.TestCase):
    """Pure-metadata gate, so these run on every runner including CPU-only."""

    def setUp(self):
        _supports_mm_fp32_out_dtype.cache_clear()
        self.addCleanup(_supports_mm_fp32_out_dtype.cache_clear)

    def test_torch_version_compare_is_prerelease_aware(self):
        # 2.9.0a0+git7bcbafe is the rocm700 CI image's torch: an alpha that
        # predates pytorch#161540 and must NOT satisfy >= 2.9.0. A .release
        # tuple comparison would read it as (2, 9, 0) and wrongly pass.
        cases = {
            "2.8.0": False,
            "2.9.0a0+git7bcbafe": False,
            "2.9.0": True,
            "2.9.1": True,
            "2.9.1+rocm7.2.0.git7e1940d4": True,
            "2.10.0a0+gitdeadbee": True,
            "not-a-version": False,
        }
        for version, expected in cases.items():
            with self.subTest(version=version):
                with patch.object(torch, "__version__", version):
                    self.assertEqual(
                        logits_processor._torch_at_least("2.9.0"), expected
                    )

    def test_rocm_gate_follows_torch_version(self):
        for has_fix in (True, False):
            with self.subTest(has_fix=has_fix):
                _supports_mm_fp32_out_dtype.cache_clear()
                with (
                    patch.object(logits_processor, "_is_hip", True),
                    patch.object(
                        logits_processor, "_TORCH_HAS_ROCM_MM_FP32_OUT", has_fix
                    ),
                ):
                    self.assertEqual(
                        _supports_mm_fp32_out_dtype("cuda", torch.bfloat16), has_fix
                    )

    def test_cuda_bf16_requires_sm80(self):
        for capability, expected in (((7, 5), False), ((8, 0), True), ((9, 0), True)):
            with self.subTest(capability=capability):
                _supports_mm_fp32_out_dtype.cache_clear()
                with (
                    patch.object(logits_processor, "_is_hip", False),
                    patch.object(
                        torch.cuda, "get_device_capability", lambda: capability
                    ),
                ):
                    self.assertEqual(
                        _supports_mm_fp32_out_dtype("cuda", torch.bfloat16), expected
                    )
                    # FP16 carries no compute-capability restriction.
                    _supports_mm_fp32_out_dtype.cache_clear()
                    self.assertTrue(_supports_mm_fp32_out_dtype("cuda", torch.float16))

    def test_non_cuda_device_never_selects_mm(self):
        self.assertFalse(_supports_mm_fp32_out_dtype("cpu", torch.bfloat16))


if __name__ == "__main__":
    unittest.main(verbosity=2)
