import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.runtime_context import get_context
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=12, stage="base-b", runner_config="1-gpu-small")
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


class TestLMHeadFP32(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() and not (
            hasattr(torch, "xpu") and torch.xpu.is_available()
        ):
            raise unittest.SkipTest("needs CUDA GPU or XPU")

    def _make_logprocessor(
        self,
        vocab_size,
        enable_fp32,
        config_enable_fp32=False,
        processor_cls=LogitsProcessor,
    ):
        # LogitsProcessor reads get_exec().features.enable_fp32_lm_head
        # from the published config.
        override = get_context().override_server_args(enable_fp32_lm_head=enable_fp32)
        override.install()
        self.addCleanup(override.restore)
        cfg = SimpleNamespace(
            vocab_size=vocab_size,
            final_logit_softcapping=None,
            enable_lm_head_fp32=config_enable_fp32,
        )
        return processor_cls(cfg, skip_all_gather=True, logit_scale=None)

    @torch.inference_mode()
    def _run_case(
        self,
        hidden_state_dtype,
        enable_fp32,
        weights_dtype,
        expected_a_dtype,
        expected_b_dtype,
        expected_operation,
        config_enable_fp32=False,
    ):
        device = get_device()
        BATCH_SIZE, HIDDEN_SIZE, VOCAB_SIZE = 2, 64, 128
        hidden_state = torch.randn(
            BATCH_SIZE, HIDDEN_SIZE, dtype=hidden_state_dtype, device=device
        )
        head = LMHeadStub(VOCAB_SIZE, HIDDEN_SIZE, dtype=weights_dtype, device=device)
        meta = DummyMeta()
        logprocessor = self._make_logprocessor(
            VOCAB_SIZE, enable_fp32, config_enable_fp32
        )

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
            if not state["called"]:
                state.update(
                    called=True,
                    operation="mm",
                    a=a.dtype,
                    b=b.dtype,
                    out_dtype=kw.get("out_dtype"),
                )
            return original_mm(a, b, *args, **kw)

        def probe_linear(x, w, bias=None):
            if not state["called"]:
                state.update(called=True, ooperationp="linear", a=x.dtype, b=w.dtype)
            return original_linear(x, w, bias)

        with (
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
        expected_out_dtype = torch.float32 if expected_operation == "mm" else None
        self.assertEqual(state["out_dtype"], expected_out_dtype)
        if expected_out_dtype is None and expected_a_dtype in (
            torch.float16,
            torch.bfloat16,
        ):
            # Preserve the ordinary GEMM's rounding; rounding an FP64 result
            # to the input dtype can differ by one output ULP.
            reference = original_matmul(
                hidden_state.to(expected_a_dtype), head.weight.to(expected_b_dtype).T
            )
        else:
            reference = F.linear(
                hidden_state.to(expected_a_dtype).double(),
                head.weight.to(expected_b_dtype).double(),
            )
        torch.testing.assert_close(logits, reference.float(), rtol=1e-5, atol=1e-5)

    def test_flag_true_fp16_activations(self):
        expected_operation = "mm" if torch.cuda.is_available() else "matmul"
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
        expected_operation = "mm" if torch.cuda.is_available() else "matmul"
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
            torch.float16,
            False,
            torch.float16,
            torch.float16,
            torch.float16,
            "matmul",
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

    @torch.inference_mode()
    def test_logits_buffer_preserves_lm_head_precision(self):
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA fp32 GEMM output")
        for dtype in (torch.bfloat16, torch.float16):
            for enable_fp32 in (False, True):
                with self.subTest(dtype=dtype, enable_fp32=enable_fp32):
                    processor = self._make_logprocessor(128, enable_fp32)
                    hidden = torch.randn(2, 64, device=get_device(), dtype=dtype)
                    head = LMHeadStub(128, 64, dtype)
                    buffer = torch.empty(2, 128, device=get_device())
                    meta = DummyMeta()
                    meta.next_token_logits_buffer = buffer
                    reference = processor._get_logits(
                        hidden, head, meta, use_logits_buffer=False
                    )
                    with patch("torch.mm", wraps=torch.mm) as mm:
                        logits = processor._get_logits(hidden, head, meta)
                    torch.testing.assert_close(logits, reference, rtol=0, atol=0)
                    self.assertIs(logits, buffer)
                    if enable_fp32:
                        self.assertIs(mm.call_args.kwargs["out"], buffer)
                    else:
                        mm.assert_not_called()

    @torch.inference_mode()
    def test_logits_buffer_shapes_and_transforms(self):
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA fp32 GEMM output")
        from sglang.srt.models.muse_spark import MuseSpark1xLogitsProcessor

        cases = [
            (128, 2, True, None, None),
            (129, 2, True, None, None),
            (128, 3, True, None, None),
            (128, 2, False, None, None),
            (128, 2, True, 0.5, None),
            (128, 2, True, None, 3.0),
            (129, 2, True, 0.5, 3.0),
        ]
        for processor_cls in (LogitsProcessor, MuseSpark1xLogitsProcessor):
            for dtype in (torch.bfloat16, torch.float16):
                for vocab, rows, use_buffer, scale, softcap in cases:
                    with self.subTest(
                        processor=processor_cls.__name__,
                        dtype=dtype,
                        case=(vocab, rows, use_buffer, scale, softcap),
                    ):
                        processor = self._make_logprocessor(
                            vocab, False, processor_cls=processor_cls
                        )
                        processor.logit_scale = scale
                        processor.final_logit_softcapping = softcap
                        padded_vocab = ((vocab + 127) // 128) * 128
                        head = LMHeadStub(padded_vocab, 64, dtype)
                        hidden = torch.randn(2, 64, device=get_device(), dtype=dtype)
                        buffer = torch.full(
                            (rows, vocab), float("nan"), device=get_device()
                        )
                        meta = DummyMeta()
                        meta.next_token_logits_buffer = buffer
                        logits = processor._get_logits(
                            hidden, head, meta, use_logits_buffer=use_buffer
                        )
                        if processor_cls is MuseSpark1xLogitsProcessor:
                            reference = F.linear(
                                hidden.double(), head.weight.double()
                            ).float()
                        else:
                            reference = torch.matmul(hidden, head.weight.T)
                        reference = reference[:, :vocab]
                        if scale is not None:
                            reference *= scale
                        reference = reference.float()
                        if softcap is not None:
                            reference = softcap * torch.tanh(reference / softcap)
                        torch.testing.assert_close(
                            logits, reference, rtol=1e-5, atol=1e-5
                        )
                        self.assertEqual(buffer.shape, (rows, vocab))
                        if use_buffer and rows == 2:
                            self.assertIs(logits, buffer)
                        else:
                            self.assertTrue(torch.isnan(buffer).all())

    @torch.inference_mode()
    def test_logits_buffer_cuda_graph_replay(self):
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA graphs")
        for enable_fp32 in (False, True):
            with self.subTest(enable_fp32=enable_fp32):
                processor = self._make_logprocessor(128, enable_fp32)
                hidden = torch.randn(2, 64, dtype=torch.bfloat16, device="cuda")
                head = LMHeadStub(128, 64, torch.bfloat16)
                meta = DummyMeta()
                meta.next_token_logits_buffer = torch.empty(2, 128, device="cuda")
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        processor._get_logits(hidden, head, meta)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    logits = processor._get_logits(hidden, head, meta)
                for _ in range(3):
                    hidden.normal_()
                    graph.replay()
                    if enable_fp32:
                        reference = F.linear(
                            hidden.double(), head.weight.double()
                        ).float()
                    else:
                        reference = torch.matmul(hidden, head.weight.T).float()
                    torch.testing.assert_close(logits, reference, rtol=1e-5, atol=1e-5)
                    self.assertIs(logits, meta.next_token_logits_buffer)

    def test_model_config_enables_fp32_without_server_flag(self):
        expected_operation = "mm" if torch.cuda.is_available() else "matmul"
        expected_dtype = (
            torch.float32 if expected_operation == "matmul" else torch.bfloat16
        )
        self._run_case(
            torch.bfloat16,
            False,
            torch.bfloat16,
            expected_dtype,
            expected_dtype,
            expected_operation,
            config_enable_fp32=True,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
