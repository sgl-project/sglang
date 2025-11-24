import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.server_args import (
    ServerArgs,
    get_global_server_args,
    set_global_server_args_for_scheduler,
)


class LMHeadStub(nn.Module):
    def __init__(self, vocab, hidden, dtype, device="cuda"):
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
        if not torch.cuda.is_available():
            raise unittest.SkipTest("needs CUDA GPU")

    def _make_logprocessor(self, vocab_size, enable_fp32):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        get_global_server_args().enable_dp_lm_head = False
        get_global_server_args().enable_fp32_lm_head = enable_fp32
        cfg = SimpleNamespace(vocab_size=vocab_size, final_logit_softcapping=None)
        return LogitsProcessor(cfg, skip_all_gather=True, logit_scale=None)

    def _run_case(
        self,
        hidden_state_dtype,
        enable_fp32,
        weights_dtype,
        expected_a_dtype,
        expected_b_dtype,
    ):
        device = "cuda"
        BATCH_SIZE, HIDDEN_SIZE, VOCAB_SIZE = 2, 64, 128
        hidden_state = torch.randn(
            BATCH_SIZE, HIDDEN_SIZE, dtype=hidden_state_dtype, device=device
        )
        head = LMHeadStub(VOCAB_SIZE, HIDDEN_SIZE, dtype=weights_dtype, device=device)
        meta = DummyMeta()
        logprocessor = self._make_logprocessor(VOCAB_SIZE, enable_fp32)

        original_matmul = torch.matmul
        original_linear = F.linear

        state = {
            "called": False,  # Whether a matmul/linear call has been intercepted yet
            "operation": None,  # Which operation was captured ("matmul" or "linear")
            "a": None,  # The dtype of the first input tensor to the operation
            "b": None,  # The dtype of the second input tensor to the operation
        }

        def probe_matmul(a, b, *args, **kw):
            if not state["called"]:
                state.update(called=True, operation="matmul", a=a.dtype, b=b.dtype)
            return original_matmul(a, b, *args, **kw)

        def probe_linear(x, w, bias=None):
            if not state["called"]:
                state.update(called=True, ooperationp="linear", a=x.dtype, b=w.dtype)
            return original_linear(x, w, bias)

        with patch("torch.matmul", new=probe_matmul), patch(
            "torch.nn.functional.linear", new=probe_linear
        ):
            logits = logprocessor._get_logits(hidden_state, head, meta)
        self.assertEqual(hidden_state.dtype, hidden_state_dtype)
        self.assertTrue(state["called"], "no call lm head matlmul/linear")
        self.assertEqual(state["a"], expected_a_dtype)
        self.assertEqual(state["b"], expected_b_dtype)

    def test_flag_true_fp16_activations(self):
        self._run_case(torch.float16, True, torch.float16, torch.float32, torch.float32)

    def test_flag_true_bf16_activations(self):
        self._run_case(
            torch.bfloat16, True, torch.bfloat16, torch.float32, torch.float32
        )

    def test_flag_false_fp16_path(self):
        self._run_case(
            torch.float16, False, torch.float16, torch.float16, torch.float16
        )

    def test_flag_false_bf16_path(self):
        self._run_case(
            torch.bfloat16, False, torch.bfloat16, torch.bfloat16, torch.bfloat16
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
