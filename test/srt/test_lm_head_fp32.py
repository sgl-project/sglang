import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.managers.schedule_batch import global_server_args_dict


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

    def _make_lp(self, vocab_size, enable_fp32):
        global_server_args_dict["enable_dp_lm_head"] = False
        global_server_args_dict["enable_lm_head_fp32"] = enable_fp32
        cfg = SimpleNamespace(vocab_size=vocab_size, final_logit_softcapping=None)
        return LogitsProcessor(cfg, skip_all_gather=True, logit_scale=None)

    def _run_case(self, hs_dtype, enable_fp32, w_dtype):
        device = "cuda"
        B, H, V = 2, 64, 128
        hs = torch.randn(B, H, dtype=hs_dtype, device=device)
        head = LMHeadStub(V, H, dtype=w_dtype, device=device)
        meta = DummyMeta()
        lp = self._make_lp(V, enable_fp32)

        orig_mm = torch.matmul
        orig_li = F.linear

        state = {"called": False, "op": None, "a": None, "b": None}

        def probe_matmul(a, b, *args, **kw):
            if not state["called"]:
                state.update(called=True, op="matmul", a=a.dtype, b=b.dtype)
            return orig_mm(a, b, *args, **kw)

        def probe_linear(x, w, bias=None):
            if not state["called"]:
                state.update(called=True, op="linear", a=x.dtype, b=w.dtype)
            return orig_li(x, w, bias)

        with patch("torch.matmul", new=probe_matmul), patch(
            "torch.nn.functional.linear", new=probe_linear
        ):
            logits = lp._get_logits(hs, head, meta)

        self.assertEqual(logits.dtype, torch.float32)
        self.assertEqual(hs.dtype, hs_dtype)
        self.assertTrue(state["called"], "no call lm head matlmul/linear")

        return state["a"], state["b"]

    def test_flag_true_fp16_activations(self):
        a, b = self._run_case(torch.float16, True, torch.float32)
        self.assertEqual(a, torch.float32)
        self.assertEqual(b, torch.float32)

    def test_flag_true_bf16_activations(self):
        a, b = self._run_case(torch.bfloat16, True, torch.float32)
        self.assertEqual(a, torch.float32)
        self.assertEqual(b, torch.float32)

    def test_flag_false_fp16_path(self):
        a, b = self._run_case(torch.float16, False, torch.float16)
        self.assertEqual(a, torch.float16)
        self.assertEqual(b, torch.float16)

    def test_flag_false_bf16_path(self):
        a, b = self._run_case(torch.bfloat16, False, torch.bfloat16)
        self.assertEqual(a, torch.bfloat16)
        self.assertEqual(b, torch.bfloat16)


if __name__ == "__main__":
    unittest.main(verbosity=2)
