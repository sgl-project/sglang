"""Full-checkpoint M4 head through the real weight-load and logits hooks.

Set SGLANG_TEST_NEMOTRON_MODEL to the local original NVFP4 checkpoint.
"""

import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors import safe_open
from sglang.kernels.ops.gemm.sm120_lossless_bf16_head import (
    try_lossless_bf16_head,
)
from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization import unquant as uq
from sglang.srt.layers.quantization.modelopt_quant import ModelOptMixedPrecisionConfig
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.models.nemotron_h import NemotronHForCausalLM
from sglang.test.test_utils import CustomTestCase


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.cuda.get_device_capability() == (12, 0)
    and os.environ.get("SGLANG_TEST_NEMOTRON_MODEL"),
    "requires SM120 and the original local Nemotron checkpoint",
)
class TestLosslessBf16HeadM4(CustomTestCase):
    def test_real_logits_and_live_graph(self):
        torch.set_num_threads(2)
        torch.manual_seed(67)
        root = Path(os.environ["SGLANG_TEST_NEMOTRON_MODEL"])
        index = json.loads((root / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        with safe_open(
            root / index["lm_head.weight"], framework="pt", device="cpu"
        ) as f:
            weight = f.get_tensor("lm_head.weight")
        self.assertEqual(weight.shape, (131072, 4096))
        self.assertEqual(weight.dtype, torch.bfloat16)
        head = ParallelLMHead.__new__(ParallelLMHead)
        torch.nn.Module.__init__(head)
        head.weight = torch.nn.Parameter(weight.cuda(), requires_grad=False)
        cfg = json.loads((root / "hf_quant_config.json").read_text())
        cfg["packed_modules_mapping"] = NemotronHForCausalLM.packed_modules_mapping
        config = ModelOptMixedPrecisionConfig.from_config(cfg)
        head.quant_method = config.get_quant_method(head, prefix="lm_head")
        self.assertIsInstance(head.quant_method, uq.UnquantizedLinearMethod)
        with (
            envs.SGLANG_SM120_LOSSLESS_BF16_HEAD.override(True),
            envs.SGLANG_SM120_LOSSLESS_BF16_HEAD_M4.override(True),
            patch.object(uq, "_BF16_GEMM_BACKEND", uq.Bf16GemmBackend.GEMV),
            patch("torch.matmul", side_effect=AssertionError("M4 head used fallback")),
        ):
            head.quant_method.process_weights_after_loading(head)
            self.assertTrue(
                torch.equal(
                    head.weight.cpu().view(torch.int16), weight.view(torch.int16)
                )
            )
            self.assertEqual(list(head.state_dict()), ["weight"])
            processor = SimpleNamespace(
                use_fp32_lm_head=False, rl_on_policy_target=None
            )
            x = torch.randn((4, 4096), dtype=torch.bfloat16, device="cuda")
            actual = LogitsProcessor._compute_lm_head(processor, x, head)
            expected = torch.mm(x, head.weight.T)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = LogitsProcessor._compute_lm_head(processor, x, head)
            for case in range(16):
                with self.subTest(case=case):
                    x.normal_()
                    if case == 0:
                        x.zero_()
                    elif case == 1:
                        x.fill_(1)
                    elif case == 2:
                        x.fill_(-1)
                    else:
                        x.mul_([0.01, 0.25, 1.0, 4.0, 128.0][case % 5])
                    expected = torch.mm(x, head.weight.T)
                    eager = LogitsProcessor._compute_lm_head(processor, x, head)
                    torch.testing.assert_close(eager, expected, rtol=0, atol=0)
                    graph.replay()
                    torch.testing.assert_close(captured, expected, rtol=0, atol=0)
            with envs.SGLANG_SM120_LOSSLESS_BF16_HEAD_M4.override(False):
                self.assertIsNone(try_lossless_bf16_head(x, head))
            self.assertIsNone(try_lossless_bf16_head(x[:3], head))
            original = head.weight
            head.weight = torch.nn.Parameter(original.detach(), requires_grad=False)
            self.assertIsNone(try_lossless_bf16_head(x, head))
            head.weight = original


if __name__ == "__main__":
    unittest.main()
