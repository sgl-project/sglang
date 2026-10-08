"""Original checkpoint input projections through real load/apply hooks on SM120."""

import gc
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors import safe_open
from sglang.kernels.ops.gemm.sm120_lossless_bf16_inproj import try_lossless_bf16_inproj
from sglang.srt.environ import envs
from sglang.srt.layers.quantization import unquant as uq
from sglang.test.test_utils import CustomTestCase


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.cuda.get_device_capability() == (12, 0)
    and os.environ.get("SGLANG_TEST_NEMOTRON_MODEL"),
    "requires SM120 and the original local Nemotron checkpoint",
)
class TestLosslessBf16Inproj(CustomTestCase):
    def test_all_original_weights_and_live_graph(self):
        torch.set_num_threads(2)
        torch.manual_seed(881)
        root = Path(os.environ["SGLANG_TEST_NEMOTRON_MODEL"])
        index = json.loads((root / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        tested = 0
        method = uq.UnquantizedLinearMethod()
        with (
            envs.SGLANG_SM120_LOSSLESS_BF16_INPROJ.override(True),
            patch.object(uq, "_BF16_GEMM_BACKEND", uq.Bf16GemmBackend.GEMV),
            patch.object(
                uq,
                "_bf16_gemm_dispatch_impl",
                side_effect=AssertionError("Used fallback"),
            ),
        ):
            for name in sorted(index):
                if not name.endswith(".mixer.in_proj.weight"):
                    continue
                with safe_open(root / index[name], framework="pt", device="cpu") as f:
                    weight = f.get_tensor(name)
                if weight.dtype != torch.bfloat16 or weight.shape != (18560, 4096):
                    continue
                with self.subTest(weight=name):
                    layer = torch.nn.Module()
                    layer.weight = torch.nn.Parameter(
                        weight.cuda(), requires_grad=False
                    )
                    method.process_weights_after_loading(layer)
                    self.assertTrue(
                        torch.equal(
                            layer.weight.cpu().view(torch.int16),
                            weight.view(torch.int16),
                        )
                    )
                    self.assertEqual(list(layer.state_dict()), ["weight"])
                    x = torch.randn((4, 4096), dtype=torch.bfloat16, device="cuda")
                    method.apply(layer, x)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        actual = method.apply(layer, x)
                    for case in range(16):
                        x.normal_()
                        if case == 0:
                            x.zero_()
                        elif case == 1:
                            x.fill_(1)
                        elif case == 2:
                            x.fill_(-1)
                        else:
                            x.mul_([0.01, 0.25, 1.0, 4.0, 128.0][case % 5])
                        expected = torch.nn.functional.linear(x, layer.weight)
                        eager = method.apply(layer, x)
                        torch.testing.assert_close(eager, expected, rtol=0, atol=0)
                        graph.replay()
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertIsNone(try_lossless_bf16_inproj(x[:1], layer))
                    self.assertIsNone(
                        try_lossless_bf16_inproj(
                            x, layer, torch.zeros(18560, device="cuda")
                        )
                    )
                    with envs.SGLANG_SM120_LOSSLESS_BF16_INPROJ.override(False):
                        self.assertIsNone(try_lossless_bf16_inproj(x, layer))
                    original = layer.weight
                    layer.weight = torch.nn.Parameter(
                        original.detach(), requires_grad=False
                    )
                    self.assertIsNone(try_lossless_bf16_inproj(x, layer))
                    print(
                        "BITWISE", name, "16 eager and changing graph cases", flush=True
                    )
                    tested += 1
                    del graph, actual, eager, expected, layer, original, x
                    gc.collect()
                    torch.cuda.empty_cache()
            self.assertEqual(tested, 12)


if __name__ == "__main__":
    unittest.main()
