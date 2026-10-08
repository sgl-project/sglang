"""Replay all checkpoint BF16 experts and changing M1--M4 CUDA Graph inputs.

This manual integration test needs an SM120 device and two offline fixtures:
SGLANG_TEST_BF16_MOE_CACHE: up/down safetensors, each with original ``weight``;
SGLANG_TEST_BF16_MOE_CAPTURE: three sample-*.safetensors containing x,
topk_ids, topk_weights and full_output from the original MTP layer.
All original BF16 weights stay resident. Real runner dispatch is asserted.
"""

import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import sglang.srt.layers.quantization  # noqa: F401 -- initialize quantization registry first
import torch
from flashinfer.fused_moe import cutlass_fused_moe
from flashinfer.fused_moe.core import ActivationType
from safetensors.torch import load_file
from sglang.kernels.ops.gemm.sm120_bf16_moe_union import bf16_moe_union
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
    FlashInferCutlassMoeQuantInfo,
    _run_flashinfer_cutlass,
)
from sglang.test.test_utils import CustomTestCase


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.cuda.get_device_capability() == (12, 0)
    and os.environ.get("SGLANG_TEST_BF16_MOE_CACHE")
    and os.environ.get("SGLANG_TEST_BF16_MOE_CAPTURE"),
    "requires SM120 and the original-weight / captured-input fixtures",
)
class TestBf16MoeUnion(CustomTestCase):
    def test_production_and_changing_live_graph(self):
        from sglang.srt.environ import envs

        with envs.SGLANG_SM120_BF16_MOE_UNION.override(True):
            root = Path(os.environ["SGLANG_TEST_BF16_MOE_CACHE"])
            torch.set_num_threads(4)
            torch.manual_seed(214)
            layer = torch.nn.Module()
            layer.moe_ep_size = layer.moe_tp_size = 1
            layer.moe_runner_config = MoeRunnerConfig(
                activation="relu2", is_gated=False, routed_scaling_factor=5.0
            )
            for kind, name in [("up_proj", "w13_weight"), ("down_proj", "w2_weight")]:
                d = load_file(
                    root / f"lossless-moe-{kind}-cache.safetensors", device="cuda"
                )
                w = (
                    d["weight"]
                    if kind == "up_proj"
                    else d["weight"][:, :, :2688].contiguous()
                )
                layer.register_parameter(
                    name, torch.nn.Parameter(w, requires_grad=False)
                )
                del d
                print("loaded", name, flush=True)
            q = FlashInferCutlassMoeQuantInfo(
                "bf16",
                layer.w13_weight,
                layer.w2_weight,
                output_dtype=torch.bfloat16,
            )
            samples = [
                load_file(p, device="cuda")
                for p in sorted(
                    Path(os.environ["SGLANG_TEST_BF16_MOE_CAPTURE"]).glob(
                        "sample-*.safetensors"
                    )
                )
            ]
            records = []
            for M in [1, 2, 3, 4]:
                x = torch.empty((M, 1024), device="cuda", dtype=torch.bfloat16)
                ids = torch.empty((M, 22), device="cuda", dtype=torch.int32)
                tw = torch.empty((M, 22), device="cuda", dtype=torch.float32)
                actual = torch.empty_like(x)
                expected = torch.empty_like(x)
                dispatch = SimpleNamespace(
                    hidden_states=x,
                    hidden_states_scale=None,
                    topk_output=SimpleNamespace(topk_ids=ids, topk_weights=tw),
                )

                def custom(dispatch=dispatch, actual=actual, require_dispatch=True):
                    with patch(
                        "sglang.kernels.ops.gemm.sm120_bf16_moe_union.bf16_moe_union",
                        wraps=bf16_moe_union,
                    ) as observed:
                        result = _run_flashinfer_cutlass(
                            dispatch_output=dispatch,
                            quant_info=q,
                            runner_config=layer.moe_runner_config,
                            output=actual,
                        )
                        self.assertEqual(observed.call_count, int(require_dispatch))
                        return result

                def stock(x=x, ids=ids, tw=tw, expected=expected, M=M):
                    cutlass_fused_moe(
                        x,
                        ids,
                        tw,
                        layer.w13_weight,
                        layer.w2_weight,
                        torch.bfloat16,
                        [],
                        output=expected,
                        activation_type=ActivationType.Relu2,
                        tune_max_num_tokens=1 << (M - 1).bit_length(),
                        use_fused_finalize=True,
                    )
                    expected.mul_(5.0)

                x.copy_(samples[0]["x"][:M])
                ids.copy_(samples[0]["topk_ids"][:M])
                tw.copy_(samples[0]["topk_weights"][:M])
                custom()
                stock()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    custom()
                for case in range(16):
                    if case < 3:
                        s = samples[case]
                        x.copy_(s["x"][:M])
                        ids.copy_(s["topk_ids"][:M])
                        tw.copy_(s["topk_weights"][:M])
                    else:
                        x.copy_(
                            torch.randn_like(x) * [0.0, 0.01, 0.3, 1.0, 4.0][case % 5]
                        )
                        tw.copy_(torch.rand_like(tw))
                        tw.div_(tw.sum(1, keepdim=True))
                        for row in range(M):
                            ids[row].copy_(torch.randperm(512, device="cuda")[:22])
                        if case % 4 == 0:
                            ids.copy_(ids[:1].expand_as(ids).clone())
                        if case == 15:
                            ids.copy_(
                                torch.arange(
                                    490, 512, device="cuda", dtype=torch.int32
                                ).expand_as(ids)
                            )
                    stock()
                    custom()
                    torch.testing.assert_close(
                        actual, expected, rtol=0, atol=0, equal_nan=True
                    )
                    g.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(
                        actual, expected, rtol=0, atol=0, equal_nan=True
                    )
                    if case < 3:
                        torch.testing.assert_close(
                            actual, samples[case]["full_output"][:M], rtol=0, atol=0
                        )
                    records.append(
                        {
                            "M": M,
                            "case": case,
                            "eager_bitwise": True,
                            "live_graph_bitwise": True,
                        }
                    )
                print("M exact eager/live cases", M, 16, flush=True)
                # Autotuning must call the original FlashInfer runner and retain its cache keys.
                with (
                    patch(
                        "sglang.srt.layers.moe.moe_runner.flashinfer_cutlass._flashinfer_autotune_active",
                        return_value=True,
                    ),
                    patch(
                        "sglang.kernels.ops.gemm.sm120_bf16_moe_union.bf16_moe_union",
                        side_effect=AssertionError("candidate used during autotune"),
                    ),
                ):
                    custom(require_dispatch=False)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(len(records), 64)
            print("all passed", flush=True)


if __name__ == "__main__":
    unittest.main()
