"""ROCm coverage for the decode-sized Qwen3.5 MoE softmax router."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.moe import topk as topk_module
from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip and is_gfx95_supported(),
    "requires AMD gfx95",
)
class TestQwen35MoeSoftmaxTopK(CustomTestCase):
    def test_triton_dispatch_matches_aiter(self):
        from aiter.fused_moe import fused_topk as aiter_fused_topk

        for num_tokens in (4, 12, 128):
            with self.subTest(num_tokens=num_tokens):
                torch.manual_seed(num_tokens)
                hidden_states = torch.randn(
                    num_tokens, 4096, device="cuda", dtype=torch.bfloat16
                )
                router_logits = torch.randn(
                    num_tokens, 512, device="cuda", dtype=torch.bfloat16
                )
                ref_weights = torch.empty(
                    num_tokens, 10, device="cuda", dtype=torch.float32
                )
                ref_ids = torch.empty(num_tokens, 10, device="cuda", dtype=torch.int32)
                ref_weights, ref_ids = aiter_fused_topk(
                    hidden_states,
                    router_logits,
                    10,
                    True,
                    topk_ids=ref_ids,
                    topk_weights=ref_weights,
                )

                with (
                    patch.object(topk_module, "_use_aiter", True),
                    patch.object(topk_module, "_is_gfx95", True),
                    patch.object(
                        topk_module,
                        "aiter_fused_topk",
                        side_effect=AssertionError("AITER top-k should be bypassed"),
                        create=True,
                    ),
                ):
                    weights, ids = topk_module.fused_topk(
                        hidden_states,
                        router_logits,
                        topk=10,
                        renormalize=True,
                    )

                torch.testing.assert_close(ids, ref_ids, rtol=0, atol=0)
                torch.testing.assert_close(weights, ref_weights, rtol=1e-5, atol=1e-6)

    def test_dispatch_envelope_is_narrow(self):
        hidden_states = torch.empty(128, 4096, device="cuda", dtype=torch.bfloat16)
        logits = torch.empty(128, 512, device="cuda", dtype=torch.bfloat16)
        packed = torch.empty(1, device="cuda")
        with (
            patch.object(topk_module, "_use_aiter", True),
            patch.object(topk_module, "_is_gfx95", True),
        ):
            self.assertTrue(
                topk_module._use_rocm_triton_softmax_topk(
                    hidden_states, logits, 10, None, 0, None
                )
            )
            self.assertFalse(
                topk_module._use_rocm_triton_softmax_topk(
                    torch.empty(129, 4096, device="cuda", dtype=torch.bfloat16),
                    torch.empty(129, 512, device="cuda", dtype=torch.bfloat16),
                    10,
                    None,
                    0,
                    None,
                )
            )
            self.assertFalse(
                topk_module._use_rocm_triton_softmax_topk(
                    hidden_states, logits.float(), 10, None, 0, None
                )
            )
            self.assertFalse(
                topk_module._use_rocm_triton_softmax_topk(
                    hidden_states[:, :2048], logits, 10, None, 0, None
                )
            )
            self.assertFalse(
                topk_module._use_rocm_triton_softmax_topk(
                    hidden_states, logits, 8, None, 0, None
                )
            )
            self.assertFalse(
                topk_module._use_rocm_triton_softmax_topk(
                    hidden_states,
                    logits,
                    10,
                    torch.empty(512, device="cuda"),
                    0,
                    None,
                )
            )
            self.assertFalse(
                topk_module._use_rocm_triton_softmax_topk(
                    hidden_states, logits, 10, None, 1, None
                )
            )
            self.assertFalse(
                topk_module._use_rocm_triton_softmax_topk(
                    hidden_states, logits, 10, None, 0, packed
                )
            )


if __name__ == "__main__":
    unittest.main()
