"""ROCm coverage for the 512-expert softmax router on aiter `topk_gating`."""

import itertools
import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.moe import topk as topk_module
from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")

# Qwen3.8 and Qwen3.8-Flash-Next.
HIDDEN_SIZES = (8192, 2560)
NUM_EXPERTS = 512
TOPK = 10


def _envelope(
    gating_output,
    hidden_size,
    correction_bias=None,
    num_fused_shared_experts=0,
    packed_out=None,
    hidden_dtype=torch.bfloat16,
):
    hidden_states = torch.empty(
        gating_output.shape[0],
        hidden_size,
        device=gating_output.device,
        dtype=hidden_dtype,
    )
    return topk_module._use_aiter_topk_gating_softmax(
        hidden_states,
        gating_output,
        correction_bias,
        num_fused_shared_experts,
        packed_out,
    )


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip and is_gfx95_supported(),
    "requires AMD gfx95",
)
class TestQwen38MoeSoftmaxTopKGating(CustomTestCase):
    def test_topk_gating_matches_legacy_launcher(self):
        from aiter.fused_moe import fused_topk as aiter_fused_topk

        for hidden_size, num_tokens in itertools.product(HIDDEN_SIZES, (4, 128, 1024)):
            with self.subTest(hidden_size=hidden_size, num_tokens=num_tokens):
                torch.manual_seed(num_tokens)
                hidden_states = torch.randn(
                    num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16
                )
                router_logits = torch.randn(
                    num_tokens, NUM_EXPERTS, device="cuda", dtype=torch.bfloat16
                )
                ref_weights = torch.empty(
                    num_tokens, TOPK, device="cuda", dtype=torch.float32
                )
                ref_ids = torch.empty(
                    num_tokens, TOPK, device="cuda", dtype=torch.int32
                )
                ref_weights, ref_ids = aiter_fused_topk(
                    hidden_states,
                    router_logits,
                    TOPK,
                    True,
                    topk_ids=ref_ids,
                    topk_weights=ref_weights,
                )

                with (
                    patch.object(topk_module, "_use_aiter", True),
                    patch.object(topk_module, "_use_aiter_topk_gating", True),
                    patch.object(topk_module, "_is_gfx95", True),
                    patch.object(
                        topk_module,
                        "aiter_fused_topk",
                        side_effect=AssertionError(
                            "the legacy launcher should be bypassed"
                        ),
                        create=True,
                    ),
                ):
                    weights, ids = topk_module.fused_topk(
                        hidden_states,
                        router_logits,
                        topk=TOPK,
                        renormalize=True,
                    )

                # BF16 ties can swap or reorder equal-logit experts.
                torch.testing.assert_close(
                    router_logits.gather(1, ids.long()).sort(dim=-1).values,
                    router_logits.gather(1, ref_ids.long()).sort(dim=-1).values,
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    weights.sort(dim=-1).values,
                    ref_weights.sort(dim=-1).values,
                    rtol=1e-5,
                    atol=1e-6,
                )
                torch.testing.assert_close(
                    weights.sum(dim=-1),
                    torch.ones(num_tokens, device="cuda"),
                    rtol=1e-5,
                    atol=1e-6,
                )

    def test_dispatch_envelope_is_narrow(self):
        logits = torch.empty(4, NUM_EXPERTS, device="cuda", dtype=torch.bfloat16)
        packed = torch.empty(1, device="cuda")
        with (
            patch.object(topk_module, "_use_aiter", True),
            patch.object(topk_module, "_use_aiter_topk_gating", True),
            patch.object(topk_module, "_is_gfx95", True),
        ):
            self.assertFalse(_envelope(logits, 4096))
            for hidden_size in HIDDEN_SIZES:
                with self.subTest(hidden_size=hidden_size):
                    self.assertTrue(_envelope(logits, hidden_size))
                    self.assertFalse(
                        _envelope(
                            torch.empty(
                                topk_module._AITER_TOPK_GATING_MAX_ROWS + 1,
                                NUM_EXPERTS,
                                device="cuda",
                                dtype=torch.bfloat16,
                            ),
                            hidden_size,
                        )
                    )
                    self.assertFalse(
                        _envelope(logits, hidden_size, hidden_dtype=torch.float16)
                    )
                    self.assertFalse(
                        _envelope(
                            torch.empty(4, 256, device="cuda", dtype=torch.bfloat16),
                            hidden_size,
                        )
                    )
                    self.assertFalse(_envelope(logits.float(), hidden_size))
                    self.assertFalse(
                        _envelope(
                            torch.empty(
                                NUM_EXPERTS, 8, device="cuda", dtype=torch.bfloat16
                            ).t(),
                            hidden_size,
                        )
                    )
                    self.assertFalse(
                        _envelope(
                            logits,
                            hidden_size,
                            correction_bias=torch.empty(NUM_EXPERTS, device="cuda"),
                        )
                    )
                    self.assertFalse(
                        _envelope(logits, hidden_size, num_fused_shared_experts=1)
                    )
                    self.assertFalse(_envelope(logits, hidden_size, packed_out=packed))
        with patch.object(topk_module, "_use_aiter_topk_gating", False):
            for hidden_size in HIDDEN_SIZES:
                self.assertFalse(_envelope(logits, hidden_size))


if __name__ == "__main__":
    unittest.main()
