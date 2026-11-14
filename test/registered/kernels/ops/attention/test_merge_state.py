"""Natural-log attention-state merging against an unsplit attention oracle.

The AOT wheel's merge_state_v2 comparison is not a registered runtime test.
These cases protect the Triton entry point independently of another merge
implementation: log-base changes, unstable exponentiation, tail masking, and
empty-partition sentinels must not silently change attention results.
"""

import unittest

import torch

from sglang.kernels.ops.attention.merge_state import merge_state_triton
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestMergeState(CustomTestCase):
    def test_split_matches_unsplit_attention(self):
        generator = torch.Generator(device="cuda").manual_seed(42)
        for head_size in (48, 128):
            # A large common offset must cancel out of the attention weights.
            logits = torch.randn(
                (3, 2, 11), device="cuda", dtype=torch.float64, generator=generator
            )
            logits += torch.tensor([-1000.0, 0.0, 1000.0], device="cuda").reshape(
                3, 1, 1
            )
            values = torch.randn(
                (3, 2, 11, head_size),
                device="cuda",
                dtype=torch.float64,
                generator=generator,
            )
            expected = (logits.softmax(-1).unsqueeze(-1) * values).sum(-2)
            expected_lse = logits.logsumexp(-1).float()
            parts = []
            for start, end in ((0, 4), (4, 11)):
                scores = logits[..., start:end]
                part = (
                    scores.softmax(-1).unsqueeze(-1) * values[..., start:end, :]
                ).sum(-2)
                parts.append((part, scores.logsumexp(-1).float()))

            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                with self.subTest(head_size=head_size, dtype=dtype):
                    prefix, prefix_lse = parts[0]
                    suffix, suffix_lse = parts[1]
                    actual, actual_lse = merge_state_triton(
                        prefix.to(dtype), prefix_lse, suffix.to(dtype), suffix_lse
                    )
                    self.assertEqual(actual.dtype, dtype)
                    # Partial outputs are rounded before merging, unlike the oracle.
                    atol = {
                        torch.float32: 1e-4,
                        torch.float16: 1e-3,
                        torch.bfloat16: 1e-2,
                    }[dtype]
                    torch.testing.assert_close(
                        actual.double(), expected, atol=atol, rtol=atol
                    )
                    torch.testing.assert_close(
                        actual_lse, expected_lse, atol=1e-4, rtol=0
                    )

    def test_empty_partition_and_preallocated_outputs(self):
        # Both +/-inf encode an empty partition. Keep the other side finite;
        # the contract for two empty partitions is intentionally not assumed.
        prefix = torch.arange(4 * 3 * 48, device="cuda", dtype=torch.float32)
        prefix = prefix.reshape(4, 3, 48) / 100
        suffix = -prefix - 1
        prefix_lse = (
            torch.tensor([float("inf"), -float("inf"), -9.0, 9.0], device="cuda")[
                :, None
            ]
            .expand(4, 3)
            .contiguous()
        )
        suffix_lse = (
            torch.tensor([7.0, -7.0, float("inf"), -float("inf")], device="cuda")[
                :, None
            ]
            .expand(4, 3)
            .contiguous()
        )
        original_prefix_lse = prefix_lse.clone()
        original_suffix_lse = suffix_lse.clone()
        output = torch.full_like(prefix, float("nan"))
        output_lse = torch.full_like(prefix_lse, float("nan"))

        actual, actual_lse = merge_state_triton(
            prefix, prefix_lse, suffix, suffix_lse, output, output_lse
        )

        self.assertIs(actual, output)
        self.assertIs(actual_lse, output_lse)
        torch.testing.assert_close(
            actual, torch.cat((suffix[:2], prefix[2:])), atol=0, rtol=0
        )
        torch.testing.assert_close(
            actual_lse, torch.cat((suffix_lse[:2], prefix_lse[2:])), atol=0, rtol=0
        )
        torch.testing.assert_close(prefix_lse, original_prefix_lse, atol=0, rtol=0)
        torch.testing.assert_close(suffix_lse, original_suffix_lse, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
