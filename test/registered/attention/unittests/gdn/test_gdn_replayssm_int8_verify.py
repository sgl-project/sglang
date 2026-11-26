"""Check the INT8 checkpoint projection with the existing D/K residual ring."""

import unittest

import torch

from sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode import (
    gdn_replayssm_spec_decode,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA GPU")
class TestGDNReplaySSMInt8Verify(unittest.TestCase):
    def test_matches_exactly_representable_bf16_checkpoint(self):
        device = "cuda"
        dtype = torch.bfloat16
        gen = torch.Generator(device=device).manual_seed(134)
        requests, slots, heads, value_heads = 2, 4, 2, 6
        dim, ring, window = 128, 32, 8
        shape = (slots, value_heads, dim, dim)

        def randn(*shape, dtype=dtype):
            return torch.randn(shape, dtype=dtype, device=device, generator=gen)

        def indices(values):
            return torch.tensor(values, dtype=torch.int32, device=device)

        quantized = torch.randint(
            -127, 128, shape, dtype=torch.int8, device=device, generator=gen
        )
        # Layer slices of a slot-major scale pool retain a strided slot axis.
        exponents = torch.randint(
            -9, -4, (slots, 2, value_heads, dim), device=device, generator=gen
        )
        scales = torch.exp2(exponents.float())[:, 1]
        self.assertFalse(scales.is_contiguous())
        # Powers of two make dequantization exact, so any output difference
        # would come from applying the scales in the wrong part of the recurrence.
        bf16_state = (quantized.float() * scales.unsqueeze(-1)).to(dtype)
        tokens = requests * window
        args = {
            "q": randn(tokens, heads, dim),
            "k": randn(tokens, heads, dim),
            "v": randn(tokens, value_heads, dim),
            "a": randn(tokens, value_heads),
            "b": randn(tokens, value_heads),
            "A_log": randn(value_heads, dtype=torch.float32),
            "dt_bias": randn(value_heads, dtype=torch.float32),
            "query_start_loc": indices([0, window, tokens]),
            "ssm_state_indices": indices([3, 1]),
            "replay_indices": indices([1, 3]),
            "write_pos": indices([0, 5, 0, 3]),
            "cache_base": indices([0, 28, 0, 7]),
            "is_flush": indices([0] * slots).to(torch.int8),
            "max_cache_len": ring,
            "max_spec_len": window,
            "launch_mode": "verify",
            "null_block_id": -1,
            "num_warps": 2,
            "num_stages": 1,
        }
        rings = {
            "d_cache": randn(slots, value_heads, ring, dim) * 0.25,
            "k_cache": randn(slots, heads, ring, dim) * 0.125,
            "g_cache": -randn(slots, value_heads, ring, dtype=torch.float32).abs(),
            "rawv_cache": randn(slots, value_heads, ring, dim) * 0.001,
            "rawk_cache": randn(slots, heads, ring, dim) * 0.001,
        }

        def verify(state, row_scales):
            updated = {name: buf.clone() for name, buf in rings.items()}
            out = torch.empty_like(args["v"])
            gdn_replayssm_spec_decode(
                **args,
                **updated,
                out=out,
                checkpoint_state=state,
                checkpoint_scale=row_scales,
                beta_cache=None,
            )
            return out, updated

        checkpoint_before = quantized.clone()
        expected, expected_rings = verify(bf16_state, None)
        actual, actual_rings = verify(quantized, scales)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(quantized, checkpoint_before, rtol=0, atol=0)
        for name, buf in expected_rings.items():
            # The low part of D exposes small FP32 accumulator differences
            # that vanish in the BF16 forward output and the high part of D.
            atol = 4e-6 if name == "rawv_cache" else 0
            torch.testing.assert_close(actual_rings[name], buf, rtol=0, atol=atol)
        for name in ("rawv_cache", "rawk_cache"):
            self.assertFalse(torch.equal(actual_rings[name], rings[name]))


if __name__ == "__main__":
    unittest.main()
