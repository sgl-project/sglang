import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestReplaySSMVerifyStages(CustomTestCase):
    def test_three_stages_bitwise(self):
        """Pipeline depth preserves output and ring bits with mixed history/padding."""
        from sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode import (
            gdn_replayssm_spec_decode,
        )

        torch.manual_seed(1)
        rows, heads, value_heads, width, capacity, window = 8, 16, 48, 128, 16, 4
        slots, tokens = rows + 1, rows * window
        dtype, device = torch.bfloat16, "cuda"
        state = (
            torch.randn(slots, value_heads, width, width, device=device, dtype=dtype)
            * 0.05
        )
        d = (
            torch.randn(slots, value_heads, capacity, width, device=device, dtype=dtype)
            * 0.1
        )
        k = torch.nn.functional.normalize(
            torch.randn(slots, heads, capacity, width, device=device), dim=-1
        ).to(dtype)
        g = -torch.rand(slots, value_heads, capacity, device=device) * 0.1
        dr = (torch.randn_like(d) * 1e-4).to(dtype)
        kr = (torch.randn_like(k) * 1e-4).to(dtype)
        rings = (d, k, g, dr, kr)
        initial = [tensor.clone() for tensor in rings]
        state_ids = (torch.randperm(rows, device=device) + 1).int()
        replay_ids = (torch.randperm(rows, device=device) + 1).long()
        kwargs = dict(
            q=torch.randn(tokens, heads, width, device=device, dtype=dtype),
            k=torch.randn(tokens, heads, width, device=device, dtype=dtype),
            v=torch.randn(tokens, value_heads, width, device=device, dtype=dtype) * 0.1,
            a=torch.randn(tokens, value_heads, device=device, dtype=dtype),
            b=torch.randn(tokens, value_heads, device=device, dtype=dtype),
            A_log=torch.log(torch.rand(value_heads, device=device) * 8 + 1).float(),
            dt_bias=torch.randn(value_heads, device=device).float(),
            checkpoint_state=state,
            d_cache=d,
            k_cache=k,
            g_cache=g,
            rawv_cache=dr,
            rawk_cache=kr,
            beta_cache=None,
            query_start_loc=torch.arange(
                0, tokens + 1, window, device=device, dtype=torch.int32
            ),
            replay_indices=replay_ids,
            cache_base=torch.randint(
                0, capacity, (slots,), device=device, dtype=torch.int32
            ),
            is_flush=torch.zeros(slots, device=device, dtype=torch.int8),
            max_cache_len=capacity,
            max_spec_len=window,
            scale=width**-0.5,
            use_qk_l2norm_in_kernel=True,
            null_block_id=-1,
            launch_mode="verify",
        )
        write_pos = torch.arange(slots, device=device, dtype=torch.int32) % 3 * 4
        state_ids[3] = -1
        results = []
        for stages in (2, 3):
            for tensor, original in zip(rings, initial):
                tensor.copy_(original)
            out = torch.zeros(tokens, value_heads, width, device=device, dtype=dtype)
            gdn_replayssm_spec_decode(
                **kwargs,
                out=out,
                ssm_state_indices=state_ids,
                write_pos=write_pos,
                num_stages=stages,
            )
            results.append([out.clone(), *[tensor.clone() for tensor in rings]])
        for before, after in zip(*results):
            integer = torch.int32 if before.dtype == torch.float32 else torch.int16
            self.assertTrue(torch.equal(before.view(integer), after.view(integer)))


if __name__ == "__main__":
    unittest.main()
