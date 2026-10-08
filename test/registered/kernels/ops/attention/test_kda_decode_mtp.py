import unittest

import torch

from sglang.kernels.ops.attention.fla.kda_replayssm_spec_decode import (
    commit_kda_replayssm_spec,
)
from sglang.kernels.ops.attention.kda_decode_mtp import fused_kda_decode_mtp_dspark
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=50, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


class TestKdaDecodeMtp(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        if get_device_sm() < 100:
            raise unittest.SkipTest("Kimi K3 compute kernels require SM100a+")

    def test_mtp_replayssm_ring(self):
        num_requests, num_heads, num_spec, key_dim = 2, 2, 2, 128
        num_tokens = num_requests * (1 + num_spec)
        num_slots, ring_size, conv_width = num_requests + 2, 16, 4

        def run(cache_ring):
            torch.manual_seed(5)
            x_q = torch.randn(
                1,
                num_tokens,
                num_heads,
                key_dim,
                device="cuda",
                dtype=torch.bfloat16,
            )
            x_k = torch.randn_like(x_q)
            x_v = torch.randn_like(x_q)
            gate = torch.randn_like(x_q)
            beta = torch.randn(
                1,
                num_tokens,
                num_heads,
                device="cuda",
                dtype=torch.bfloat16,
            )
            conv_weight = [
                torch.randn(
                    num_heads * key_dim,
                    conv_width,
                    device="cuda",
                )
                * 0.1
                for _ in range(3)
            ]
            conv_state = [
                torch.randn(
                    num_slots,
                    num_heads * key_dim,
                    conv_width - 1,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                for _ in range(3)
            ]
            slots = torch.arange(1, num_requests + 1, device="cuda", dtype=torch.int32)
            scratch = torch.arange(num_requests, device="cuda", dtype=torch.int32)
            state = torch.randn(
                num_slots,
                num_heads,
                key_dim,
                key_dim,
                device="cuda",
            )
            intermediate_conv = torch.zeros(
                num_requests,
                1 + num_spec,
                num_heads * key_dim,
                conv_width - 1,
                device="cuda",
                dtype=torch.bfloat16,
            )
            kwargs = dict(
                x_q=x_q,
                x_k=x_k,
                x_v=x_v,
                w_q=conv_weight[0],
                w_k=conv_weight[1],
                w_v=conv_weight[2],
                cs_q=conv_state[0],
                cs_k=conv_state[1],
                cs_v=conv_state[2],
                g=gate,
                beta=beta,
                A_log=torch.randn(num_heads, device="cuda"),
                dt_bias=torch.randn(num_heads * key_dim, device="cuda"),
                recurrent_state=state,
                intermediate_state_indices=scratch,
                intermediate_conv_q=intermediate_conv.clone(),
                intermediate_conv_k=intermediate_conv.clone(),
                intermediate_conv_v=intermediate_conv.clone(),
                ssm_state_indices=slots,
                cu_seqlens=torch.arange(
                    0,
                    num_tokens + 1,
                    1 + num_spec,
                    device="cuda",
                    dtype=torch.int32,
                ),
                lower_bound=-5.0,
            )
            if not cache_ring:
                intermediate = torch.zeros(
                    num_requests,
                    1 + num_spec,
                    num_heads,
                    key_dim,
                    key_dim,
                    device="cuda",
                )
                output = fused_kda_decode_mtp_dspark(
                    intermediate_ssm=intermediate,
                    **kwargs,
                )
                return output, intermediate, slots, scratch

            raw_v = torch.zeros(
                num_slots,
                num_heads,
                ring_size,
                key_dim,
                device="cuda",
                dtype=torch.bfloat16,
            )
            raw_k = torch.zeros_like(raw_v)
            ring_gate = torch.zeros(
                num_slots,
                num_heads,
                ring_size,
                key_dim,
                device="cuda",
            )
            ring_beta = torch.zeros(
                num_slots,
                num_heads,
                ring_size,
                device="cuda",
            )
            output = fused_kda_decode_mtp_dspark(
                intermediate_ssm=None,
                replayssm_rawv=raw_v,
                replayssm_rawk=raw_k,
                replayssm_g=ring_gate,
                replayssm_beta=ring_beta,
                **kwargs,
            )
            return (
                output,
                state,
                slots,
                (raw_v, raw_k, ring_gate, ring_beta),
            )

        baseline, intermediate, slots, scratch = run(cache_ring=False)
        ring_output, checkpoint, ring_slots, rings = run(cache_ring=True)
        self.assertTrue(torch.equal(ring_output, baseline))
        commit_kda_replayssm_spec(
            checkpoint,
            *rings,
            ring_slots,
            torch.full(
                (num_requests,),
                1 + num_spec,
                device="cuda",
                dtype=torch.int32,
            ),
            max_cache_len=ring_size,
            num_k_heads=num_heads,
            use_qk_l2norm_in_kernel=True,
            null_block_id=-1,
        )
        for request in range(num_requests):
            expected = intermediate[scratch[request], num_spec]
            actual = checkpoint[slots[request]]
            relative_error = (
                actual - expected
            ).abs().max() / expected.abs().max().clamp_min(1e-6)
            self.assertLess(relative_error.item(), 2e-2)


if __name__ == "__main__":
    unittest.main()
