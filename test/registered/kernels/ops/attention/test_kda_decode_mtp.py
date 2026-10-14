import unittest
from unittest import mock

import torch

from sglang.kernels.ops.attention import kda_decode_mtp
from sglang.kernels.ops.attention.fla.kda_replayssm_spec_decode import (
    commit_kda_replayssm_spec,
)
from sglang.kernels.ops.attention.kda_decode_mtp import fused_kda_decode_mtp_dspark
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=50, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def _make_inputs(num_requests, num_spec, key_dim=128, num_heads=2):
    num_tokens = num_requests * (1 + num_spec)
    num_slots, conv_width = num_requests + 2, 4
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
    return dict(
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


class TestKdaDecodeMtp(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        if get_device_sm() < 100:
            raise unittest.SkipTest("Kimi K3 compute kernels require SM100a+")

    def test_mtp_replayssm_ring(self):
        num_requests, num_heads, num_spec, key_dim = 2, 2, 2, 128
        num_slots, ring_size = num_requests + 2, 16

        def run(cache_ring):
            kwargs = _make_inputs(num_requests, num_spec, key_dim, num_heads)
            state = kwargs["recurrent_state"]
            slots = kwargs["ssm_state_indices"]
            scratch = kwargs["intermediate_state_indices"]
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

    def _run(self, num_requests, num_spec, *, onorm, block_threads=None):
        kwargs = _make_inputs(num_requests, num_spec)
        _, num_tokens, num_heads, key_dim = kwargs["x_v"].shape
        intermediate = torch.zeros(
            num_requests,
            1 + num_spec,
            num_heads,
            key_dim,
            key_dim,
            device="cuda",
        )
        gate = torch.randn(
            1, num_tokens, num_heads, key_dim, device="cuda", dtype=torch.bfloat16
        )
        weight = torch.rand(key_dim, device="cuda") + 0.5
        if onorm:
            kwargs.update(onorm_gate=gate, onorm_weight=weight, onorm_eps=1e-6)
        with mock.patch.object(
            kda_decode_mtp,
            "_block_threads",
            (lambda **_: block_threads)
            if block_threads
            else kda_decode_mtp._block_threads,
        ):
            output = fused_kda_decode_mtp_dspark(
                intermediate_ssm=intermediate, **kwargs
            )
        return output, intermediate, gate, weight

    def test_output_norm(self):
        for num_spec in (0, 2):
            with self.subTest(num_spec=num_spec):
                raw, raw_states, gate, weight = self._run(4, num_spec, onorm=False)
                fused, fused_states, _, _ = self._run(4, num_spec, onorm=True)
                raw = raw.float()
                expected = (
                    raw
                    * torch.rsqrt(raw.pow(2).mean(-1, keepdim=True) + 1e-6)
                    * weight
                    * torch.sigmoid(gate.float())
                )
                torch.testing.assert_close(
                    fused.float(), expected, atol=3e-2, rtol=3e-2
                )
                torch.testing.assert_close(fused_states, raw_states, atol=0, rtol=0)

    def test_streaming_state_matches_resident(self):
        for onorm in (False, True):
            with self.subTest(onorm=onorm):
                resident = self._run(
                    4, 0, onorm=onorm, block_threads=kda_decode_mtp.BLOCK_THREADS_WIDE
                )
                streaming = self._run(
                    4, 0, onorm=onorm, block_threads=kda_decode_mtp.BLOCK_THREADS_NARROW
                )
                torch.testing.assert_close(
                    streaming[0], resident[0], atol=2e-2, rtol=2e-2
                )
                torch.testing.assert_close(
                    streaming[1], resident[1], atol=1e-4, rtol=1e-4
                )


if __name__ == "__main__":
    unittest.main()
