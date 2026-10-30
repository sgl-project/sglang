"""MoE TMA down path must stay bit-identical to the non-TMA path.

The TMA down path keeps the activation and the down-input quant in the sorted,
block-padded buffer sized for the worst case and skips the padding rows the down
GEMM never reads. Skipping must not change any result, so the whole layer is
compared bitwise; a tolerance would hide a row mix-up.
"""

import unittest

import torch

from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

E, TOPK, HIDDEN, INTER, BLOCK = 256, 8, 2048, 512, [128, 128]
CASES = ((1, 16), (32, 16), (300, 32), (2048, 64))  # (num_tokens, BLOCK_SIZE_M)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9,
    "requires sm90+ for the TMA down path",
)
class TestMoeTmaDownPaddingRows(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.weights = {
            False: (
                (torch.randn(E, 2 * INTER, HIDDEN, device="cuda") / HIDDEN**0.5).to(
                    torch.bfloat16
                ),
                (torch.randn(E, HIDDEN, INTER, device="cuda") / INTER**0.5).to(
                    torch.bfloat16
                ),
                None,
                None,
            ),
            True: (
                torch.randn(E, 2 * INTER, HIDDEN, device="cuda").to(
                    torch.float8_e4m3fn
                ),
                torch.randn(E, HIDDEN, INTER, device="cuda").to(torch.float8_e4m3fn),
                torch.rand(E, 2 * INTER // 128, HIDDEN // 128, device="cuda") / 32
                + 0.01,
                torch.rand(E, HIDDEN // 128, INTER // 128, device="cuda") / 16 + 0.02,
            ),
        }

    def _layer(self, m, block_m, down_tma, fp8):
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            _fused_moe_kernel_sequence,
        )
        from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
            moe_align_block_size,
        )

        torch.manual_seed(m)
        hidden = torch.randn(m, HIDDEN, device="cuda", dtype=torch.bfloat16)
        topk_ids = torch.rand(m, E, device="cuda").topk(TOPK, dim=-1).indices
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = torch.rand(m, TOPK, device="cuda").softmax(-1)
        sorted_ids, expert_ids, num_post = moe_align_block_size(topk_ids, block_m, E)
        config = dict(
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_N=64,
            BLOCK_SIZE_K=128,
            GROUP_SIZE_M=1,
            num_warps=4,
            num_stages=3,
        )
        w1, w2, w1_s, w2_s = self.weights[fp8]
        return _fused_moe_kernel_sequence(
            hidden,
            w1,
            w2,
            topk_weights,
            topk_ids,
            sorted_ids,
            expert_ids,
            num_post,
            config,
            dict(config, BLOCK_SIZE_N=128),
            down_tma,
            False,
            b1=None,
            b2=None,
            use_fp8_w8a8=fp8,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            w1_scale=w1_s,
            w2_scale=w2_s,
            w1_zp=None,
            w2_zp=None,
            a1_scale=None,
            a2_scale=None,
            block_shape=BLOCK if fp8 else None,
            activation="silu",
            is_gated=True,
            no_combine=False,
            inplace=False,
            apply_router_weight_on_input=False,
            routed_scaling_factor=None,
            gemm1_alpha=None,
            gemm1_limit=None,
            filter_expert=False,
        )

    def test_tma_down_layer_bitwise_matches_non_tma(self):
        with (
            get_context().override_server_args(enable_deterministic_inference=False),
            get_parallel().override(tp_group=None),
        ):
            for fp8 in (True, False):
                for m, block_m in CASES:
                    with self.subTest(fp8=fp8, m=m, block_m=block_m):
                        tma = self._layer(m, block_m, down_tma=True, fp8=fp8)
                        ref = self._layer(m, block_m, down_tma=False, fp8=fp8)
                        self.assertFalse(tma.isnan().any())
                        self.assertGreater(ref.abs().max().item(), 0.1)
                        self.assertTrue(torch.equal(tma, ref))


if __name__ == "__main__":
    unittest.main()
