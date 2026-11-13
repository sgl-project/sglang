"""
Tests the W8A16 small-M MoE for 128x128 block-FP8 experts against an fp32
dequantized reference, and the conditions under which the Triton MoE runner
may route to it.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-large")


def _is_sm90() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


def _make_experts(num_experts, hidden, inter, padded_scale=False):
    w13 = (torch.randn(num_experts, 2 * inter, hidden, device="cuda") * 0.5).to(
        torch.float8_e4m3fn
    )
    w2 = (torch.randn(num_experts, hidden, inter, device="cuda") * 0.5).to(
        torch.float8_e4m3fn
    )
    scales = []
    for rows, cols in ((2 * inter // 128, hidden // 128), (hidden // 128, inter // 128)):
        pad = 3 if padded_scale else 0
        # Padding is NaN so any read outside the logical block grid shows up.
        s = torch.full((num_experts, rows + pad, cols + pad), float("nan"), device="cuda")
        s[:, :rows, :cols] = torch.rand(num_experts, rows, cols, device="cuda") * 0.02 + 1e-3
        scales.append(s[:, :rows, :cols])
    return w13, w2, scales[0], scales[1]


def _reference(x, w13, w2, s13, s2, topk_weights, topk_ids, scaling):
    inter = w2.shape[2]
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for m in range(x.shape[0]):
        for j in range(topk_ids.shape[1]):
            e = topk_ids[m, j].item()
            a = w13[e].float() * s13[e].repeat_interleave(128, 0).repeat_interleave(128, 1)
            b = w2[e].float() * s2[e].repeat_interleave(128, 0).repeat_interleave(128, 1)
            h = a @ x[m].float()
            h = torch.nn.functional.silu(h[:inter]) * h[inter:]
            out[m] += topk_weights[m, j] * (b @ h)
    return out * scaling


@unittest.skipIf(not _is_sm90(), "small-M block-FP8 MoE is enabled on SM90 only")
class TestSmallMBlockFp8Moe(unittest.TestCase):
    def _run_case(self, m, num_experts, hidden, inter, topk, **kw):
        from sglang.kernels.ops.moe.small_m_block_fp8_moe import small_m_block_fp8_moe

        torch.manual_seed(m * 1000 + inter)
        w13, w2, s13, s2 = _make_experts(
            num_experts, hidden, inter, padded_scale=kw.get("padded_scale", False)
        )
        x = torch.randn(m, hidden, dtype=torch.bfloat16, device="cuda")
        ids = torch.stack(
            [torch.randperm(num_experts, device="cuda")[:topk] for _ in range(m)]
        ).to(torch.int32)
        if kw.get("shared_slot"):
            # Every token hits the last expert, like a fused shared expert.
            ids[:, -1] = num_experts - 1
            ids[:, :-1] = torch.where(ids[:, :-1] == num_experts - 1, 0, ids[:, :-1])
        weights = torch.softmax(torch.randn(m, topk, device="cuda"), -1)
        scaling = kw.get("scaling")
        out = torch.zeros_like(x) if kw.get("inplace") else None
        y = small_m_block_fp8_moe(
            x,
            w13=w13,
            w2=w2,
            w13_scale=s13,
            w2_scale=s2,
            topk_weights=weights,
            topk_ids=ids,
            routed_scaling_factor=scaling,
            out=out,
        )
        if out is not None:
            self.assertEqual(y.data_ptr(), out.data_ptr())
        ref = _reference(x, w13, w2, s13, s2, weights, ids, scaling or 1.0)
        rel = ((y.float() - ref).norm() / ref.norm()).item()
        self.assertLess(rel, 5e-3, (m, hidden, inter, rel))

    def test_tokens_share_experts(self):
        # Shared-expert slot repeats across tokens; each token must still read
        # its own activation row and top-k weights.
        for m in (1, 2, 4):
            self._run_case(m, 33, 2048, 512, 9, shared_slot=True)

    def test_non_power_of_two_dims_and_strided_scales(self):
        # I = 768 and H = 7168 force a K tile smaller than the default; the
        # scale views are non-contiguous slices of a NaN-padded tensor.
        self._run_case(3, 16, 2048, 768, 8, padded_scale=True)
        self._run_case(1, 8, 7168, 256, 4, padded_scale=True, scaling=2.5)

    def test_out_buffer(self):
        self._run_case(2, 16, 2048, 512, 8, inplace=True)

    def test_predicate(self):
        from sglang.kernels.ops.moe.small_m_block_fp8_moe import (
            use_small_m_block_fp8_moe,
        )

        w13, w2, s13, s2 = _make_experts(4, 256, 128)
        ids = torch.zeros(1, 2, dtype=torch.int32, device="cuda")
        x = torch.empty(1, 256, dtype=torch.bfloat16, device="cuda")
        self.assertTrue(use_small_m_block_fp8_moe(x, w13, w2, s13, s2, ids))
        # More than 4 tokens, fp32 activations, bf16 weights and non-fp32
        # scales must all fall back to the regular fused MoE.
        x5 = torch.empty(5, 256, dtype=torch.bfloat16, device="cuda")
        self.assertFalse(use_small_m_block_fp8_moe(x5, w13, w2, s13, s2, ids))
        self.assertFalse(use_small_m_block_fp8_moe(x.float(), w13, w2, s13, s2, ids))
        self.assertFalse(
            use_small_m_block_fp8_moe(x, w13.to(torch.bfloat16), w2, s13, s2, ids)
        )
        self.assertFalse(use_small_m_block_fp8_moe(x, w13, w2, s13.half(), s2, ids))

    def test_runner_gate(self):
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        from sglang.srt.layers.moe.moe_runner.triton import (
            TritonMoeQuantInfo,
            _use_small_m_w8a16,
        )

        w13, w2, s13, s2 = _make_experts(4, 256, 128)
        x = torch.empty(1, 256, dtype=torch.bfloat16, device="cuda")
        ids = torch.zeros(1, 2, dtype=torch.int32, device="cuda")

        def gate(quant=None, **cfg):
            info = TritonMoeQuantInfo(
                **{
                    "w13_weight": w13,
                    "w2_weight": w2,
                    "w13_scale": s13,
                    "w2_scale": s2,
                    "use_fp8_w8a8": True,
                    "block_shape": [128, 128],
                    "small_m_w8a16": True,
                    **(quant or {}),
                }
            )
            config = MoeRunnerConfig(
                **{"num_experts": 4, "num_local_experts": 4, **cfg}
            )
            return _use_small_m_w8a16(x, ids, info, config)

        self.assertTrue(gate())
        # Expert parallelism can leave non-local (filtered) ids in topk_ids,
        # and these epilogues are not implemented by the small-M kernels.
        self.assertFalse(gate(num_local_experts=2))
        self.assertFalse(gate(apply_router_weight_on_input=True))
        self.assertFalse(gate(swiglu_limit=7.0))
        self.assertFalse(gate(activation="gelu"))
        self.assertFalse(gate(quant={"small_m_w8a16": False}))
        self.assertFalse(
            gate(quant={"b13": torch.zeros(4, 256, device="cuda")})
        )


if __name__ == "__main__":
    unittest.main()
