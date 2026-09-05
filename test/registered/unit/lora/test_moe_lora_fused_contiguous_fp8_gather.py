"""The fused contiguous fp8 gather must match gather-then-quant.

Token-group quantization is a pure per-row function, so quantizing during the
gather (one BF16 read per routed pair) must reproduce the reference's scales
bitwise. Row bytes may differ by one fp8 code on rounding-boundary values
(Triton lowers the scalar and vector reciprocals differently); the masked
fp8 fill has the same relationship to the reference.
"""

import unittest

import torch

from sglang.srt.lora.moe.kernels.dispatch_contiguous import (
    dispatch_fill_rows_contiguous_bf16,
    dispatch_fill_rows_contiguous_fp8,
    dispatch_layout_contiguous,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")


def _reference(hidden, topk_ids, num_experts, top_k, alignment):
    num_pairs = topk_ids.numel()
    from sglang.srt.lora.moe.kernels.dispatch_contiguous import contiguous_m_pad_ceiling

    m_pad = contiguous_m_pad_ceiling(num_pairs, num_experts, alignment)
    device = hidden.device
    seg_counts = torch.empty(num_experts, dtype=torch.int32, device=device)
    seg_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
    pair_to_row = torch.empty(num_pairs, dtype=torch.int32, device=device)
    compact = torch.zeros((m_pad, hidden.size(1)), dtype=torch.bfloat16, device=device)
    dispatch_layout_contiguous(
        hidden,
        topk_ids,
        num_experts,
        top_k,
        alignment,
        seg_counts_out=seg_counts,
        seg_offsets_out=seg_offsets,
        pair_to_row_out=pair_to_row,
    )
    dispatch_fill_rows_contiguous_bf16(
        hidden, topk_ids, pair_to_row, hidden_compact_out=compact
    )
    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    q_ref, s_ref = sglang_per_token_group_quant_fp8(compact, 128)
    return pair_to_row, q_ref, s_ref, m_pad


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestFusedContiguousFp8Gather(unittest.TestCase):
    def test_matches_gather_then_quant(self):
        torch.manual_seed(7)
        device = "cuda"
        num_tokens, hidden_size, num_experts, top_k, alignment = 96, 512, 32, 6, 128
        hidden = torch.randn(
            (num_tokens, hidden_size), device=device, dtype=torch.bfloat16
        )
        topk_ids = torch.randint(
            0, num_experts, (num_tokens, top_k), device=device, dtype=torch.int32
        )
        # Sentinel pairs must be skipped by both paths.
        topk_ids[3, 1] = -1
        topk_ids[40, 5] = -1

        pair_to_row, q_ref, s_ref, m_pad = _reference(
            hidden, topk_ids, num_experts, top_k, alignment
        )
        rows_fp8 = torch.empty(
            (m_pad, hidden_size), dtype=torch.float8_e4m3fn, device=device
        )
        sf = torch.empty(
            (m_pad, hidden_size // 128), dtype=torch.float32, device=device
        )
        dispatch_fill_rows_contiguous_fp8(
            hidden, topk_ids, pair_to_row, rows_fp8_out=rows_fp8, scale_out=sf
        )

        real = pair_to_row[topk_ids.view(-1) >= 0].long()
        self.assertTrue(torch.equal(sf[real], s_ref[real]))
        delta = (
            rows_fp8[real].view(torch.uint8).int() - q_ref[real].view(torch.uint8).int()
        ).abs()
        self.assertLessEqual(int(delta.max()), 1)
        self.assertLess(
            float((delta > 0).float().mean()),
            0.005,
            "more than 0.5% of bytes moved off the reference",
        )


if __name__ == "__main__":
    unittest.main()
