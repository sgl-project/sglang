"""Regression for #53618: stale shrink allocations must not affect dense LoRA."""

import os
import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.lora.dense.trtllm_lora_temp.sgemm_lora_a import (
    sgemm_lora_a_fwd,
)
from sglang.kernels.ops.lora.dense.trtllm_lora_temp.sgemm_lora_b import (
    sgemm_lora_b_fwd,
    shared_sink_sgemm_lora_b_fwd,
)
from sglang.srt.lora.utils import LoRABatchInfo
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestTRTLLMDenseLoRARankTail(CustomTestCase):
    def test_shrink_clears_nonfinite_rank_tail(self):
        """Zero B padding must not multiply NaN/Inf outside an adapter's rank.

        Exercise both the bf16 shrink output and the fp32 split-K accumulator,
        with contiguous segments and adapter-sorted token indirection. Supply
        a NaN/Inf-filled allocation to the real shrink, avoiding allocator luck.
        Its epilogue must overwrite the padding before the dense expand reads it.
        """
        with patch.dict(
            os.environ,
            {
                "SGLANG_EXPERIMENTAL_LORA_OPTI": "1",
                "SGLANG_ENABLE_LORA_SHRINK_SPLIT_K": "1",
                "SGLANG_OPT_LORA_CUBLAS": "0",
                "SGLANG_OPT_LORA_CUBLAS_A": "0",
                "SGLANG_OPT_LORA_CUBLAS_B": "0",
            },
        ):
            for input_dim in (3584, 4096):
                for poison, sorted_tokens in (
                    (float("nan"), False),
                    (float("inf"), True),
                ):
                    with self.subTest(
                        input_dim=input_dim, poison=poison, sorted_tokens=sorted_tokens
                    ):
                        self._check_rank_tail(input_dim, poison, sorted_tokens)

    def _check_rank_tail(self, input_dim, poison, sorted_tokens):
        device = "cuda"
        dtype = torch.bfloat16
        generator = torch.Generator(device=device).manual_seed(53618)
        ranks = [7, 16, 64, 0]
        lengths = [7, 9, 13, 3]
        # Rank/scaling metadata are indexed by adapter slot, not segment number.
        slots = [2, 0, 1, 3]
        scalings = [0.5, 1.25, 2.0, 1.0]
        num_tokens = sum(lengths)
        max_rank = max(ranks)
        output_dim = 272  # Also exercise a partial output tile.
        permutation = (
            torch.randperm(num_tokens, generator=generator, device=device).int()
            if sorted_tokens
            else None
        )
        info = LoRABatchInfo(
            use_cuda_graph=False,
            bs=len(lengths),
            num_segments=len(lengths),
            seg_indptr=torch.tensor(
                [0, 7, 16, 29, 32], device=device, dtype=torch.int32
            ),
            weight_indices=torch.tensor(slots, device=device, dtype=torch.int32),
            lora_ranks=torch.tensor(ranks, device=device, dtype=torch.int32),
            scalings=torch.tensor(scalings, device=device, dtype=torch.float32),
            max_len=max(lengths),
            seg_lens=torch.tensor(lengths, device=device, dtype=torch.int32),
            permutation=permutation,
        )
        x = torch.randn(
            num_tokens, input_dim, device=device, dtype=dtype, generator=generator
        )
        a = (
            torch.randn(
                len(ranks), max_rank, input_dim, device=device, generator=generator
            )
            * 0.02
        ).to(dtype)
        b = (
            torch.randn(
                len(ranks), output_dim, max_rank, device=device, generator=generator
            )
            * 0.05
        ).to(dtype)
        for slot, rank in enumerate(ranks):
            b[slot, :, rank:] = 0  # Match the memory pool's zero-padded B tail.
        base = torch.randn(
            num_tokens, output_dim, device=device, dtype=dtype, generator=generator
        )

        poisoned = torch.full(
            (num_tokens, max_rank), poison, device=device, dtype=dtype
        )
        real_empty = torch.empty

        def allocate(shape, *args, **kwargs):
            # Intercept only the bf16 shrink allocation, not compiler/runtime
            # allocations. The split-K path initializes its fp32 buffer itself.
            if (
                shape == poisoned.shape
                and kwargs.get("device") == x.device
                and kwargs.get("dtype") == dtype
            ):
                return poisoned
            return real_empty(shape, *args, **kwargs)

        with patch.object(torch, "empty", side_effect=allocate):
            intermediate = sgemm_lora_a_fwd(
                x,
                a,
                info,
                out_alloc_stream=(
                    torch.cuda.current_stream() if sorted_tokens else None
                ),
            )
        if input_dim == 3584:
            self.assertIs(intermediate, poisoned)
        self.assertEqual(
            intermediate.dtype, dtype if input_dim == 3584 else torch.float32
        )
        expected = base.float().clone()
        shared_input = intermediate.clone()
        padded_rows = []
        start = 0
        for length, slot in zip(lengths, slots):
            rows = (
                permutation[start : start + length].long()
                if permutation is not None
                else torch.arange(start, start + length, device=device)
            )
            rank = ranks[slot]
            if rank:
                # Independent torch matmuls, with the same intermediate bf16
                # rounding that expand applies to the fp32 split-K accumulator.
                shrink = x[rows].float() @ a[slot, :rank].float().T
                torch.testing.assert_close(
                    intermediate[rows, :rank].float(), shrink, rtol=1e-2, atol=1e-2
                )
                delta = scalings[slot] * (
                    shrink.to(dtype).float() @ b[slot, :, :rank].float().T
                )
                expected[rows] += delta.to(dtype).float()
            if 0 < rank < max_rank:
                padded_rows.append(intermediate[rows, rank:])
            # This independent caller uses its existing dynamic-rank mask.
            shared_input[rows, rank:] = poison
            start += length

        out = sgemm_lora_b_fwd(intermediate, b, info, base.clone())
        self.assertTrue(torch.isfinite(out).all().item())
        torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)

        for tail in padded_rows:
            self.assertTrue((tail == 0).all().item())

        # The shared-sink caller retains its rank-masked, non-atomic writeback.
        shared_out = shared_sink_sgemm_lora_b_fwd(
            shared_input,
            b,
            info,
            base.clone(),
            apply_scaling=True,
            padded_rank=False,
        )
        self.assertTrue(torch.isfinite(shared_out).all().item())
        torch.testing.assert_close(shared_out.float(), expected, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    unittest.main()
