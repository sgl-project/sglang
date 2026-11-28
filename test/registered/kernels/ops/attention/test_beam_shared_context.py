import itertools
import math
import unittest

import torch

from sglang.kernels.ops.attention.beam_shared_context import shared_context_attention
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")


def _check_against_reference(first_slot):
    """Two groups of 16 beams whose K/V slots start at `first_slot`, vs FP32."""
    device = torch.device("cuda")
    heads, kv_heads, head_dim, width, capacity = 4, 2, 64, 16, 3
    context_lens, decode_lens = [70, 300], [2, 1]
    num_context = sum(context_lens)
    num_used = num_context + len(context_lens) * capacity * width
    torch.manual_seed(0)
    # One tensor serves as both K and V, to halve the memory a large cache takes.
    kv = torch.empty(
        first_slot + num_used, kv_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    kv[first_slot:] = torch.randn(
        num_used, kv_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    slots = torch.arange(first_slot, first_slot + num_used, device=device)
    rows = torch.randperm(len(context_lens) * width, device=device).view(-1, width)
    context_slots = slots[:num_context].int()
    decode_slots = slots[num_context:].view(len(context_lens), -1).int()
    cu_seqlens = [0, *itertools.accumulate(context_lens)]
    q = torch.randn(rows.numel(), heads, head_dim, dtype=torch.bfloat16, device=device)
    scale = 1.0 / math.sqrt(head_dim)

    output = shared_context_attention(
        q,
        kv,
        kv,
        rows.int(),
        context_slots,
        torch.tensor(cu_seqlens, dtype=torch.int32, device=device),
        decode_slots,
        torch.tensor(decode_lens, dtype=torch.int32, device=device),
        softmax_scale=scale,
    )

    q_per_kv = heads // kv_heads
    for group, decode_len in enumerate(decode_lens):
        context = context_slots[cu_seqlens[group] : cu_seqlens[group + 1]].long()
        decode = decode_slots[group].view(-1, width)[:decode_len].T.long()
        k = torch.cat((kv[context].expand(width, -1, -1, -1), kv[decode]), dim=1)
        k = k.repeat_interleave(q_per_kv, dim=2).float()
        query = q[rows[group]].float()
        probabilities = (torch.einsum("bhd,blhd->bhl", query, k) * scale).softmax(-1)
        expected = torch.einsum("bhl,blhd->bhd", probabilities, k)
        torch.testing.assert_close(
            output[rows[group]].float(), expected, rtol=1e-2, atol=1e-2
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestSharedContextAttention(CustomTestCase):
    def test_small_kv_cache(self):
        _check_against_reference(first_slot=100)

    def test_kv_cache_past_2_31_elements(self):
        """In 32 bits the slot offsets wrap and the kernel reads outside the cache."""
        if torch.cuda.mem_get_info()[0] < 6 << 30:
            self.skipTest("needs 6 GB of free GPU memory")
        # Every slot sits past 2^31 elements: 2 KV heads x 64 dims per slot.
        _check_against_reference(first_slot=(1 << 31) // (2 * 64) + 16)


if __name__ == "__main__":
    unittest.main()
