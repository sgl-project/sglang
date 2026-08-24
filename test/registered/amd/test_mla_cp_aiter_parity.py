"""Aiter absorbed-MLA prefill parity for the CP zigzag geometry."""

import unittest

import torch

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd-mi35x")

KV_LORA_RANK = 512
ROPE_DIM = 64
HEAD_DIM = KV_LORA_RANK + ROPE_DIM
DEVICE = "cuda"
DTYPE = torch.bfloat16


def _aiter_mla_available():
    if not is_hip() or not torch.cuda.is_available():
        return False
    try:
        from aiter.mla import mla_prefill_fwd  # noqa: F401
    except Exception:
        return False
    return True


def _run_mla(q, kv, q_len, kv_len, num_heads, scaling):
    from aiter.mla import mla_prefill_fwd

    output = torch.empty(q_len, num_heads, KV_LORA_RANK, dtype=DTYPE, device=DEVICE)
    mla_prefill_fwd(
        q.contiguous(),
        kv.view(-1, 1, 1, HEAD_DIM),
        output,
        torch.tensor([0, q_len], dtype=torch.int32, device=DEVICE),
        torch.tensor([0, kv_len], dtype=torch.int32, device=DEVICE),
        torch.arange(kv_len, dtype=torch.int32, device=DEVICE),
        torch.ones(1, dtype=torch.int32, device=DEVICE),
        q_len,
        scaling,
        0.0,
    )
    return output


@unittest.skipUnless(_aiter_mla_available(), "requires ROCm + aiter MLA prefill")
class TestAiterMlaPrefillCPParity(CustomTestCase):
    def test_zigzag_halves_match_full_prefill(self):
        torch.manual_seed(7)
        prefix_len = 8
        extend_len = 64
        cp_size = 2
        block_count = 2 * cp_size
        block_len = extend_len // block_count
        scaling = HEAD_DIM**-0.5

        for num_heads in (16, 128):
            with self.subTest(num_heads=num_heads):
                kv = (
                    torch.randn(
                        prefix_len + extend_len,
                        1,
                        HEAD_DIM,
                        dtype=DTYPE,
                        device=DEVICE,
                    )
                    * 0.2
                )
                q = (
                    torch.randn(
                        extend_len,
                        num_heads,
                        HEAD_DIM,
                        dtype=DTYPE,
                        device=DEVICE,
                    )
                    * 0.2
                )
                reference = _run_mla(
                    q,
                    kv,
                    q_len=extend_len,
                    kv_len=prefix_len + extend_len,
                    num_heads=num_heads,
                    scaling=scaling,
                )

                for rank in range(cp_size):
                    block_ids = (rank, block_count - rank - 1)
                    for block_id in block_ids:
                        start = block_id * block_len
                        end = start + block_len
                        kv_len = prefix_len + end
                        actual = _run_mla(
                            q[start:end],
                            kv,
                            q_len=block_len,
                            kv_len=kv_len,
                            num_heads=num_heads,
                            scaling=scaling,
                        )
                        torch.testing.assert_close(
                            actual.float(),
                            reference[start:end].float(),
                            atol=6e-3,
                            rtol=2e-3,
                        )


if __name__ == "__main__":
    unittest.main()
