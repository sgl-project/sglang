import unittest

import torch

from sglang.kernels.ops.attention.dsa.triton_sparse_mla_decode import (
    triton_sparse_mla_decode_splitk,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")

D_V, D_ROPE, SM_SCALE = 512, 64, 0.0625


def _reference(q_nope, q_rope, kv, indices, lengths, sink):
    bs, heads, _ = q_nope.shape
    out = torch.zeros(bs, heads, D_V, dtype=torch.float32, device=q_nope.device)
    q = torch.cat([q_nope, q_rope], dim=-1).float()
    kf = kv[:, 0].float()
    for b in range(bs):
        idx = indices[b, 0, : int(lengths[b])]
        k = kf[idx.long()]
        scores = (q[b] @ k.T) * SM_SCALE
        if sink is not None:
            scores = torch.cat([scores, sink[:, None]], dim=-1)
        p = torch.softmax(scores, dim=-1)[:, : k.shape[0]]
        out[b] = p @ k[:, :D_V]
    return out


@unittest.skipUnless(torch.cuda.is_available(), "Triton sparse MLA needs CUDA")
class TestTritonSparseMlaDecodeSink(CustomTestCase):
    def _run(self, bs, heads, topk, kv_splits, with_sink):
        torch.manual_seed(bs * 131 + heads * 7 + topk + (kv_splits or 0))
        num_slots = 4 * topk
        q_nope = torch.randn(bs, heads, D_V, device="cuda", dtype=torch.bfloat16)
        q_rope = torch.randn(bs, heads, D_ROPE, device="cuda", dtype=torch.bfloat16)
        kv = torch.randn(
            num_slots, 1, D_V + D_ROPE, device="cuda", dtype=torch.bfloat16
        )
        lengths = torch.randint(1, topk + 1, (bs,), device="cuda", dtype=torch.int32)
        lengths[0] = topk
        indices = torch.full((bs, 1, topk), -1, device="cuda", dtype=torch.int32)
        for b in range(bs):
            n = int(lengths[b])
            indices[b, 0, :n] = torch.randperm(num_slots, device="cuda")[:n].int()
        sink = (
            torch.randn(heads, device="cuda", dtype=torch.float32) * 2.0
            if with_sink
            else None
        )
        out = triton_sparse_mla_decode_splitk(
            q_nope=q_nope,
            q_rope=q_rope,
            kv=kv,
            indices=indices,
            sm_scale=SM_SCALE,
            d_v=D_V,
            kv_splits=kv_splits,
            lengths=lengths,
            attn_sink=sink,
        )
        expected = _reference(q_nope, q_rope, kv, indices, lengths, sink)
        torch.testing.assert_close(
            out.squeeze(0).float(), expected, atol=2e-2, rtol=2e-2
        )

    def test_sink_matches_reference(self):
        for bs, heads, topk in ((1, 4, 512), (7, 8, 2048), (32, 16, 1024)):
            for kv_splits in (1, None):
                with self.subTest(bs=bs, heads=heads, topk=topk, kv_splits=kv_splits):
                    self._run(bs, heads, topk, kv_splits, with_sink=True)

    def test_without_sink_unchanged(self):
        for kv_splits in (1, None):
            with self.subTest(kv_splits=kv_splits):
                self._run(5, 8, 1024, kv_splits, with_sink=False)


if __name__ == "__main__":
    unittest.main()
