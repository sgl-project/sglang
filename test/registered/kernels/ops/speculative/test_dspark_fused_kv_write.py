from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-small")

import unittest

import torch

from sglang.kernels.ops.speculative.dspark.fused_kv_write import (
    fused_kv_norm_rope_write,
)
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.test.test_utils import CustomTestCase

DEVICE = torch.device("cuda")
LAYERS, HEADS, HEAD_DIM, POOL, MAX_POS = 3, 2, 128, 64, 4096
KV = HEADS * HEAD_DIM
EPS = 1e-6


class TestFusedKvNormRopeWrite(CustomTestCase):
    def _run(self, *, rotary_dim, is_neox_style):
        g = torch.Generator(device=DEVICE).manual_seed(rotary_dim)
        bs, width = 3, 7
        rows = bs * width
        rope = get_rope(
            HEAD_DIM,
            rotary_dim=rotary_dim,
            max_position=MAX_POS,
            base=10_000_000,
            is_neox_style=is_neox_style,
        ).to(DEVICE)
        norms = [GemmaRMSNorm(HEAD_DIM, eps=EPS).to(DEVICE) for _ in range(LAYERS)]
        for norm in norms:
            norm.weight.data.normal_(0, 0.1, generator=g)
        kv = torch.randn(
            rows, LAYERS * 2 * KV, device=DEVICE, dtype=torch.bfloat16, generator=g
        )
        positions = torch.randint(0, MAX_POS, (rows,), device=DEVICE, generator=g)
        locs = torch.randperm(POOL, device=DEVICE, generator=g)[:rows]
        commit_lens = torch.tensor([width, 3, 0], device=DEVICE, dtype=torch.int32)
        k_bufs = [
            torch.zeros(POOL, HEADS, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
            for _ in range(LAYERS)
        ]
        v_bufs = [torch.zeros_like(k) for k in k_bufs]
        meta = torch.tensor(
            [
                [k.data_ptr(), v.data_ptr(), k.stride(0), v.stride(0)]
                for k, v in zip(k_bufs, v_bufs)
            ],
            device=DEVICE,
            dtype=torch.int64,
        )
        fused_kv_norm_rope_write(
            kv,
            meta,
            torch.stack([1.0 + n.weight.float() for n in norms]),
            rope.cos_sin_cache,
            positions,
            locs,
            LAYERS,
            KV,
            HEAD_DIM,
            EPS,
            commit_lens=commit_lens,
            locs_row_width=width,
            is_neox_style=is_neox_style,
            rotary_dim=rotary_dim,
        )

        committed = (
            torch.arange(width, device=DEVICE)[None] < commit_lens[:, None]
        ).reshape(-1)
        view = kv.view(rows, LAYERS, 2, HEADS, HEAD_DIM)
        for layer in range(LAYERS):
            k = norms[layer].forward_native(view[:, layer, 0].reshape(-1, HEAD_DIM))
            dummy_q = torch.zeros_like(k.view(rows, KV))
            _, k = rope.forward_native(positions, dummy_q, k.view(rows, KV))
            k = k.view(rows, HEADS, HEAD_DIM)
            got_k, got_v = k_bufs[layer][locs], v_bufs[layer][locs]
            torch.testing.assert_close(
                got_k[committed], k[committed], atol=1.6e-2, rtol=1.6e-2
            )
            self.assertTrue(torch.equal(got_v[committed], view[committed, layer, 1]))
            # rows past a request's commit length leave the pool untouched
            self.assertEqual(got_k[~committed].abs().sum().item(), 0)
            self.assertEqual(got_v[~committed].abs().sum().item(), 0)

    def test_partial_rope_gemma_norm(self):
        self._run(rotary_dim=64, is_neox_style=True)

    def test_full_rope_interleaved(self):
        self._run(rotary_dim=HEAD_DIM, is_neox_style=False)


if __name__ == "__main__":
    unittest.main()
