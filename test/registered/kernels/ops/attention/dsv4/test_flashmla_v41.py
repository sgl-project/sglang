import unittest

import torch

from sglang.kernels.ops.attention.dsv4 import flashmla_v41
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.deepseek_v4.dsv41_kv_quant_reference import (
    dequantize_k_cache_v41,
    dequantize_k_cache_v41_fp4,
    quantize_k_cache_v41,
    quantize_k_cache_v41_fp4,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

D = 512


def paged_view(raw, page_size, bytes_per_token):
    return raw[:, : page_size * bytes_per_token].view(
        raw.shape[0], page_size, 1, bytes_per_token
    )


def unpack_output(data, scale):
    tokens, groups, hidden = data.shape
    sf = scale.contiguous().view(torch.uint8).float()
    values = data.float().view(tokens, groups, hidden // 32, 32)
    return (values * torch.exp2(sf - 127).unsqueeze(-1)).view(tokens, groups, hidden)


def to_page_split(raw, page_size, data_bytes, scale_bytes, page_align):
    bpt = data_bytes + scale_bytes
    tokens = raw[:, : page_size * bpt].view(raw.shape[0], page_size, bpt)
    width = -(-page_size * bpt // page_align) * page_align
    out = torch.zeros(raw.shape[0], width, dtype=torch.uint8, device=raw.device)
    out[:, : page_size * data_bytes] = tokens[..., :data_bytes].reshape(
        raw.shape[0], -1
    )
    out[:, page_size * data_bytes : page_size * bpt] = tokens[..., data_bytes:].reshape(
        raw.shape[0], -1
    )
    return out


def reference_attention(q, k_rows, valid, sink, sm_scale):
    logits = torch.einsum("nhd,ntd->nht", q.float(), k_rows.float()) * sm_scale
    logits = logits.masked_fill(~valid[:, None, :], float("-inf"))
    m = torch.maximum(logits.amax(-1), sink[None, :])
    p = torch.exp(logits - m[..., None])
    denom = p.sum(-1) + torch.exp(sink[None, :] - m)
    return torch.einsum("nht,ntd->nhd", p, k_rows.float()) / denom[..., None]


class Case:
    def __init__(self, n, topk, extra_topk, seed, page=64, extra_page=64):
        g = torch.Generator(device="cuda").manual_seed(seed)
        dev = "cuda"
        self.n, self.topk, self.extra_topk = n, topk, extra_topk
        self.page, self.extra_page = page, extra_page
        main_pages, extra_pages = 6, 9
        k = torch.randn(main_pages, page, D, generator=g, device=dev) / 4
        ek = torch.randn(extra_pages, extra_page, D, generator=g, device=dev) / 4
        self.main_raw = quantize_k_cache_v41(k.bfloat16())
        self.extra_raw = quantize_k_cache_v41_fp4(ek.bfloat16())
        self.main = paged_view(self.main_raw, page, 528)
        self.extra = paged_view(self.extra_raw, extra_page, 288)
        self.main_k = dequantize_k_cache_v41(self.main_raw, page).view(-1, D)
        self.extra_k = dequantize_k_cache_v41_fp4(self.extra_raw, extra_page).view(
            -1, D
        )
        self.indices = torch.randint(
            0, main_pages * page, (n, topk), generator=g, device=dev, dtype=torch.int32
        )
        self.extra_indices = torch.randint(
            0,
            extra_pages * extra_page,
            (n, extra_topk),
            generator=g,
            device=dev,
            dtype=torch.int32,
        )
        self.lengths = torch.randint(
            1, topk + 1, (n,), generator=g, device=dev, dtype=torch.int32
        )
        self.extra_lengths = torch.randint(
            0, extra_topk + 1, (n,), generator=g, device=dev, dtype=torch.int32
        )
        self.extra_indices[0, 0] = -1
        self.sink = torch.randn(64, generator=g, device=dev)

    def reference(self, q, sm_scale):
        pos = torch.arange(self.topk, device="cuda")
        epos = torch.arange(self.extra_topk, device="cuda")
        valid = pos[None] < self.lengths[:, None]
        evalid = (epos[None] < self.extra_lengths[:, None]) & (self.extra_indices >= 0)
        k_rows = torch.cat(
            [
                self.main_k[self.indices.clamp_min(0).long()],
                self.extra_k[self.extra_indices.clamp_min(0).long()],
            ],
            dim=1,
        )
        return reference_attention(
            q, k_rows, torch.cat([valid, evalid], dim=1), self.sink, sm_scale
        )


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] == 10
    and flashmla_v41.is_available(),
    "flashmla_v41_ops needs SM100/SM103 and the built library",
)
class TestFlashMLAV41(CustomTestCase):
    SM_SCALE = D**-0.5

    def test_sparse_decode_matches_reference(self):
        for n, topk, extra_topk in ((5, 128, 512), (64, 128, 64), (3, 256, 0)):
            with self.subTest(n=n, topk=topk, extra_topk=extra_topk):
                c = Case(n, topk, max(extra_topk, 64), seed=n + topk)
                q = (torch.randn(n, 64, D, device="cuda") / 2).bfloat16()
                kw = dict(topk_length=c.lengths, attn_sink=c.sink)
                if extra_topk:
                    kw.update(
                        extra_kv=c.extra,
                        extra_indices=c.extra_indices[:, None],
                        extra_topk_length=c.extra_lengths,
                    )
                else:
                    c.extra_lengths.zero_()
                out, _ = flashmla_v41.sparse_decode(
                    q[:, None], c.main, c.indices[:, None], self.SM_SCALE, **kw
                )
                ref = c.reference(q, self.SM_SCALE)
                torch.testing.assert_close(out[:, 0].float(), ref, atol=2e-2, rtol=2e-2)

    def test_sparse_decode_matches_page_split_kernel(self):
        from sgl_kernel.flash_mla import flash_mla_with_kvcache, get_mla_metadata

        n = 32
        c = Case(n, 128, 512, seed=3)
        q = (torch.randn(n, 64, D, device="cuda") / 2).bfloat16()
        out, _ = flashmla_v41.sparse_decode(
            q[:, None],
            c.main,
            c.indices[:, None],
            self.SM_SCALE,
            topk_length=c.lengths,
            attn_sink=c.sink,
            extra_kv=c.extra,
            extra_indices=c.extra_indices[:, None],
            extra_topk_length=c.extra_lengths,
        )
        old_main = paged_view(
            to_page_split(c.main_raw, c.page, 512, 16, 512), c.page, 528
        )
        old_extra = paged_view(
            to_page_split(c.extra_raw, c.extra_page, 256, 32, 256), c.extra_page, 288
        )
        old = flash_mla_with_kvcache(
            q=q[:, None],
            k_cache=old_main,
            head_dim_v=D,
            block_table=None,
            cache_seqlens=None,
            tile_scheduler_metadata=get_mla_metadata()[0],
            softmax_scale=self.SM_SCALE,
            is_fp8_kvcache=True,
            indices=c.indices[:, None],
            topk_length=c.lengths,
            attn_sink=c.sink,
            extra_k_cache=old_extra,
            extra_indices_in_kvcache=c.extra_indices[:, None],
            extra_topk_length=c.extra_lengths,
        )[0]
        torch.testing.assert_close(out.float(), old.float(), atol=1e-2, rtol=1e-2)

    def test_mega_decode_matches_page_split_kernel(self):
        from sgl_kernel.flash_mla_mega import flash_mla_mega_decode

        from sglang.srt.layers.attention.deepseek_v4_mega_backend import pack_query

        n = 48
        c = Case(n, 128, 512, seed=5, page=256)
        q = pack_query((torch.randn(n, 64, D, device="cuda") / 2).bfloat16(), 64)
        positions = torch.randint(0, 4096, (n,), device="cuda", dtype=torch.int32)
        angle = (
            torch.arange(4096, device="cuda")[:, None]
            / torch.arange(1, 33, device="cuda")[None]
        )
        table = torch.cat((angle.cos(), angle.sin()), dim=-1).float()
        args = (positions, table, self.SM_SCALE)
        kw = dict(
            attn_sink=c.sink,
            topk_length=c.lengths,
            extra_indices=c.extra_indices,
            extra_topk_length=c.extra_lengths,
        )
        new = flashmla_v41.mega_decode(
            q, c.main, c.indices, *args, extra_kv=c.extra, **kw
        )
        old_main = paged_view(
            to_page_split(c.main_raw, c.page, 512, 16, 512), c.page, 528
        )
        old_extra = paged_view(
            to_page_split(c.extra_raw, c.extra_page, 256, 32, 256), c.extra_page, 288
        )
        old = flash_mla_mega_decode(
            q, old_main, c.indices, *args, extra_kv=old_extra, **kw
        )
        torch.testing.assert_close(new[2], old[2], atol=1e-3, rtol=1e-3)
        got, want = unpack_output(new[0], new[1]), unpack_output(old[0], old[1])
        rel = (got - want).norm() / want.norm()
        self.assertLess(rel.item(), 1e-2)


if __name__ == "__main__":
    unittest.main()
