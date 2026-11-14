"""The current FlashMLA V4.1 kernels on the interleaved caches against the
legacy block-split kernels on the same tokens, and the schedule handling sglang
adds around them (precomputed metadata, reuse across layers, CUDA graphs)."""

import unittest

import torch

from sglang.kernels.ops.attention import flash_mla_interface
from sglang.kernels.ops.attention.flash_mla_interface import SPLIT_KV_MIN_TOPK
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.deepseek_v4.dsv41_kv_quant_reference import (
    dequantize_k_cache_v41,
    dequantize_k_cache_v41_fp4,
    interleave_pages,
    quantize_k_cache_v41,
    quantize_k_cache_v41_fp4,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

D = 512
H = 64
SM_SCALE = D**-0.5


def paged_view(raw, page_size, bytes_per_token):
    return raw[:, : page_size * bytes_per_token].view(
        raw.shape[0], page_size, 1, bytes_per_token
    )


def unpack_mega_output(data, scale):
    tokens, groups, hidden = data.shape
    sf = scale.contiguous().view(torch.uint8).float()
    values = data.float().view(tokens, groups, hidden // 32, 32)
    return (values * torch.exp2(sf - 127).unsqueeze(-1)).view(tokens, groups, hidden)


def reference_attention(q, k_rows, valid, sink):
    logits = torch.einsum("nhd,ntd->nht", q.float(), k_rows.float()) * SM_SCALE
    logits = logits.masked_fill(~valid[:, None, :], float("-inf"))
    m = torch.maximum(logits.amax(-1), sink[None, :])
    p = torch.exp(logits - m[..., None])
    denom = p.sum(-1) + torch.exp(sink[None, :] - m)
    return torch.einsum("nht,ntd->nhd", p, k_rows.float()) / denom[..., None]


class Case:
    """An fp8 main cache and an fp4 extra cache, block-split and interleaved."""

    def __init__(self, n, topk, extra_topk, seed, page=64, extra_page=64):
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.n, self.topk, self.extra_topk = n, topk, extra_topk
        main_pages, extra_pages = 6, 9
        k = torch.randn(main_pages, page, D, generator=g, device="cuda") / 4
        ek = torch.randn(extra_pages, extra_page, D, generator=g, device="cuda") / 4
        main_split = quantize_k_cache_v41(k.bfloat16())
        extra_split = quantize_k_cache_v41_fp4(ek.bfloat16())
        self.main_k = dequantize_k_cache_v41(main_split, page).view(-1, D)
        self.extra_k = dequantize_k_cache_v41_fp4(extra_split, extra_page).view(-1, D)
        self.split = (
            paged_view(main_split, page, 528),
            paged_view(extra_split, extra_page, 288),
        )
        self.interleaved = (
            paged_view(interleave_pages(main_split, page, 512, 16), page, 528),
            paged_view(
                interleave_pages(extra_split, extra_page, 256, 32), extra_page, 288
            ),
        )
        self.indices = torch.randint(
            0,
            main_pages * page,
            (n, topk),
            generator=g,
            device="cuda",
            dtype=torch.int32,
        )
        self.extra_indices = torch.randint(
            0,
            extra_pages * extra_page,
            (n, extra_topk),
            generator=g,
            device="cuda",
            dtype=torch.int32,
        )
        self.extra_indices[0, 0] = -1
        self.lengths = torch.randint(
            1, topk + 1, (n,), generator=g, device="cuda", dtype=torch.int32
        )
        self.extra_lengths = torch.randint(
            0, extra_topk + 1, (n,), generator=g, device="cuda", dtype=torch.int32
        )
        self.sink = torch.randn(H, generator=g, device="cuda")
        self.q = (torch.randn(n, H, D, generator=g, device="cuda") / 2).bfloat16()

    def reference(self):
        valid = torch.arange(self.topk, device="cuda")[None] < self.lengths[:, None]
        evalid = (
            torch.arange(self.extra_topk, device="cuda")[None]
            < self.extra_lengths[:, None]
        ) & (self.extra_indices >= 0)
        k_rows = torch.cat(
            [
                self.main_k[self.indices.long()],
                self.extra_k[self.extra_indices.clamp_min(0).long()],
            ],
            dim=1,
        )
        return reference_attention(
            self.q, k_rows, torch.cat([valid, evalid], dim=1), self.sink
        )

    def decode(self, sched_meta=None, *, use_legacy=False):
        main, extra = self.split if use_legacy else self.interleaved
        return flash_mla_interface.flash_mla_with_kvcache(
            q=self.q[:, None],
            k_cache=main,
            head_dim_v=D,
            block_table=None,
            cache_seqlens=None,
            tile_scheduler_metadata=(
                flash_mla_interface.get_mla_metadata()[0]
                if sched_meta is None
                else sched_meta
            ),
            softmax_scale=SM_SCALE,
            is_fp8_kvcache=True,
            indices=self.indices[:, None],
            topk_length=self.lengths,
            attn_sink=self.sink,
            extra_k_cache=extra,
            extra_indices_in_kvcache=self.extra_indices[:, None],
            extra_topk_length=self.extra_lengths,
            use_legacy=use_legacy,
        )[0][:, 0]


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
    "the current FlashMLA V4.1 kernels need SM100/SM103",
)
class TestFlashMLAV41(CustomTestCase):
    # The kernel splits the KV only past 640 indices: the first three shapes
    # run unsplit, the rest take the split-KV schedule.
    SHAPES = (
        (5, 128, 512),
        (64, 128, 64),
        (1, 128, 128),
        (16, 128, 1024),
        (32, 512, 512),
        (1, 128, 1024),
    )
    SPLIT = (16, 128, 1024)

    def test_sparse_decode_matches_reference(self):
        for n, topk, extra_topk in self.SHAPES:
            with self.subTest(n=n, topk=topk, extra_topk=extra_topk):
                c = Case(n, topk, extra_topk, seed=n + topk)
                torch.testing.assert_close(
                    c.decode().float(), c.reference(), atol=2e-2, rtol=2e-2
                )

    def test_sparse_decode_matches_block_split_kernel(self):
        for n, topk, extra_topk in self.SHAPES:
            with self.subTest(n=n, topk=topk, extra_topk=extra_topk):
                c = Case(n, topk, extra_topk, seed=7 * n + topk)
                torch.testing.assert_close(
                    c.decode(), c.decode(use_legacy=True), atol=1e-2, rtol=1e-2
                )

    def test_precomputed_schedule_matches_own_schedule(self):
        """The backend's `flashmla_sched_meta` precompute feeds the new kernel."""
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            _maybe_precompute_flashmla_sched_meta,
        )

        for n, topk, extra_topk in self.SHAPES:
            with self.subTest(n=n, topk=topk, extra_topk=extra_topk):
                c = Case(n, topk, extra_topk, seed=11 * n + topk)
                meta = flash_mla_interface.get_mla_metadata()[0]
                _maybe_precompute_flashmla_sched_meta(
                    meta,
                    q=c.q[:, None],
                    indices=c.indices[:, None],
                    topk_length=c.lengths,
                    extra_indices=c.extra_indices[:, None],
                    extra_topk_length=c.extra_lengths,
                )
                self.assertIsNotNone(meta.tile_scheduler_metadata)
                self.assertTrue(torch.equal(c.decode(meta), c.decode()))

    def test_backend_precomputes_only_split_schedules(self):
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            _flashmla_v41_splits_kv,
        )

        for n, topk, extra_topk in self.SHAPES:
            with self.subTest(n=n, topk=topk, extra_topk=extra_topk):
                c = Case(n, topk, extra_topk, seed=n)
                meta = flash_mla_interface.get_mla_metadata()[0]
                c.decode(meta)
                self.assertEqual(
                    meta.tile_scheduler_metadata is not None,
                    _flashmla_v41_splits_kv(
                        c.indices, c.extra_indices, SPLIT_KV_MIN_TOPK
                    ),
                )

    def test_schedule_is_shared_across_layers(self):
        """The first call fills the backend's metadata object and later layers reuse it."""
        c = Case(*self.SPLIT, seed=13)
        meta = flash_mla_interface.get_mla_metadata()[0]
        first = c.decode(meta)
        self.assertTrue(meta.have_initialized)
        schedule = meta.tile_scheduler_metadata
        self.assertIsNotNone(schedule)
        second = c.decode(meta)
        self.assertIs(meta.tile_scheduler_metadata, schedule)
        self.assertTrue(torch.equal(first, second))

    def test_cuda_graph_replay(self):
        """A graph captured with an empty schedule recomputes it on replay, for
        new topk lengths, the way the backend's captured decode does."""
        for shape in ((8, 128, 512), (8, 128, 1024)):
            with self.subTest(shape=shape):
                c = Case(*shape, seed=17)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    c.decode()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out = c.decode(flash_mla_interface.get_mla_metadata()[0])
                fresh = Case(*shape, seed=19)
                for name in (
                    "q",
                    "indices",
                    "extra_indices",
                    "lengths",
                    "extra_lengths",
                ):
                    getattr(c, name).copy_(getattr(fresh, name))
                graph.replay()
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(out, c.decode()))

    def test_mega_decode_matches_block_split_kernel(self):
        from sglang.srt.layers.attention.deepseek_v4_mega_backend import pack_query

        for page in (64, 256):
            with self.subTest(page=page):
                c = Case(48, 128, 512, seed=5 + page, page=page)
                q = pack_query(c.q, H)
                positions = torch.randint(
                    0, 4096, (c.n,), device="cuda", dtype=torch.int32
                )
                angle = (
                    torch.arange(4096, device="cuda")[:, None]
                    / torch.arange(1, 33, device="cuda")[None]
                )
                table = torch.cat((angle.cos(), angle.sin()), dim=-1).float()
                kw = dict(
                    attn_sink=c.sink,
                    topk_length=c.lengths,
                    extra_indices=c.extra_indices,
                    extra_topk_length=c.extra_lengths,
                )
                new = flash_mla_interface.flash_mla_mega_decode(
                    q,
                    c.interleaved[0],
                    c.indices,
                    positions,
                    table,
                    SM_SCALE,
                    extra_kv=c.interleaved[1],
                    use_legacy=False,
                    **kw,
                )
                old = flash_mla_interface.flash_mla_mega_decode(
                    q,
                    c.split[0],
                    c.indices,
                    positions,
                    table,
                    SM_SCALE,
                    extra_kv=c.split[1],
                    **kw,
                )
                torch.testing.assert_close(new[2], old[2], atol=1e-3, rtol=1e-3)
                got = unpack_mega_output(new[0], new[1])
                want = unpack_mega_output(old[0], old[1])
                rel = (got - want).norm() / want.norm()
                self.assertLess(rel.item(), 1e-2)


if __name__ == "__main__":
    unittest.main()
