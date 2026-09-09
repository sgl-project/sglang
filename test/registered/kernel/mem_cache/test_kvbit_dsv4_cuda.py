"""SM90 conformance for the experimental branch; no model or server required."""

import unittest

import torch

from sglang.srt.mem_cache.kvbit_dsv4_codec import (
    DSV4_INT4_ALIGNED_LAYOUT,
    DSV4_INT4_LAYOUT,
    decode_dsv4_int4_reference,
    encode_dsv4_int4_reference,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_SM90 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


@unittest.skipUnless(_SM90, "DSV4 INT4 conformance requires SM90")
class TestDSV4INT4CUDA(unittest.TestCase):
    def test_writer_bytes_strides_pages_and_invalid_locations(self):
        from sglang.srt.mem_cache.kvbit_dsv4 import write_dsv4_int4_packed

        generator = torch.Generator().manual_seed(42)
        source = torch.randn(6, 640, generator=generator, dtype=torch.bfloat16) * 0.1
        source[0, 128:160] = 0
        source[1, 128:160] = 1e30
        source[2, 128:160] = -1e30
        for group, scale in enumerate((1.25, 1.75, 2.5, 3.5)):
            ties = torch.tensor([7, -7, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5]).repeat(4)
            source[3, 128 + group * 32 : 160 + group * 32] = ties * scale
        kv = source.cuda()[:, 128:]
        for page_size in (2, 64, 256):
            for layout in (DSV4_INT4_LAYOUT, DSV4_INT4_ALIGNED_LAYOUT):
                for dtype in (torch.int32, torch.int64):
                    with self.subTest(
                        page_size=page_size, layout=layout.layout_id, dtype=dtype
                    ):
                        loc_storage = torch.zeros(12, dtype=dtype, device="cuda")
                        loc = loc_storage[::2]
                        loc.copy_(
                            torch.tensor(
                                [
                                    0,
                                    page_size - 1,
                                    page_size,
                                    2 * page_size - 1,
                                    -1,
                                    2 * page_size,
                                ]
                            )
                        )
                        packed = torch.full(
                            (2, page_size * layout.row_bytes),
                            0xAB,
                            dtype=torch.uint8,
                            device="cuda",
                        )
                        expected = packed.cpu().view(-1, layout.row_bytes)
                        expected[loc[:4].cpu().long()] = encode_dsv4_int4_reference(
                            source[:4, 128:], layout=layout
                        )
                        write_dsv4_int4_packed(kv, loc, packed, page_size=page_size)
                        self.assertTrue(
                            torch.equal(packed.cpu().view_as(expected), expected)
                        )
                        write_dsv4_int4_packed(
                            kv[:0], loc[:0], packed, page_size=page_size
                        )
                        self.assertTrue(
                            torch.equal(packed.cpu().view_as(expected), expected)
                        )

    def _case(
        self, layout, extra_page_size, *, batch=3, width=128, queries=1, schedule="auto"
    ):
        from sgl_kernel.flash_mla import FlashMLASchedMeta
        from sgl_kernel.kvbit_flash_mla import require_kvbit_int4_extension

        require_kvbit_int4_extension()
        generator = torch.Generator().manual_seed(42)
        q = (
            (torch.randn(batch, queries, 64, 512, generator=generator) * 0.1)
            .bfloat16()
            .cuda()
        )
        sink = torch.linspace(-5, 5, 64, device="cuda")
        arguments = dict(
            q=q,
            head_dim_v=512,
            softmax_scale=512**-0.5,
            sched_meta=FlashMLASchedMeta(),
            attn_sink=sink,
        )
        sources = []
        for extra, page_size in ((False, 256), (True, extra_page_size)):
            if page_size is None:
                continue
            kv = torch.randn(page_size * 2, 512, generator=generator).bfloat16() * 0.1
            packed = encode_dsv4_int4_reference(kv, layout=layout).cuda()
            carrier = packed.view(2, page_size, 1, layout.row_bytes)
            indices = torch.randint(
                0,
                page_size * 2,
                (batch, queries, width),
                generator=generator,
                dtype=torch.int32,
            )
            indices[..., :3] = torch.tensor([-1, -7, 2**31 - 1], dtype=torch.int32)
            lengths = torch.full((batch,), width - 3, dtype=torch.int32)
            if batch > 1:
                lengths[-1] = 0
            indices = indices.cuda()
            lengths = lengths.cuda()
            arguments.update(
                {
                    "extra_k_cache" if extra else "k_cache": carrier,
                    "extra_packed_kcache" if extra else "packed_kcache": packed,
                    "extra_indices_in_kvcache" if extra else "indices": indices,
                    "extra_topk_length" if extra else "topk_length": lengths,
                }
            )
            sources.append(
                (
                    decode_dsv4_int4_reference(packed.cpu()).float(),
                    indices.cpu(),
                    lengths.cpu(),
                )
            )
        if schedule != "auto":
            parts = max(
                torch.cuda.get_device_properties(q.device).multi_processor_count
                // queries,
                1,
            )
            metadata = torch.zeros(parts, 8, dtype=torch.int32)
            metadata[:, 0] = batch
            if schedule == "no_split":
                # The last request is sink-only and still has one SWA tile.
                metadata[0] = torch.tensor([0, batch - 1, 0, 1, 0, 0, 0, 0])
                splits = torch.arange(batch + 1, dtype=torch.int32)
            else:
                self.assertEqual(batch, 1)
                self.assertGreaterEqual(parts, 2)
                blocks = sum((int(lengths[0]) + 63) // 64 for _, _, lengths in sources)
                metadata[0] = torch.tensor([0, 0, 0, blocks // 2, 0, 1, 1, 0])
                metadata[1] = torch.tensor([0, 0, blocks // 2, blocks, 1, 1, 1, 0])
                splits = torch.tensor([0, 2], dtype=torch.int32)
            arguments["sched_meta"].tile_scheduler_metadata = metadata.cuda()
            arguments["sched_meta"].num_splits = splits.cuda()
        return arguments, sources

    def _reference(self, q, sink, sources):
        outputs, lses = [], []
        for b in range(q.shape[0]):
            query_outputs, query_lses = [], []
            for query in range(q.shape[1]):
                selected = []
                for rows, indices, lengths in sources:
                    loc = indices[b, query, : int(lengths[b])]
                    loc = loc[(loc >= 0) & (loc < rows.shape[0])].long()
                    selected.append(rows[loc])
                kv = torch.cat(selected)
                scores = q[b, query].cpu().float() @ kv.T * (512**-0.5)
                scores = torch.cat((scores, sink.cpu()[:, None]), dim=-1)
                query_outputs.append(
                    (torch.softmax(scores, -1)[:, :-1] @ kv).bfloat16()
                )
                query_lses.append(torch.logsumexp(scores, -1))
            outputs.append(torch.stack(query_outputs))
            lses.append(torch.stack(query_lses).T)
        return torch.stack(outputs).cuda(), torch.stack(lses).cuda()

    def test_aot_swa_and_compressed_sources_match_reference(self):
        from sgl_kernel.kvbit_flash_mla import kvbit_int4_flash_mla_with_kvcache

        for layout in (DSV4_INT4_LAYOUT, DSV4_INT4_ALIGNED_LAYOUT):
            for page_size in (None, 64, 2):
                for batch, width, queries, schedule in (
                    (3, 128, 1, "auto"),
                    (3, 128, 2, "no_split"),
                    (1, 1024, 1, "split"),
                ):
                    with self.subTest(
                        layout=layout.layout_id,
                        extra=page_size,
                        batch=batch,
                        width=width,
                        queries=queries,
                        schedule=schedule,
                    ):
                        args, sources = self._case(
                            layout,
                            page_size,
                            batch=batch,
                            width=width,
                            queries=queries,
                            schedule=schedule,
                        )
                        output, lse = kvbit_int4_flash_mla_with_kvcache(**args)
                        expected, expected_lse = self._reference(
                            args["q"], args["attn_sink"], sources
                        )
                        torch.testing.assert_close(
                            output, expected, atol=0.002, rtol=0.02
                        )
                        torch.testing.assert_close(
                            lse, expected_lse, atol=2e-4, rtol=2e-4
                        )

    def test_cuda_graph_replay_observes_updated_query(self):
        from sgl_kernel.kvbit_flash_mla import kvbit_int4_flash_mla_with_kvcache

        args, sources = self._case(DSV4_INT4_ALIGNED_LAYOUT, 64)
        kvbit_int4_flash_mla_with_kvcache(**args)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output, lse = kvbit_int4_flash_mla_with_kvcache(**args)
        args["q"].zero_()
        graph.replay()
        expected, expected_lse = self._reference(args["q"], args["attn_sink"], sources)
        torch.testing.assert_close(output, expected, atol=0.002, rtol=0.02)
        torch.testing.assert_close(lse, expected_lse, atol=2e-4, rtol=2e-4)

    def test_norm_rope_strided_tail_matches_contiguous_triton(self):
        from sglang.kernels.ops.attention.deepseek_v4_rope import (
            fused_norm_rope_inplace_triton,
        )

        generator = torch.Generator().manual_seed(42)
        storage = torch.randn(6, 640, generator=generator).bfloat16().cuda()
        prefix = storage[:, :128].clone()
        kv = storage[:, 128:]
        contiguous = kv.contiguous()
        angles = torch.randn(8, 32, generator=generator).cuda()
        freqs = torch.polar(torch.ones_like(angles), angles)
        positions = torch.tensor([0, 7, 1, 6, 2, 5], device="cuda")
        weight = torch.ones(512, dtype=torch.bfloat16, device="cuda")
        fused_norm_rope_inplace_triton(kv, weight, 1e-6, freqs, positions)
        fused_norm_rope_inplace_triton(contiguous, weight, 1e-6, freqs, positions)
        torch.testing.assert_close(kv, contiguous, atol=0, rtol=0)
        torch.testing.assert_close(storage[:, :128], prefix, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
