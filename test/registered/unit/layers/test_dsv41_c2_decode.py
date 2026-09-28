"""Hopper C2 dispatch preserves pending pairs and existing fused cache outputs."""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.attention.dsv4.c2_decode_pool import c2_decode_pool
from sglang.kernels.ops.attention.dsv4.fp4_indexer_rope import (
    index_k_norm_rope_pack_store,
)
from sglang.kernels.ops.attention.dsv4.low_ratio_compress import (
    c2_decode_norm_rope_store,
)
from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
from sglang.srt.layers.attention.dsv4.dsv41_sparse import (
    DeepseekV41Compressor,
    DeepseekV41Indexer,
    RMSNorm,
)
from sglang.srt.mem_cache.deepseek_v4_compress_state import (
    CompressStatePool,
    KVAndScore,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4IndexerPool,
    DeepSeekV4SingleKVPool,
    KVLayout,
)
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="base-b", runner_config="1-gpu-large")


def assert_bits(a, b):
    torch.testing.assert_close(
        a.contiguous().view(torch.uint8),
        b.contiguous().view(torch.uint8),
        rtol=0,
        atol=0,
    )


class Indexer(torch.nn.Module):
    index_keys = DeepseekV41Indexer.index_keys

    def __init__(self):
        super().__init__()
        self.wk = torch.nn.Linear(512, 128, bias=False, dtype=torch.bfloat16).cuda()
        self.k_norm = RMSNorm(128, 1e-6).cuda()
        self.owns_k, self.index_head_dim, self.rope_head_dim = True, 128, 64

    def forward_wk(self, latent):
        self.latent = latent
        return DeepseekV41Indexer.forward_wk(self, latent)


class Pool:
    def __init__(self, initial, ring):
        self.state = object.__new__(CompressStatePool)
        self.state.ring_size = ring
        self.state.kv_score_buffer = KVAndScore(initial.clone())
        self.main = DeepSeekV4SingleKVPool(
            2048,
            128,
            torch.float8_e4m3fn,
            448,
            64,
            1,
            "cuda",
            False,
            kv_layout=KVLayout.V4,
        )
        self.index = DeepSeekV4IndexerPool(
            2048, 64, torch.float8_e4m3fn, 128, 1, "cuda", False, use_fp4_indexer=True
        )
        self.index.index_k_rne = True
        self.main.kv_buffer[0].zero_()
        self.index.index_k_with_scale_buffer[0].zero_()

    def get_attention_compress_states(self, _):
        return self.state

    def get_extra_key_layout(self, _):
        return KVLayout.V4

    def get_extra_key_buffer(self, _):
        return self.main.get_key_buffer(0)

    def get_extra_key_page_size(self, _):
        return 128

    def get_index_k_with_scale_buffer(self, _):
        return self.index.index_k_with_scale_buffer[0]

    def set_index_k_fp4(self, layer_id, loc, cache_k):
        self.index.set_index_fp4(0, loc, cache_k)

    def set_extra_key_buffer_fused(self, layer_id, loc, cache_k, freqs_cis=None):
        self.main.set_key_buffer_fused(0, loc, cache_k, freqs_cis)

    def buffers(self):
        return (
            self.state.kv_score_buffer.kv_score,
            self.main.kv_buffer[0],
            self.index.index_k_with_scale_buffer[0],
        )


class Harness:
    _low_ratio_compress_fused = DeepseekV4AttnBackend._low_ratio_compress_fused
    _low_ratio_write_group = DeepseekV4AttnBackend._low_ratio_write_group
    _low_ratio_pair_partners = DeepseekV4AttnBackend._low_ratio_pair_partners
    _low_ratio_compress_torch = DeepseekV4AttnBackend._low_ratio_compress_torch
    decode = DeepseekV4AttnBackend._low_ratio_compress_decode

    def __init__(self, initial, ring):
        self.token_to_kv_pool = Pool(initial, ring)
        self.forward_metadata = SimpleNamespace(core_metadata=None)

    def metadata(self, req, pos, raw=None):
        raw = 512 + req * 256 + pos if raw is None else raw
        out = torch.where(pos % 2 == 1, raw // 2, -1)
        self.forward_metadata.core_metadata = SimpleNamespace(
            raw_out_loc=raw, c2_out_loc=out
        )


class TestDSV41C2Decode(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
            9,
            0,
        ):
            raise unittest.SkipTest("Hopper runtime dispatch requires SM90")
        torch.manual_seed(41294)
        with set_default_torch_dtype(torch.bfloat16):
            compressor = DeepseekV41Compressor(
                512, 512, 2, 1e-6, fused_compress=False
            ).cuda()
            indexer = Indexer()
        compressor.requires_grad_(False)
        indexer.requires_grad_(False)
        compressor.norm.weight.copy_(torch.rand(512, device="cuda") + 0.5)
        phase = torch.randn(128, 32, device="cuda")
        cls.layer = SimpleNamespace(
            compressor=compressor,
            indexer=indexer,
            layer_id=0,
            compress_ratio=2,
            freqs_cis=torch.polar(torch.ones_like(phase), phase),
            rope_head_dim=64,
        )
        cls.inputs = torch.randn(128, 512, device="cuda", dtype=torch.bfloat16)

    def pair(self, ring):
        initial = torch.randn(2 * ring + 1, 1024, device="cuda")
        initial[-1, :512], initial[-1, 512:] = 0, -torch.inf
        return initial, Harness(initial, ring), Harness(initial, ring)

    def reference(self, h, x, req, pos, *, fallback=False):
        pool, layer = h.token_to_kv_pool, self.layer
        core = h.forward_metadata.core_metadata
        kv, score = layer.compressor.project(x)
        state = pool.state.kv_score_buffer
        if fallback:
            pooled, group, slots = c2_decode_pool(
                kv,
                score,
                pos,
                core.raw_out_loc,
                core.c2_out_loc,
                req,
                state.kv,
                state.score,
                state.shape[0] - 1,
                ring_size=pool.state.ring_size,
            )
            h._low_ratio_write_group(layer, pooled, slots, group)
            return layer.indexer.latent
        freqs = torch.view_as_real(layer.freqs_cis).flatten(-2)
        latent = torch.zeros_like(kv, dtype=torch.bfloat16)
        c2_decode_norm_rope_store(
            torch.cat((kv, score), -1),
            state.kv_score,
            layer.compressor.norm.weight,
            pos,
            req,
            core.raw_out_loc,
            layer.compressor.norm.eps,
            freqs,
            pool.get_extra_key_buffer(0),
            page_size=128,
            ring_size=pool.state.ring_size,
            layout=KVLayout.V4,
            out=latent,
        )
        index = layer.indexer
        index_k_norm_rope_pack_store(
            index.forward_wk(latent),
            index.k_norm.weight,
            index.k_norm.eps,
            freqs,
            pos,
            core.c2_out_loc,
            pool.get_index_k_with_scale_buffer(0),
            ratio=2,
        )
        return latent

    def compare(self, a, b, *, state=True):
        start = 0 if state else 1
        for aa, bb in zip(
            a.token_to_kv_pool.buffers()[start:], b.token_to_kv_pool.buffers()[start:]
        ):
            assert_bits(aa, bb)

    def test_decode_pending_rows_and_padding(self):
        """Raw request zero is live; pad/even rows must not publish stale latents."""
        for ring in (2, 8):
            _, a, b = self.pair(ring)
            req = torch.tensor([0, 1, 0], device="cuda")
            for step in range(4):
                pos = torch.tensor([step, step + 1, 0], device="cuda")
                raw = 512 + req * 256 + pos
                raw[-1] = 0
                a.metadata(req, pos, raw)
                b.metadata(req, pos, raw)
                x = self.inputs[step : step + 3]
                expected = a.token_to_kv_pool.buffers()[0].clone()
                kv, score = self.layer.compressor.project(x)
                even = (pos % 2 == 0) & (raw != 0)
                expected[req[even] * ring + pos[even] % ring] = torch.cat(
                    (kv, score), -1
                )[even]
                a.decode(self.layer, x, req, pos)
                actual = self.layer.indexer.latent.clone()
                assert_bits(actual, self.reference(b, x, req, pos))
                assert_bits(
                    actual[~((pos % 2 == 1) & (raw != 0))],
                    torch.zeros_like(actual[~((pos % 2 == 1) & (raw != 0))]),
                )
                assert_bits(a.token_to_kv_pool.buffers()[0], expected)
                self.compare(a, b)
            # A fallback prefill chunk can follow decode and leave the next pair pending.
            pos = torch.arange(4, 7, device="cuda")
            req = torch.zeros_like(pos)
            for h in (a, b):
                h.metadata(req, pos)
                h._low_ratio_compress_torch(self.layer, self.inputs[4:7], req, pos)
            pos, req = pos[-1:] + 1, req[-1:]
            a.metadata(req, pos)
            b.metadata(req, pos)
            a.decode(self.layer, self.inputs[7:8], req, pos)
            actual = self.layer.indexer.latent.clone()
            assert_bits(actual, self.reference(b, self.inputs[7:8], req, pos))
            # Fallback incomplete rows share sink slot 0; only live pages matter.
            for i, (aa, bb) in enumerate(
                zip(a.token_to_kv_pool.buffers(), b.token_to_kv_pool.buffers())
            ):
                assert_bits(aa if i == 0 else aa[1:], bb if i == 0 else bb[1:])

    def test_verify_rollback_preserves_pending_even_predecessor(self):
        """Rejected future rows cannot replace a retained pair's even predecessor."""
        for start in (7, 8):
            for kept in (0, 1, 6):
                with self.subTest(start=start, kept=kept):
                    _, a, b = self.pair(8)
                    for h, stop in ((a, start + 6), (b, start + kept)):
                        pos = torch.arange(stop, device="cuda")
                        req = torch.zeros_like(pos)
                        h.metadata(req, pos)
                        h._low_ratio_compress_torch(
                            self.layer, self.inputs[:stop], req, pos
                        )
                    for p in range(start + kept, start + kept + 4):
                        pos, req = (
                            torch.tensor([p], device="cuda"),
                            torch.zeros(1, device="cuda", dtype=torch.int64),
                        )
                        x = self.inputs[64 + p : 65 + p]
                        # Isolate this step's writes; rejected future slots are not visible.
                        for h in (a, b):
                            h.metadata(req, pos)
                            for cache in h.token_to_kv_pool.buffers()[1:]:
                                cache.zero_()
                        if p % 2:
                            assert_bits(
                                a.token_to_kv_pool.buffers()[0][(p - 1) % 8],
                                b.token_to_kv_pool.buffers()[0][(p - 1) % 8],
                            )
                        expected = a.token_to_kv_pool.buffers()[0].clone()
                        if p % 2 == 0:
                            expected[p % 8] = torch.cat(
                                self.layer.compressor.project(x), -1
                            )[0]
                        a.decode(self.layer, x, req, pos)
                        actual = self.layer.indexer.latent.clone()
                        assert_bits(actual, self.reference(b, x, req, pos))
                        assert_bits(a.token_to_kv_pool.buffers()[0], expected)
                        self.compare(a, b, state=False)

    def test_graph_replay_refreshes_metadata_and_zero_latents(self):
        """In-place metadata changes must reach captured C2 and index writers."""
        initial, a, b = self.pair(8)
        req = torch.tensor([0, 1, 0], device="cuda")
        pos = torch.tensor([7, 8, 0], device="cuda")
        raw = torch.tensor([519, 776, 0], device="cuda")
        x = self.inputs[:3].clone()
        a.metadata(req, pos, raw)
        core = a.forward_metadata.core_metadata
        for _ in range(3):
            a.decode(self.layer, x, req, pos)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            a.decode(self.layer, x, req, pos)
        captured = self.layer.indexer.latent
        for positions, locations in (
            ([8, 9, 0], [520, 777, 0]),
            ([9, 10, 0], [521, 0, 0]),
        ):
            pos.copy_(torch.tensor(positions, device="cuda"))
            raw.copy_(torch.tensor(locations, device="cuda"))
            core.c2_out_loc.copy_(torch.where(pos % 2 == 1, raw // 2, -1))
            for h in (a, b):
                h.token_to_kv_pool.buffers()[0].copy_(initial)
                for cache in h.token_to_kv_pool.buffers()[1:]:
                    cache.zero_()
            graph.replay()
            b.metadata(req, pos, raw)
            assert_bits(captured, self.reference(b, x, req, pos))
            self.compare(a, b)

    def test_unsupported_norm_weights_preserve_original_fallback(self):
        """BF16 vector loads cannot consume FP32, strided or two-byte-offset weights."""
        for norm in (self.layer.compressor.norm, self.layer.indexer.k_norm):
            original = norm.weight
            dim = original.numel()
            weights = (
                original.float(),
                torch.empty(2 * dim, device="cuda", dtype=torch.bfloat16)[::2],
                torch.empty(dim + 1, device="cuda", dtype=torch.bfloat16)[1:],
            )
            self.assertFalse(weights[1].is_contiguous())
            self.assertEqual(weights[2].data_ptr() % 4, 2)
            try:
                for weight in weights:
                    weight.copy_(original)
                    norm.weight = torch.nn.Parameter(weight, requires_grad=False)
                    _, a, b = self.pair(8)

                    def reject_fused(*args, **kwargs):
                        self.fail("unsupported norm weight entered fused C2")

                    a._low_ratio_compress_fused = reject_fused
                    pos, req = (
                        torch.tensor([7], device="cuda"),
                        torch.zeros(1, device="cuda", dtype=torch.int64),
                    )
                    a.metadata(req, pos)
                    b.metadata(req, pos)
                    a.decode(self.layer, self.inputs[:1], req, pos)
                    actual = self.layer.indexer.latent.clone()
                    assert_bits(
                        actual,
                        self.reference(b, self.inputs[:1], req, pos, fallback=True),
                    )
                    self.compare(a, b)
            finally:
                norm.weight = original

    def test_default_fused_projection_matches_supplied_input(self):
        with set_default_torch_dtype(torch.bfloat16):
            compressor = DeepseekV41Compressor(
                512, 512, 2, 1e-6, fused_compress=True
            ).cuda()
        compressor.requires_grad_(False)
        layer = SimpleNamespace(**{**vars(self.layer), "compressor": compressor})
        _, a, b = self.pair(8)
        pos, req = (
            torch.tensor([7, 9], device="cuda"),
            torch.tensor([0, 1], device="cuda"),
        )
        a.metadata(req, pos)
        b.metadata(req, pos)
        x = self.inputs[:2]
        packed = compressor.project_fused(x)
        a._low_ratio_compress_fused(layer, x, req, pos)
        actual = layer.indexer.latent.clone()
        b._low_ratio_compress_fused(layer, x, req, pos, kv_score_input=packed)
        assert_bits(actual, layer.indexer.latent)
        self.compare(a, b)


if __name__ == "__main__":
    unittest.main()
