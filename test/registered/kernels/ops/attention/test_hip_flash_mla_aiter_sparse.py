"""The ``aiter_sparse`` ROCm attention backend and its split-KV combine must match the torch reference and aiter's own reduce on the served packed fp8 KV layout, bitwise repeatable and batch-invariant."""

import math
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")


NOPE, ROPE, D = 448, 64, 512
PAGE = 256
BYTES = 584
SCALE = D**-0.5


# Relative tolerance for the short-list case: about 3x the measured aiter error
# (~3e-3) and 40x under what attending one stray key on a 5-key list costs (~0.4).
TOL_SHORT = 1e-2


def _pack_cache(num_blocks, device, gen, *, fp8_view=True):
    """Random bf16 keys in the packed fp8 layout: cache [num_blocks, PAGE, 1, BYTES] and the dequantized keys [slots, D] fp32."""
    slots = num_blocks * PAGE
    k = torch.randn(slots, D, generator=gen) * 0.5
    nope = k[:, :NOPE].reshape(slots, NOPE // 64, 64)
    amax = nope.abs().amax(-1, keepdim=True).clamp(min=1e-6)
    exp = torch.ceil(torch.log2(amax / 448.0)).clamp(min=-127, max=127)
    scale = torch.pow(2.0, exp)
    nope_fp8 = (nope / scale).to(torch.float8_e4m3fn)
    nope_deq = nope_fp8.float() * scale
    rope = k[:, NOPE:].to(torch.bfloat16)
    raw = torch.zeros(num_blocks, PAGE * BYTES, dtype=torch.uint8)
    data = raw[:, : PAGE * 576].view(num_blocks, PAGE, 576)
    data[:, :, :NOPE] = nope_fp8.view(torch.uint8).reshape(num_blocks, PAGE, NOPE)
    data[:, :, NOPE:] = rope.view(torch.uint8).reshape(num_blocks, PAGE, 2 * ROPE)
    scales = raw[:, PAGE * 576 :].view(num_blocks, PAGE, 8)
    scales[:, :, :7] = (exp.reshape(num_blocks, PAGE, 7) + 127).to(torch.uint8)
    cache = raw.view(num_blocks, PAGE, 1, BYTES)
    if fp8_view:
        cache = cache.view(torch.float8_e4m3fn)
    deq = torch.cat([nope_deq.reshape(slots, NOPE), rope.float()], dim=1).to(device)
    return cache.to(device), deq


def _masked(indices, lengths):
    """One length-folded list; ``_fold_lengths_into_index_lists`` returns the (main, extra) pair."""
    from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
        _fold_lengths_into_index_lists,
    )

    return _fold_lengths_into_index_lists(indices, lengths)[0]


def _decode_case(batch, heads, gen, dev):
    """Two SWA and five top-k pages of packed keys, a query per row, the sink, and one
    random 128-slot SWA list plus one 512-slot top-k list per row ([b, 1, w] int32)."""
    swa_cache, swa_deq = _pack_cache(2, dev, gen)
    topk_cache, topk_deq = _pack_cache(5, dev, gen)
    q = (torch.randn(batch, 1, heads, D, generator=gen) * 0.5).to(torch.bfloat16)
    sink = (torch.randn(heads, generator=gen) * 0.5).to(dev)
    swa_idx = torch.stack(
        [torch.randperm(2 * PAGE, generator=gen)[:128] for _ in range(batch)]
    )
    topk_idx = torch.stack(
        [torch.randperm(5 * PAGE, generator=gen)[:512] for _ in range(batch)]
    )
    return SimpleNamespace(
        swa_cache=swa_cache,
        swa_deq=swa_deq,
        topk_cache=topk_cache,
        topk_deq=topk_deq,
        q=q.to(dev),
        sink=sink,
        swa_idx=swa_idx.to(torch.int32).unsqueeze(1).to(dev),
        topk_idx=topk_idx.to(torch.int32).unsqueeze(1).to(dev),
    )


def _reference(q, sink, sets):
    """Decode form of `_reference_prefill`: sets = [(deq_keys, indices [b, 1, w],
    lengths [b])], a slot at or past its row's length being padding."""
    folded = []
    for deq, idx, length in sets:
        pos = torch.arange(idx.shape[-1], device=idx.device)
        folded.append((deq, torch.where(pos < length[:, None, None], idx, -1)))
    return _reference_prefill(q, sink, folded)


@unittest.skipUnless(
    is_hip() and is_gfx95_supported(), "aiter gluon kernel is gfx950-only"
)
class TestAiterSparseBackend(CustomTestCase):
    def _assert_matches_reference(
        self, batch, heads, swa_lengths, topk_lengths, seed=0, tol=3e-2
    ):
        from sglang.srt.layers.attention.hip_flash_mla import (
            flash_mla_with_kvcache_entrypoint,
        )

        gen = torch.Generator(device="cpu").manual_seed(seed)
        dev = torch.device("cuda")
        c = _decode_case(batch, heads, gen, dev)
        q, sink, swa_idx, topk_idx = c.q, c.sink, c.swa_idx, c.topk_idx
        swa_len = torch.tensor(swa_lengths, dtype=torch.int32, device=dev)
        topk_len = torch.tensor(topk_lengths, dtype=torch.int32, device=dev)
        # Some -1 padding inside the length too: must be skipped by the kernel.
        topk_idx[:, 0, 3] = -1
        ref = _reference(
            q, sink, [(c.swa_deq, swa_idx, swa_len), (c.topk_deq, topk_idx, topk_len)]
        )
        # The backend folds the lengths into the index lists before this call.
        kwargs = dict(
            backend="aiter_sparse",
            q=q,
            k_cache=c.swa_cache,
            head_dim_v=D,
            block_table=None,
            cache_seqlens=None,
            tile_scheduler_metadata=None,
            softmax_scale=SCALE,
            is_fp8_kvcache=True,
            attn_sink=sink,
            extra_k_cache=c.topk_cache,
            indices=_masked(swa_idx, swa_len),
            topk_length=swa_len,
            extra_indices_in_kvcache=_masked(topk_idx, topk_len),
            extra_topk_length=topk_len,
        )
        got = flash_mla_with_kvcache_entrypoint(**kwargs)[0]
        self.assertEqual(got.shape, q.shape)
        self.assertEqual(got.dtype, torch.bfloat16)
        err = (got.float() - ref).abs().max().item() / ref.abs().max().item()
        self.assertLess(err, tol, f"aiter vs reference {err:.4f}")
        for _ in range(3):
            self.assertTrue(
                torch.equal(flash_mla_with_kvcache_entrypoint(**kwargs)[0], got)
            )

    def test_matches_reference(self):
        """Full lists, contexts shorter than the window and the top-k width (the length
        masks live slots left in the list), and the model's 64-padded heads."""
        for batch, heads, swa_lengths, topk_lengths, seed in (
            (3, 16, [101, 128, 5], [100, 512, 1], 1),
        ):
            with self.subTest(batch=batch, heads=heads, seed=seed):
                self._assert_matches_reference(
                    batch, heads, swa_lengths, topk_lengths, seed=seed
                )

    def test_short_lists_skip_the_padding_key(self):
        """The -1 inside the length (index 3 of the top-k list) must not be attended: on
        a 5-key list a stray key moves the softmax mass past this tolerance, where the
        640-key cases absorb it."""
        self._assert_matches_reference(2, 16, [2, 3], [4, 5], seed=6, tol=TOL_SHORT)

    def test_fold_cache_follows_the_index_source(self):
        """Layers between two index sources fold the first source's list once; the
        layers after the second source must fold its list, not reuse the first one's."""
        from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
            DSV4AttnMetadata,
            _fold_lengths_for_aiter_sparse,
        )

        dev = "cuda"
        i32 = dict(dtype=torch.int32, device=dev)
        swa = torch.arange(2 * 1 * 64, **i32).view(2, 1, 64)
        swa_len = torch.tensor([64, 64], **i32)
        core = DSV4AttnMetadata(
            page_size=64,
            page_table=torch.zeros(1, **i32),
            raw_out_loc=torch.zeros(1, **i32),
            cuda_int32_kwargs={},
            seq_lens_casual=torch.zeros(2, **i32),
            positions_casual=torch.zeros(2, **i32),
            swa_page_indices=swa,
            swa_topk_lengths=swa_len,
            index_topk=64,
        )
        core.c2_sparse_page_indices = torch.arange(2 * 1 * 64, **i32).view(2, 1, 64)
        core.c2_sparse_topk_lengths = torch.tensor([3, 64], **i32)
        extra = core.sparse_page_indices(2)

        _, first = _fold_lengths_for_aiter_sparse(
            core, 2, swa, swa_len, extra, core.c2_sparse_topk_lengths
        )
        self.assertEqual(first[0, 0, :4].tolist(), [0, 1, 2, -1])
        # the next index source writes new picks into the same buffer
        core.drop_folded_sparse_indices(2)
        extra.add_(1000)
        core.c2_sparse_topk_lengths.fill_(2)
        _, second = _fold_lengths_for_aiter_sparse(
            core, 2, swa, swa_len, extra, core.c2_sparse_topk_lengths
        )
        self.assertEqual(second[0, 0, :4].tolist(), [1000, 1001, -1, -1])
        # ratio 1's folds survive a ratio-2 rewrite
        core._aiter_sparse_masked_indices[(1, 0, (), None, None)] = "kept"
        core.drop_folded_sparse_indices(2)
        self.assertEqual(
            list(core._aiter_sparse_masked_indices), [(1, 0, (), None, None)]
        )

    def test_indexer_dispatch_drops_the_ratio_folds(self):
        """Every indexer body rewrites the ratio's page indices, so the dispatcher must
        drop that ratio's folded lists before any body runs."""
        import sglang.srt.layers.attention.deepseek_v4_backend_hip_radix as module
        from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
            DeepseekV4HipRadixBackend,
        )
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        backend = object.__new__(DeepseekV4HipRadixBackend)
        dropped, calls = [], []
        backend.forward_metadata = SimpleNamespace(
            core_metadata=SimpleNamespace(drop_folded_sparse_indices=dropped.append)
        )
        backend._low_ratio_index_topk_torch = lambda *a, **k: calls.append("torch")
        layer = SimpleNamespace(compress_ratio=2)
        with mock.patch.object(
            module,
            "low_ratio_index_topk_hip_decode",
            lambda *a, **k: calls.append("decode"),
        ):
            for mode in (ForwardMode.DECODE, ForwardMode.EXTEND):
                forward_batch = SimpleNamespace(
                    forward_mode=mode, seq_lens_cpu=None, extend_seq_lens_cpu=None
                )
                backend._low_ratio_index_topk(
                    layer, None, None, None, None, forward_batch
                )
        self.assertEqual(dropped, [2, 2])
        # extend without CPU lengths falls back to the torch oracle
        self.assertEqual(calls, ["decode", "torch"])


SWA, TOPK = 128, 512


def _prefill_lists(num_tokens, device, gen):
    """Causal prefill lists as the indexer emits them: SWA holds the 128 most recent
    slots, top-k 512 random earlier slots, both -1 padded."""
    pos = torch.arange(num_tokens)
    swa = pos[:, None] - torch.arange(SWA)[None, :]
    swa = torch.where(swa >= 0, swa, torch.full_like(swa, -1))
    topk = torch.argsort(torch.rand(num_tokens, num_tokens, generator=gen), dim=1)
    topk = topk[:, :TOPK]
    topk = torch.where(topk <= pos[:, None], topk, torch.full_like(topk, -1))
    topk = topk.sort(dim=1, descending=True).values
    return (
        swa.to(torch.int32).view(num_tokens, 1, SWA).to(device),
        topk.to(torch.int32).view(num_tokens, 1, TOPK).to(device),
    )


def _reference_prefill(q, sink, sets, chunk=256):
    """q [t, 1, h, D] bf16; sets = [(deq_keys [slots, D], indices [t, 1, w])]."""
    t, _, h, _ = q.shape
    out = torch.empty(t, 1, h, D, device=q.device, dtype=torch.float32)
    for t0 in range(0, t, chunk):
        t1 = min(t, t0 + chunk)
        idx = torch.cat([i[t0:t1, 0] for _, i in sets], dim=1)
        keys = torch.cat(
            [deq[i[t0:t1, 0].clamp(min=0).long()] for deq, i in sets], dim=1
        )
        s = torch.einsum("chd,cnd->chn", q[t0:t1, 0].float(), keys) * SCALE
        s = torch.where((idx >= 0)[:, None, :], s, torch.full_like(s, float("-inf")))
        logits = torch.cat([s, sink[None, :, None].expand(t1 - t0, h, 1)], dim=-1)
        p = torch.softmax(logits, dim=-1)[..., :-1]
        out[t0:t1, 0] = torch.einsum("chn,cnd->chd", p, keys)
    return out


@unittest.skipUnless(
    is_hip() and is_gfx95_supported(), "aiter gluon kernel is gfx950-only"
)
class TestAiterSparsePrefill(CustomTestCase):
    NUM_TOKENS = 2048
    HEADS = 16

    @classmethod
    def setUpClass(cls):
        gen = torch.Generator(device="cpu").manual_seed(0)
        dev = torch.device("cuda")
        blocks = cls.NUM_TOKENS // PAGE + 1
        cls.swa_cache, swa_deq = _pack_cache(blocks, dev, gen, fp8_view=False)
        cls.topk_cache, topk_deq = _pack_cache(blocks, dev, gen, fp8_view=False)
        cls.q = (
            (torch.randn(cls.NUM_TOKENS, 1, cls.HEADS, D, generator=gen) * 0.5)
            .to(torch.bfloat16)
            .to(dev)
        )
        cls.sink = (torch.randn(cls.HEADS, generator=gen) * 0.5).to(dev)
        cls.swa_idx, cls.topk_idx = _prefill_lists(cls.NUM_TOKENS, dev, gen)
        cls.ref = _reference_prefill(
            cls.q, cls.sink, [(swa_deq, cls.swa_idx), (topk_deq, cls.topk_idx)]
        )

    def _run(self, backend, rows=slice(None)):
        from sglang.srt.layers.attention.hip_flash_mla import (
            flash_mla_with_kvcache_entrypoint,
        )

        return flash_mla_with_kvcache_entrypoint(
            backend=backend,
            q=self.q[rows],
            k_cache=self.swa_cache,
            head_dim_v=D,
            block_table=None,
            cache_seqlens=None,
            tile_scheduler_metadata=None,
            softmax_scale=SCALE,
            is_fp8_kvcache=True,
            indices=self.swa_idx[rows],
            topk_length=None,
            attn_sink=self.sink,
            extra_k_cache=self.topk_cache,
            extra_indices_in_kvcache=self.topk_idx[rows],
            extra_topk_length=None,
        )[0]

    def test_matches_reference(self):
        got = self._run("aiter_sparse")
        self.assertEqual(got.shape, self.q.shape)
        self.assertEqual(got.dtype, torch.bfloat16)
        err = (got.float() - self.ref).abs().max().item() / self.ref.abs().max().item()
        # Measured 2.5e-3 (bf16 q, bf16 probabilities).
        self.assertLess(err, 1e-2, f"aiter vs reference {err:.2e}")
        for _ in range(3):
            self.assertTrue(torch.equal(self._run("aiter_sparse"), got))

    def test_batch_invariant(self):
        from sglang.srt.layers.attention import hip_flash_mla

        # both batches are at or above the unsplit threshold, so each row runs one program in the same order
        half = self.NUM_TOKENS // 2
        self.assertGreaterEqual(
            half, hip_flash_mla._AITER_SPARSE_SINGLE_SPLIT_MIN_TOKENS
        )
        full = self._run("aiter_sparse")
        part = self._run("aiter_sparse", rows=slice(0, half))
        self.assertTrue(torch.equal(full[:half], part))


def _freqs(device, max_pos=8192, seed=0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    angles = torch.rand(max_pos, ROPE // 2, generator=gen) * 2 * math.pi
    freqs_cis = torch.polar(torch.ones_like(angles), angles).to(device)
    return torch.view_as_real(freqs_cis).flatten(-2).contiguous()


def _model_inverse_rope(x, freqs_real, positions):
    """The model's standalone inverse RoPE of the attention output:
    `fused_rope_inplace(..., inverse=True)` with the batched flat kernel
    (`set_batched_rope(True)`)."""
    from sglang.kernels.ops.attention.deepseek_v4_rope import set_batched_rope
    from sglang.kernels.ops.attention.dsv4.elementwise import fused_rope_inplace

    set_batched_rope(True)
    freqs_cis = torch.view_as_complex(freqs_real.view(freqs_real.shape[0], -1, 2))
    fused_rope_inplace(x, None, freqs_cis, positions, inverse=True)


@unittest.skipUnless(
    is_hip() and is_gfx95_supported(), "aiter gluon kernel is gfx950-only"
)
class TestAiterSparseDecodeReduce(CustomTestCase):
    def _inputs(self, batch, heads, seed, swa_len=128, topk_len=512):
        """The kernel's own shapes: q [b, h, D], uint8 caches, flat length-folded lists."""
        gen = torch.Generator(device="cpu").manual_seed(seed)
        dev = torch.device("cuda")
        c = _decode_case(batch, heads, gen, dev)

        def lengths(n):
            return torch.full((batch,), n, dtype=torch.int32, device=dev)

        return dict(
            q=c.q.squeeze(1),
            sink=c.sink,
            swa_cache=c.swa_cache.view(torch.uint8).squeeze(2),
            topk_cache=c.topk_cache.view(torch.uint8).squeeze(2),
            swa_idx=_masked(c.swa_idx, lengths(swa_len)).reshape(-1),
            topk_idx=_masked(c.topk_idx, lengths(topk_len)).reshape(-1),
        )

    @staticmethod
    def _indptr(n, width):
        return torch.arange(0, (n + 1) * width, width, dtype=torch.int32, device="cuda")

    def _aiter(self, inputs, kv_splits, skip_reduce):
        from aiter.ops.triton.attention.pa_decode_sparse import pa_decode_sparse

        n = inputs["q"].shape[0]
        return pa_decode_sparse(
            inputs["q"],
            inputs["swa_cache"],
            inputs["swa_idx"],
            self._indptr(n, 128),
            inputs["sink"],
            SCALE,
            extra_cache=inputs["topk_cache"],
            extra_indices=inputs["topk_idx"],
            extra_indptr=self._indptr(n, 512),
            kv_splits=kv_splits,
            skip_reduce=skip_reduce,
        )

    def test_bitwise_against_aiter_reduce(self):
        from sglang.kernels.ops.attention.aiter_sparse_decode_reduce import (
            aiter_sparse_split_reduce,
        )

        for batch, heads, splits, seed in [
            (1, 16, 4, 0),
            (3, 16, 4, 3),
        ]:
            with self.subTest(batch=batch, heads=heads, splits=splits):
                # Partial lists on the larger batches: some splits come out empty.
                lens = (128, 512) if batch == 1 else (77, 301)
                inputs = self._inputs(batch, heads, seed, *lens)
                ref = self._aiter(inputs, splits, skip_reduce=False)
                acc, m, lsum = self._aiter(inputs, splits, skip_reduce=True)
                self.assertEqual(tuple(acc.shape), (batch, splits, heads, D))
                got = aiter_sparse_split_reduce(acc, m, lsum, inputs["sink"])
                self.assertEqual(got.dtype, torch.bfloat16)
                self.assertTrue(torch.equal(got, ref))
                self.assertTrue(
                    torch.equal(
                        aiter_sparse_split_reduce(acc, m, lsum, inputs["sink"]), got
                    )
                )

    def test_inverse_rope_matches_flat_kernel(self):
        from sglang.kernels.ops.attention.aiter_sparse_decode_reduce import (
            aiter_sparse_split_reduce,
        )

        dev = torch.device("cuda")
        fr = _freqs(dev)
        for batch, heads, splits, pos_dtype in [
            (1, 16, 4, torch.int64),
            (5, 16, 4, torch.int32),
        ]:
            with self.subTest(batch=batch, heads=heads, splits=splits):
                inputs = self._inputs(batch, heads, 20 + batch)
                acc, m, lsum = self._aiter(inputs, splits, skip_reduce=True)
                pos = torch.randint(0, 8192, (batch,), device=dev, dtype=pos_dtype)
                plain = aiter_sparse_split_reduce(acc, m, lsum, inputs["sink"])
                ref = plain.clone()
                _model_inverse_rope(ref[..., -ROPE:], fr, pos)
                got = aiter_sparse_split_reduce(
                    acc, m, lsum, inputs["sink"], inv_rope=(fr, pos)
                )
                self.assertTrue(torch.equal(got, ref))
                self.assertTrue(torch.equal(got[..., :-ROPE], plain[..., :-ROPE]))


@unittest.skipUnless(
    is_hip() and is_gfx95_supported(), "aiter gluon kernel is gfx950-only"
)
class TestAiterSparseDecodeSplitPin(CustomTestCase):
    """``SGLANG_OPT_HIP_ATTN_KV_SPLITS`` pins the split-KV count so a row's output is
    bitwise the same at every batch size; aiter's cost model changes the count past 64
    rows."""

    ROWS = 96  # aiter picks 2 splits here, 4 at 1..64 rows

    def _run(self, batch):
        from sglang.srt.layers.attention.hip_flash_mla import (
            flash_mla_with_kvcache_entrypoint,
        )

        # row i of every batch is row i of the same case
        dev = torch.device("cuda")
        c = _decode_case(
            self.ROWS, 16, torch.Generator(device="cpu").manual_seed(40), dev
        )
        lens = lambda n: torch.full((batch,), n, dtype=torch.int32, device=dev)
        return flash_mla_with_kvcache_entrypoint(
            backend="aiter_sparse",
            q=c.q[:batch],
            k_cache=c.swa_cache,
            head_dim_v=D,
            block_table=None,
            cache_seqlens=None,
            tile_scheduler_metadata=None,
            softmax_scale=SCALE,
            is_fp8_kvcache=True,
            attn_sink=c.sink,
            extra_k_cache=c.topk_cache,
            indices=_masked(c.swa_idx[:batch], lens(128)),
            extra_indices_in_kvcache=_masked(c.topk_idx[:batch], lens(512)),
        )[0]

    def test_pinned_splits_are_batch_invariant(self):
        from sglang.srt.environ import envs

        with envs.SGLANG_OPT_HIP_ATTN_KV_SPLITS.override(4):
            one, eight, many = self._run(1), self._run(8), self._run(self.ROWS)
        self.assertTrue(torch.equal(one, eight[:1]))
        self.assertTrue(torch.equal(eight, many[:8]))


if __name__ == "__main__":
    unittest.main()
