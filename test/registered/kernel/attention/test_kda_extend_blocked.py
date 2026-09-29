"""Token-blocked KDA extend must match the single-call chunk_kda extend.

With SGLANG_KDA_EXTEND_BLOCK_TOKENS set, TritonKDAKernel.extend runs chunk_kda
over blocks of at most that many tokens: consecutive whole sequences are
grouped per call and longer sequences are split with the recurrent state
chained between blocks. The GPU tests compare outputs, final pool states and
per-chunk intermediate states against the unblocked path from identical pool
copies, with a spy around the real chunk_kda proving that the calls were
actually blocked. The CPU tests drive the same code with a stand-in kernel to
check the call plan, the metadata fallback, the state carry and the padded-row
sentinel without a GPU.

Numerical contract: the single call and the blocked calls run the same
per-chunk math on the same chunks (blocks are chunk-aligned), the state is
accumulated in fp32 in both and chained between blocks through an fp32 scratch
slot (cast back once, like the single call's final store). When every call
dispatches the same intra-chunk kernel variant (B * num_chunks * H <= 256 on
both sides, fla/kda.py) the results are bitwise equal and the tests assert
that. When blocking moves a call across that threshold the two variants round
differently and the test asserts a tight tolerance instead.
"""

import contextlib
import os
import unittest
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.linear.kernels import kda_triton
from sglang.srt.layers.attention.linear.kernels.kda_triton import (
    TritonKDAKernel,
    _cached_cu_seqlens,
    _extend_block_tokens,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

BLOCK = 2048
CHUNK = 64
# Straddle the block: below one block, exactly one, one plus a chunk, two plus
# a chunk. Expected calls: [64], [2048], 2112 -> 2048 + 64, 4160 -> 3 blocks.
SEQ_LENS = [64, BLOCK, BLOCK + 64, 2 * BLOCK + 64]
CALLS = [64, BLOCK, BLOCK, 64, BLOCK, BLOCK, 64]
# Final blocks that are not chunk multiples, and tiny sequences between them.
SEQ_LENS_RAGGED = [BLOCK + 37, 3, 2 * BLOCK + 1, 100]
CALLS_RAGGED = [BLOCK, 37, 3, BLOCK, BLOCK, 1, 100]
NUM_HEADS = 1
HEAD_DIM = 64
# Non-consecutive slots, so a wrong slot read/write shows up as a mismatch.
SLOT_INDICES = [4, 1, 5, 2]


def _inputs(
    seq_lens,
    slot_indices,
    *,
    pool_dtype=torch.float32,
    device="cuda",
    seed=0,
    num_heads=NUM_HEADS,
):
    generator = torch.Generator(device=device).manual_seed(seed)
    total = sum(seq_lens)
    shape = (1, total, num_heads, HEAD_DIM)
    pool_slots = max(slot_indices) + 2

    def rand(*s, dtype=torch.bfloat16, scale=1.0):
        return (
            scale
            * torch.randn(*s, generator=generator, device=device, dtype=torch.float32)
        ).to(dtype)

    q = rand(*shape)
    k = rand(*shape)
    # v as a strided view of a wider buffer, like the caller's split view.
    v_wide = rand(1, total, num_heads, 2 * HEAD_DIM, scale=0.1)
    v = v_wide[..., :HEAD_DIM]
    g = rand(*shape)
    beta = rand(*shape[:-1])
    A_log = rand(num_heads, dtype=torch.float32)
    dt_bias = rand(num_heads * HEAD_DIM, dtype=torch.float32)
    # Every slot nonzero (slot 0 in particular: a padded row that gathers it
    # instead of starting from zero must show up).
    pool = rand(pool_slots, num_heads, HEAD_DIM, HEAD_DIM, dtype=pool_dtype, scale=0.1)
    cache_indices = torch.tensor(slot_indices, dtype=torch.int32, device=device)
    cu_seqlens = torch.tensor(
        [0] + list(torch.tensor(seq_lens).cumsum(0)), dtype=torch.int32, device=device
    )
    return dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        pool=pool,
        cache_indices=cache_indices,
        cu_seqlens=cu_seqlens,
        seq_lens=list(seq_lens),
    )


class _ChunkKdaSpy:
    """Wraps chunk_kda: records each call's token count and cu_seqlens, then
    invokes the wrapped kernel unchanged."""

    def __init__(self, real):
        self.real = real
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append((kwargs["q"].shape[1], kwargs["cu_seqlens"].tolist()))
        return self.real(**kwargs)

    @property
    def tokens(self):
        return [t for t, _ in self.calls]


class _IntraSpy:
    """Wraps fla's chunk_kda_fwd_intra: records which intra-chunk kernel
    variant each call dispatched (fuse_diagonal, from the grid heuristic)."""

    def __init__(self, real):
        self.real = real
        self.fused = []

    def __call__(self, **kwargs):
        self.fused.append(bool(kwargs["fuse_diagonal"]))
        return self.real(**kwargs)


_TRUE_LENS = object()


def _run(
    block, inputs, *, return_intermediate_states=False, kernel=None, seq_lens=_TRUE_LENS
):
    """Run extend with SGLANG_KDA_EXTEND_BLOCK_TOKENS=block on a fresh pool
    copy. ``kernel`` replaces kda_triton.chunk_kda for the call; ``seq_lens``
    overrides extend_seq_lens_cpu (default: the true lengths)."""
    if seq_lens is _TRUE_LENS:
        seq_lens = inputs["seq_lens"]
    pool = inputs["pool"].clone()
    patch = (
        mock.patch.object(kda_triton, "chunk_kda", kernel, create=True)
        if kernel is not None
        else contextlib.nullcontext()
    )
    with envs.SGLANG_KDA_EXTEND_BLOCK_TOKENS.override(block), patch:
        out = TritonKDAKernel().extend(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["g"],
            inputs["beta"],
            ssm_states=pool,
            cache_indices=inputs["cache_indices"],
            query_start_loc=inputs["cu_seqlens"],
            A_log=inputs["A_log"],
            dt_bias=inputs["dt_bias"],
            lower_bound=-5.0,
            extend_seq_lens_cpu=seq_lens,
            return_intermediate_states=return_intermediate_states,
        )
    if return_intermediate_states:
        o, h = out
    else:
        o, h = out, None
    if o.is_cuda:
        torch.cuda.synchronize()
    return o, h, pool


class _Checks:
    def assertSame(self, got, want, what):
        self.assertEqual(tuple(got.shape), tuple(want.shape), what)
        self.assertEqual(got.dtype, want.dtype, what)
        if not torch.equal(got, want):
            diff = (got.float() - want.float()).abs().max().item()
            self.fail(f"{what}: max |diff| = {diff}")

    def assertClose(self, got, want, what, *, atol, rtol):
        self.assertEqual(tuple(got.shape), tuple(want.shape), what)
        self.assertEqual(got.dtype, want.dtype, what)
        torch.testing.assert_close(
            got.float(),
            want.float(),
            atol=atol,
            rtol=rtol,
            msg=lambda m: f"{what}: {m}",
        )

    def assertUntouched(self, pool, inputs):
        untouched = [
            i for i in range(pool.shape[0]) if i not in inputs["cache_indices"].tolist()
        ]
        self.assertSame(pool[untouched], inputs["pool"][untouched], "untouched slots")


class TestKdaExtendBlocked(CustomTestCase, _Checks):
    """GPU: blocked path against the single call on the real kernels."""

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")

    @torch.inference_mode()
    def _compare(
        self, inputs, block, calls, *, return_intermediate_states=False, close=None
    ):
        """``close=(atol, rtol)`` compares with a tolerance instead of exactly
        (only for cases whose calls dispatch different kernel variants)."""
        o_ref, h_ref, pool_ref = _run(
            0, inputs, return_intermediate_states=return_intermediate_states
        )
        spy = _ChunkKdaSpy(kda_triton.chunk_kda)
        o_blk, h_blk, pool_blk = _run(
            block,
            inputs,
            return_intermediate_states=return_intermediate_states,
            kernel=spy,
        )
        # Blocking happened: the exact plan, and every call within the block.
        self.assertEqual(spy.tokens, calls)
        self.assertTrue(all(t <= block for t in spy.tokens))
        if close is None:
            check = self.assertSame
        else:
            atol, rtol = close

            def check(got, want, what):
                self.assertClose(got, want, what, atol=atol, rtol=rtol)

        check(o_blk, o_ref, "output")
        check(pool_blk, pool_ref, "pool")
        self.assertUntouched(pool_blk, inputs)
        if return_intermediate_states:
            # Per-chunk states, packed across sequences in chunk order.
            self.assertEqual(
                h_ref.shape[1], sum(-(-n // CHUNK) for n in inputs["seq_lens"])
            )
            check(h_blk, h_ref, "intermediate states")
        else:
            self.assertIsNone(h_blk)
        return spy

    def test_blocked_matches_unblocked(self):
        self._compare(_inputs(SEQ_LENS, SLOT_INDICES), BLOCK, CALLS)

    def test_blocked_matches_unblocked_intermediate_states(self):
        self._compare(
            _inputs(SEQ_LENS, SLOT_INDICES),
            BLOCK,
            CALLS,
            return_intermediate_states=True,
        )

    def test_reduced_precision_pool_matches_unblocked(self):
        # The kernel stores the state in the pool dtype, so a split sequence
        # is chained through an fp32 scratch slot and cast back once; the
        # single call also rounds exactly once (its final store), so the
        # results must be identical, not merely close.
        for pool_dtype in (torch.bfloat16, torch.float16):
            with self.subTest(pool_dtype=pool_dtype):
                inputs = _inputs(SEQ_LENS, SLOT_INDICES, pool_dtype=pool_dtype)
                self._compare(inputs, BLOCK, CALLS, return_intermediate_states=True)

    def test_padded_row_matches_unblocked(self):
        # A padded row carries cache index -1: the kernel starts it from a zero
        # state and never stores it. Split, it must chain through a zero
        # scratch state (not slot 0, which is nonzero here) and leave the
        # whole pool, slot 0 included, untouched.
        # int64 too: ``to(torch.long)`` then returns the caller's tensor, so an
        # in-place clamp would turn the caller's -1 into 0.
        for pool_dtype, index_dtype in (
            (torch.float32, torch.int32),
            (torch.bfloat16, torch.int32),
            (torch.float32, torch.int64),
            (torch.bfloat16, torch.int64),
        ):
            with self.subTest(pool_dtype=pool_dtype, index_dtype=index_dtype):
                inputs = _inputs([2 * BLOCK], [-1], pool_dtype=pool_dtype)
                inputs["cache_indices"] = inputs["cache_indices"].to(index_dtype)
                self.assertTrue(bool((inputs["pool"][0] != 0).any()))
                self._compare(
                    inputs, BLOCK, [BLOCK, BLOCK], return_intermediate_states=True
                )
                self.assertEqual(inputs["cache_indices"].tolist(), [-1])

    def test_final_blocks_not_chunk_multiples(self):
        self._compare(
            _inputs(SEQ_LENS_RAGGED, SLOT_INDICES),
            BLOCK,
            CALLS_RAGGED,
            return_intermediate_states=True,
        )

    def test_short_sequences_share_one_call(self):
        # 64 sequences of 64 tokens fit one 4096-token block together, so they
        # must go through a single chunk_kda call (one cu_seqlens of 65
        # entries), not one call per sequence; the long sequence after them
        # forces the blocked path and is split on its own.
        short = 64
        seq_lens = [CHUNK] * short + [4096 + CHUNK]
        slots = list(range(short + 1))
        spy = self._compare(_inputs(seq_lens, slots), 4096, [4096, 4096, CHUNK])
        tokens, cu = spy.calls[0]
        self.assertEqual(len(cu), short + 1)
        self.assertEqual(cu, [CHUNK * i for i in range(short + 1)])

    def test_multi_head_across_intra_kernel_threshold(self):
        # chunk_kda picks the intra-chunk kernel variant per call from
        # B * num_chunks * H <= 256 (fla/kda.py). With 8 heads, the single call
        # over 2112 tokens has 33 * 8 = 264 (non-fused variant) while the 2048-
        # and 64-token blocks have 256 and 8 (fused variant): different kernels
        # with different rounding of the bf16 intermediates, so exact equality
        # is not the contract here. The tolerance is a few bf16 ulps of the
        # values involved; it has not been calibrated on a GPU in this revision.
        from sglang.kernels.ops.attention.fla import kda as fla_kda

        intra = _IntraSpy(fla_kda.chunk_kda_fwd_intra)
        with mock.patch.object(fla_kda, "chunk_kda_fwd_intra", intra):
            self._compare(
                _inputs([BLOCK + 64], [1], num_heads=8),
                BLOCK,
                [BLOCK, 64],
                return_intermediate_states=True,
                close=(1e-2, 1e-2),
            )
        # The single call ran the non-fused variant, both blocks the fused one:
        # the threshold really was crossed.
        self.assertEqual(intra.fused, [False, True, True])


class _FakeChunkKda:
    """CPU stand-in for chunk_kda with the side effects the blocked path relies
    on, and results that depend on the loaded initial state so a wrong state
    source shows up. Per sequence: the state starts from
    ``initial_state[index]``, or zero for index < 0 (the real fwd_h kernel
    skips both the load and the store for the padded-row sentinel); ``h``
    holds the state entering each 64-token chunk in call order; each token's
    output ``2 * v + state[..., 0]`` is written into ``v`` in place (o=v); the
    state accumulates each chunk's ``v`` in fp32; and the final state is stored
    to ``initial_state[index]`` in the state's dtype (INPLACE_UPDATE). Values
    are small integers so fp32 sums are exact however the tokens are split."""

    def __init__(self):
        self.calls = []

    def __call__(
        self,
        *,
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        initial_state,
        initial_state_indices,
        output_intermediate_states=False,
        **kwargs,
    ):
        cu = cu_seqlens.tolist()
        indices = initial_state_indices.tolist()
        self.calls.append(
            dict(
                tokens=q.shape[1],
                cu=cu,
                indices=indices,
                state=initial_state,
                cu_seqlens=cu_seqlens,
                v=v,
            )
        )
        v = v.contiguous()
        hs = []
        for n in range(len(cu) - 1):
            s, e = cu[n], cu[n + 1]
            if indices[n] >= 0:
                state = initial_state[indices[n]].float()
            else:
                state = torch.zeros(initial_state.shape[1:], dtype=torch.float32)
            for c in range(s, e, CHUNK):
                chunk = v[0, c : min(c + CHUNK, e)].float()
                hs.append(state.clone())
                v[0, c : min(c + CHUNK, e)] = (2 * chunk + state[..., 0]).to(v.dtype)
                state = state + chunk.sum(0)[..., None]
            if e > s and indices[n] >= 0:
                initial_state[indices[n]] = state.to(initial_state.dtype)
        if output_intermediate_states:
            return v, torch.stack(hs, 0).unsqueeze(0)
        return v


def _int_inputs(seq_lens, slot_indices, *, pool_dtype=torch.float32):
    inputs = _inputs(seq_lens, slot_indices, device="cpu")
    total = sum(seq_lens)
    generator = torch.Generator().manual_seed(1)
    v_wide = torch.randint(
        -2, 3, (1, total, NUM_HEADS, 2 * HEAD_DIM), generator=generator
    ).to(torch.bfloat16)
    inputs["v"] = v_wide[..., :HEAD_DIM]
    inputs["pool"] = torch.randint(
        -2,
        3,
        (inputs["pool"].shape[0], NUM_HEADS, HEAD_DIM, HEAD_DIM),
        generator=generator,
    ).to(pool_dtype)
    return inputs


class TestKdaExtendBlockedPlan(CustomTestCase, _Checks):
    """CPU: call plan, metadata fallback and state carry with a fake kernel."""

    def test_block_rounding(self):
        # Unset (0, the default) or anything below the 64-token chunk
        # disables; else round down.
        self.assertEqual(envs.SGLANG_KDA_EXTEND_BLOCK_TOKENS.default, 0)
        with mock.patch.dict(os.environ):
            envs.SGLANG_KDA_EXTEND_BLOCK_TOKENS.clear()
            self.assertEqual(_extend_block_tokens(), 0)
        for raw, want in [
            (0, 0),
            (-1, 0),
            (1, 0),
            (63, 0),
            (64, 64),
            (100, 64),
            (2048, 2048),
            (2100, 2048),
        ]:
            with envs.SGLANG_KDA_EXTEND_BLOCK_TOKENS.override(raw):
                self.assertEqual(_extend_block_tokens(), want)

    def _plan(self, inputs, block, *, return_intermediate_states=True):
        ref = _FakeChunkKda()
        o_ref, h_ref, pool_ref = _run(
            0, inputs, return_intermediate_states=return_intermediate_states, kernel=ref
        )
        self.assertEqual(len(ref.calls), 1)
        fake = _FakeChunkKda()
        o_blk, h_blk, pool_blk = _run(
            block,
            inputs,
            return_intermediate_states=return_intermediate_states,
            kernel=fake,
        )
        self.assertSame(o_blk, o_ref, "output")
        self.assertSame(pool_blk, pool_ref, "pool")
        self.assertUntouched(pool_blk, inputs)
        if return_intermediate_states:
            self.assertSame(h_blk, h_ref, "intermediate states")
        return fake

    def test_plan_groups_and_splits(self):
        inputs = _int_inputs(SEQ_LENS, SLOT_INDICES)
        fake = self._plan(inputs, BLOCK)
        self.assertEqual([c["tokens"] for c in fake.calls], CALLS)
        self.assertEqual(
            [c["cu"] for c in fake.calls],
            [[0, 64], [0, BLOCK], [0, BLOCK], [0, 64], [0, BLOCK], [0, BLOCK], [0, 64]],
        )
        # Whole sequences run against the pool; each split sequence runs
        # against its own fp32 scratch slot (index 0), even on an fp32 pool.
        self.assertEqual(
            [c["indices"] for c in fake.calls], [[4], [1], [0], [0], [0], [0], [0]]
        )
        pool_calls, split_calls = fake.calls[:2], fake.calls[2:]
        self.assertTrue(all(c["state"].shape[0] > 1 for c in pool_calls))
        for c in split_calls:
            self.assertIs(c["state"].dtype, torch.float32)
            self.assertEqual(
                tuple(c["state"].shape), (1, NUM_HEADS, HEAD_DIM, HEAD_DIM)
            )
        self.assertIs(split_calls[0]["state"], split_calls[1]["state"])
        self.assertIsNot(split_calls[1]["state"], split_calls[2]["state"])
        # Identical layouts share one cu_seqlens tensor (chunk_kda's identity
        # cache), within the extend and across extends (layers, steps).
        by_layout = {}
        for c in fake.calls:
            by_layout.setdefault(tuple(c["cu"]), []).append(c["cu_seqlens"])
        for layout, tensors in by_layout.items():
            self.assertTrue(all(t is tensors[0] for t in tensors), layout)
        again = _FakeChunkKda()
        _run(BLOCK, inputs, kernel=again)
        for c1, c2 in zip(fake.calls, again.calls):
            self.assertIs(c1["cu_seqlens"], c2["cu_seqlens"])

    def test_cu_seqlens_cache_is_bounded(self):
        bound = kda_triton._CU_SEQLENS_CACHE_SIZE
        first = _cached_cu_seqlens([0, 1], torch.int32, "cpu")
        self.assertIs(_cached_cu_seqlens([0, 1], torch.int32, "cpu"), first)
        for n in range(2, bound + 2):
            _cached_cu_seqlens([0, n], torch.int32, "cpu")
        self.assertLessEqual(len(kda_triton._cu_seqlens_cache), bound)
        # The least recently used layout was evicted and is rebuilt.
        self.assertIsNot(_cached_cu_seqlens([0, 1], torch.int32, "cpu"), first)

    def test_plan_ragged_and_grouped(self):
        fake = self._plan(_int_inputs(SEQ_LENS_RAGGED, SLOT_INDICES), BLOCK)
        self.assertEqual([c["tokens"] for c in fake.calls], CALLS_RAGGED)
        short = [CHUNK] * 64 + [4096 + CHUNK]
        fake = self._plan(_int_inputs(short, list(range(65))), 4096)
        self.assertEqual([c["tokens"] for c in fake.calls], [4096, 4096, CHUNK])
        self.assertEqual(fake.calls[0]["indices"], list(range(64)))
        self.assertEqual(fake.calls[1]["indices"], [0])

    def test_reduced_precision_pool_carries_fp32(self):
        # Split sequences run against an fp32 scratch slot and the pool
        # receives one final cast; grouped sequences use the pool directly.
        inputs = _int_inputs(SEQ_LENS, SLOT_INDICES, pool_dtype=torch.bfloat16)
        fake = self._plan(inputs, BLOCK)
        self.assertEqual([c["tokens"] for c in fake.calls], CALLS)
        for call in fake.calls[:2]:
            self.assertIs(call["state"].dtype, torch.bfloat16)
        for call in fake.calls[2:]:
            self.assertIs(call["state"].dtype, torch.float32)
            self.assertEqual(call["indices"], [0])

    def test_padded_row_starts_from_zero_state(self):
        # A -1 (padded) index starts from a zero state in the single call. The
        # split sequence must chain through a zero scratch state, not through
        # slot 0's contents (nonzero here), and leave the pool untouched; the
        # fake's output and h depend on the state, so gathering slot 0 fails.
        for pool_dtype in (torch.float32, torch.bfloat16):
            for index_dtype in (torch.int32, torch.int64):
                with self.subTest(pool_dtype=pool_dtype, index_dtype=index_dtype):
                    inputs = _int_inputs(
                        [2 * BLOCK + 64, 64], [-1, 3], pool_dtype=pool_dtype
                    )
                    inputs["cache_indices"] = inputs["cache_indices"].to(index_dtype)
                    inputs["pool"][0].fill_(3)
                    fake = self._plan(inputs, BLOCK)
                    self.assertEqual(
                        [c["tokens"] for c in fake.calls], [BLOCK, BLOCK, 64, 64]
                    )
                    for call in fake.calls[:3]:
                        self.assertEqual(call["indices"], [0])
                        self.assertIs(call["state"].dtype, torch.float32)
                    self.assertEqual(fake.calls[3]["indices"], [3])
                    # The caller's index tensor is not mutated by the clamp.
                    self.assertEqual(inputs["cache_indices"].tolist(), [-1, 3])

    def test_metadata_mismatch_falls_back(self):
        inputs = _int_inputs(SEQ_LENS, SLOT_INDICES)
        bad = [
            None,
            SEQ_LENS[:-1],
            SEQ_LENS + [0],
            [BLOCK, BLOCK] + SEQ_LENS[2:],
            SEQ_LENS[:-1] + [SEQ_LENS[-1] - 1],
            [-64, 128] + SEQ_LENS[2:],
        ]
        for seq_lens in bad:
            with self.subTest(seq_lens=seq_lens):
                fake = _FakeChunkKda()
                o, _, pool = _run(BLOCK, inputs, kernel=fake, seq_lens=seq_lens)
                self.assertEqual(len(fake.calls), 1)
                self.assertIs(fake.calls[0]["cu_seqlens"], inputs["cu_seqlens"])
                self.assertIs(fake.calls[0]["v"], inputs["v"])
                self.assertIs(fake.calls[0]["state"], pool)
                self.assertEqual(fake.calls[0]["indices"], SLOT_INDICES)
        # A tensor of lengths is accepted like a list.
        fake = _FakeChunkKda()
        _run(BLOCK, inputs, kernel=fake, seq_lens=torch.tensor(SEQ_LENS))
        self.assertEqual([c["tokens"] for c in fake.calls], CALLS)

    def test_within_block_is_single_call(self):
        inputs = _int_inputs([64, 128], [1, 2])
        for block in (0, 63, 192, 4096):
            with self.subTest(block=block):
                fake = _FakeChunkKda()
                _run(block, inputs, kernel=fake)
                self.assertEqual(len(fake.calls), 1)
                self.assertIs(fake.calls[0]["cu_seqlens"], inputs["cu_seqlens"])


if __name__ == "__main__":
    unittest.main()
