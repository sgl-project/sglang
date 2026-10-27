"""The CUDA-graph captured KDA extend (linear/kda_prefill_graph.py) against the
eager KDAAttnBackend.forward_extend.

One bucket is captured once, then replayed for batches of different shapes:
several sequences, cached prefixes, padding past the batch, chunk-aligned and
unaligned prefix-cache snapshots, and a batch without tracking. Each replay must
match the eager path on the output, the conv / SSM state pools and the track
slots, and must leave every other pool slot untouched. Comparisons are bitwise
except FlashInfer's small padded plans and the eager Triton fallback for
one-token sequences, which can use different floating-point operation orders.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.linear.kda_backend import (
    KDAAttnBackend,
    KDAKernelDispatcher,
)
from sglang.srt.layers.attention.linear.kda_prefill_graph import KDAPrefillGraphMetadata
from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

HEADS, HEAD_DIM, CONV = 2, 128, 4
DIM = 3 * HEADS * HEAD_DIM
BUCKET, MAX_SEQS = 512, 16
NUM_SLOTS = 48
CHUNK = 64


def _layer():
    g = torch.Generator(device="cuda").manual_seed(7)

    def randn(*shape, dtype=torch.bfloat16, scale=0.2):
        return (torch.randn(*shape, generator=g, device="cuda") * scale).to(dtype)

    return SimpleNamespace(
        layer_id=0,
        num_q_heads=HEADS,
        num_k_heads=HEADS,
        num_v_heads=HEADS,
        head_q_dim=HEAD_DIM,
        head_k_dim=HEAD_DIM,
        head_v_dim=HEAD_DIM,
        q_dim=HEADS * HEAD_DIM,
        k_dim=HEADS * HEAD_DIM,
        v_dim=HEADS * HEAD_DIM,
        conv_weights=randn(DIM, CONV),
        bias=randn(DIM),
        A_log=randn(1, 1, HEADS, 1, dtype=torch.float32),
        dt_bias=randn(HEADS * HEAD_DIM, dtype=torch.float32),
        lower_bound=-5.0,
    )


def _backend(conv_pool, ssm_pool, prefill_backend=LinearAttnKernelBackend.TRITON):
    """The production backend on test pools: slot of request i is i + 1."""
    backend = KDAAttnBackend.__new__(KDAAttnBackend)
    cache = SimpleNamespace(conv=[conv_pool], temporal=ssm_pool)
    backend.req_to_token_pool = SimpleNamespace(
        mamba2_layer_cache=lambda layer_id: cache,
        get_mamba_indices=lambda req_pool_indices: (req_pool_indices + 1).to(
            torch.int32
        ),
        translate_mamba_indices=lambda indices: indices,
    )
    backend.device = torch.device("cuda")
    backend._mamba_chunk_size = CHUNK
    backend.conv_states_shape = conv_pool.transpose(-1, -2).shape
    backend.accept_lens_pool = None
    backend.prefill_graph_metadata = None
    backend.kernel_dispatcher = KDAKernelDispatcher(
        LinearAttnKernelBackend.TRITON,
        prefill_backend,
        LinearAttnKernelBackend.TRITON,
    )
    return backend


def _batch(extend_lens, prefix_lens, track=None):
    """A ForwardBatch stand-in. ``track[row] = tracked length`` (from the
    prefix) marks prefix-cache snapshot rows; their track slots follow the
    cache slots (NUM_SLOTS // 2 + row)."""
    bs = len(extend_lens)
    dev = "cuda"
    starts = [sum(extend_lens[:i]) for i in range(bs)]
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        logical_forward_mode=ForwardMode.EXTEND,
        batch_size=bs,
        _original_batch_size=None,
        req_pool_indices=torch.arange(bs, device=dev),
        extend_seq_lens=torch.tensor(extend_lens, device=dev),
        extend_prefix_lens=torch.tensor(prefix_lens, device=dev),
        extend_start_loc=torch.tensor(starts, device=dev),
        extend_seq_lens_cpu=list(extend_lens),
        extend_prefix_lens_cpu=list(prefix_lens),
        tbo_parent_token_range=None,
        mamba_track_mask=None,
        mamba_track_indices=None,
        mamba_track_seqlens=None,
        mamba_prefill_track_mask_cpu=None,
        mamba_track_seqlens_cpu=None,
    )
    if track is not None:
        mask = [row in track for row in range(bs)]
        track_lens = [prefix_lens[r] + track.get(r, 0) for r in range(bs)]
        batch.mamba_prefill_track_mask_cpu = mask
        batch.mamba_track_seqlens_cpu = track_lens
        batch.mamba_track_mask = torch.tensor(mask, device=dev)
        batch.mamba_track_seqlens = torch.tensor(track_lens, device=dev)
        batch.mamba_track_indices = torch.arange(
            NUM_SLOTS // 2, NUM_SLOTS // 2 + bs, device=dev, dtype=torch.int32
        )

    def aligned_lens():
        lens = batch.mamba_track_seqlens - batch.extend_prefix_lens
        return (lens // CHUNK) * CHUNK

    batch.mamba_track_aligned_lens = aligned_lens
    return batch


class TestKDAPrefillGraph(CustomTestCase):
    prefill_backend = LinearAttnKernelBackend.TRITON

    def setUp(self):
        self.layer = _layer()
        g = torch.Generator(device="cuda").manual_seed(11)
        self.conv_pool = (
            torch.randn(NUM_SLOTS, CONV - 1, DIM, generator=g, device="cuda") * 0.3
        ).to(torch.bfloat16)
        self.ssm_pool = (
            torch.randn(
                NUM_SLOTS, HEADS, HEAD_DIM, HEAD_DIM, generator=g, device="cuda"
            )
            * 0.05
        )
        self.backend = _backend(self.conv_pool, self.ssm_pool, self.prefill_backend)
        # Static inputs the captured graph reads.
        self.mixed = torch.zeros(BUCKET, DIM, device="cuda", dtype=torch.bfloat16)
        self.a = torch.zeros(1, BUCKET, HEADS * HEAD_DIM, device="cuda")
        self.a = self.a.to(torch.bfloat16)
        self.b = torch.zeros(1, BUCKET, HEADS, device="cuda", dtype=torch.bfloat16)

    def _fill(self, meta, batch):
        with patch(
            "sglang.srt.layers.attention.linear.kda_backend.mamba_cache_chunk_size",
            return_value=CHUNK,
        ):
            self.backend._fill_prefill_graph_metadata(
                meta=meta,
                forward_batch=batch,
                cache_indices=self.backend._prefill_graph_cache_indices(batch),
            )

    def _capture(self, *, track=True, num_tokens=BUCKET, max_seqs=MAX_SEQS):
        from sglang.srt.runtime_context import get_context

        capture_batch = _batch([num_tokens], [0])
        capture_batch.extend_num_tokens = num_tokens
        capture_batch.positions = torch.arange(num_tokens, device="cuda")
        self.backend.prefill_graph_max_seqs = max_seqs
        with get_context().override_server_args(
            disable_radix_cache=not track, mamba_radix_cache_strategy="extra_buffer"
        ):
            meta = self.backend.init_prefill_graph_metadata(capture_batch)
        snapshot = (self.conv_pool.clone(), self.ssm_pool.clone())

        def run():
            return self.backend.forward_extend(
                self.layer,
                capture_batch,
                self.mixed[:num_tokens],
                self.a[:, :num_tokens],
                self.b[:, :num_tokens],
            )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()  # warmup: Triton compile / autotune outside the capture
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = run()
        # Warmup and capture ran on the pools; restore them.
        self.conv_pool.copy_(snapshot[0])
        self.ssm_pool.copy_(snapshot[1])
        return meta, graph, out

    def _check_replay(self, meta, graph, out, *, lens, prefixes, track, seed):
        batch = _batch(lens, prefixes, track=track)
        total = sum(batch.extend_seq_lens_cpu)
        g = torch.Generator(device="cuda").manual_seed(seed)
        # Fresh requests start from zeroed slots (the pool clears on alloc).
        for row, prefix in enumerate(batch.extend_prefix_lens_cpu):
            if prefix == 0:
                self.conv_pool[row + 1].zero_()
                self.ssm_pool[row + 1].zero_()
        # Garbage past the batch must not leak into anything.
        self.mixed.copy_(torch.randn(BUCKET, DIM, generator=g, device="cuda"))
        self.a.copy_(
            torch.randn(1, BUCKET, HEADS * HEAD_DIM, generator=g, device="cuda")
        )
        self.b.copy_(torch.randn(1, BUCKET, HEADS, generator=g, device="cuda"))

        # Eager reference on cloned pools.
        ref_conv, ref_ssm = self.conv_pool.clone(), self.ssm_pool.clone()
        ref_backend = _backend(ref_conv, ref_ssm, self.prefill_backend)
        ref_batch = _batch(lens, prefixes, track=track)
        ref_backend.init_forward_metadata(ref_batch)
        ref_out = ref_backend.forward_extend(
            self.layer,
            ref_batch,
            self.mixed[:total],
            self.a[:, :total],
            self.b[:, :total],
        )

        self._fill(meta, batch)
        graph.replay()
        torch.cuda.synchronize()

        different_kernel_schedule = (
            self.prefill_backend == LinearAttnKernelBackend.FLASHINFER
            and (max(lens) == 1 or meta.num_tokens <= meta.max_seqs)
        )
        if different_kernel_schedule:
            # Planned small FlashInfer buckets and the eager Triton fallback
            # need numerical comparison. Unused slots must remain bitwise equal.
            torch.testing.assert_close(
                out[:, :total], ref_out[:, :total], rtol=1e-2, atol=1e-4
            )
            touched = list(range(1, len(lens) + 1))
            touched += [NUM_SLOTS // 2 + row for row in (track or {})]
            untouched = [slot for slot in range(NUM_SLOTS) if slot not in touched]
            torch.testing.assert_close(
                self.ssm_pool[touched], ref_ssm[touched], rtol=1e-2, atol=1e-3
            )
            self.assertTrue(torch.equal(self.ssm_pool[untouched], ref_ssm[untouched]))
        else:
            self.assertTrue(torch.equal(out[:, :total], ref_out[:, :total]))
            self.assertTrue(torch.equal(self.ssm_pool, ref_ssm), "ssm pool differs")
        self.assertTrue(bool((out[:, total:] == 0).all()), "padded rows not zeroed")
        self.assertTrue(torch.equal(self.conv_pool, ref_conv), "conv pool differs")

    def test_without_radix_tracking(self):
        meta, graph, out = self._capture(track=False)
        self._check_replay(
            meta, graph, out, lens=[129, 7], prefixes=[64, 0], track=None, seed=300
        )

    def test_replays_match_eager(self):
        meta, graph, out = self._capture()
        cases = [
            # (extend lens, prefix lens, track {row: tracked length from prefix}).
            # Tracked lengths are >= one chunk and within the row's extend, as
            # the scheduler sets them; a chunk multiple snapshots the final state.
            ([37, 100, 5, 192], [0, 64, 130, 0], {1: 70, 3: 192}),
            ([BUCKET], [256], {0: 300}),
            (
                [1, 2, 3, 5, 8, 13, 21, 34, 3, 9, 1, 30, 17, 4, 64, 20],
                [0, 5, 0, 64, 0, 3, 0, 128, 0, 0, 7, 0, 0, 1, 0, 64],
                {14: 64},
            ),
            ([200, 11], [0, 0], None),
            ([1], [900], None),
            ([1] * MAX_SEQS, [64] * MAX_SEQS, None),
        ]
        for i, (lens, prefixes, track) in enumerate(cases):
            with self.subTest(case=i):
                self._check_replay(
                    meta,
                    graph,
                    out,
                    lens=lens,
                    prefixes=prefixes,
                    track=track,
                    seed=100 + i,
                )

    def test_capacity(self):
        meta = KDAPrefillGraphMetadata.allocate(
            num_tokens=BUCKET,
            max_seqs_cap=MAX_SEQS,
            track=False,
            cache_index_dtype=torch.int32,
            device=torch.device("cuda"),
        )
        self.assertTrue(meta.can_fit([BUCKET // MAX_SEQS] * MAX_SEQS))
        self.assertFalse(meta.can_fit([1] * (MAX_SEQS + 1)))
        self.assertFalse(meta.can_fit([BUCKET + 1]))
        # Every (sequence, chunk) pair of a worst-case batch fits the tables.
        lens = [1] * (MAX_SEQS - 1) + [BUCKET - MAX_SEQS + 1]
        chunks = sum((n + CHUNK - 1) // CHUNK for n in lens)
        self.assertLessEqual(chunks, meta.max_chunks)


class TestFlashInferKDAPrefillGraph(TestKDAPrefillGraph):
    prefill_backend = LinearAttnKernelBackend.FLASHINFER

    def test_small_buckets_with_padded_sequences(self):
        for num_tokens in (1, 8, 16, 128):
            with self.subTest(num_tokens=num_tokens):
                meta, graph, out = self._capture(num_tokens=num_tokens, max_seqs=128)
                for lens in ([num_tokens], [1] * min(num_tokens, MAX_SEQS)):
                    self._check_replay(
                        meta,
                        graph,
                        out,
                        lens=lens,
                        prefixes=[64] * len(lens),
                        track={0: 70} if lens == [128] else None,
                        seed=400 + num_tokens,
                    )

    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            raise unittest.SkipTest("FlashInfer KDA prefill requires SM100/SM103")
        from flashinfer import kda

        if not hasattr(kda, "RecurrentKDAPrefillWrapper"):
            raise unittest.SkipTest("FlashInfer lacks the KDA prefill graph wrapper")


if __name__ == "__main__":
    unittest.main()
