"""Inkling's short-conv metadata must be resolved exactly ONCE per forward step.

A decoder layer holds FOUR ``ShortConvolution`` modules, and per-layer ownership
would recompute the whole set once per module. Pinned here: one resolution per step
however many modules ask, every module gets the *same* tensors, and the graph-path
destinations stay address-stable across steps -- including across a later
``init_cuda_graph_state``, where reallocating would move an address an
already-captured prefill graph reads.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

NUM_LAYERS = 4
NUM_SCONV_STREAMS = 6  # pool-wide streams: k/v full, k/v local, attn, mlp
NUM_MODULES_PER_LAYER = 4  # k_sconv, v_sconv, attn_sconv, mlp_sconv
POOL_SLOTS = 32
CONV_KERNEL = 4
CONV_DIM = 8


class _MockMambaPool:
    enable_linear_replayssm = False

    def __init__(self):
        conv = [
            torch.zeros(
                (NUM_LAYERS, POOL_SLOTS + 1, CONV_KERNEL - 1, CONV_DIM),
                dtype=torch.bfloat16,
                device="cuda",
            )
            for _ in range(NUM_SCONV_STREAMS)
        ]
        self.mamba_cache = SimpleNamespace(conv=conv, temporal=None)

    def mamba2_layer_cache(self, layer_id: int):
        return SimpleNamespace(
            conv=[c[layer_id] for c in self.mamba_cache.conv],
            intermediate_conv_window=None,
        )


class _MockReqToTokenPool:
    """The four methods the backend calls, plus ``size`` (its max-bs bound)."""

    def __init__(self):
        self.size = POOL_SLOTS
        self.mamba_pool = _MockMambaPool()
        self.req_index_to_mamba_index_mapping = torch.arange(
            POOL_SLOTS + 1, dtype=torch.int32, device="cuda"
        )
        self.gather_calls = 0

    def get_mamba_indices(self, req_indices: torch.Tensor) -> torch.Tensor:
        self.gather_calls += 1
        return self.req_index_to_mamba_index_mapping[req_indices]

    def translate_mamba_indices(self, mamba_indices: torch.Tensor) -> torch.Tensor:
        return mamba_indices

    def mamba2_layer_cache(self, layer_id: int):
        return self.mamba_pool.mamba2_layer_cache(layer_id)

    def get_speculative_mamba2_params_all_layers(self):
        return self.mamba_pool.mamba_cache


def _decode_batch(bs: int):
    return SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        batch_size=bs,
        req_pool_indices=torch.arange(bs, dtype=torch.int64, device="cuda"),
        seq_lens=torch.full((bs,), 64, dtype=torch.int64, device="cuda"),
        spec_info=None,
        mamba_track_mask=None,
        mamba_track_seqlens=None,
        mamba_track_indices=None,
    )


def _extend_batch(seq_lens):
    bs = len(seq_lens)
    lens = torch.tensor(seq_lens, dtype=torch.int64, device="cuda")
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        batch_size=bs,
        req_pool_indices=torch.arange(bs, dtype=torch.int64, device="cuda"),
        seq_lens=lens,
        extend_seq_lens=lens,
        extend_prefix_lens=torch.zeros(bs, dtype=torch.int64, device="cuda"),
        extend_num_tokens=int(sum(seq_lens)),
        spec_info=None,
        mamba_track_mask=torch.ones(bs, dtype=torch.bool, device="cuda"),
        mamba_track_seqlens=lens,
        mamba_track_indices=torch.arange(bs, dtype=torch.int64, device="cuda"),
    )


class TestInklingSconvMetadataOnce(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("Inkling's conv metadata kernels are CUDA-only.")
        server_args = ServerArgs(
            model_path="dummy",
            page_size=1,
            # Skips the model-config load in the Inkling prefill-graph default.
            disable_prefill_cuda_graph=True,
            disable_cuda_graph=True,
        )
        # Pre-seed the cached property so it does not reach for a real HF config.
        server_args._mamba_cache_chunk_size = 64
        set_global_server_args_for_scheduler(server_args)

    def _build_backend(self):
        from sglang.srt.layers.attention.linear.inkling_sconv_backend import (
            InklingShortConvAttnBackend,
        )

        pool = _MockReqToTokenPool()
        from sglang.srt.runtime_context import get_server_args

        runner = SimpleNamespace(
            device="cuda",
            server_args=get_server_args(),
            is_draft_worker=False,
            req_to_token_pool=pool,
            token_to_kv_pool=None,
        )
        return InklingShortConvAttnBackend(runner), pool

    def _count_fused_calls(self, backend):
        """Wrap the two fused metadata entry points with counters."""
        import sglang.srt.layers.attention.linear.inkling_sconv_backend as mod

        counts = {"decode": 0, "extend": 0}
        real_decode = mod.fused_decode_sconv_metadata
        real_extend = mod.fused_extend_sconv_metadata

        def decode(*a, **kw):
            counts["decode"] += 1
            return real_decode(*a, **kw)

        def extend(*a, **kw):
            counts["extend"] += 1
            return real_extend(*a, **kw)

        mod.fused_decode_sconv_metadata = decode
        mod.fused_extend_sconv_metadata = extend
        self.addCleanup(setattr, mod, "fused_decode_sconv_metadata", real_decode)
        self.addCleanup(setattr, mod, "fused_extend_sconv_metadata", real_extend)
        return counts

    def _drain_all_conv_modules(self, backend, forward_batch):
        """Mimic every ShortConvolution in the model asking for its handle."""
        handles = []
        for layer_id in range(NUM_LAYERS):
            for _module in range(NUM_MODULES_PER_LAYER):
                handles.append(backend.conv_state_metadata(layer_id, forward_batch))
        return handles

    def test_decode_resolves_once_per_step(self):
        backend, pool = self._build_backend()
        counts = self._count_fused_calls(backend)
        fb = _decode_batch(bs=3)

        backend.init_forward_metadata(fb)
        handles = self._drain_all_conv_modules(backend, fb)

        self.assertEqual(counts["decode"], 1)
        self.assertEqual(pool.gather_calls, 1)
        self.assertEqual(len(handles), NUM_LAYERS * NUM_MODULES_PER_LAYER)
        first = handles[0]
        for h in handles[1:]:
            self.assertIs(h.cache_indices, first.cache_indices)
            self.assertIs(h.precomputed, first.precomputed)
            self.assertIs(h.query_start_loc, first.query_start_loc)
            self.assertIs(h.has_initial_state, first.has_initial_state)

    def test_extend_resolves_once_per_step(self):
        backend, pool = self._build_backend()
        counts = self._count_fused_calls(backend)
        fb = _extend_batch([7, 5, 3])

        backend.init_forward_metadata(fb)
        handles = self._drain_all_conv_modules(backend, fb)

        self.assertEqual(counts["extend"], 1)
        self.assertEqual(pool.gather_calls, 1)
        first = handles[0]
        self.assertIsNotNone(first.track_conv_indices)
        self.assertEqual(tuple(first.track_conv_indices.shape), (3, CONV_KERNEL - 1))
        for h in handles[1:]:
            self.assertIs(h.track_conv_indices, first.track_conv_indices)
            self.assertIs(h.precomputed, first.precomputed)

    def test_each_step_re_resolves(self):
        """A second forward must recompute; nothing may leak across steps."""
        backend, pool = self._build_backend()
        counts = self._count_fused_calls(backend)
        fb = _decode_batch(bs=2)

        for _ in range(3):
            backend.init_forward_metadata(fb)
            self._drain_all_conv_modules(backend, fb)

        self.assertEqual(counts["decode"], 3)
        self.assertEqual(pool.gather_calls, 3)

    def test_graph_destinations_are_address_stable(self):
        for slots_in_graph in (False, True):
            with self.subTest(slots_in_graph=slots_in_graph):
                self._check_address_stable(slots_in_graph)

    def _check_address_stable(self, slots_in_graph: bool):
        """A captured graph holds each metadata tensor's address, so steps refill in
        place and a later ``init_cuda_graph_state`` must not reallocate."""
        backend, _pool = self._build_backend()
        # Cover both halves of the slot split (the mock's translate is not the base
        # one, so slots would otherwise always stay eager).
        backend._slot_gather_recordable = slots_in_graph
        fb = _decode_batch(bs=2)

        # Mirrors the decode runner: out-of-graph prep, then the recorded hook.
        backend.init_forward_metadata_out_graph(fb, in_capture=True)
        backend.init_forward_metadata_in_graph(fb)
        h0 = backend.conv_state_metadata(0, fb)
        ptrs = (
            h0.cache_indices.data_ptr(),
            h0.query_start_loc.data_ptr(),
            h0.has_initial_state.data_ptr(),
            h0.precomputed["cache_mask"].data_ptr(),
            h0.precomputed["safe_idx"].data_ptr(),
            h0.precomputed["cu"].data_ptr(),
            h0.precomputed["si"].data_ptr(),
        )

        backend.init_cuda_graph_state(max_bs=8, max_num_tokens=8)
        backend.init_forward_metadata_out_graph(fb)
        backend.init_forward_metadata_in_graph(fb)
        h1 = backend.conv_state_metadata(0, fb)
        self.assertEqual(
            ptrs,
            (
                h1.cache_indices.data_ptr(),
                h1.query_start_loc.data_ptr(),
                h1.has_initial_state.data_ptr(),
                h1.precomputed["cache_mask"].data_ptr(),
                h1.precomputed["safe_idx"].data_ptr(),
                h1.precomputed["cu"].data_ptr(),
                h1.precomputed["si"].data_ptr(),
            ),
        )


class TestInklingMtpVerifyCommit(CustomTestCase):
    """The commit runs after the forward context exits, so the per-step slot buffer
    may already belong to a later forward. Sourcing slot ids from
    ``forward_metadata`` (as the generic mamba path does) therefore mismatches the
    verify batch; they must come from the passed ``req_pool_indices``.
    """

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("Inkling's conv-state kernels are CUDA-only.")
        TestInklingSconvMetadataOnce.setUpClass()

    def _build_wrapper(self):
        from sglang.srt.layers.attention.linear.inkling_sconv_backend import (
            InklingShortConvAttnBackend,
            InklingShortConvHybridAttnBackend,
        )
        from sglang.srt.runtime_context import get_server_args

        pool = _MockReqToTokenPool()
        runner = SimpleNamespace(
            device="cuda",
            server_args=get_server_args(),
            is_draft_worker=False,
            req_to_token_pool=pool,
            token_to_kv_pool=None,
        )
        sidecar = InklingShortConvAttnBackend(runner)
        full = SimpleNamespace(
            token_to_kv_pool=None,
            req_to_token_pool=pool,
            needs_cpu_seq_lens=True,
        )
        wrapper = InklingShortConvHybridAttnBackend(
            full, sidecar, list(range(NUM_LAYERS))
        )
        return wrapper, sidecar, pool

    def test_commit_uses_passed_req_pool_indices_not_step_metadata(self):
        wrapper, sidecar, pool = self._build_wrapper()

        # The hazard: a later forward left a SHORTER slot buffer than the verify
        # batch this commit is for.
        sidecar.init_forward_metadata(_decode_batch(bs=3))
        self.assertEqual(sidecar._cache_indices.shape[0], 3)

        seen = {}

        def fake_scatter(caches, state_indices, last_correct, track, steps):
            seen["state_indices"] = state_indices

        import sglang.srt.layers.attention.linear.inkling_sconv_backend as mod

        real = mod.scatter_mamba_states_after_mtp_verify
        mod.scatter_mamba_states_after_mtp_verify = fake_scatter
        self.addCleanup(setattr, mod, "scatter_mamba_states_after_mtp_verify", real)

        req_pool_indices = torch.arange(5, dtype=torch.int64, device="cuda")
        wrapper.update_mamba_state_after_mtp_verify(
            last_correct_step_indices=torch.zeros(5, dtype=torch.int64, device="cuda"),
            mamba_track_indices=None,
            mamba_steps_to_track=None,
            model=None,
            req_pool_indices=req_pool_indices,
        )
        # 5 rows from req_pool_indices, not the 3 on the step buffer.
        self.assertEqual(seen["state_indices"].shape[0], 5)
        self.assertTrue(
            torch.equal(seen["state_indices"], pool.get_mamba_indices(req_pool_indices))
        )

    def test_commit_requires_req_pool_indices(self):
        """The generic caller signature makes it optional; Inkling cannot guess it."""
        wrapper, _sidecar, _pool = self._build_wrapper()
        with self.assertRaises(AssertionError):
            wrapper.update_mamba_state_after_mtp_verify(
                last_correct_step_indices=torch.zeros(
                    2, dtype=torch.int64, device="cuda"
                ),
                mamba_track_indices=None,
                mamba_steps_to_track=None,
                model=None,
            )


if __name__ == "__main__":
    unittest.main()
