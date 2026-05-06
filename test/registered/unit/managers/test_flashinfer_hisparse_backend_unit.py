"""Unit tests for FlashInferHiSparseDecodeBackend.

These tests verify the **integration glue** in ``forward_decode``:

  1. quest.retrieve_topk is called with (q, layer_id, req_pool_indices, seq_lens)
  2. coord.swap_in_selected_pages is called with retrieve_topk's output
  3. The result is copied verbatim into the wrapper's pre-allocated
     kv_indices buffer (so subsequent layers / cuda graph replays see fresh
     indices)
  4. wrapper.forward is invoked with the right shape

Quest, the coordinator, and the FlashInfer wrapper are all mocked.  Real
end-to-end exercise is deferred to the integration sub-chunks.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.utils import is_cuda, is_hip, is_npu, is_xpu

BS = 4
TOP_K = 64
NUM_QO_HEADS = 8
NUM_KV_HEADS = 2
HEAD_DIM = 64
MAX_BS = 8
POOL_SIZE = 4096


@unittest.skipUnless(
    torch.cuda.is_available()
    and not is_npu()
    and not is_xpu()
    and (is_cuda() or is_hip()),
    "FlashInfer hisparse backend tests require CUDA/ROCm.",
)
class TestFlashInferHiSparseDecodeBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda")
        # FlashInfer must be importable for the backend to construct.
        try:
            import flashinfer  # noqa: F401
        except ImportError as e:
            raise unittest.SkipTest(f"flashinfer not available: {e}")

    def _make_quest_mock(self, retrieve_value: torch.Tensor):
        """Create a Quest mock returning ``(retrieve_value, actual_lens)`` from
        retrieve_topk. The hisparse backend currently consumes only the
        positions (Task 3 will use actual_lens for variable-length packing)."""
        m = MagicMock()
        m.top_k = TOP_K
        bs = retrieve_value.shape[0]
        actual_lens = torch.full(
            (bs,), retrieve_value.shape[1],
            dtype=torch.int32, device=retrieve_value.device,
        )
        m.retrieve_topk = MagicMock(return_value=(retrieve_value, actual_lens))
        return m

    def _make_coord_mock(self, swap_in_value: torch.Tensor):
        """Create a Coordinator mock returning ``swap_in_value`` from swap_in_selected_pages."""
        m = MagicMock()
        m.swap_in_selected_pages = MagicMock(return_value=swap_in_value)
        return m

    def _make_backend(self, quest, coord):
        from sglang.srt.layers.attention.flashinfer_hisparse_backend import (
            FlashInferHiSparseDecodeBackend,
        )

        return FlashInferHiSparseDecodeBackend(
            quest_algorithm=quest,
            coordinator=coord,
            num_qo_heads=NUM_QO_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            max_bs=MAX_BS,
            kv_data_type=torch.bfloat16,
            q_data_type=torch.bfloat16,
            device=self.device,
        )

    def _make_forward_batch(self, bs: int, seq_len: int = 256):
        # Per-token KV pool with set_kv_buffer / get_kv_buffer that the
        # backend exercises.  We mock the whole pool — the test only cares
        # that the backend invokes them correctly.
        k_buf = torch.zeros(POOL_SIZE, NUM_KV_HEADS, HEAD_DIM,
                            dtype=torch.bfloat16, device=self.device)
        v_buf = torch.zeros_like(k_buf)
        pool_mock = MagicMock()
        pool_mock.set_kv_buffer = MagicMock()
        pool_mock.get_kv_buffer = MagicMock(return_value=(k_buf, v_buf))

        out_cache_loc = torch.arange(bs, dtype=torch.int64, device=self.device)
        return SimpleNamespace(
            batch_size=bs,
            req_pool_indices=torch.arange(bs, dtype=torch.int64, device=self.device),
            seq_lens=torch.full((bs,), seq_len, dtype=torch.int32, device=self.device),
            out_cache_loc=out_cache_loc,
            token_to_kv_pool=pool_mock,
        ), pool_mock

    def _make_layer(self, layer_id: int = 0):
        layer = SimpleNamespace(
            layer_id=layer_id,
            tp_q_head_num=NUM_QO_HEADS,
            tp_k_head_num=NUM_KV_HEADS,
            tp_v_head_num=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            v_head_dim=HEAD_DIM,
            scaling=1.0 / (HEAD_DIM ** 0.5),
            logit_cap=0.0,
            k_scale=None,
            v_scale=None,
            k_scale_float=None,
            v_scale_float=None,
            is_cross_attention=False,
        )
        return layer

    # -------------------------------------------------------- construction

    def test_construction_allocates_buffers(self):
        quest = self._make_quest_mock(
            torch.zeros(BS, TOP_K, dtype=torch.int32, device=self.device)
        )
        coord = self._make_coord_mock(
            torch.zeros(BS, TOP_K, dtype=torch.int32, device=self.device)
        )
        backend = self._make_backend(quest, coord)

        self.assertEqual(backend._kv_indptr_buf.shape, (MAX_BS + 1,))
        # +1 trailing scratch slot for graph-safe scatter-pack of invalid positions.
        self.assertEqual(backend._kv_indices_buf.shape, (MAX_BS * TOP_K + 1,))
        self.assertEqual(backend._kv_last_page_len_buf.shape, (MAX_BS,))
        # last_page_len is hardcoded to ones (page_size = 1).
        self.assertTrue(torch.all(backend._kv_last_page_len_buf == 1))
        # kv_indptr template should be cumulative top_k.
        self.assertTrue(
            torch.all(backend._kv_indptr_template
                      == torch.arange(0, (MAX_BS + 1) * TOP_K, step=TOP_K,
                                      dtype=torch.int32, device=self.device))
        )

    def test_init_forward_metadata_stamps_kv_indptr(self):
        quest = self._make_quest_mock(
            torch.zeros(BS, TOP_K, dtype=torch.int32, device=self.device)
        )
        coord = self._make_coord_mock(
            torch.zeros(BS, TOP_K, dtype=torch.int32, device=self.device)
        )
        backend = self._make_backend(quest, coord)
        # Pre-populate the wrapper cache so init_forward_metadata reuses
        # the cached entry instead of constructing a real flashinfer wrapper.
        # The cumulative-top_k stamp happens at wrapper-construction time,
        # so we stamp here too to reproduce the on-creation effect.
        backend._kv_indptr_buf[: BS + 1].copy_(
            backend._kv_indptr_template[: BS + 1]
        )
        self._stub_wrapper(backend, BS)
        forward_batch, _ = self._make_forward_batch(BS)
        backend.init_forward_metadata(forward_batch)
        expected = torch.arange(0, (BS + 1) * TOP_K, step=TOP_K,
                                dtype=torch.int32, device=self.device)
        torch.testing.assert_close(backend._kv_indptr_buf[: BS + 1], expected)
        self.assertEqual(backend._current_bs, BS)

    def test_init_forward_metadata_rejects_bs_over_max(self):
        quest = self._make_quest_mock(
            torch.zeros(MAX_BS + 1, TOP_K, dtype=torch.int32, device=self.device)
        )
        coord = self._make_coord_mock(
            torch.zeros(MAX_BS + 1, TOP_K, dtype=torch.int32, device=self.device)
        )
        backend = self._make_backend(quest, coord)
        forward_batch, _ = self._make_forward_batch(MAX_BS + 1)
        with self.assertRaises(RuntimeError):
            backend.init_forward_metadata(forward_batch)

    # ----------------------------------------------------- forward_decode

    def _stub_wrapper(self, backend, bs: int):
        """Pre-populate backend._wrappers[bs] with a mock so that
        init_forward_metadata short-circuits real wrapper construction.

        Also no-ops fast_decode_plan + monkey-patches the coordinator's
        req_to_host_pool / req_to_device_buffer to writable tensors so the
        capture-time pre-init can run without exploding on MagicMock state.
        """
        wrapper = MagicMock()
        wrapper.forward = MagicMock(
            return_value=torch.zeros(bs, NUM_QO_HEADS, HEAD_DIM,
                                     dtype=torch.bfloat16, device=self.device),
        )
        wrapper.run = wrapper.forward  # new forward_decode uses .run()
        backend._wrappers[bs] = wrapper
        # Skip fast_decode_plan in mock-driven tests; it pokes a lot of
        # internal wrapper state that's not worth replicating for unit cov.
        backend._fast_decode_plan = MagicMock()
        # Provide a real device tensor for coord swap-in pre-init writes
        # done in init_forward_metadata_capture_cuda_graph.
        backend.coord.req_to_host_pool = torch.zeros(
            backend.max_bs, dtype=torch.int64, device=self.device,
        )
        backend.coord.req_to_device_buffer = torch.zeros(
            backend.max_bs, dtype=torch.int64, device=self.device,
        )
        return wrapper

    def test_forward_decode_call_sequence(self):
        # Quest returns a known top-k pattern; coord returns a known
        # device-buffer-index pattern.  Verify both are called with the
        # right args, and the device-buffer indices end up in the wrapper
        # buffer.
        topk_positions = torch.arange(BS * TOP_K, dtype=torch.int32,
                                      device=self.device).view(BS, TOP_K) % 256
        device_buffer_indices = (
            torch.arange(BS * TOP_K, dtype=torch.int32, device=self.device).view(BS, TOP_K)
            + 1000
        )
        quest = self._make_quest_mock(topk_positions)
        coord = self._make_coord_mock(device_buffer_indices)
        backend = self._make_backend(quest, coord)
        wrapper_mock = self._stub_wrapper(backend, BS)

        forward_batch, pool_mock = self._make_forward_batch(BS, seq_len=256)
        backend.init_forward_metadata(forward_batch)

        layer = self._make_layer(layer_id=3)
        q = torch.randn(BS, NUM_QO_HEADS * HEAD_DIM,
                        dtype=torch.bfloat16, device=self.device)
        k = torch.randn(BS, NUM_KV_HEADS * HEAD_DIM,
                        dtype=torch.bfloat16, device=self.device)
        v = torch.randn(BS, NUM_KV_HEADS * HEAD_DIM,
                        dtype=torch.bfloat16, device=self.device)

        out = backend.forward_decode(q, k, v, layer, forward_batch)

        # 1. Quest invoked with the right per-step arguments.
        quest.retrieve_topk.assert_called_once()
        kw = quest.retrieve_topk.call_args.kwargs
        self.assertIs(kw["queries"], q)
        self.assertEqual(kw["layer_id"], 3)
        self.assertIs(kw["req_pool_indices"], forward_batch.req_pool_indices)
        self.assertIs(kw["seq_lens"], forward_batch.seq_lens)

        # 2. Coordinator invoked with Quest's output as top_k_result.
        coord.swap_in_selected_pages.assert_called_once()
        kw = coord.swap_in_selected_pages.call_args.kwargs
        self.assertIs(kw["req_pool_indices"], forward_batch.req_pool_indices)
        self.assertIs(kw["seq_lens"], forward_batch.seq_lens)
        self.assertIs(kw["top_k_result"], topk_positions)
        self.assertEqual(kw["layer_id"], 3)

        # 3. Coordinator's swap-in result landed in the wrapper buffer.
        torch.testing.assert_close(
            backend._kv_indices_buf[: BS * TOP_K], device_buffer_indices.view(-1),
        )

        # 4. Pool's set_kv_buffer was invoked exactly once with cache_loc.
        pool_mock.set_kv_buffer.assert_called_once()
        args, kwargs = pool_mock.set_kv_buffer.call_args
        self.assertIs(args[1], forward_batch.out_cache_loc)
        self.assertIs(args[2], k)
        self.assertIs(args[3], v)

        # 5. Wrapper.forward was called exactly once and the output flows back.
        wrapper_mock.forward.assert_called_once()
        self.assertEqual(out.shape, (BS, NUM_QO_HEADS * HEAD_DIM))

    def test_forward_decode_save_kv_cache_false_skips_set_kv_buffer(self):
        quest = self._make_quest_mock(
            torch.zeros(BS, TOP_K, dtype=torch.int32, device=self.device)
        )
        coord = self._make_coord_mock(
            torch.zeros(BS, TOP_K, dtype=torch.int32, device=self.device)
        )
        backend = self._make_backend(quest, coord)
        self._stub_wrapper(backend, BS)

        forward_batch, pool_mock = self._make_forward_batch(BS)
        backend.init_forward_metadata(forward_batch)

        layer = self._make_layer()
        q = torch.randn(BS, NUM_QO_HEADS * HEAD_DIM,
                        dtype=torch.bfloat16, device=self.device)
        k = torch.randn(BS, NUM_KV_HEADS * HEAD_DIM,
                        dtype=torch.bfloat16, device=self.device)
        v = torch.randn(BS, NUM_KV_HEADS * HEAD_DIM,
                        dtype=torch.bfloat16, device=self.device)

        backend.forward_decode(q, k, v, layer, forward_batch, save_kv_cache=False)
        pool_mock.set_kv_buffer.assert_not_called()
        # But Quest + Coord still get called.
        quest.retrieve_topk.assert_called_once()
        coord.swap_in_selected_pages.assert_called_once()

    def test_forward_decode_requires_init_forward_metadata(self):
        quest = self._make_quest_mock(
            torch.zeros(BS, TOP_K, dtype=torch.int32, device=self.device)
        )
        coord = self._make_coord_mock(
            torch.zeros(BS, TOP_K, dtype=torch.int32, device=self.device)
        )
        backend = self._make_backend(quest, coord)
        # Note: don't call init_forward_metadata; current_bs stays None.

        forward_batch, _ = self._make_forward_batch(BS)
        layer = self._make_layer()
        q = torch.randn(BS, NUM_QO_HEADS * HEAD_DIM,
                        dtype=torch.bfloat16, device=self.device)
        k = torch.randn(BS, NUM_KV_HEADS * HEAD_DIM,
                        dtype=torch.bfloat16, device=self.device)
        v = torch.randn(BS, NUM_KV_HEADS * HEAD_DIM,
                        dtype=torch.bfloat16, device=self.device)
        with self.assertRaises(RuntimeError):
            backend.forward_decode(q, k, v, layer, forward_batch)

    def test_forward_decode_replaces_kv_indices_each_call(self):
        """Two consecutive forward_decodes with different swap_in results
        must overwrite the buffer (no stale data from layer N-1)."""
        quest = self._make_quest_mock(
            torch.zeros(BS, TOP_K, dtype=torch.int32, device=self.device)
        )
        # Coordinator returns DIFFERENT values on each call.
        first = torch.full((BS, TOP_K), 11, dtype=torch.int32, device=self.device)
        second = torch.full((BS, TOP_K), 22, dtype=torch.int32, device=self.device)
        coord = MagicMock()
        coord.swap_in_selected_pages = MagicMock(side_effect=[first, second])

        backend = self._make_backend(quest, coord)
        self._stub_wrapper(backend, BS)

        forward_batch, _ = self._make_forward_batch(BS)
        backend.init_forward_metadata(forward_batch)
        layer = self._make_layer(layer_id=0)
        q = torch.randn(BS, NUM_QO_HEADS * HEAD_DIM,
                        dtype=torch.bfloat16, device=self.device)
        k = torch.randn(BS, NUM_KV_HEADS * HEAD_DIM,
                        dtype=torch.bfloat16, device=self.device)
        v = torch.randn(BS, NUM_KV_HEADS * HEAD_DIM,
                        dtype=torch.bfloat16, device=self.device)

        backend.forward_decode(q, k, v, layer, forward_batch)
        self.assertTrue(torch.all(backend._kv_indices_buf[: BS * TOP_K] == 11))

        # Layer 1 — different swap-in result must overwrite layer 0's.
        layer.layer_id = 1
        backend.forward_decode(q, k, v, layer, forward_batch)
        self.assertTrue(torch.all(backend._kv_indices_buf[: BS * TOP_K] == 22))


    # -------------------------------------------------- cuda graph wrapper cache

    def test_prepare_wrappers_for_bs_caches_each(self):
        quest = self._make_quest_mock(
            torch.zeros(BS, TOP_K, dtype=torch.int32, device=self.device)
        )
        coord = self._make_coord_mock(
            torch.zeros(BS, TOP_K, dtype=torch.int32, device=self.device)
        )
        backend = self._make_backend(quest, coord)
        # Pre-stub three bs values so prepare_wrappers_for_bs short-circuits
        # real flashinfer construction for each.
        for bs in (2, 4, 8):
            self._stub_wrapper(backend, bs)
        backend.prepare_wrappers_for_bs([2, 4, 8])
        # Cache should have all three (no real construction happened).
        self.assertEqual(set(backend._wrappers.keys()), {2, 4, 8})

    def _capture_args(self, bs):
        """Build the full AttentionBackend.init_forward_metadata_capture_cuda_graph
        argument tuple — most are unused by our backend but required by the
        interface."""
        return dict(
            bs=bs,
            num_tokens=bs,
            req_pool_indices=torch.arange(bs, dtype=torch.int64, device=self.device),
            seq_lens=torch.full((bs,), 128, dtype=torch.int32, device=self.device),
            encoder_lens=None,
            forward_mode=None,
            spec_info=None,
        )

    def _replay_args(self, bs):
        return dict(
            bs=bs,
            req_pool_indices=torch.arange(bs, dtype=torch.int64, device=self.device),
            seq_lens=torch.full((bs,), 128, dtype=torch.int32, device=self.device),
            seq_lens_sum=bs * 128,
            encoder_lens=None,
            forward_mode=None,
            spec_info=None,
        )

    def test_capture_then_replay_uses_same_wrapper(self):
        quest = self._make_quest_mock(
            torch.zeros(BS, TOP_K, dtype=torch.int32, device=self.device)
        )
        coord = self._make_coord_mock(
            torch.zeros(BS, TOP_K, dtype=torch.int32, device=self.device)
        )
        backend = self._make_backend(quest, coord)
        wrapper_mock = self._stub_wrapper(backend, BS)

        backend.init_forward_metadata_capture_cuda_graph(**self._capture_args(BS))
        captured_wrapper = backend._wrapper
        self.assertIs(captured_wrapper, wrapper_mock)

        # Switch away (simulate a different bs being active).
        other_mock = self._stub_wrapper(backend, BS // 2)
        backend.init_forward_metadata_capture_cuda_graph(**self._capture_args(BS // 2))
        self.assertIs(backend._wrapper, other_mock)

        # Replay at BS — must return to the originally captured wrapper.
        backend.init_forward_metadata_replay_cuda_graph(**self._replay_args(BS))
        self.assertIs(backend._wrapper, captured_wrapper)
        self.assertEqual(backend._current_bs, BS)

    def test_replay_without_capture_raises(self):
        quest = self._make_quest_mock(
            torch.zeros(BS, TOP_K, dtype=torch.int32, device=self.device)
        )
        coord = self._make_coord_mock(
            torch.zeros(BS, TOP_K, dtype=torch.int32, device=self.device)
        )
        backend = self._make_backend(quest, coord)
        # No wrapper for bs=BS in cache.
        with self.assertRaises(RuntimeError):
            backend.init_forward_metadata_replay_cuda_graph(**self._replay_args(BS))


if __name__ == "__main__":
    unittest.main()
