"""Decoder SWA bounded replay inside the HIP breakable prefill CUDA graph: the
late-layer tail block's persistent buffers, its padding rows, and the tail switch."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.attention import deepseek_v4_backend
from sglang.srt.layers.attention.deepseek_v4_backend import SWA_WINDOW
from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
    DeepseekV4HipRadixBackend,
    DSV4Metadata,
)
from sglang.srt.model_executor.forward_mode import CaptureHiddenMode, ForwardMode
from sglang.srt.models import deepseek_v4_replay_tail as replay_tail
from sglang.srt.models.deepseek_v4_mhc import HcPending, HcState
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=8, suite="stage-b-test-1-gpu-large-amd")

CPU = torch.device("cpu")
MAX_REQS = 4
BLOCK_TOKENS = 1024


def _backend():
    backend = object.__new__(DeepseekV4HipRadixBackend)
    backend.enable_decoder_swa_bounded_replay = True
    backend.is_draft_worker = False
    backend.is_dspark = False
    backend._bcg_tail_capacity = None
    backend._bcg_tail_buffers = None
    backend._bcg_tail_metadata = {}
    backend._bcg_tail_num_token_non_padded = None
    backend.prefill_graph_late_layer_saved = None
    backend.candidate_masks = None
    backend.tail_source_kv_layer_id = None
    backend.model_runner = SimpleNamespace(
        server_args=SimpleNamespace(max_running_requests=MAX_REQS),
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    # full slot s maps to SWA slot s + 1000, the dummy slot 0 to 0
    mapping = torch.arange(1 << 16) + 1000
    mapping[0] = 0
    backend.token_to_kv_pool = SimpleNamespace(
        translate_loc_from_full_to_swa=lambda loc: mapping[loc],
        request_window=None,
    )
    backend._refresh_fp4_prefill_workspace = lambda *args, **kwargs: None
    backend.built = []

    def init_forward_metadata_prefill(**kwargs):
        backend.built.append(kwargs)
        return DSV4Metadata(
            core_attn_metadata=MagicMock(request_window_layout=None),
            indexer_metadata=None,
        )

    backend.init_forward_metadata_prefill = init_forward_metadata_prefill
    return backend


def _batches(extend_lens, prefix=1000):
    """The live batch and its static copy padded to BLOCK_TOKENS rows."""
    n = sum(extend_lens)
    seq_lens = [prefix + e for e in extend_lens]
    positions = torch.cat([torch.arange(prefix, s) for s in seq_lens])
    out_cache_loc = torch.arange(1, n + 1) + 5 * len(extend_lens)

    def batch(rows):
        pad = rows - n
        return SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens_cpu=list(extend_lens),
            extend_logprob_start_lens_cpu=None,
            return_logprob=False,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            seq_lens_cpu=torch.tensor(seq_lens),
            seq_lens=torch.tensor(seq_lens),
            req_pool_indices=torch.arange(1, len(extend_lens) + 1),
            positions=torch.cat([positions, positions.new_zeros(pad)]),
            out_cache_loc=torch.cat([out_cache_loc, out_cache_loc.new_zeros(pad)]),
            global_num_token_non_padded_cpu=n,
            global_num_token_non_padded=torch.tensor([n]),
            num_token_non_padded=torch.tensor([n]),
        )

    return batch(n), batch(max(n, BLOCK_TOKENS))


def _expected_rows(extend_lens):
    rows, start = [], 0
    for e in extend_lens:
        rows += list(range(start + max(e - SWA_WINDOW, 0), start + e))
        start += e
    return rows


class TestHipPrefillGraphTailBlock(CustomTestCase):
    def _tail(self, backend, extend_lens, capture=False):
        live, static = _batches(extend_lens)
        metadata = backend._prefill_graph_tail_metadata(
            live, None, static_forward_batch=static, capture=capture
        )
        return metadata, live, static

    def test_block_rows_and_padding(self):
        cap = MAX_REQS * SWA_WINDOW
        for extend_lens in ([300], [200, 50, 300]):
            with self.subTest(extend_lens=extend_lens):
                backend = _backend()
                metadata, live, _ = self._tail(backend, extend_lens)
                tail = metadata.late_layer_tail
                rows = _expected_rows(extend_lens)
                n = len(rows)
                self.assertEqual(tail.real_rows_count, n)
                self.assertIsNone(tail.contiguous_start)
                for buf in (tail.token_indices, tail.positions, tail.scatter_indices):
                    self.assertEqual(buf.shape[0], cap)
                self.assertEqual(tail.token_indices[:n].tolist(), rows)
                self.assertEqual(tail.scatter_indices[:n].tolist(), rows)
                self.assertTrue(bool((tail.token_indices[n:] == 0).all()))
                self.assertTrue(bool((tail.scatter_indices[n:] == BLOCK_TOKENS).all()))
                self.assertTrue(bool((tail.positions[n:] == 0).all()))
                self.assertTrue(
                    torch.equal(tail.positions[:n], live.positions[torch.tensor(rows)])
                )
                # Real rows store to their SWA slots, padding rows to the dummy slot.
                expected_swa = live.out_cache_loc[torch.tensor(rows)] + 1000
                self.assertTrue(torch.equal(tail.swa_out_cache_loc[:n], expected_swa))
                self.assertTrue(bool((tail.swa_out_cache_loc[n:] == 0).all()))
                self.assertIs(
                    metadata.core_attn_metadata.swa_out_cache_loc,
                    tail.swa_out_cache_loc,
                )
                # The break-time metadata covers the live rows only.
                self.assertEqual(backend.built[-1]["num_tokens"], n)
                self.assertEqual(backend.built[-1]["out_cache_loc"].shape[0], n)
                self.assertEqual(metadata.low_ratio_req_indices.shape[0], n)

    def test_captures_share_one_object_per_block_size(self):
        backend = _backend()
        first, _, _ = self._tail(backend, [300], capture=True)
        first.q_pad_buffer = torch.zeros(1)
        second, _, _ = self._tail(backend, [200, 50, 300], capture=True)
        self.assertIs(second, first)
        self.assertIsNotNone(second.q_pad_buffer)

    def test_replay_rewrites_the_captured_buffers_in_place(self):
        backend = _backend()
        captured, _, _ = self._tail(backend, [300], capture=True)
        captured_tail = captured.late_layer_tail
        captured_core = captured.core_attn_metadata
        ptrs = [
            t.data_ptr()
            for t in (
                captured_tail.token_indices,
                captured_tail.positions,
                captured_tail.scatter_indices,
                captured_tail.swa_out_cache_loc,
            )
        ]
        captured.q_pad_buffer = torch.zeros(1)
        replayed, _, _ = self._tail(backend, [200, 50, 300])
        # A replay holds its own live-width build; the captured object is untouched.
        self.assertIsNot(replayed, captured)
        self.assertIs(captured.late_layer_tail, captured_tail)
        self.assertIs(captured.core_attn_metadata, captured_core)
        self.assertEqual(captured_tail.real_rows_count, SWA_WINDOW)
        self.assertEqual(backend.built[-1]["max_seq_len"], 1000 + 300)
        tail = replayed.late_layer_tail
        self.assertEqual(
            ptrs,
            [
                t.data_ptr()
                for t in (
                    tail.token_indices,
                    tail.positions,
                    tail.scatter_indices,
                    tail.swa_out_cache_loc,
                )
            ],
        )
        self.assertEqual(tail.real_rows_count, len(_expected_rows([200, 50, 300])))
        self.assertIsNotNone(captured.q_pad_buffer)

    def test_keep_fields_survive_a_refresh(self):
        held = DSV4Metadata(
            core_attn_metadata=MagicMock(),
            indexer_metadata=None,
            prefill_graph_tail=True,
        )
        held.core_attn_metadata.low_ratios = (1, 2)
        fresh = DSV4Metadata(core_attn_metadata=MagicMock(), indexer_metadata=None)
        held.refresh_for_breakable_cuda_graph_replay_(fresh)
        self.assertTrue(held.prefill_graph_tail)


class TestHipPrefillGraphLiveWidth(CustomTestCase):
    """The full rows' metadata spans the live context at any length, and a replay
    leaves the graph bucket's capture-time metadata in place."""

    def _replay(self, *, seq_len, with_tail=False):
        backend = _backend()
        backend.low_ratios = (1, 2)
        backend._opus_prefill_state = None
        backend.enable_decoder_swa_bounded_replay = with_tail
        widths = []

        def build(forward_batch, *, max_seq_len_override):
            widths.append(max_seq_len_override)
            return DSV4Metadata(
                core_attn_metadata=SimpleNamespace(swa_out_cache_loc=None),
                indexer_metadata=None,
            )

        def in_graph(forward_batch):
            # the live SWA store target, from the padded static batch
            backend.forward_metadata.core_attn_metadata.swa_out_cache_loc = (
                forward_batch.out_cache_loc.to(torch.int32) + 1000
            )

        backend._prefill_metadata_for_batch = build
        backend.init_forward_metadata_in_graph = in_graph
        captured_loc = torch.zeros(BLOCK_TOKENS, dtype=torch.int32)
        captured_core = SimpleNamespace(
            swa_out_cache_loc=captured_loc, page_table="cap"
        )
        capture = DSV4Metadata(
            core_attn_metadata=captured_core,
            indexer_metadata=None,
            prefill_graph_tail=with_tail,
        )
        capture.q_pad_buffer = torch.zeros(1)
        live, static = _batches([300], prefix=seq_len - 300)
        # A user-set --cuda-graph-prefill-max-context no longer pins the width.
        live.max_seq_len_override = static.max_seq_len_override = 16384
        backend.prepare_forward_metadata_for_breakable_cuda_graph_replay(
            capture, live, static_forward_batch=static
        )
        return backend, capture, captured_core, captured_loc, widths, static

    def test_replay_keeps_the_bucket_and_refreshes_only_the_store_target(self):
        backend, capture, core, loc, widths, static = self._replay(seq_len=200_000)
        live = backend.forward_metadata
        self.assertIsNot(live, capture)
        self.assertIs(capture.core_attn_metadata, core)
        self.assertEqual(core.page_table, "cap")
        self.assertIsNotNone(capture.q_pad_buffer)
        self.assertIs(live.core_attn_metadata.swa_out_cache_loc, loc)
        self.assertTrue(torch.equal(loc, static.out_cache_loc.to(torch.int32) + 1000))
        self.assertEqual(widths, [200_000])
        self.assertIsNone(backend.tail_forward_metadata)

    def test_replay_builds_a_fresh_tail(self):
        backend, _, _, _, _, _ = self._replay(seq_len=200_000, with_tail=True)
        tail = backend.tail_forward_metadata
        self.assertIsNotNone(tail.late_layer_tail.scatter_indices)
        self.assertEqual(backend.built[-1]["max_seq_len"], 200_000)
        self.assertEqual(backend._bcg_tail_metadata, {})

    def test_no_default_replay_limit(self):
        self.assertIsNone(_backend().prefill_cuda_graph_default_max_seq_len)


class TestHipPrefillGraphTailSwitch(CustomTestCase):
    def test_enter_bounds_the_rows_and_exit_restores(self):
        backend = _backend()
        live, static = _batches([200, 50, 300])
        backend.tail_forward_metadata = backend._prefill_graph_tail_metadata(
            live, None, static_forward_batch=static
        )
        real = backend.tail_forward_metadata.late_layer_tail.real_rows_count
        full_metadata = object()
        backend.forward_metadata = full_metadata
        backend.tail_source_kv_layer_id = 21
        before = (
            static.global_num_token_non_padded_cpu,
            static.global_num_token_non_padded,
            static.num_token_non_padded,
        )
        backend.prefill_graph_enter_late_layer_tail(static)
        self.assertIs(backend.forward_metadata, backend.tail_forward_metadata)
        self.assertEqual(static.global_num_token_non_padded_cpu, real)
        self.assertEqual(static.global_num_token_non_padded.tolist(), [real])
        self.assertIs(static.num_token_non_padded, static.global_num_token_non_padded)
        bound = static.global_num_token_non_padded
        backend.prefill_graph_exit_late_layer_tail(static)
        self.assertIs(backend.forward_metadata, full_metadata)
        self.assertIsNone(backend.tail_source_kv_layer_id)
        self.assertEqual(
            before,
            (
                static.global_num_token_non_padded_cpu,
                static.global_num_token_non_padded,
                static.num_token_non_padded,
            ),
        )
        # A second replay clamps into the same buffer the captured MoE reads.
        backend.tail_source_kv_layer_id = 21
        backend.prefill_graph_enter_late_layer_tail(static)
        self.assertEqual(
            static.global_num_token_non_padded.data_ptr(), bound.data_ptr()
        )

    def test_graph_check_and_vote_run_on_hip(self):
        backend = _backend()
        with (
            patch.object(
                deepseek_v4_backend, "_prefill_graph_max_seq_len", return_value=1 << 20
            ),
            patch.object(
                deepseek_v4_backend,
                "get_server_return_hidden_states_mode",
                return_value=CaptureHiddenMode.NULL,
            ),
        ):
            for num_reqs, fits in ((MAX_REQS, True), (MAX_REQS + 1, False)):
                live, _ = _batches([300] * num_reqs)
                schedule_batch = SimpleNamespace(
                    extend_lens=[300] * num_reqs,
                    extend_logprob_start_lens=None,
                    return_logprob=False,
                    reqs=[
                        SimpleNamespace(
                            return_hidden_states_mode=CaptureHiddenMode.NULL
                        )
                    ]
                    * num_reqs,
                )
                self.assertEqual(backend.can_run_prefill_cuda_graph(live), fits)
                self.assertEqual(backend.vote_prefill_cuda_graph(schedule_batch), fits)


class TestHipDecoderTailGraphs(CustomTestCase):
    """The decoder tail's row-bucket graphs on HIP: the fused boundary's pending post
    as graph inputs, and the pinned store target their captured KV stores read."""

    def test_pending_post_round_trips_through_the_graph_inputs(self):
        rows = [torch.randn(5, 3) for _ in range(4)]
        for pre in (None, torch.randn(5, 2)):
            with self.subTest(pre=pre is not None):
                state = HcState(HcPending(*rows), pre)
                leaves, has_pre, rebuild = replay_tail._flatten_state(state)
                self.assertEqual(len(leaves), 5 if pre is not None else 4)
                self.assertEqual(has_pre, pre is not None)
                rebuilt = rebuild(leaves)
                self.assertIsInstance(rebuilt.streams, HcPending)
                for a, b in zip(rebuilt.streams, rows):
                    self.assertIs(a, b)
                self.assertIs(rebuilt.pre, pre)
        plain = HcState(torch.randn(5, 3), None)
        leaves, _, rebuild = replay_tail._flatten_state(plain)
        self.assertIs(rebuild(leaves).streams, plain.streams)

    def _store_backend(self, capacity=8):
        backend = _backend()
        backend._tail_graph_swa_loc = torch.zeros(capacity, dtype=torch.int32)
        return backend

    def _prepare(self, backend, live):
        backend.forward_metadata = SimpleNamespace(
            late_layer_tail=SimpleNamespace(swa_out_cache_loc=live)
        )
        backend.prepare_decoder_tail_store(live.shape[0])

    def test_store_target_pads_with_the_dummy_slot_at_a_fixed_address(self):
        backend = self._store_backend()
        ptr = backend._tail_graph_swa_loc.data_ptr()
        self._prepare(backend, torch.arange(11, 17, dtype=torch.int32))
        self._prepare(backend, torch.tensor([21, 22, 23], dtype=torch.int32))
        # The previous step's longer rows are not left in the padding.
        self.assertEqual(
            backend._tail_graph_swa_loc.tolist(), [21, 22, 23, 0, 0, 0, 0, 0]
        )
        with replay_tail._replay_tail_scope():
            # A capture or replay of a 4-row bucket reads its padding row too.
            target = backend._decoder_tail_store_target(
                SimpleNamespace(global_num_token_non_padded_cpu=4)
            )
            self.assertEqual(target.tolist(), [21, 22, 23, 0])
            self.assertEqual(target.data_ptr(), ptr)
            # An eager fallback past the buffer reads the tail's own rows.
            self.assertIsNone(
                backend._decoder_tail_store_target(
                    SimpleNamespace(global_num_token_non_padded_cpu=9)
                )
            )
        self.assertIsNone(
            backend._decoder_tail_store_target(
                SimpleNamespace(global_num_token_non_padded_cpu=4)
            )
        )

    def test_rows_past_the_buffer_leave_it_untouched(self):
        backend = self._store_backend(capacity=4)
        self._prepare(backend, torch.tensor([5, 6], dtype=torch.int32))
        self._prepare(backend, torch.arange(1, 7, dtype=torch.int32))
        self.assertEqual(backend._tail_graph_swa_loc.tolist(), [5, 6, 0, 0])

    def test_tail_graphs_on_hip_without_attention_dp(self):
        model = SimpleNamespace(
            tail_writes_source_kv=True, late_layer_start=21, end_layer=43
        )
        for dp_size, created in ((1, True), (2, False)):
            with (
                self.subTest(dp_size=dp_size),
                patch.object(replay_tail, "is_cuda", return_value=False),
                patch.object(replay_tail, "is_hip", return_value=True),
                patch.object(
                    replay_tail,
                    "get_parallel",
                    return_value=SimpleNamespace(attn_dp_size=dp_size),
                ),
            ):
                tail = replay_tail.DecoderReplayTail.maybe_create(model)
                self.assertEqual(tail is not None, created)


if __name__ == "__main__":
    unittest.main()
