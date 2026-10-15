"""Eager replay graphs of DeepSeek-V4 layer ranges: bucketing, the state's static-buffer
round trip, refill without recapture, no capture outside the startup scope, the eager
fallback, the late layers as one prefill-graph break, and the replay pointer guard.
CPU only; capture runs on GPU."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.models import deepseek_v4_replay_graphs as rg
from sglang.srt.models.deepseek_v4_mhc import HcPending, HcState
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def setUpModule():
    # The default probes read the live KV pool and flashinfer; CPU tests use none.
    patcher = patch.dict(rg._pointer_probes, clear=True)
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)


def _graphs(buckets=(128, 256, 384, 512), tail_rows=True):
    graphs = rg.EagerReplayGraphs(
        name="test",
        model=None,
        run_layers=None,
        buckets=list(buckets),
        tail_rows=tail_rows,
    )
    graphs._break_context = lambda batch: nullcontext()
    return graphs


def _fake_capture(graphs, captured):
    def capture(
        *, rows, num_rows, shape_key, static, num_leaves, rebuild, names, forward_batch
    ):
        out = [t[:rows] * 2 for t in static[:num_leaves]]
        graph = rg._ReplayGraph(
            SimpleNamespace(replay=MagicMock()),
            rows,
            lambda n: rebuild([t[:n] for t in out]),
            {},
            {
                "num_token_non_padded": torch.zeros((), dtype=torch.int32),
                "q_pad_buffer": None,
            },
            None,
        )
        captured.append(graph)
        return graph

    graphs._capture = MagicMock(side_effect=capture)


def _step(graphs, n, seed=0):
    torch.manual_seed(seed)
    state = HcState(torch.randn(n, 4))
    out = graphs.run(
        state=state,
        forward_batch=SimpleNamespace(global_num_token_non_padded_cpu=None),
        positions=torch.arange(n),
        hash_ids=None,
    )
    return state, out


class TestBuckets(CustomTestCase):
    def test_smallest_bucket_at_or_above_rows(self):
        graphs = _graphs(buckets=(256, 512, 768))
        self.assertIsNone(graphs.bucket_rows(0))
        self.assertEqual(graphs.bucket_rows(1), 256)
        self.assertEqual(graphs.bucket_rows(256), 256)
        self.assertEqual(graphs.bucket_rows(257), 512)
        self.assertIsNone(graphs.bucket_rows(769))


class TestFlattenState(CustomTestCase):
    def test_pending_round_trip_and_structure_key(self):
        pending = HcPending(*(torch.randn(3, 4) for _ in range(4)))
        leaves, structure, rebuild = rg._flatten_state(
            HcState(pending, torch.randn(3, 4))
        )
        self.assertEqual(len(leaves), 5)
        back = rebuild(leaves)
        self.assertIsInstance(back.streams, HcPending)
        self.assertIs(back.pre, leaves[-1])
        self.assertNotEqual(structure, rg._flatten_state(HcState(torch.randn(3, 4)))[1])


class TestRunAndCapture(CustomTestCase):
    def test_no_capture_outside_startup_scope(self):
        graphs, captured = _graphs(), []
        _fake_capture(graphs, captured)
        _, out = _step(graphs, 100)
        self.assertIsNone(out)
        self.assertEqual(graphs._capture.call_count, 0)

    def test_refill_without_recapture_and_row_count(self):
        graphs, captured = _graphs(), []
        _fake_capture(graphs, captured)
        with graphs.capture_scope():
            _step(graphs, 100)
        state, out = _step(graphs, 90, seed=1)
        self.assertEqual(graphs._capture.call_count, 1)
        static = next(iter(graphs._static.values()))
        self.assertEqual(static[0].shape[0], 512)
        torch.testing.assert_close(static[0][:90], state.streams)
        torch.testing.assert_close(static[1][:90], torch.arange(90))
        self.assertEqual(captured[0].graph.replay.call_count, 2)
        self.assertEqual(int(captured[0].owned["num_token_non_padded"]), 90)
        self.assertEqual(out.streams.shape[0], 90)


class TestEagerFallback(CustomTestCase):
    def _body_graphs(self):
        graphs, captured = _graphs(), []
        _fake_capture(graphs, captured)
        calls = []

        def body(state, *, forward_batch, **inputs):
            calls.append(forward_batch.global_num_token_non_padded_cpu)
            return HcState(state.streams * 3)

        graphs._run_layers = body
        return graphs, captured, calls

    def test_runs_the_body_when_no_graph_exists(self):
        graphs, _, calls = self._body_graphs()
        state = HcState(torch.randn(100, 4))
        out = graphs.run(
            state=state,
            forward_batch=SimpleNamespace(global_num_token_non_padded_cpu=None),
            eager_fallback=True,
            positions=torch.arange(100),
        )
        torch.testing.assert_close(out.streams, state.streams * 3)
        self.assertEqual(calls, [100])
        self.assertEqual(graphs._capture.call_count, 0)

    def test_past_the_last_bucket_and_without_graphs(self):
        graphs, captured, calls = self._body_graphs()
        with graphs.capture_scope():
            _step(graphs, 100)
        batch = SimpleNamespace(global_num_token_non_padded_cpu=None)
        for n, use_graph in ((600, True), (100, False)):
            graphs.run(
                state=HcState(torch.randn(n, 4)),
                forward_batch=batch,
                eager_fallback=True,
                use_graph=use_graph,
                positions=torch.arange(n),
            )
        self.assertEqual(calls, [600, 100])
        self.assertEqual(captured[0].graph.replay.call_count, 1)


class TestLateLayersInPrefillGraph(CustomTestCase):
    def test_bridge_views_are_fixed_per_block_and_share_storage(self):
        rg._late_bridge.clear()
        self.addCleanup(rg._late_bridge.clear)
        leaves = [torch.zeros(8, 2, 4), torch.zeros(8, 2)]
        big = rg._late_bridge_outputs(leaves, 512)
        small = rg._late_bridge_outputs(leaves, 256)
        self.assertIs(rg._late_bridge_outputs(leaves, 256), small)
        self.assertEqual([t.shape[0] for t in small], [256, 256])
        self.assertEqual(small[0].data_ptr(), big[0].data_ptr())

    def test_live_tail_rows_first_and_metadata_restored(self):
        rg._late_bridge.clear()
        self.addCleanup(rg._late_bridge.clear)
        # Bucket of 16 rows, 10 live tokens, tail rows 7..9; the block holds 8 rows.
        bucket, live, tail_idx = 16, 10, torch.tensor([7, 8, 9])
        events = []
        tail = SimpleNamespace(
            rows=lambda t: t[tail_idx], positions=torch.tensor([7, 8, 9])
        )
        block = SimpleNamespace(
            late_layer_tail=SimpleNamespace(token_indices=torch.zeros(8))
        )
        backend = SimpleNamespace(
            tail_forward_metadata=block,
            enter_live_late_layer_tail=lambda fb, num_tokens: (
                events.append(("enter", num_tokens)) or ("saved", tail)
            ),
            exit_late_layer_tail=lambda saved, fb: events.append(("exit", saved)),
        )
        batch = SimpleNamespace(
            extend_seq_lens_cpu=[6, 4], contains_mm_inputs=lambda: False
        )

        def run(*, state, forward_batch, eager_fallback, use_graph, **inputs):
            self.assertTrue(eager_fallback and use_graph)
            torch.testing.assert_close(inputs["input_ids"], tail_idx)
            return HcState(state.streams + 1, state.pre)

        model = SimpleNamespace(decoder_replay_graphs=SimpleNamespace(run=run))
        streams = torch.arange(bucket, dtype=torch.float32)[:, None].repeat(1, 4)
        pre = torch.ones(bucket, 2)
        leaves, _, rebuild = rg._flatten_state(HcState(streams, pre))
        ids = torch.arange(bucket)
        with (
            patch.object(rg, "get_attn_backend", lambda: backend),
            patch.object(
                rg,
                "get_tc_piecewise_forward_context",
                lambda: SimpleNamespace(forward_batch=batch),
            ),
        ):
            out = rg._late_layers_in_prefill_graph(
                model, tuple(leaves), rebuild, torch.arange(bucket), ids, ids
            )
            stub = rg._late_layers_capture_stub(
                model, tuple(leaves), rebuild, None, None, None
            )
        self.assertEqual(events, [("enter", live), ("exit", "saved")])
        self.assertIs(out, stub)
        self.assertEqual(out[0].shape[0], 8)
        torch.testing.assert_close(out[0][:3], streams[tail_idx] + 1)
        torch.testing.assert_close(out[1][:3], pre[tail_idx])


class TestLateConsumersInPrefillGraph(CustomTestCase):
    """Inside a prefill-graph step the captured source layer publishes no candidates;
    the late layers' eager consumers then take the full top-k, as the graph does."""

    def _route(self, *, published, cover_window):
        try:
            from sglang.srt.layers.attention.deepseek_v4_backend import (
                DeepseekV4AttnBackend,
            )
        except (ImportError, OSError) as e:
            self.skipTest(f"backend not importable here: {e}")
        backend = SimpleNamespace(
            forward_metadata=SimpleNamespace(
                candidate_metadata=published, candidates_cover_window=cover_window
            ),
            _make_low_ratio_prefill_indexer_inputs=lambda *a: "inputs",
            prefill_candidates=MagicMock(),
            full_topk_indexer=MagicMock(),
        )
        layer = SimpleNamespace(
            compress_ratio=2,
            indexer=SimpleNamespace(is_candidate_source=False, uses_candidates=True),
        )
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_decode=lambda: False, is_target_verify=lambda: False
            )
        )
        DeepseekV4AttnBackend._low_ratio_index_topk(
            backend, layer, None, None, None, None, batch
        )
        return backend

    def test_full_topk_only_without_candidates_inside_the_window(self):
        b = self._route(published=None, cover_window=True)
        b.full_topk_indexer.topk_prefill.assert_called_once_with("inputs")
        b.prefill_candidates.consume_prefill.assert_not_called()
        b = self._route(published="table", cover_window=True)
        b.prefill_candidates.consume_prefill.assert_called_once_with("inputs", "table")
        b = self._route(published=None, cover_window=False)
        b.prefill_candidates.consume_prefill.assert_called_once_with("inputs", None)


class TestEnterLiveLateLayerTail(CustomTestCase):
    def test_live_rows_only_and_block_metadata_restored(self):
        try:
            from sglang.srt.layers.attention.deepseek_v4_backend import (
                DeepseekV4AttnBackend,
                LateLayerTail,
            )
        except (ImportError, OSError) as e:
            self.skipTest(f"backend not importable here: {e}")
        # One request of 10 tokens padded to a 16-token bucket; its tail is rows 2..9.
        built, entered = [], []
        block = SimpleNamespace(name="block")

        def build(live):
            built.append((live.positions.shape[0], live.out_cache_loc.shape[0]))
            return SimpleNamespace(
                late_layer_tail=LateLayerTail(
                    token_indices=torch.arange(2, 10),
                    positions=torch.arange(2, 10),
                    extend_seq_lens=torch.tensor([8]),
                    extend_seq_lens_cpu=[8],
                    swa_out_cache_loc=torch.arange(8),
                    contiguous_start=2,
                ),
                candidates_cover_window=False,
            )

        def enter(fb):
            entered.append(backend.tail_forward_metadata)
            return "saved"

        backend = SimpleNamespace(
            tail_forward_metadata=block,
            _build_late_layer_tail_metadata=build,
            enter_late_layer_tail=enter,
        )
        batch = SimpleNamespace(
            positions=torch.arange(16), out_cache_loc=torch.arange(16)
        )
        saved, tail = DeepseekV4AttnBackend.enter_live_late_layer_tail(
            backend, batch, num_tokens=10
        )
        self.assertEqual(built, [(10, 10)])
        self.assertEqual(saved, "saved")
        self.assertEqual(tail.real_rows_count, 8)
        self.assertTrue(entered[0].candidates_cover_window)
        self.assertIs(entered[0].late_layer_tail, tail)
        self.assertIs(backend.tail_forward_metadata, block)
        # The padded full-width buffer's contiguous gather, cut as enter does.
        full = torch.arange(16)
        self.assertEqual(
            tail.real_rows(full)[: tail.real_rows_count].tolist(), list(range(2, 10))
        )


class TestBreakContextNesting(CustomTestCase):
    def test_keeps_the_prefill_graph_flag_set_after_a_nested_run(self):
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
            context,
        )

        graphs = rg.EagerReplayGraphs(
            name="test", model=None, run_layers=None, buckets=[128], tail_rows=True
        )
        graphs._layers = SimpleNamespace(
            attention_layers=[],
            moe_layers=[],
            moe_fusions=[],
            dsa_indexers=[],
            mha_companion_layers=[],
        )
        with patch.object(
            rg, "set_tc_piecewise_forward_context", lambda *a, **k: nullcontext()
        ):
            with context.enable_breakable_cuda_graph():
                with graphs._break_context(SimpleNamespace()):
                    self.assertTrue(rg.is_in_breakable_cuda_graph())
                self.assertTrue(rg.is_in_breakable_cuda_graph())
            self.assertFalse(rg.is_in_breakable_cuda_graph())
            with graphs._break_context(SimpleNamespace()):
                self.assertTrue(rg.is_in_breakable_cuda_graph())
            self.assertFalse(rg.is_in_breakable_cuda_graph())


class TestStepCaches(CustomTestCase):
    def test_off_under_the_prefill_graph_unless_in_a_replay_graph(self):
        for breakable, replay, allowed in (
            (False, False, True),
            (True, False, False),
            (True, True, True),
        ):
            with (
                patch.object(rg, "is_in_breakable_cuda_graph", lambda: breakable),
                patch.object(rg, "_in_replay_graph", replay),
            ):
                self.assertEqual(rg.step_caches_allowed(), allowed)


class TestPointerGuard(CustomTestCase):
    def test_assert_fires_when_a_probed_buffer_is_reallocated(self):
        holder = {"buf": torch.zeros(16)}
        rg.register_pointer_probe("fake", lambda: {"buf": holder["buf"].data_ptr()})
        owned = {"q_pad_buffer": torch.zeros(4)}
        recorded = rg._pointer_snapshot(owned)
        rg.check_pointers(recorded, owned, "test graph")
        holder["buf"] = torch.zeros(32)
        with self.assertRaisesRegex(AssertionError, "fake.buf"):
            rg.check_pointers(recorded, owned, "test graph")
        holder["buf"] = torch.zeros(16)
        recorded = rg._pointer_snapshot(owned)
        owned["q_pad_buffer"] = torch.zeros(4)
        with self.assertRaisesRegex(AssertionError, "owned.q_pad_buffer"):
            rg.check_pointers(recorded, owned, "test graph")


class TestStartupPlan(CustomTestCase):
    def test_full_chunk_first_then_every_bucket_largest_first(self):
        runs = []
        runner = SimpleNamespace(
            _alloc_dummy_decode_buffers=lambda bs, num_tokens_per_req: SimpleNamespace(
                positions=torch.zeros(bs * num_tokens_per_req, dtype=torch.int64)
            ),
            _dummy_run=lambda bs, forward_mode_override, buffers, extend_num_tokens_per_req: (
                runs.append((bs, extend_num_tokens_per_req))
            ),
        )
        full = _graphs(buckets=(256, 512), tail_rows=False)
        tail = _graphs(buckets=(128, 256), tail_rows=True)
        with patch.object(rg, "get_parallel", lambda: SimpleNamespace(tp_rank=1)):
            rg.capture_at_startup(
                eager_runner=runner, request_window=None, graphs=[full, tail]
            )
        self.assertEqual(runs[0], (1, 512))
        self.assertEqual(sorted(runs[1:], key=lambda r: -r[0] * r[1]), runs[1:])
        self.assertEqual(set(runs[1:]), {(1, 512), (1, 256), (1, 128), (2, 128)})
        self.assertFalse(full._capture_open or tail._capture_open)


class TestIndexerStepRowsCache(CustomTestCase):
    """Lean breaks share the DeepGEMM prefill indexer's row plumbing across the index
    layers of one step; a new step (new page table) or other rows must never reuse it."""

    def setUp(self):
        from sglang.srt.layers.attention.dsv4.v41_indexer import scoring

        self.scoring = scoring
        scoring._step_rows_cache.clear()
        self.addCleanup(scoring._step_rows_cache.clear)
        self.req_to_token = torch.arange(4 * 64).reshape(4, 64)
        self.positions = torch.arange(40)

    def _inputs(self, page_table, positions):
        return SimpleNamespace(
            compress_ratio=2,
            positions=positions,
            seq_lens_cpu=[24, 16],
            req_pool_indices=self.req_pool,
            kv_page_table=page_table,
            rows_per_request_device=torch.tensor([24, 16]),
        )

    def test_reused_within_a_step_only(self):
        self.req_pool = torch.tensor([1, 3])
        step1 = torch.zeros(40, 8, dtype=torch.int32)
        first = self.scoring._step_rows(
            self._inputs(step1, self.positions), self.req_to_token
        )
        # Another index layer of the same step passes a fresh view of the same rows.
        again = self.scoring._step_rows(
            self._inputs(step1, self.positions[:40]), self.req_to_token
        )
        self.assertIs(again, first)
        torch.testing.assert_close(
            first["k_slots"][:12], self.req_to_token[1, 0:24:2] // 2
        )
        step2 = torch.zeros(40, 8, dtype=torch.int32)
        self.assertIsNot(
            self.scoring._step_rows(
                self._inputs(step2, self.positions), self.req_to_token
            ),
            first,
        )

    def test_not_cached_when_lean_breaks_off(self):
        from sglang.srt.environ import envs

        self.req_pool = torch.tensor([1, 3])
        page_table = torch.zeros(40, 8, dtype=torch.int32)
        with envs.SGLANG_DSV4_EAGER_GRAPH_LEAN_BREAKS.override(False):
            first = self.scoring._step_rows(
                self._inputs(page_table, self.positions), self.req_to_token
            )
            again = self.scoring._step_rows(
                self._inputs(page_table, self.positions), self.req_to_token
            )
        self.assertIsNot(again, first)


if __name__ == "__main__":
    unittest.main()
