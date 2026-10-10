"""Eager replay graphs of DeepSeek-V4 layer ranges: bucketing, the state's static-buffer
round trip, refill without recapture, no capture outside the startup scope, and the
replay pointer guard. CPU only; capture runs on GPU."""

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
