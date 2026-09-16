"""Unit tests for MLX extend-batch request routing in ``MlxTpModelWorker``.

Regression guard for the single-token chunked-prefill continuation bug: a
continuation whose final chunk is exactly one token (prompt length ==
k * chunked_prefill_size + 1) must be routed to the **extend** path, not the
**decode** path.

The old routing keyed on ``seq_len > 1`` as a proxy for "is this a
continuation"; a 1-token continuation is indistinguishable, by length, from a
genuine single-token decode step mixed into the batch, so it was misrouted to
decode. The decode path ignores the batch's real token and feeds the model its
own stored prediction from the previous chunk -> the true last prompt token is
silently dropped and generation is conditioned on a corrupted prompt. The
correct discriminator is ``batch.decoding_reqs``, not the chunk length.

The routing decision was duplicated across the sync and async paths (the bug
therefore existed in both). It now lives in the shared
``MlxTpModelWorker._route_extend_request`` helper, and the sync entry point
launches through the async one rather than re-implementing it. These tests
cover:

  * the helper decision directly;
  * the async wiring, by driving ``_async_extend_batch``;
  * the sync entry point, by driving ``_forward_batch_generation_mlx`` --
    which also guards the delegation, since a divergence there would show up
    as a routing or token-ordering difference between the two.

They mock the MLX runner and load no model. Apple-Silicon-only because
``tp_worker`` imports ``mlx.core`` at module load.
"""

from __future__ import annotations

import importlib.util
import platform
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

# CPU marker is AST-parsed "this test exists"; actual CPU-side execution is
# gated by the @skipUnless guard below. MLX marker runs for real on the MLX
# lane's stage-a (model-free: mocks the runner, loads no model).
register_mlx_ci(est_time=10, suite="stage-a-unit-test-mlx")

_IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"
_HAS_MLX = importlib.util.find_spec("mlx") is not None
_SKIP_REASON = "Apple-Silicon-only (tp_worker imports mlx.core at module load)"


class _FakeRunner:
    """Records which routing path each request took (both worker paths
    drive the runner through the same start/finalize surface)."""

    def __init__(self, known_rids):
        self._known = set(known_rids)
        self.calls: list[tuple[str, str]] = []  # (op, rid)
        # (op, rid) -> needs_logits as received; guards the worker's
        # chunk-finality derivation reaching the runner intact.
        self.logits_flags: dict[tuple[str, str], bool] = {}
        self._req_caches: dict[str, list] = {}
        self._req_penalty_counts = {rid: object() for rid in known_rids}
        self._req_penalty_seed_ids = {rid: [7] for rid in known_rids}
        self.penalty_states: dict[tuple[str, str], object] = {}
        self.remove_sync_flags: dict[str, bool] = {}
        self.prefill_inputs: dict[str, tuple[object, list[int]]] = {}
        self._counter = 0

    # --- shared ---
    def has_request(self, rid):
        return rid in self._known

    def flush_all_decode_kv(self):
        pass

    def remove_request(self, rid, *, sync_to_pool=True):
        self.calls.append(("remove_request", rid))
        self.remove_sync_flags[rid] = sync_to_pool
        self._known.discard(rid)
        self._req_caches.pop(rid, None)
        self._req_penalty_counts.pop(rid, None)
        self._req_penalty_seed_ids.pop(rid, None)

    def store_auxiliary_state_for_request(self, rid):
        self.calls.append(("store_auxiliary_state", rid))

    def ops_for(self, rid):
        return [op for op, r in self.calls if r == rid]

    @staticmethod
    def _fake_cache_layer():
        import mlx.core as mx

        return SimpleNamespace(state=[mx.array([0.0], dtype=mx.float32)])

    # --- start/finalize surface (shared by the sync and async worker paths) ---
    def extend_start(
        self,
        req_id,
        new_token_ids,
        new_slot_ids,
        needs_logits=True,
        logit_edit_row=None,
        logprob_spec=None,
    ):
        import mlx.core as mx

        self.calls.append(("extend_start", req_id))
        self.logits_flags[("extend_start", req_id)] = needs_logits
        self._req_caches[req_id] = [self._fake_cache_layer()]
        penalty_state = mx.array([10 + self._counter], dtype=mx.uint32)
        self.penalty_states[("extend_start", req_id)] = penalty_state
        return SimpleNamespace(
            lazy_token=mx.array([0], dtype=mx.int32),
            cache=self._req_caches[req_id],
            req_id=req_id,
            lazy_logprobs=None,
            penalty_states=(penalty_state,),
        )

    def prefill_start(
        self,
        req_id,
        new_token_ids,
        full_token_ids,
        prefix_slot_ids,
        new_slot_ids,
        req_pool_idx,
        req=None,
        needs_logits=True,
        logit_edit_row=None,
        logprob_spec=None,
    ):
        import mlx.core as mx

        self.calls.append(("prefill_start", req_id))
        self.logits_flags[("prefill_start", req_id)] = needs_logits
        self.prefill_inputs[req_id] = (req, list(full_token_ids))
        self._known.add(req_id)
        penalty_state = mx.array([20 + self._counter], dtype=mx.uint32)
        self.penalty_states[("prefill_start", req_id)] = penalty_state
        return SimpleNamespace(
            lazy_token=mx.array([0], dtype=mx.int32),
            cache=[self._fake_cache_layer()],
            req_id=req_id,
            lazy_logprobs=None,
            penalty_states=(penalty_state,),
        )

    def decode_batch_start(
        self, rids, edit_rows=None, logprob_spec=None, logits_hook=None
    ):
        import mlx.core as mx

        for rid in rids:
            self.calls.append(("decode_start", rid))
        states = tuple(mx.array([30 + i], dtype=mx.uint32) for i in range(len(rids)))
        for rid, state in zip(rids, states):
            self.penalty_states[("decode_start", rid)] = state
        return SimpleNamespace(
            lazy_tokens=mx.array([0] * len(rids), dtype=mx.int32),
            caches=[[self._fake_cache_layer()] for _ in rids],
            req_ids=list(rids),
            lazy_logprobs=None,
            penalty_states=states,
        )

    def decode_batch_start_chained(self, prev):
        import mlx.core as mx

        states = tuple(
            mx.array([40 + i], dtype=mx.uint32) for i in range(len(prev.req_ids))
        )
        for rid, state in zip(prev.req_ids, states):
            self.penalty_states[("chained_decode", rid)] = state
        return SimpleNamespace(
            lazy_tokens=mx.array([0] * len(prev.req_ids), dtype=mx.int32),
            caches=prev.caches,
            req_ids=list(prev.req_ids),
            lazy_logprobs=None,
            penalty_states=states,
        )

    def prefill_finalize(self, pending):
        return 3000

    def extend_finalize(self, pending):
        self._counter += 1
        return 1000 + self._counter

    def decode_batch_finalize(self, pending):
        return [2000 + i for i in range(len(pending.req_ids))]

    def collect_logprobs(self, lazy_logprobs):
        return None

    def eval_pending(self, pending):
        pass

    @staticmethod
    def cache_state_arrays(caches):
        return [s for cache_list in caches for c in cache_list for s in c.state]


class _FakeReq:
    def __init__(self, rid, req_pool_idx=0, *, fill_ids=None, output_ids=None):
        self.rid = rid
        self.prefix_indices = torch.empty(0, dtype=torch.long)
        self.fill_ids = list([0] if fill_ids is None else fill_ids)
        self.output_ids = list(() if output_ids is None else output_ids)
        self.kv = ReqKvInfo(req_pool_idx=req_pool_idx)
        # Mirrors Req's chunk-finality contract read by
        # MlxTpModelWorker._chunk_needs_logits: extend_range=None means
        # "not truncated" (final chunk / plain prefill).
        self.extend_range = None
        self.full_untruncated_fill_ids = self.fill_ids
        self.is_retracted = False
        self.retraction_count = 0
        self._finished = False

    def get_fill_ids(self):
        return self.fill_ids

    def finished(self):
        return self._finished


class _FakeBatch:
    def __init__(self, forward_mode, reqs, extend_lens, decoding_reqs=None):
        total = sum(extend_lens)
        self.forward_mode = forward_mode
        self.reqs = reqs
        self.extend_lens = list(extend_lens)
        self.decoding_reqs = decoding_reqs
        self.sampling_info = None
        self.return_logprob = False
        # Arbitrary but correctly-sized token / slot arrays.
        self.input_ids = torch.arange(total, dtype=torch.long)
        self.out_cache_loc = torch.arange(total, dtype=torch.long)


@unittest.skipUnless(_IS_APPLE_SILICON and _HAS_MLX, _SKIP_REASON)
class TestMlxExtendRouting(CustomTestCase):
    """Routing contract for MlxTpModelWorker: shared helper + sync + async."""

    @classmethod
    def setUpClass(cls):
        # The worker reads --mlx-enable-sampling off the device config bag,
        # which fails closed before a publish. Routing itself is orthogonal
        # to sampling, so pin it off for the whole case.
        cls._config = get_context().override_server_args(mlx_enable_sampling=False)
        cls._config.install()
        cls.addClassCleanup(cls._config.restore)

    @staticmethod
    def _worker(known_rids):
        from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker

        worker = MlxTpModelWorker.__new__(MlxTpModelWorker)
        worker._mlx_runner = _FakeRunner(known_rids)
        worker._mlx_active_rids = set()
        worker._mlx_active_reqs = {}
        # The sync entry point delegates to the async launch, which guards
        # pool creation behind this flag; forward_batch_generation has
        # already run it for real by the time either path is reached.
        worker._mlx_pool_initialized = True
        return worker

    def assertAsyncEvaluated(self, state, evaluated):
        self.assertTrue(any(arg is state for arg in evaluated))

    def test_startup_weight_overlap_is_rejected_before_mlx_model_load(self):
        from sglang.srt.hardware_backend.mlx.model_runner_stub import (
            MlxModelRunnerStub,
        )
        from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker
        from sglang.srt.runtime_context import get_context

        worker = MlxTpModelWorker.__new__(MlxTpModelWorker)

        # The guard reads `get_model().is_startup_weight_load_overlap`, which is
        # derived from `startup_weight_load_mode`. Stating it on a `server_args`
        # of the worker's own no longer reaches it.
        with get_context().override_server_args(startup_weight_load_mode="overlap"):
            with self.assertRaisesRegex(ValueError, "CUDA only"):
                MlxModelRunnerStub.validate_startup_weight_load_mode()
            with self.assertRaisesRegex(ValueError, "CUDA only"):
                worker._init_model_runner()

    # ---------- the shared decision helper ----------
    # The helper takes no seq_len: length cannot distinguish a 1-token
    # continuation from a genuine decode -- request state does.

    def test_route_unseen_request_is_prefill(self):
        worker = self._worker(known_rids=set())
        self.assertEqual(worker._route_extend_request("r1", set()), "prefill")

    def test_route_seen_non_decode_is_continuation(self):
        worker = self._worker(known_rids={"r1"})
        self.assertEqual(worker._route_extend_request("r1", set()), "continuation")

    def test_route_seen_and_in_decoding_reqs_is_decode(self):
        worker = self._worker(known_rids={"r1"})
        self.assertEqual(worker._route_extend_request("r1", {"r1"}), "decode")

    # ---------- sync path: _forward_batch_generation_mlx ----------

    def _run_sync(self, reqs, extend_lens, known_rids, decoding_reqs, forward_mode):
        worker = self._worker(known_rids)
        batch = _FakeBatch(forward_mode, reqs, extend_lens, decoding_reqs)
        result = worker._forward_batch_generation_mlx(batch)
        assert result.next_token_ids.numel() == len(reqs)
        return worker._mlx_runner

    def test_sync_one_token_continuation_routes_to_extend(self):
        """THE REGRESSION (sync): a 1-token continuation must extend, not decode."""
        runner = self._run_sync([_FakeReq("r1")], [1], {"r1"}, None, ForwardMode.EXTEND)
        self.assertEqual(runner.ops_for("r1"), ["extend_start"])
        # Untruncated (extend_range None) => final chunk => logits required.
        self.assertIs(runner.logits_flags[("extend_start", "r1")], True)

    def test_sync_non_final_chunk_skips_logits(self):
        """Head-skip derivation: a scheduler-truncated chunk (extend_range.end
        below the request's full untruncated length) reaches the runner with
        needs_logits=False; its next-token output is popped as the stale
        intermediate token, so computing the vocab head for it is pure waste.
        Everything else about routing is unchanged."""
        req = _FakeReq("r1")
        req.full_untruncated_fill_ids = list(range(8))
        req.extend_range = SimpleNamespace(start=0, end=4)  # 4 < 8: non-final
        runner = self._run_sync([req], [4], {"r1"}, None, ForwardMode.EXTEND)
        self.assertEqual(runner.ops_for("r1"), ["extend_start"])
        self.assertIs(runner.logits_flags[("extend_start", "r1")], False)

    def test_sync_genuine_mixed_decode_routes_to_decode(self):
        p, d = _FakeReq("p1"), _FakeReq("d1")
        runner = self._run_sync([p, d], [4, 1], {"d1"}, [d], ForwardMode.MIXED)
        self.assertEqual(runner.ops_for("p1"), ["prefill_start"])
        self.assertEqual(runner.ops_for("d1"), ["decode_start"])

    # ---------- async path: _async_extend_batch ----------

    def _run_async(self, reqs, extend_lens, known_rids, decoding_reqs, forward_mode):
        from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker

        worker = MlxTpModelWorker.__new__(MlxTpModelWorker)
        worker._mlx_runner = _FakeRunner(known_rids)
        batch = _FakeBatch(forward_mode, reqs, extend_lens, decoding_reqs)
        launch = worker._async_extend_batch(batch)
        return worker._mlx_runner, launch

    def test_async_one_token_continuation_routes_to_extend(self):
        """THE REGRESSION (async): a 1-token continuation must extend, not decode."""
        runner, launch = self._run_async(
            [_FakeReq("r1")], [1], {"r1"}, None, ForwardMode.EXTEND
        )
        self.assertEqual(runner.ops_for("r1"), ["extend_start"])
        self.assertIs(runner.logits_flags[("extend_start", "r1")], True)
        self.assertEqual(len(launch.extends), 1)  # one pending extend
        self.assertIsNone(launch.decode)  # no mixed decode

    def test_async_non_final_chunk_skips_logits(self):
        """Async twin of the head-skip derivation guard."""
        req = _FakeReq("r1")
        req.full_untruncated_fill_ids = list(range(8))
        req.extend_range = SimpleNamespace(start=0, end=4)
        runner, _ = self._run_async([req], [4], {"r1"}, None, ForwardMode.EXTEND)
        self.assertEqual(runner.ops_for("r1"), ["extend_start"])
        self.assertIs(runner.logits_flags[("extend_start", "r1")], False)

    def test_async_genuine_mixed_decode_routes_to_decode(self):
        p, d = _FakeReq("p1"), _FakeReq("d1")
        runner, launch = self._run_async([p, d], [4, 1], {"d1"}, [d], ForwardMode.MIXED)
        self.assertEqual(runner.ops_for("p1"), ["prefill_start"])
        self.assertEqual(runner.ops_for("d1"), ["decode_start"])
        self.assertIsNotNone(launch.decode)  # pending mixed decode present

    def test_async_eval_receives_fresh_decode_penalty_states(self):
        """Dropping the sibling count output would leave the next decode stale."""
        req = _FakeReq("d1")
        worker = self._worker({"d1"})
        batch = _FakeBatch(ForwardMode.DECODE, [req], [1])

        with patch("mlx.core.async_eval") as async_eval:
            launch = worker.async_forward_batch_generation_mlx(batch)

        self.assertAsyncEvaluated(
            launch.decode.penalty_states[0], async_eval.call_args.args
        )

    def test_async_eval_receives_prefill_extend_and_mixed_penalty_states(self):
        """Every pending row in an extend/mixed launch is scheduled explicitly."""
        prefill = _FakeReq("p1", req_pool_idx=11)
        extend = _FakeReq("e1", req_pool_idx=12)
        decode = _FakeReq("d1", req_pool_idx=13)
        worker = self._worker({"e1", "d1"})
        batch = _FakeBatch(
            ForwardMode.MIXED,
            [prefill, extend, decode],
            [2, 2, 1],
            decoding_reqs=[decode],
        )

        with patch("mlx.core.async_eval") as async_eval:
            launch = worker.async_forward_batch_generation_mlx(batch)

        evaluated = async_eval.call_args.args
        pending_states = [
            *(state for pending in launch.prefills for state in pending.penalty_states),
            *(state for pending in launch.extends for state in pending.penalty_states),
            *(launch.decode.penalty_states if launch.decode is not None else ()),
        ]
        self.assertEqual(len(pending_states), 3)
        for state in pending_states:
            self.assertAsyncEvaluated(state, evaluated)

    def test_async_eval_receives_chained_decode_penalty_states(self):
        """A chained token does not itself evaluate its downstream count row."""
        worker = self._worker({"d1"})
        previous = worker._mlx_runner.decode_batch_start(["d1"])

        with patch("mlx.core.async_eval") as async_eval:
            launch = worker.async_chained_decode_mlx(previous)

        self.assertAsyncEvaluated(
            launch.decode.penalty_states[0], async_eval.call_args.args
        )

    def test_idle_boundary_clears_counts_and_deferred_seeds(self):
        req = _FakeReq("old", req_pool_idx=21)
        worker = self._worker({"old"})
        worker._mlx_active_rids = {"old"}
        worker._mlx_active_reqs = {"old": (req, 21, 0)}

        worker.cleanup_idle_request_state()

        self.assertEqual(worker._mlx_active_rids, set())
        self.assertEqual(worker._mlx_active_reqs, {})
        self.assertEqual(worker._mlx_runner._req_penalty_counts, {})
        self.assertEqual(worker._mlx_runner._req_penalty_seed_ids, {})
        self.assertFalse(worker._mlx_runner.remove_sync_flags["old"])

    def test_finished_prefill_is_retired_without_dropping_live_extend(self):
        finished = _FakeReq("finished", req_pool_idx=31)
        live = _FakeReq("live", req_pool_idx=32)
        worker = self._worker({"finished", "live"})
        worker._mlx_active_rids = {"finished", "live"}
        worker._mlx_active_reqs = {
            "finished": (finished, 31, 0),
            "live": (live, 32, 0),
        }
        finished._finished = True

        worker.async_forward_batch_generation_mlx(
            _FakeBatch(ForwardMode.EXTEND, [live], [1])
        )

        self.assertNotIn("finished", worker._mlx_runner._req_penalty_counts)
        self.assertNotIn("finished", worker._mlx_runner._req_penalty_seed_ids)
        self.assertIn("live", worker._mlx_runner._req_penalty_counts)
        self.assertEqual(worker._mlx_runner.ops_for("live"), ["extend_start"])
        self.assertFalse(worker._mlx_runner.remove_sync_flags["finished"])

    def test_finished_decode_release_immediately_retires_worker_state(self):
        req = _FakeReq("finished", req_pool_idx=33)
        worker = self._worker({"finished"})
        worker._mlx_active_rids = {"finished"}
        worker._mlx_active_reqs = {"finished": (req, 33, 0)}

        worker.prepare_for_kv_cache_release(req)

        self.assertEqual(
            worker._mlx_runner.ops_for("finished"),
            ["store_auxiliary_state", "remove_request"],
        )
        self.assertNotIn("finished", worker._mlx_active_rids)
        self.assertNotIn("finished", worker._mlx_active_reqs)
        self.assertTrue(worker._mlx_runner.remove_sync_flags["finished"])

    def test_aborted_or_retracted_request_is_retired_on_next_boundary(self):
        aborted = _FakeReq("aborted", req_pool_idx=34)
        live = _FakeReq("live", req_pool_idx=35)
        worker = self._worker({"aborted", "live"})
        worker._mlx_active_rids = {"aborted", "live"}
        worker._mlx_active_reqs = {
            "aborted": (aborted, 34, 0),
            "live": (live, 35, 0),
        }
        aborted.is_retracted = True

        worker.async_forward_batch_generation_mlx(
            _FakeBatch(ForwardMode.EXTEND, [live], [1])
        )

        self.assertNotIn("aborted", worker._mlx_runner._req_penalty_counts)
        self.assertIn("live", worker._mlx_runner._req_penalty_counts)
        self.assertFalse(worker._mlx_runner.remove_sync_flags["aborted"])

    def test_extend_boundary_preserves_absent_but_live_request(self):
        current = _FakeReq("current", req_pool_idx=36)
        parked = _FakeReq("parked", req_pool_idx=37)
        worker = self._worker({"current", "parked"})
        worker._mlx_active_rids = {"current", "parked"}
        worker._mlx_active_reqs = {
            "current": (current, 36, 0),
            "parked": (parked, 37, 0),
        }

        worker.async_forward_batch_generation_mlx(
            _FakeBatch(ForwardMode.EXTEND, [current], [1])
        )

        self.assertIn("parked", worker._mlx_runner._req_penalty_counts)
        self.assertNotIn(("remove_request", "parked"), worker._mlx_runner.calls)
        self.assertEqual(worker._mlx_active_rids, {"current", "parked"})
        self.assertEqual(
            worker._mlx_active_reqs,
            {"current": (current, 36, 0), "parked": (parked, 37, 0)},
        )

    def test_retracted_request_reprefills_with_accepted_output_seed(self):
        req = _FakeReq(
            "same",
            req_pool_idx=41,
            fill_ids=[101, 102, 7, 8],
            output_ids=[7, 8],
        )
        worker = self._worker({"same"})
        worker._mlx_active_rids = {"same"}
        worker._mlx_active_reqs = {"same": (req, 41, 0)}
        old_count = worker._mlx_runner._req_penalty_counts["same"]
        old_seed = worker._mlx_runner._req_penalty_seed_ids["same"]

        # release_kv_cache + the next prefill allocation changes the scheduler
        # pool registration while retaining the same Req/output_ids.
        req.kv.req_pool_idx = 42
        worker.async_forward_batch_generation_mlx(
            _FakeBatch(ForwardMode.EXTEND, [req], [1])
        )

        self.assertEqual(
            worker._mlx_runner.ops_for("same"),
            ["remove_request", "prefill_start"],
        )
        self.assertNotIn(old_count, worker._mlx_runner._req_penalty_counts.values())
        self.assertNotIn(old_seed, worker._mlx_runner._req_penalty_seed_ids.values())
        self.assertFalse(worker._mlx_runner.remove_sync_flags["same"])
        forwarded_req, forwarded_full_token_ids = worker._mlx_runner.prefill_inputs[
            "same"
        ]
        self.assertIs(forwarded_req, req)
        self.assertEqual(forwarded_req.output_ids, [7, 8])
        self.assertEqual(forwarded_full_token_ids, [101, 102, 7, 8])
        self.assertEqual(worker._mlx_active_rids, {"same"})
        self.assertEqual(worker._mlx_active_reqs, {"same": (req, 42, 0)})

    def test_same_rid_reuse_drops_old_counts_and_seed_before_prefill(self):
        old_req = _FakeReq("same", req_pool_idx=51)
        new_req = _FakeReq("same", req_pool_idx=51)
        worker = self._worker({"same"})
        worker._mlx_active_rids = {"same"}
        worker._mlx_active_reqs = {"same": (old_req, 51, 0)}
        old_count = worker._mlx_runner._req_penalty_counts["same"]
        old_seed = worker._mlx_runner._req_penalty_seed_ids["same"]

        worker.async_forward_batch_generation_mlx(
            _FakeBatch(ForwardMode.EXTEND, [new_req], [1])
        )

        self.assertEqual(
            worker._mlx_runner.ops_for("same"),
            ["remove_request", "prefill_start"],
        )
        self.assertNotIn(old_count, worker._mlx_runner._req_penalty_counts.values())
        self.assertNotIn(old_seed, worker._mlx_runner._req_penalty_seed_ids.values())
        self.assertFalse(worker._mlx_runner.remove_sync_flags["same"])
        self.assertEqual(worker._mlx_active_rids, {"same"})
        self.assertEqual(worker._mlx_active_reqs, {"same": (new_req, 51, 0)})

    def test_retraction_reuses_same_pool_row_before_reprefill(self):
        """A retraction epoch change must retire stale runner state even if
        the allocator gives the request its previous row back."""
        req = _FakeReq("same", req_pool_idx=61)
        worker = self._worker({"same"})
        worker._mlx_active_rids = {"same"}
        worker._mlx_active_reqs = {"same": (req, 61, 0)}
        old_count = worker._mlx_runner._req_penalty_counts["same"]
        old_seed = worker._mlx_runner._req_penalty_seed_ids["same"]

        req.retraction_count = 1
        req.is_retracted = False
        worker.async_forward_batch_generation_mlx(
            _FakeBatch(ForwardMode.EXTEND, [req], [1])
        )

        self.assertEqual(
            worker._mlx_runner.ops_for("same"),
            ["remove_request", "prefill_start"],
        )
        self.assertNotIn(old_count, worker._mlx_runner._req_penalty_counts.values())
        self.assertNotIn(old_seed, worker._mlx_runner._req_penalty_seed_ids.values())
        self.assertFalse(worker._mlx_runner.remove_sync_flags["same"])
        self.assertEqual(worker._mlx_active_rids, {"same"})
        self.assertEqual(worker._mlx_active_reqs, {"same": (req, 61, 1)})


if __name__ == "__main__":
    unittest.main()
