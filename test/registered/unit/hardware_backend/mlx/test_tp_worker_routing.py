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
from array import array
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import Req, ReqKvInfo, ScheduleBatch
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
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
        self._counter = 0

    # --- shared ---
    def has_request(self, rid):
        return rid in self._known

    def remove_request(self, rid):
        self.calls.append(("remove_request", rid))
        self._known.discard(rid)
        self._req_caches.pop(rid, None)

    def store_auxiliary_state_for_request(self, rid):
        pass

    def flush_all_decode_kv(self):
        pass

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
        return SimpleNamespace(
            lazy_token=mx.array([0], dtype=mx.int32),
            cache=self._req_caches[req_id],
            req_id=req_id,
            lazy_logprobs=None,
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
        return SimpleNamespace(
            lazy_token=mx.array([0], dtype=mx.int32),
            cache=[self._fake_cache_layer()],
            req_id=req_id,
            lazy_logprobs=None,
        )

    def decode_batch_start(
        self, rids, edit_rows=None, logprob_spec=None, logits_hook=None
    ):
        import mlx.core as mx

        for rid in rids:
            self.calls.append(("decode_start", rid))
        return SimpleNamespace(
            lazy_tokens=mx.array([0] * len(rids), dtype=mx.int32),
            caches=[[self._fake_cache_layer()] for _ in rids],
            req_ids=list(rids),
            lazy_logprobs=None,
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
    def __init__(self, rid, req_pool_idx=0):
        self.rid = rid
        self.prefix_indices = torch.empty(0, dtype=torch.long)
        self.fill_ids = [0]
        self.kv = ReqKvInfo(req_pool_idx=req_pool_idx)
        # Mirrors Req's chunk-finality contract read by
        # MlxTpModelWorker._chunk_needs_logits: extend_range=None means
        # "not truncated" (final chunk / plain prefill).
        self.extend_range = None
        self.full_untruncated_fill_ids = self.fill_ids

    def get_fill_ids(self):
        return self.fill_ids


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
        worker._mlx_finished_rids = set()
        # The sync entry point delegates to the async launch, which guards
        # pool creation behind this flag; forward_batch_generation has
        # already run it for real by the time either path is reached.
        worker._mlx_pool_initialized = True
        return worker

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


def _finished_req(rid):
    """Minimal Req stand-in for prepare_for_kv_cache_release."""
    return SimpleNamespace(rid=rid, kv=SimpleNamespace(mamba_last_track_seqlen=None))


@unittest.skipUnless(_IS_APPLE_SILICON and _HAS_MLX, _SKIP_REASON)
class TestMlxFinishedRequestRelease(CustomTestCase):
    """Finished prefills must release worker state before the next request wave."""

    @classmethod
    def setUpClass(cls):
        cls._config = get_context().override_server_args(mlx_enable_sampling=False)
        cls._config.install()
        cls.addClassCleanup(cls._config.restore)

    @staticmethod
    def _worker(active_rids):
        from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker

        worker = MlxTpModelWorker.__new__(MlxTpModelWorker)
        worker._mlx_runner = _FakeRunner(active_rids)
        worker._mlx_active_rids = set(active_rids)
        worker._mlx_finished_rids = set()
        worker._mlx_pool_initialized = True
        return worker

    def test_finished_requests_released_on_extend_launch(self):
        worker = self._worker({"a", "b"})
        worker.prepare_for_kv_cache_release(_finished_req("a"))
        worker.prepare_for_kv_cache_release(_finished_req("b"))

        worker._cleanup_stale_rids(ForwardMode.EXTEND, {"c"})

        self.assertFalse(worker._mlx_runner.has_request("a"))
        self.assertFalse(worker._mlx_runner.has_request("b"))
        self.assertEqual(worker._mlx_active_rids, {"c"})
        self.assertEqual(worker._mlx_finished_rids, set())

    def test_unfinished_requests_survive_extend_launch(self):
        worker = self._worker({"a", "b"})
        worker.prepare_for_kv_cache_release(_finished_req("a"))

        worker._cleanup_stale_rids(ForwardMode.EXTEND, {"c"})

        self.assertFalse(worker._mlx_runner.has_request("a"))
        self.assertTrue(worker._mlx_runner.has_request("b"))
        self.assertEqual(worker._mlx_active_rids, {"b", "c"})

    def test_decode_launch_still_drops_silently_departed_requests(self):
        # An abort leaves the batch without a finish notification; the
        # decode-mode set difference must keep catching it.
        worker = self._worker({"a", "b"})

        worker._cleanup_stale_rids(ForwardMode.DECODE, {"b"})

        self.assertFalse(worker._mlx_runner.has_request("a"))
        self.assertTrue(worker._mlx_runner.has_request("b"))
        self.assertEqual(worker._mlx_active_rids, {"b"})

    def test_unknown_request_is_not_marked(self):
        worker = self._worker({"a"})
        worker.prepare_for_kv_cache_release(_finished_req("ghost"))
        self.assertEqual(worker._mlx_finished_rids, set())

    @staticmethod
    def _req(rid, max_new_tokens=0):
        # score_prompts (including Decisions) uses this prefill-only contract.
        sampling_params = SamplingParams(max_new_tokens=max_new_tokens)
        sampling_params.normalize(None)
        return Req(
            rid=rid,
            origin_input_text="prompt",
            origin_input_ids=array("q", [1, 2]),
            sampling_params=sampling_params,
            return_logprob=True,
            token_ids_logprob=[3, 4],
            vocab_size=128,
        )

    def _processor(self, worker, *, overlap=False, tree_cache=None):
        logprob_processor = Mock()
        logprob_processor.calculate_num_input_logprobs.return_value = 0
        return SchedulerBatchResultProcessor(
            is_generation=True,
            disaggregation_mode=None,
            enable_overlap=False,
            enable_overlap_mlx=overlap,
            model_config=SimpleNamespace(think_end_ids=None),
            token_to_kv_pool_allocator=Mock(),
            tree_cache=tree_cache,
            hisparse_coordinator=None,
            req_to_token_pool=None,
            decode_offload_manager=None,
            metrics_collector=None,
            metrics_reporter=Mock(num_generated_tokens=0, forward_ct_decode=0),
            draft_worker=None,
            model_worker=worker,
            logprob_result_processor=logprob_processor,
            output_streamer=Mock(),
            beam_coordinator=Mock(),
            abort_request=Mock(),
        )

    def _process_prefill(self, worker, reqs):
        processor = self._processor(worker)
        batch = SimpleNamespace(
            reqs=reqs,
            forward_mode=ForwardMode.EXTEND,
            decoding_reqs=[],
            return_logprob=True,
            return_hidden_states=False,
            return_hidden_states_mode=CaptureHiddenMode.NULL,
            spec_info=None,
            prefill_stats=None,
        )
        result = SimpleNamespace(
            copy_done=None,
            auxiliary_host_output=None,
            routed_experts_output=None,
            indexer_topk_output=None,
            logits_output=LogitsProcessorOutput(next_token_logits=None),
            next_token_ids=torch.full((len(reqs),), 3, dtype=torch.int64),
            extend_input_len_per_req=[2] * len(reqs),
            extend_logprob_start_len_per_req=[1] * len(reqs),
            grammar_advanced=False,
            can_run_cuda_graph=False,
        )

        def release(req, *_args, **_kwargs):
            # The notification must precede scheduler-side KV release, while
            # auxiliary state still belongs to this request's pool row.
            if hasattr(worker, "_mlx_finished_rids"):
                self.assertIn(req.rid, worker._mlx_finished_rids)
                self.assertTrue(worker._mlx_runner.has_request(req.rid))

        module = "sglang.srt.managers.scheduler_components.batch_result_processor"
        with (
            patch(f"{module}.release_kv_cache", side_effect=release) as released,
            patch(f"{module}.checkpoint_kv_cache"),
        ):
            processor.process_batch_result_prefill(batch, result)
        return released

    def test_prefill_only_waves_release_at_next_extend(self):
        worker = self._worker(set())
        for rid in ("a", "b", "c"):
            worker._cleanup_stale_rids(ForwardMode.EXTEND, {rid})
            self.assertFalse(worker._mlx_runner._known)
            worker._mlx_runner._known.add(rid)  # the new prefill finalized
            req = self._req(rid)

            released = self._process_prefill(worker, [req])

            self.assertTrue(req.finished())
            self.assertEqual(req.finished_len, 0)
            released.assert_called_once_with(req, None, checkpoint=True)
            self.assertEqual(worker._mlx_finished_rids, {rid})
            # Reclamation remains deferred, preserving overlap safety.
            self.assertEqual(worker._mlx_runner._known, {rid})

    def test_finished_rid_reuse_starts_a_new_prefill(self):
        worker = self._worker({"same"})
        runner = worker._mlx_runner
        old_cache = [runner._fake_cache_layer()]
        runner._req_caches["same"] = old_cache
        self._process_prefill(worker, [self._req("same")])

        new_req = _FakeReq("same")
        batch = _FakeBatch(ForwardMode.EXTEND, [new_req], [2])
        launch = worker.async_forward_batch_generation_mlx(batch)

        self.assertEqual(runner.ops_for("same"), ["remove_request", "prefill_start"])
        self.assertEqual(worker._mlx_active_rids, {"same"})
        self.assertFalse(worker._mlx_finished_rids)
        self.assertEqual(len(launch.prefills), 1)
        self.assertFalse(launch.extends)
        self.assertIsNot(launch.prefills[0].cache, old_cache)
        worker.finalize_mlx_result(launch, [new_req])

    def test_overlap_drains_chained_decode_before_mixed_launch_cleanup(self):
        import mlx.core as mx

        from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner
        from sglang.srt.hardware_backend.mlx.scheduler_mixin import (
            SchedulerMlxOverlapMixin,
        )

        events = []
        worker = self._worker({"finished", "live"})
        runner = worker._mlx_runner
        # Exercise remove_request's radix-enabled pool flush, with synthetic
        # cache tensors / scheduler rows and a mocked pool-write destination.
        runner.disable_radix_cache = False
        runner._cache_layout = SimpleNamespace(
            has_auxiliary_state=False,
            first_attention_layer_index=0,
            full_attention_layer_indices=[0],
        )
        runner._cache_pool = []
        runner._req_token_ids = {rid: [1, 2] for rid in runner._known}
        runner._req_sampling = {}
        runner._req_pool_idx = {"finished": 0, "live": 1}
        runner._req_synced_offset = {rid: 1 for rid in runner._known}
        runner._req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(8).reshape(2, 4)
        )
        runner._attention_kv_pool = Mock()
        for rid in runner._known:
            state = mx.zeros((1, 1, 2, 1))
            runner._req_caches[rid] = [
                SimpleNamespace(keys=state, values=state, offset=2, state=[state])
            ]
        for name in (
            "_sync_decode_kv_to_pool",
            "_sync_new_kv_to_pool",
            "_first_attention_cache",
            "_release_cache",
            "flush_all_decode_kv",
        ):
            setattr(runner, name, getattr(MlxModelRunner, name).__get__(runner))

        def remove(rid):
            events.append(("remove", rid))
            MlxModelRunner.remove_request(runner, rid)
            runner._known.discard(rid)

        runner.remove_request = remove
        original_decode_start = runner.decode_batch_start
        original_prefill_start = runner.prefill_start

        def chain(previous):
            events.append(("chain", None))
            pending = original_decode_start(previous.req_ids)
            pending.lazy_tokens = previous.lazy_tokens + 1
            pending.chained = True
            return pending

        def finalize(pending):
            pending.lazy_tokens.tolist()
            if getattr(pending, "chained", False):
                # Both requests' state must survive the first result's finish
                # notification until the already-launched decode has drained.
                self.assertTrue(runner.has_request("finished"))
                self.assertTrue(runner.has_request("live"))
                events.append(("finalize_chain", None))
            return [3] * len(pending.req_ids)

        def prefill(**kwargs):
            events.append(("prefill", kwargs["req_id"]))
            return original_prefill_start(**kwargs)

        runner.decode_batch_start_chained = chain
        runner.decode_batch_finalize = finalize
        runner.prefill_start = prefill

        # Fake computation still needs to publish the new request's state at
        # the same boundary as MlxModelRunner.prefill_finalize.
        def prefill_finalize(pending):
            runner._known.add(pending.req_id)
            runner._req_caches[pending.req_id] = pending.cache
            return 3

        runner.prefill_finalize = prefill_finalize
        finished = self._req("finished", max_new_tokens=1)
        live = self._req("live", max_new_tokens=8)
        fresh = self._req("fresh")
        for req in (finished, live, fresh):
            req.return_logprob = False
            req.full_untruncated_fill_ids = array("q", [1, 2])
            req.extend_range = SimpleNamespace(end=2, length=2)
            req.prefix_indices = torch.empty(0, dtype=torch.long)

        def batch(mode, reqs, lengths, decoding_reqs=None):
            out = ScheduleBatch(reqs=reqs)
            out.forward_mode = mode
            out.extend_lens = lengths
            out.decoding_reqs = decoding_reqs
            out.input_ids = torch.arange(sum(lengths))
            out.out_cache_loc = torch.arange(sum(lengths))
            out.return_logprob = False
            out.enable_overlap = True
            out.spec_algorithm = SimpleNamespace(is_none=lambda: True)
            return out

        decode = batch(ForwardMode.DECODE, [finished, live], [1, 1])
        mixed = batch(ForwardMode.MIXED, [fresh, live], [2, 1], [live])
        processor = self._processor(worker, overlap=True, tree_cache=Mock())
        scheduler = SchedulerMlxOverlapMixin()
        scheduler.tp_worker = worker
        scheduler.forward_ct = 0
        scheduler._sched_idled = False
        scheduler.gracefully_exit = False
        scheduler._engine_paused = False
        scheduler.waiting_queue = []
        scheduler.result_queue = deque()
        scheduler.running_batch = decode
        scheduler.last_batch = None
        scheduler.profiler_manager = Mock()
        scheduler.future_map = Mock()
        scheduler.ingest_requests = Mock()
        scheduler.invariant_checker = Mock()
        scheduler.get_next_batch_to_run = Mock(
            side_effect=[
                SimpleNamespace(running_batch=decode, batch_to_run=decode),
                SimpleNamespace(running_batch=mixed, batch_to_run=mixed),
                StopIteration,
            ]
        )

        def process(batch, result):
            if batch.forward_mode.is_decode():
                processor.process_batch_result_decode(batch, result)
            else:
                processor.process_batch_result_prefill(batch, result)

        scheduler.process_batch_result = process

        def release(req, *_args, **_kwargs):
            self.assertIn(req.rid, worker._mlx_finished_rids)
            self.assertTrue(runner.has_request(req.rid))
            events.append(("finish", req.rid))

        module = "sglang.srt.managers.scheduler_components.batch_result_processor"
        with (
            get_context().override_server_args(
                disable_radix_cache=False, disable_overlap_schedule=False
            ),
            patch(f"{module}.release_kv_cache", side_effect=release),
            patch(f"{module}.checkpoint_kv_cache"),
            patch(
                "sglang.srt.hardware_backend.mlx.scheduler_mixin.resolve_forward_inputs"
            ),
            self.assertRaises(StopIteration),
        ):
            scheduler.event_loop_overlap_mlx()

        self.assertLess(
            events.index(("chain", None)), events.index(("finish", "finished"))
        )
        self.assertLess(
            events.index(("finish", "finished")), events.index(("finalize_chain", None))
        )
        self.assertLess(
            events.index(("finalize_chain", None)), events.index(("remove", "finished"))
        )
        self.assertLess(
            events.index(("remove", "finished")), events.index(("prefill", "fresh"))
        )
        self.assertNotIn(("remove", "live"), events)
        self.assertEqual(worker._mlx_active_rids, {"live", "fresh"})
        self.assertEqual(worker._mlx_finished_rids, {"fresh"})
        self.assertFalse(live.finished())
        pool_writes = runner._attention_kv_pool.set_kv_all_layers.call_args_list
        self.assertEqual([call.args[0].tolist() for call in pool_writes], [[1], [5]])
        self.assertEqual(len(scheduler.result_queue), 0)

    def test_generation_finished_during_prefill_is_marked(self):
        worker = self._worker({"a"})
        self._process_prefill(worker, [self._req("a", max_new_tokens=1)])
        self.assertEqual(worker._mlx_finished_rids, {"a"})

    def test_unfinished_and_middle_prefills_are_not_marked(self):
        worker = self._worker({"unfinished", "middle"})
        unfinished = self._req("unfinished", max_new_tokens=2)
        middle = self._req("middle")
        middle.inflight_middle_chunks = 1

        released = self._process_prefill(worker, [unfinished, middle])

        released.assert_not_called()
        self.assertEqual(worker._mlx_finished_rids, set())
        self.assertEqual(worker._mlx_runner._known, {"unfinished", "middle"})
        self.assertFalse(unfinished.finished())
        self.assertFalse(middle.finished())

    def test_prefill_completion_without_worker_hook(self):
        req = self._req("a")
        released = self._process_prefill(SimpleNamespace(), [req])
        self.assertTrue(req.finished())
        released.assert_called_once_with(req, None, checkpoint=True)


if __name__ == "__main__":
    unittest.main()
