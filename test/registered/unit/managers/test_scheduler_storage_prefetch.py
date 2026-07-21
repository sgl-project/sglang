"""Unit tests for durable HiCache storage-prefetch admission state."""

import threading
import unittest
from array import array
from queue import Queue
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.cache_controller import (
    HiCacheController,
    PrefetchOperation,
)
from sglang.srt.managers.io_struct import (
    GenerateReqInput,
    TokenizedGenerateReqInput,
    msgpack_decode,
    msgpack_encode,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    StorageOperation as HybridStorageOperation,
)
from sglang.srt.mem_cache.storage_prefetch import (
    StoragePrefetchState,
    StoragePrefetchTracker,
    StorageWriteTracker,
)
from sglang.srt.sampling.sampling_params import SamplingParams

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeNode:
    backuped = False
    parent = None

    def get_last_hash_value(self):
        return None


class _FakeTreeCache:
    def __init__(self, attempt_states: list[StoragePrefetchState]):
        self.root_node = _FakeNode()
        self.hicache_storage_pass_prefix_keys = False
        self.attempt_states = attempt_states
        self.state = StoragePrefetchState.NOT_ATTEMPTED
        self.loaded_tokens = 0
        self.prefetch_attempts = 0
        self.progress_checks = 0
        self.released_requests: list[str] = []
        self.synchronized_flag: bool | None = None

    def storage_prefetch_timeout(self, num_tokens: int) -> float:
        assert num_tokens == 256
        return 5.0

    def synchronize_storage_prefetch_flag(self, flag: bool) -> bool:
        return flag if self.synchronized_flag is None else self.synchronized_flag

    def prefetch_from_storage(self, request_id, *args):
        assert request_id == "request-a"
        self.prefetch_attempts += 1
        self.state = self.attempt_states.pop(0)
        return self.state

    def check_prefetch_progress(self, request_id: str) -> bool:
        assert request_id == "request-a"
        self.progress_checks += 1
        if self.progress_checks == 1:
            self.state = StoragePrefetchState.READING
            return False
        self.state = StoragePrefetchState.READY
        self.loaded_tokens = 256
        return True

    def get_storage_prefetch_state(self, request_id: str) -> StoragePrefetchState:
        assert request_id == "request-a"
        return self.state

    def pop_prefetch_loaded_tokens(self, request_id: str) -> int:
        assert request_id == "request-a"
        loaded_tokens = self.loaded_tokens
        self.loaded_tokens = 0
        return loaded_tokens

    def release_aborted_request(self, request_id: str) -> None:
        self.released_requests.append(request_id)
        self.state = StoragePrefetchState.NOT_ATTEMPTED


def _make_request(tree_cache: _FakeTreeCache):
    request = SimpleNamespace(
        rid="request-a",
        storage_prefetch_state=StoragePrefetchState.NOT_ATTEMPTED,
        storage_prefetch_deadline=None,
        storage_hit_length=0,
        storage_checkpoint_dependency=None,
        last_host_node=tree_cache.root_node,
        prefix_indices=[],
        host_hit_length=0,
        full_untruncated_fill_ids=array("q", range(257)),
    )
    request.init_next_round_input = lambda *args, **kwargs: None
    request._compute_max_prefix_len = lambda input_len: input_len - 1
    return request


class TestStoragePrefetchTracker(unittest.TestCase):
    def test_forget_restores_not_attempted(self):
        tracker = StoragePrefetchTracker()

        tracker.set("request-a", StoragePrefetchState.DEFERRED)
        self.assertEqual(tracker.get("request-a"), StoragePrefetchState.DEFERRED)

        tracker.forget("request-a")
        self.assertEqual(tracker.get("request-a"), StoragePrefetchState.NOT_ATTEMPTED)

    def test_pending_write_clears_only_after_every_overlapping_operation(self):
        tracker = StorageWriteTracker()
        tracker.register(1, ["page-a", "page-b"])
        tracker.register(2, ["page-b"])

        completion = tracker.complete(1, durable_pages=2)

        self.assertIsNotNone(completion)
        self.assertEqual(completion.durable_pages, 2)
        self.assertFalse(tracker.has_pending(["page-a"]))
        self.assertTrue(tracker.has_pending(["page-b"]))

        tracker.complete(2, durable_pages=1)
        self.assertFalse(tracker.has_pending(["page-b"]))


class TestSchedulerStoragePrefetch(unittest.TestCase):
    def _make_scheduler(self, tree_cache: _FakeTreeCache) -> Scheduler:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_hicache_storage = True
        scheduler.tree_cache = tree_cache
        return scheduler

    def test_deferred_prefetch_retries_until_ready(self):
        tree_cache = _FakeTreeCache(
            [StoragePrefetchState.DEFERRED, StoragePrefetchState.QUERYING]
        )
        scheduler = self._make_scheduler(tree_cache)
        request = _make_request(tree_cache)

        with patch("sglang.srt.managers.scheduler.time.monotonic", return_value=10.0):
            scheduler._prefetch_kvcache(request)

        self.assertEqual(request.storage_prefetch_state, StoragePrefetchState.DEFERRED)
        self.assertEqual(request.storage_prefetch_deadline, 15.0)
        self.assertEqual(tree_cache.prefetch_attempts, 1)

        with patch("sglang.srt.managers.scheduler.time.monotonic", return_value=11.0):
            self.assertFalse(scheduler._storage_prefetch_ready_for_admission(request))

        self.assertEqual(request.storage_prefetch_state, StoragePrefetchState.READING)
        self.assertEqual(tree_cache.prefetch_attempts, 2)

        with patch("sglang.srt.managers.scheduler.time.monotonic", return_value=12.0):
            self.assertTrue(scheduler._storage_prefetch_ready_for_admission(request))

        self.assertEqual(request.storage_prefetch_state, StoragePrefetchState.READY)
        self.assertEqual(request.storage_hit_length, 256)

    def test_deadline_cancels_prefetch_and_falls_back(self):
        tree_cache = _FakeTreeCache([])
        scheduler = self._make_scheduler(tree_cache)
        request = _make_request(tree_cache)
        request.storage_prefetch_state = StoragePrefetchState.DEFERRED
        request.storage_prefetch_deadline = 10.0

        with patch("sglang.srt.managers.scheduler.time.monotonic", return_value=10.0):
            self.assertTrue(scheduler._storage_prefetch_ready_for_admission(request))

        self.assertEqual(request.storage_prefetch_state, StoragePrefetchState.FAILED)
        self.assertEqual(tree_cache.released_requests, ["request-a"])
        self.assertEqual(tree_cache.prefetch_attempts, 0)

    def test_deadline_expiry_on_another_rank_falls_back_locally(self):
        tree_cache = _FakeTreeCache([])
        tree_cache.synchronized_flag = True
        scheduler = self._make_scheduler(tree_cache)
        request = _make_request(tree_cache)
        request.storage_prefetch_state = StoragePrefetchState.DEFERRED
        request.storage_prefetch_deadline = 11.0

        with patch("sglang.srt.managers.scheduler.time.monotonic", return_value=10.0):
            self.assertTrue(scheduler._storage_prefetch_ready_for_admission(request))

        self.assertEqual(request.storage_prefetch_state, StoragePrefetchState.FAILED)
        self.assertEqual(tree_cache.released_requests, ["request-a"])
        self.assertEqual(tree_cache.prefetch_attempts, 0)

    def test_later_scheduler_retry_preserves_positive_storage_hit(self):
        tree_cache = _FakeTreeCache([])
        scheduler = self._make_scheduler(tree_cache)
        request = _make_request(tree_cache)
        request.storage_prefetch_state = StoragePrefetchState.READY
        request.storage_hit_length = 256

        self.assertTrue(scheduler._storage_prefetch_ready_for_admission(request))
        self.assertEqual(request.storage_hit_length, 256)


class TestStorageCheckpointRequestNormalization(unittest.TestCase):
    def test_batch_checkpoint_fields_follow_each_request(self):
        request = GenerateReqInput(
            input_ids=[[1, 2], [3, 4]],
            storage_checkpoint=[True, False],
            storage_checkpoint_dependency=[None, "hicache:prior-request"],
        )

        request.normalize_batch_and_arguments()

        self.assertTrue(request[0].storage_checkpoint)
        self.assertIsNone(request[0].storage_checkpoint_dependency)
        self.assertFalse(request[1].storage_checkpoint)
        self.assertEqual(
            request[1].storage_checkpoint_dependency, "hicache:prior-request"
        )

    def test_single_request_rejects_batch_checkpoint_values(self):
        request = GenerateReqInput(input_ids=[1, 2], storage_checkpoint=[True])

        with self.assertRaisesRegex(ValueError, "should be a bool"):
            request.normalize_batch_and_arguments()

    def test_checkpoint_fields_survive_scheduler_ipc_serialization(self):
        request = TokenizedGenerateReqInput(
            rid="request-a",
            input_text="prompt",
            input_ids=array("q", [1, 2]),
            input_embeds=None,
            mm_inputs=None,
            token_type_ids=None,
            sampling_params=SamplingParams(),
            return_logprob=False,
            logprob_start_len=-1,
            top_logprobs_num=0,
            token_ids_logprob=None,
            stream=False,
            storage_checkpoint=True,
            storage_checkpoint_dependency="hicache:prior-request",
        )

        payload = msgpack_encode(request)
        decoded = msgpack_decode(payload)

        self.assertIsInstance(decoded, TokenizedGenerateReqInput)
        self.assertTrue(decoded.storage_checkpoint)
        self.assertEqual(decoded.storage_checkpoint_dependency, "hicache:prior-request")


class TestStorageCheckpointDisaggregation(unittest.IsolatedAsyncioTestCase):
    async def test_checkpoint_fields_are_rejected_for_pd_disaggregation(self):
        for request_kwargs in (
            {"storage_checkpoint": True},
            {"storage_checkpoint_dependency": "hicache:prior-request"},
        ):
            with self.subTest(request_kwargs=request_kwargs):
                manager = TokenizerManager.__new__(TokenizerManager)
                manager.auto_create_handle_loop = lambda: None
                manager._set_default_priority = lambda _request: None
                manager.disaggregation_mode = DisaggregationMode.PREFILL
                request = GenerateReqInput(input_ids=[1, 2], **request_kwargs)

                with self.assertRaisesRegex(ValueError, "not supported with P/D"):
                    await anext(manager.generate_request(request))


class TestStorageCheckpointSchedulerValidation(unittest.TestCase):
    def test_checkpoint_requires_unified_serving_with_l3(self):
        scheduler = Scheduler.__new__(Scheduler)
        request = SimpleNamespace(
            storage_checkpoint=True,
            storage_checkpoint_dependency=None,
        )

        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.enable_hicache_storage = False
        self.assertIn(
            "attached HiCache L3",
            scheduler._storage_checkpoint_request_error(request),
        )

        scheduler.enable_hicache_storage = True
        self.assertIsNone(scheduler._storage_checkpoint_request_error(request))

        scheduler.disaggregation_mode = DisaggregationMode.PREFILL
        self.assertIn(
            "P/D disaggregation",
            scheduler._storage_checkpoint_request_error(request),
        )


class TestHybridStorageDurability(unittest.TestCase):
    def _make_operation(self) -> HybridStorageOperation:
        operation = HybridStorageOperation(
            host_indices=torch.tensor([0, 1]),
            token_ids=[10, 11],
            hash_value=["page-a", "page-b"],
            pool_transfers=[
                PoolTransfer(name=PoolName.MAMBA, keys=["page-b"]),
                PoolTransfer(name=PoolName.SWA, keys=["page-a", "page-b"]),
            ],
        )
        operation.completed_tokens = 2
        return operation

    def test_auxiliary_pool_ack_is_required_for_durability(self):
        operation = self._make_operation()
        self.assertEqual(operation.durable_page_count(1, backup_skip=False), 0)

        operation.pool_storage_result.extra_pool_hit_pages[PoolName.MAMBA] = 1
        self.assertEqual(operation.durable_page_count(1, backup_skip=False), 0)

        operation.pool_storage_result.extra_pool_hit_pages[PoolName.SWA] = 2
        self.assertEqual(operation.durable_page_count(1, backup_skip=False), 2)

    def test_non_owner_rank_defers_to_tp_min_reduction(self):
        operation = self._make_operation()
        self.assertEqual(operation.durable_page_count(1, backup_skip=True), 2)


class TestStorageQuerySynchronization(unittest.TestCase):
    def test_query_retries_when_write_starts_during_lookup(self):
        controller = HiCacheController.__new__(HiCacheController)
        controller.page_size = 1
        controller.storage_stop_event = threading.Event()
        controller.storage_write_tracker = StorageWriteTracker()
        controller.get_hash_str = lambda *_args, **_kwargs: ["page-a"]
        first_query_finished = threading.Event()

        class _Backend:
            calls = 0

            def batch_exists(self, _hashes, _extra_info):
                self.calls += 1
                if self.calls == 1:
                    controller.storage_write_tracker.register(1, ["page-a"])
                    first_query_finished.set()
                    return 0
                return 1

        controller.storage_backend = _Backend()
        operation = PrefetchOperation("request-a", [1])
        query_result = []
        worker = threading.Thread(
            target=lambda: query_result.append(controller._storage_hit_query(operation))
        )
        worker.start()
        self.assertTrue(first_query_finished.wait(timeout=1))

        controller.storage_write_tracker.complete(1, durable_pages=1)
        worker.join(timeout=1)

        self.assertFalse(worker.is_alive())
        self.assertEqual(controller.storage_backend.calls, 2)
        self.assertEqual(query_result, [(["page-a"], 1)])

    def test_query_exception_reports_failed_state(self):
        controller = HiCacheController.__new__(HiCacheController)
        controller.prefetch_queue = Queue()
        controller.prefetch_hit_queue = Queue()
        controller.prefetch_revoke_queue = Queue()
        controller.storage_stop_event = threading.Event()
        controller.prefetch_threshold = 1
        controller._all_reduce_prefetch_groups = lambda _value, _op: None

        def fail_query(_operation):
            raise RuntimeError("storage query failed")

        controller._storage_hit_query = fail_query
        operation = PrefetchOperation("request-a", [1])
        controller.prefetch_queue.put(operation)
        worker = threading.Thread(target=controller.prefetch_thread_func)
        worker.start()

        revoke = controller.prefetch_revoke_queue.get(timeout=1)
        controller.storage_stop_event.set()
        controller.prefetch_queue.put(None)
        controller.prefetch_buffer.put(None)
        worker.join(timeout=1)
        controller.prefetch_io_aux_thread.join(timeout=1)

        self.assertFalse(worker.is_alive())
        self.assertFalse(controller.prefetch_io_aux_thread.is_alive())
        self.assertTrue(operation.is_terminated())
        self.assertEqual(revoke.request_id, "request-a")
        self.assertEqual(revoke.state, StoragePrefetchState.FAILED)


if __name__ == "__main__":
    unittest.main()
