"""Unit tests for external-linker load failure handling.

A failed remote read used to raise out of the model forward pass and take the
scheduler process down with it. These tests pin the replacement contract: the
failure travels as a value, the tree releases what the load pinned, and the
affected requests are reported so the scheduler can abort them.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import inspect
import unittest
from http import HTTPStatus
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import (
    EXTERNAL_KV_LOAD_ERR_TYPE,
    is_external_kv_load_failure,
)
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.storage.umbp.umbp_direct_linker import (
    LayerWiseLoadCounter,
    UMBPDirectLinker,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    UnifiedCacheLinkerWrapper,
)
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore


class FakeLinker:
    """Stands in for a backend, replaying a scripted sequence of outcomes."""

    def __init__(self, completions):
        self._completions = list(completions)
        self.queued = {}
        self.offloaded = []

    def load(self, rid, transfers):
        self.queued[rid] = transfers
        return True

    def cancel_queued_load(self, rid):
        return self.queued.pop(rid, None) is not None

    def num_completed_loads(self):
        return len(self._completions)

    def pop_completed_load(self):
        return self._completions.pop(0)

    def offload(self, transfers):
        self.offloaded.append(transfers)
        return True

    def num_completed_offloads(self):
        return 0

    def reset(self):
        self._completions = []


class FakeKey:
    """A key whose child_key is the node id, so children map id -> node."""

    def __init__(self, token):
        self._token = token

    def child_key(self, page_size):
        return self._token


class FakeNode:
    def __init__(self, node_id, parent=None):
        self.id = node_id
        self.parent = parent
        self.external_cache_stored = True
        self.detached = False
        self.write_through_pending_id = None
        self.key = FakeKey(node_id)
        self.children = {}
        if parent is not None:
            parent.children[node_id] = self


class FakeCache:
    """Just the tree surface drain_loads touches."""

    def __init__(self, nodes):
        self.nodes = nodes
        self.released = []
        self.locks = 0
        # No components: _offload_node then builds an empty transfer list, which
        # is enough to observe whether it decided to offload at all.
        self._components_tuple = ()

    def resolve_node_handle(self, node_id):
        return self.nodes[node_id]

    def inc_lock_ref(self, node_id):
        self.locks += 1

        class _Params:
            def to_dec_params(self):
                return ("dec", node_id)

        return _Params()

    def dec_lock_ref(self, node_id, params):
        self.locks -= 1
        self.released.append(node_id)


def _bare_core(arena=None, **overrides):
    """A UnifiedTreeCore with only what the detach path touches.

    Deliberately the real class: the linker delegates the whole cut to it, so
    stubbing it here would leave the cut untested everywhere.
    """
    core = UnifiedTreeCore.__new__(UnifiedTreeCore)
    core._node_arena = dict(arena or {})
    core._detached_roots = {}
    core.page_size = 1
    core.components = ()
    core.full_host_duplicates = {}
    # Deleting a node clears it from both leaf sets, so they have to exist.
    core.evictable_device_leaves = set()
    core.evictable_host_leaves = set()
    core.root_node = SimpleNamespace(id=-1, parent=None, children={})
    core._update_evictable_leaf_sets = lambda node: None
    for key, value in overrides.items():
        setattr(core, key, value)
    return core


def _make_wrapper(completions, chain_len=3):
    """A wrapper with a linker and a chain, bypassing backend construction."""
    wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
    nodes = {}
    parent = None
    for node_id in range(chain_len):
        parent = FakeNode(node_id, parent)
        nodes[node_id] = parent
    wrapper.cache = FakeCache(nodes)
    wrapper.cache.tree_core = _bare_core(nodes)
    wrapper.cache_linker = FakeLinker(completions)
    wrapper.hit_markers = {}
    wrapper.pending_loads = {}
    wrapper.pending_offloads = []
    wrapper.failed_chains = {}
    wrapper.taken_loads = []
    return wrapper


def _drain(wrapper, finish_count):
    """take + commit on the local verdict, the way one rank sees it.

    The production path reduces the verdict across the attention group between
    these two calls; see TestLoadVerdictIsReduced.
    """
    successes = wrapper.take_completed_loads(finish_count)
    return wrapper.commit_completed_loads(successes)


class TestUnifiedCacheLinkerLoadFailure(CustomTestCase):
    def test_failed_batch_releases_locks_and_reports_rids(self):
        wrapper = _make_wrapper([(["rid-a", "rid-b"], False)])
        for rid in ("rid-a", "rid-b"):
            wrapper._queue_load(rid, 2, ["transfer"], anchor=0)
        self.assertEqual(wrapper.cache.locks, 2)

        failed = _drain(wrapper, 1)

        self.assertEqual(sorted(failed), ["rid-a", "rid-b"])
        # The load must not keep pinning the chain it can no longer fill.
        self.assertEqual(wrapper.cache.locks, 0)
        self.assertEqual(wrapper.pending_loads, {})

    def test_failed_batch_detaches_the_chain_it_published(self):
        """The published chain must never be offloaded back into the store."""
        wrapper = _make_wrapper([(["rid-a"], False)])
        wrapper._queue_load("rid-a", 2, ["transfer"], anchor=0)

        _drain(wrapper, 1)

        # Nodes 1 and 2 were published by this load; node 0 is the anchor the
        # request already had, and must be left alone.
        self.assertTrue(wrapper.cache.nodes[2].detached)
        self.assertTrue(wrapper.cache.nodes[1].detached)
        self.assertFalse(wrapper.cache.nodes[0].detached)
        self.assertEqual(wrapper.take_failed_chain("rid-a"), [2, 1])
        self.assertEqual(wrapper.take_failed_chain("rid-a"), [])

    def test_successful_batch_reports_nothing_and_keeps_the_chain(self):
        wrapper = _make_wrapper([(["rid-a"], True)])
        wrapper._queue_load("rid-a", 2, ["transfer"], anchor=0)

        failed = _drain(wrapper, 1)

        self.assertEqual(failed, [])
        self.assertEqual(wrapper.cache.locks, 0)
        self.assertTrue(wrapper.cache.nodes[2].external_cache_stored)
        self.assertFalse(wrapper.cache.nodes[2].detached)
        self.assertEqual(wrapper.failed_chains, {})

    def test_drain_tolerates_a_request_released_while_queued(self):
        """release_request cancels an unstarted load; its rid still comes back."""
        wrapper = _make_wrapper([(["rid-a", "rid-b"], False)])
        for rid in ("rid-a", "rid-b"):
            wrapper._queue_load(rid, 2, ["transfer"], anchor=0)
        wrapper.release_request("rid-a")
        self.assertEqual(wrapper.cache.locks, 1)

        failed = _drain(wrapper, 1)

        self.assertEqual(failed, ["rid-b"])
        self.assertEqual(wrapper.cache.locks, 0)

    def test_take_reports_the_local_verdict_without_touching_the_tree(self):
        """take must not detach or unlock -- the verdict is not final yet."""
        wrapper = _make_wrapper([(["rid-a"], False)])
        wrapper._queue_load("rid-a", 2, ["transfer"], anchor=0)

        successes = wrapper.take_completed_loads(1)

        self.assertEqual(successes, [False])
        # Still pinned and still on the tree until commit applies the verdict.
        self.assertEqual(wrapper.cache.locks, 1)
        self.assertEqual(wrapper.pending_loads.keys(), {"rid-a"})
        self.assertFalse(wrapper.cache.nodes[2].detached)
        self.assertEqual(wrapper.failed_chains, {})

    def test_commit_honours_a_verdict_the_rank_did_not_reach_itself(self):
        """A rank whose own get succeeded must still abort when a peer's failed.

        This is the divergence the MIN-reduce exists to stop: on hardware, one
        rank's batch_get missing a key made that rank abort while the other
        seven served the request, and the rank owning the output stream emitted
        KV that never arrived.
        """
        wrapper = _make_wrapper([(["rid-a"], True)])
        wrapper._queue_load("rid-a", 2, ["transfer"], anchor=0)

        self.assertEqual(wrapper.take_completed_loads(1), [True])
        failed = wrapper.commit_completed_loads([False])  # reduced from a peer

        self.assertEqual(failed, ["rid-a"])
        self.assertEqual(wrapper.cache.locks, 0)
        self.assertTrue(wrapper.cache.nodes[2].detached)
        self.assertEqual(wrapper.take_failed_chain("rid-a"), [2, 1])

    def test_commit_keeps_a_chain_when_the_group_agrees_it_landed(self):
        wrapper = _make_wrapper([(["rid-a"], True)])
        wrapper._queue_load("rid-a", 2, ["transfer"], anchor=0)

        self.assertEqual(wrapper.take_completed_loads(1), [True])
        self.assertEqual(wrapper.commit_completed_loads([True]), [])
        self.assertTrue(wrapper.cache.nodes[2].external_cache_stored)
        self.assertFalse(wrapper.cache.nodes[2].detached)
        self.assertEqual(wrapper.failed_chains, {})

    def test_reset_drops_batches_taken_but_never_committed(self):
        wrapper = _make_wrapper([(["rid-a"], False)])
        wrapper._queue_load("rid-a", 2, ["transfer"], anchor=0)
        wrapper.take_completed_loads(1)

        wrapper.reset()

        self.assertEqual(wrapper.taken_loads, [])

    def test_layer_counter_does_not_raise_into_the_forward(self):
        counter = LayerWiseLoadCounter(4)
        index = counter.update_producer()
        counter.set_consumer(index)
        counter.complete(index, 0)
        counter.fail(index, RuntimeError("transport reset"))

        # Every layer wait must return; raising here kills the engine.
        for layer in range(4):
            counter.wait_until(layer)

        self.assertNotIn(index, counter._futures)


class TestFailedLoadDoesNotPinTheForward(CustomTestCase):
    """A failed load must not keep the model forward's frames alive.

    fail() publishes one exception to every layer's future, and each layer's
    wait_until re-raises it. Python appends a traceback entry per raise, and
    each entry roots a frame chain reaching that layer's forward frame, keeping
    its locals -- that layer's activations -- alive. On DeepSeek-V4 that
    retained one full activation set per layer: +31.9 GiB on a single faulted
    forward at --chunked-prefill-size 2048, stacking across failures until the
    engine OOMed inside the attention kernel.
    """

    def test_the_published_failure_carries_no_traceback(self):
        counter = LayerWiseLoadCounter(8)
        index = counter.update_producer()
        counter.set_consumer(index)
        counter.fail(index, RuntimeError("transport reset"))

        # The type and message have to survive the copy that drops the frames.
        first = counter._futures[index][0].exception()
        self.assertIn("transport reset", str(first))
        self.assertIn("RuntimeError", str(first))

        for layer in range(8):
            counter.wait_until(layer)
            published = counter._futures.get(index)
            if published is None:  # popped on the last layer
                break
            error = published[layer].exception()
            depth = 0
            tb = error.__traceback__
            while tb is not None:
                depth += 1
                tb = tb.tb_next
            self.assertEqual(
                depth,
                0,
                f"layer {layer}: traceback survived wait_until, so every frame "
                f"it reaches -- including that layer's forward -- stays alive",
            )

    def test_the_caller_keeps_its_own_exception_intact(self):
        """fail() must not strip the traceback of the exception handed to it:
        the loader thread logs it with logger.exception() right after."""
        counter = LayerWiseLoadCounter(4)
        index = counter.update_producer()
        counter.set_consumer(index)
        try:
            raise RuntimeError("transport reset")
        except RuntimeError as caller_error:
            counter.fail(index, caller_error)
            for layer in range(4):
                counter.wait_until(layer)
            self.assertIsNotNone(caller_error.__traceback__)


class TestUMBPLinkerCompletionChannel(CustomTestCase):
    def test_linker_is_concrete(self):
        """Adding a UnifiedCacheLinker method without implementing it here
        leaves UMBPDirectLinker abstract, and the mori backend then dies at
        construction with TypeError, nowhere near the cause."""
        self.assertFalse(
            inspect.isabstract(UMBPDirectLinker),
            "UMBPDirectLinker cannot be constructed while it leaves "
            "UnifiedCacheLinker methods unimplemented",
        )

    def _linker_with_queues(self, run_result):
        linker = UMBPDirectLinker.__new__(UMBPDirectLinker)
        linker._load_queue = Queue()
        linker._completed_loads = Queue()
        linker._run_layer_wise_batch = run_result
        return linker

    # The load task carries the event guarding the batch's KV; the stubs below
    # replace the batch itself, so it only has to be there to be unpacked.
    _READY = SimpleNamespace(synchronize=lambda: None)

    def test_load_thread_publishes_failure(self):
        linker = self._linker_with_queues(lambda index, plans, ready_event: False)
        linker._load_queue.put((0, ["rid-a"], [], self._READY))
        linker._load_queue.put(None)
        linker._load_thread_func()

        self.assertEqual(linker.num_completed_loads(), 1)
        self.assertEqual(linker.pop_completed_load(), (["rid-a"], False))

    def test_load_thread_publishes_even_when_the_batch_raises(self):
        """The tree's locks hang on this batch coming back, however it ends."""

        def boom(index, plans, ready_event):
            raise RuntimeError("unexpected")

        linker = self._linker_with_queues(boom)
        linker._load_queue.put((0, ["rid-a"], [], self._READY))
        with self.assertRaises(RuntimeError):
            linker._load_thread_func()

        self.assertEqual(linker.pop_completed_load(), (["rid-a"], False))

    def test_load_thread_publishes_success(self):
        linker = self._linker_with_queues(lambda index, plans, ready_event: True)
        linker._load_queue.put((0, ["rid-a"], [], self._READY))
        linker._load_queue.put(None)
        linker._load_thread_func()

        self.assertEqual(linker.pop_completed_load(), (["rid-a"], True))


class TestInvalidateExternalLoadChain(CustomTestCase):
    """The tree-core guard: drop the chain only when nothing else owns it."""

    def _core(self, node, is_device_leaf=True):
        core = UnifiedTreeCore.__new__(UnifiedTreeCore)
        core._node_arena = {node.id: node} if node is not None else {}
        core.root_node = object()
        core._is_device_leaf = lambda n: is_device_leaf
        core.deleted = []
        core._delete_unbacked_device_leaf = (
            lambda n, tracker, device_frees, host_frees: core.deleted.append(n.id)
        )
        return core

    def _node(self, **overrides):
        class _Component:
            host_lock_ref = 0

        class _Node:
            id = 7
            backuped = False
            write_through_pending_id = None
            load_back_pending_id = None
            component_data = (_Component(),)

        node = _Node()
        node.children = {}
        for key, value in overrides.items():
            setattr(node, key, value)
        return node

    def test_drops_an_unowned_chain(self):
        node = self._node()
        core = self._core(node)
        self.assertTrue(core.invalidate_external_load_chain(7).is_dropped)
        self.assertEqual(core.deleted, [7])

    def test_declines_for_a_backed_up_or_in_flight_node(self):
        for field in ("backuped", "write_through_pending_id", "load_back_pending_id"):
            node = self._node(**{field: True})
            core = self._core(node)
            self.assertFalse(
                core.invalidate_external_load_chain(7).is_dropped,
                f"{field} should block the drop",
            )
            self.assertEqual(core.deleted, [])

    def test_declines_for_a_missing_node_or_a_non_leaf(self):
        """Non-leaf covers a since-adopted chain: locked, or grown a device child."""
        for core in (self._core(None), self._core(self._node(), is_device_leaf=False)):
            self.assertFalse(core.invalidate_external_load_chain(7).is_dropped)
            self.assertEqual(core.deleted, [])

    def test_declines_for_any_child_not_only_a_device_bearing_one(self):
        """`_is_device_leaf` is not a child check, so it cannot be the one here.

        It only rejects children holding Full KV *on device*. A child that
        holds host-only KV, or none, leaves the node looking free -- and
        deleting it strands that child in the arena with a parent the tree no
        longer holds, where no walk from the root or a detached root reaches it.
        """
        node = self._node()
        node.children = {"tok": object()}
        core = self._core(node, is_device_leaf=True)

        self.assertFalse(core.invalidate_external_load_chain(7).is_dropped)
        self.assertEqual(core.deleted, [])


class TestSchedulerMarkHook(CustomTestCase):
    """The scheduler side: mark the affected requests through ``to_finish``.

    Setting ``finished_reason`` here instead would make every result processor
    skip the request, so it would never be freed and never answer. The marker
    lets ``update_finish_state`` finish it inside the loop that already owns
    the free and the streaming.
    """

    def _make_req(self, rid, last_node=None):
        req = MagicMock()
        req.rid = rid
        req.last_node = last_node
        req.finished.return_value = False
        req.finished_reason = None
        req.to_finish = None
        req.skip_radix_cache_insert = False
        return req

    def _run(
        self,
        failed_rids,
        batch_reqs,
        running_reqs=(),
        chunked_req=None,
        detached_nodes=(),
    ):
        released = []

        tree_cache = MagicMock()
        tree_cache.drain_linker_loads.return_value = list(failed_rids)
        tree_cache.has_outstanding_failed_linker_chains.return_value = bool(
            detached_nodes
        )
        tree_cache.is_on_failed_linker_chain.side_effect = lambda node_id: (
            node_id in set(detached_nodes)
        )

        batch = MagicMock()
        batch.reqs = list(batch_reqs)
        running_batch = MagicMock()
        running_batch.reqs = list(running_reqs)
        running_batch.is_empty.return_value = not running_reqs

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.tree_cache = tree_cache
        scheduler.ipc_channels = MagicMock()
        scheduler.running_batch = running_batch
        scheduler.chunked_req = chunked_req
        scheduler._pending_chunked_abort_req = None
        scheduler._deferred_linker_rids = set()
        scheduler._release_aborted_request = lambda rid: released.append(rid)

        with patch("sglang.srt.managers.scheduler.release_kv_cache") as release_kv:
            scheduler._mark_failed_linker_loads(batch)

        return scheduler, released, release_kv

    def test_marks_only_the_failed_requests(self):
        failed, kept = self._make_req("rid-a"), self._make_req("rid-b")

        _scheduler, released, _release_kv = self._run(["rid-a"], [failed, kept])

        self.assertIsInstance(failed.to_finish, FINISH_ABORT)
        self.assertEqual(failed.to_finish.status_code, HTTPStatus.INTERNAL_SERVER_ERROR)
        self.assertIsNone(kept.to_finish)
        self.assertEqual(released, ["rid-a"])

    def test_never_finishes_or_frees_the_request_here(self):
        # A req finished before the result processors run is skipped by all of
        # them, so it would leak its KV and never respond; the free and the
        # response belong to the loop that promotes to_finish.
        req = self._make_req("rid-a")

        scheduler, _released, release_kv = self._run(["rid-a"], [req])

        self.assertIsNone(req.finished_reason)
        self.assertFalse(release_kv.called)
        self.assertFalse(scheduler.ipc_channels.send_to_tokenizer.send_output.called)

    def test_suppresses_the_radix_insert_of_the_unloaded_tail(self):
        # The eventual release_kv_cache defaults to is_insert=True. Without
        # this the unloaded pages go back into the tree, and they also pin the
        # chain cache_finished_req has to free.
        req = self._make_req("rid-a")

        self._run(["rid-a"], [req])

        self.assertTrue(req.skip_radix_cache_insert)

    def test_marks_a_request_that_has_moved_to_the_running_batch(self):
        # The batch count is MIN-reduced, so a lagging rank can defer the
        # verdict past the extend batch that consumed the load. The request is
        # still decoding over KV that never arrived.
        in_batch, running = self._make_req("rid-a"), self._make_req("rid-late")

        _scheduler, released, _release_kv = self._run(
            ["rid-late"], [in_batch], running_reqs=[running]
        )

        self.assertIsInstance(running.to_finish, FINISH_ABORT)
        self.assertIsNone(in_batch.to_finish)
        self.assertEqual(released, ["rid-late"])

    def test_defers_a_mid_chunk_request_to_the_safe_point(self):
        # Tearing it down here would leave self.chunked_req pointing at a freed
        # request for the next step to stash and re-prefill.
        req = self._make_req("rid-a")

        scheduler, _released, _release_kv = self._run(["rid-a"], [req], chunked_req=req)

        self.assertIs(scheduler._pending_chunked_abort_req, req)
        self.assertIsInstance(req.to_finish, FINISH_ABORT)

    def test_a_request_appearing_twice_is_marked_once(self):
        # A mixed batch has the same req in both lists.
        req = self._make_req("rid-a")

        _scheduler, released, _release_kv = self._run(
            ["rid-a"], [req], running_reqs=[req]
        )

        self.assertEqual(released, ["rid-a"])

    def test_leaves_an_already_finished_request_alone(self):
        req = self._make_req("rid-a")
        req.finished.return_value = True

        _scheduler, released, _release_kv = self._run(["rid-a"], [req])

        self.assertIsNone(req.to_finish)
        self.assertEqual(released, [])

    def test_no_failures_does_no_work(self):
        req = self._make_req("rid-a")

        _scheduler, released, release_kv = self._run([], [req])

        self.assertIsNone(req.to_finish)
        self.assertEqual(released, [])
        self.assertFalse(release_kv.called)

    def test_holds_a_rid_that_is_not_scheduled_here(self):
        req = self._make_req("rid-a")

        scheduler, released, _release_kv = self._run(["rid-gone"], [req])

        self.assertEqual(released, [])
        self.assertIsNone(req.to_finish)
        # Held, not dropped -- the request may be in a batch already launched
        # but not yet merged into running_batch.
        self.assertEqual(scheduler._deferred_linker_rids, {"rid-gone"})


class TestSchedulerAbortsChainCoOwners(CustomTestCase):
    """A failed load's chain can be held by a request that issued no load.

    The chain goes into the tree before the transfer is verified, so a request
    that arrives while the load is in flight matches it and is repointed onto
    exactly those pages. It appears in no rid list -- the linker knows only the
    rids it queued -- so naming is not enough to find it; it has to be found by
    where it points. Left alone it is served KV that never arrived, at HTTP
    200, which is the one outcome this whole path exists to prevent, and its
    next ``cache_unfinished_req`` re-inserts the chain's pages under a fresh
    node on top of that.
    """

    _make_req = TestSchedulerMarkHook._make_req
    _run = TestSchedulerMarkHook._run

    def test_aborts_a_request_that_holds_a_chain_but_named_no_load(self):
        loader = self._make_req("rid-a", last_node=5)
        co_owner = self._make_req("rid-b", last_node=5)

        _scheduler, released, _release_kv = self._run(
            ["rid-a"], [loader, co_owner], detached_nodes=[5]
        )

        self.assertIsInstance(co_owner.to_finish, FINISH_ABORT)
        self.assertTrue(co_owner.skip_radix_cache_insert)
        self.assertCountEqual(released, ["rid-a", "rid-b"])

    def test_leaves_a_request_that_points_somewhere_else_alone(self):
        loader = self._make_req("rid-a", last_node=5)
        elsewhere = self._make_req("rid-b", last_node=11)

        self._run(["rid-a"], [loader, elsewhere], detached_nodes=[5])

        self.assertIsNone(elsewhere.to_finish)

    def test_a_co_owner_in_the_running_batch_is_found_too(self):
        # It has already left the extend batch, and it is decoding over the
        # pages the load never filled.
        loader = self._make_req("rid-a", last_node=5)
        co_owner = self._make_req("rid-b", last_node=5)

        self._run(["rid-a"], [loader], running_reqs=[co_owner], detached_nodes=[5])

        self.assertIsInstance(co_owner.to_finish, FINISH_ABORT)

    def test_the_sweep_still_runs_when_no_new_load_failed(self):
        # The verdict and the co-owner do not have to land in the same step:
        # a chain detached earlier is still in the tree until its last owner
        # releases, and a request can match it right up to that point.
        co_owner = self._make_req("rid-b", last_node=5)

        self._run([], [co_owner], detached_nodes=[5])

        self.assertIsInstance(co_owner.to_finish, FINISH_ABORT)

    def test_no_outstanding_chain_means_the_walk_is_never_run(self):
        # The sweep costs a parent walk per request per step, so it must be
        # gated on there being a chain to find owners of.
        req = self._make_req("rid-a", last_node=5)

        scheduler, _released, _release_kv = self._run(["rid-a"], [req])

        self.assertFalse(scheduler.tree_cache.is_on_failed_linker_chain.called)


class TestReclaimFailedLinkerChain(CustomTestCase):
    """The reclaim is keyed by rid, and keeps what it could not free.

    ``cache_finished_req`` is where the loading request's whole chain becomes
    reclaimable: Full is a path-unlock, so that single ``dec_lock_ref`` clears
    ``lock_ref`` on every node of the chain at once. But the loading request is
    not always the last owner -- a request that matched the chain in the tree
    while the load was still in flight holds it too -- so a node that declines
    is carried to the next pass rather than abandoned. Nothing polls: the retry
    rides on the next request to finish, and the list is empty in the ordinary
    case.
    """

    def _cache(self, chains, drop_results, gone=()):
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.linker = MagicMock()
        cache.linker.take_failed_chain.side_effect = lambda rid: chains.pop(rid, [])
        cache._free_values = MagicMock()
        cache._stranded_linker_nodes = []

        attempts = []

        def invalidate(node_id):
            attempts.append(node_id)
            result = MagicMock()
            result.is_dropped = drop_results.pop(0)
            result.device_frees = {}
            result.host_frees = {}
            return result

        cache.tree_core = MagicMock()
        cache.tree_core.invalidate_external_load_chain.side_effect = invalidate
        cache.tree_core.holds_detached_node.side_effect = lambda node_id: (
            node_id not in gone
        )
        return cache, attempts

    def test_frees_the_whole_chain_endpoint_first(self):
        # Deleting the endpoint does not cascade, and a parent only becomes a
        # device leaf once its child is gone.
        cache, attempts = self._cache({"rid-a": [3, 2, 1]}, [True] * 3)

        cache._reclaim_failed_linker_chain("rid-a")

        self.assertEqual(attempts, [3, 2, 1])
        self.assertEqual(cache._free_values.call_count, 3)

    def test_a_request_with_no_failed_load_touches_the_tree_not_at_all(self):
        cache, attempts = self._cache({}, [])

        cache._reclaim_failed_linker_chain("rid-a")

        self.assertEqual(attempts, [])

    def test_another_rid_reclaims_nothing(self):
        cache, attempts = self._cache({"rid-a": [1]}, [True])

        cache._reclaim_failed_linker_chain("rid-b")

        self.assertEqual(attempts, [])

    def test_a_chain_someone_else_owns_is_retried_until_it_is_freed(self):
        # Abandoning it is not safe. The chain keeps its device slots while it
        # sits in the arena, and the request still pointing into it goes on to
        # insert those same slots under a fresh node -- one set of pages, two
        # owners, which is the pool-accounting leak. Eviction is no answer
        # either: a locked node is not a device leaf, so eviction skips it for
        # exactly as long as the other owner holds it.
        cache, attempts = self._cache({"rid-a": [1]}, [False, True])

        cache._reclaim_failed_linker_chain("rid-a")
        self.assertEqual(cache._stranded_linker_nodes, [1])

        # The other owner releases; the next request to finish frees it.
        cache._reclaim_failed_linker_chain("rid-b")

        self.assertEqual(attempts, [1, 1])
        self.assertEqual(cache._stranded_linker_nodes, [])

    def test_a_node_eviction_reached_first_leaves_the_retry_list(self):
        # Otherwise the list grows without bound: the node is gone from the
        # arena, so invalidate can never report it dropped.
        cache, attempts = self._cache({"rid-a": [1]}, [False, False], gone={1})

        cache._reclaim_failed_linker_chain("rid-a")

        self.assertEqual(attempts, [1])
        self.assertEqual(cache._stranded_linker_nodes, [])

    def test_the_retry_list_is_bounded(self):
        # Switching from "give up" to "retry" is only safe if the list cannot
        # grow: a node that stays owned is retried on every finishing request,
        # so naming it twice would accumulate a pass per duplicate.
        cache, attempts = self._cache({"rid-a": [1]}, [False] * 50)

        cache._reclaim_failed_linker_chain("rid-a")
        for i in range(20):
            cache._reclaim_failed_linker_chain(f"rid-{i}")

        self.assertEqual(cache._stranded_linker_nodes, [1])
        self.assertEqual(len(attempts), 21, "one attempt per pass, no more")

    def test_the_same_node_is_never_retried_twice_in_one_pass(self):
        cache, attempts = self._cache({"rid-a": [4], "rid-b": [4]}, [False] * 4)

        cache._reclaim_failed_linker_chain("rid-a")
        cache._reclaim_failed_linker_chain("rid-b")

        self.assertEqual(cache._stranded_linker_nodes, [4])

    def test_a_retry_is_attempted_before_a_freshly_failed_chain(self):
        # Endpoint-first ordering is what makes a chain free-able at all, and
        # an older chain's endpoint is not below a newer one.
        cache, attempts = self._cache(
            {"rid-a": [1], "rid-b": [9, 8]}, [False, True, True, True]
        )

        cache._reclaim_failed_linker_chain("rid-a")
        cache._reclaim_failed_linker_chain("rid-b")

        self.assertEqual(attempts, [1, 1, 9, 8])


class TestCacheFinishedReqReclaimsAfterTheUnlock(CustomTestCase):
    """The wiring: the reclaim runs, and runs after the tree lock is dropped.

    Freeing before ``_dec_req_lock`` would decline on every node -- the
    request still owns them -- and the chain would silently fall through to
    eviction on every abort.
    """

    def _cache(self, order):
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.session = MagicMock()
        cache.session.try_cache_finished_req.return_value = False
        cache.disable = False
        cache.enable_session_radix_cache = False
        cache._components_tuple = ()
        cache.req_to_token_pool = MagicMock()
        cache.token_to_kv_pool_allocator = MagicMock()
        cache._dec_req_lock = lambda req, skip_swa=False: order.append("unlock")
        cache._reclaim_failed_linker_chain = lambda rid: order.append(rid)
        # What is under test is the order of the two calls above, not the row
        # accounting cache_finished_req does on its way there.
        cache.free_kv_row = lambda kv, ranges: None
        return cache

    def test_the_reclaim_follows_the_unlock(self):
        order = []
        req = MagicMock()
        req.rid = "rid-a"
        req.origin_input_ids = [1, 2]
        req.output_ids = []
        req.kv.cache_protected_len = 0

        self._cache(order).cache_finished_req(req, is_insert=False, kv_len_to_handle=2)

        self.assertEqual(order, ["unlock", "rid-a"])


class TestFailedChainNeverReachesTheStore(CustomTestCase):
    """B1. A chain published by a failed load must not be offloaded.

    ``load_back`` sets ``external_cache_stored = True`` because those pages
    came *from* the store, which is also what makes them ineligible for
    offload -- so clearing it on failure would schedule the *unfilled* pages
    into the store, where the corruption outlives a ``/flush_cache``. The
    chain is cut out of the tree instead, and the write-through paths assert
    rather than test: reaching one of these nodes means the tree is corrupt.
    """

    def test_offloading_a_failed_chain_fails_loudly(self):
        wrapper = _make_wrapper([(["rid-a"], False)])
        wrapper._queue_load("rid-a", 2, ["transfer"], anchor=0)
        _drain(wrapper, 1)

        with self.assertRaises(AssertionError):
            wrapper.offload_nodes([2, 1])

        self.assertEqual(wrapper.cache_linker.offloaded, [])
        self.assertEqual(wrapper.pending_offloads, [])

    def test_a_failed_chain_keeps_the_flag_it_was_loaded_with(self):
        """The chain did come from the store; that fact did not change."""
        wrapper = _make_wrapper([(["rid-a"], False)])
        wrapper._queue_load("rid-a", 2, ["transfer"], anchor=0)

        _drain(wrapper, 1)

        for node_id in (1, 2):
            self.assertTrue(wrapper.cache.nodes[node_id].external_cache_stored)
            self.assertTrue(wrapper.cache.nodes[node_id].detached)
        # The anchor was the request's own node, not part of this load.
        self.assertFalse(wrapper.cache.nodes[0].detached)

    def test_an_ordinary_unstored_node_still_offloads(self):
        """The refusal must be detach-specific, not a blanket one."""
        wrapper = _make_wrapper([])
        wrapper.cache.nodes[1].external_cache_stored = False

        wrapper.offload_nodes([1])

        self.assertEqual(len(wrapper.cache_linker.offloaded), 1)


class TestDetachedNodeIsRefusedByTheTree(CustomTestCase):
    """B1, tree side: write-through must not fire for a detached node.

    Match is deliberately *not* taught to refuse one -- see the note on
    ``UnifiedTreeNode.detached``. The insert walk cannot reach a detached
    node either, so this is an assert, not a skip.
    """

    def _core(self, **overrides):
        core = UnifiedTreeCore.__new__(UnifiedTreeCore)
        core.is_write_back = False
        core.enable_hicache = False
        core.enable_external_cache_linker = True
        core.write_through_threshold = 1
        core.page_size = 1
        for key, value in overrides.items():
            setattr(core, key, value)
        return core

    def _node(self, **overrides):
        node = SimpleNamespace(
            id=7,
            evicted=False,
            detached=False,
            external_cache_stored=False,
            hit_count=0,
        )
        for key, value in overrides.items():
            setattr(node, key, value)
        return node

    def test_write_through_on_a_detached_node_fails_loudly(self):
        core = self._core()
        with self.assertRaises(AssertionError):
            core._inc_hit_count_and_check(self._node(detached=True))

    def test_write_through_still_fires_for_an_ordinary_node(self):
        core = self._core()
        self.assertTrue(core._inc_hit_count_and_check(self._node()))


class TestSchedulerDefersUnmatchedRids(CustomTestCase):
    """B3. A failed rid that matches no live request must not be dropped.

    Under overlap, ``pop_and_process()`` for batch N runs after
    ``get_next_batch_to_run`` has merged N into ``running_batch`` and after
    ``run_batch(N+1)`` launched N+1 -- so N+1's requests are in neither
    ``batch.reqs`` (which is N) nor ``running_batch``. A load for N+1 that fails
    fast, which is what a missing key does, lands in exactly that window.
    Dropping the verdict there serves the request over KV that never arrived,
    at HTTP 200.
    """

    def _scheduler(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.tree_cache = MagicMock()
        scheduler.ipc_channels = MagicMock()
        scheduler.running_batch = MagicMock()
        scheduler.running_batch.reqs = []
        scheduler.running_batch.is_empty.return_value = True
        scheduler.chunked_req = None
        scheduler._pending_chunked_abort_req = None
        scheduler._deferred_linker_rids = set()
        scheduler._release_aborted_request = lambda rid: None
        return scheduler

    def _req(self, rid):
        req = MagicMock()
        req.rid = rid
        req.finished.return_value = False
        req.finished_reason = None
        req.to_finish = None
        req.skip_radix_cache_insert = False
        return req

    def _step(self, scheduler, drained, batch_reqs, running_reqs=()):
        scheduler.tree_cache.drain_linker_loads.return_value = list(drained)
        scheduler.running_batch.reqs = list(running_reqs)
        scheduler.running_batch.is_empty.return_value = not running_reqs
        batch = MagicMock()
        batch.reqs = list(batch_reqs)
        with patch("sglang.srt.managers.scheduler.release_kv_cache"):
            scheduler._mark_failed_linker_loads(batch)

    def test_a_rid_not_yet_in_any_list_is_aborted_on_the_next_pass(self):
        scheduler = self._scheduler()
        late = self._req("rid-late")

        # Step 1: the verdict lands while N+1 is launched but not yet merged.
        self._step(scheduler, ["rid-late"], [self._req("rid-a")])
        self.assertIsNone(late.to_finish)

        # Step 2: N+1 has been merged into running_batch by the next
        # get_next_batch_to_run, so the retained verdict finds it.
        self._step(scheduler, [], [], running_reqs=[late])

        self.assertIsInstance(late.to_finish, FINISH_ABORT)
        self.assertEqual(late.to_finish.status_code, HTTPStatus.INTERNAL_SERVER_ERROR)

    def test_the_retry_is_bounded_to_one_pass(self):
        """One is provable, not arbitrary: a batch launched during step k is
        merged into running_batch by step k+1, so a rid still unmatched after a
        second pass is genuinely gone and must not accumulate forever."""
        scheduler = self._scheduler()

        self._step(scheduler, ["rid-ghost"], [])
        self.assertEqual(scheduler._deferred_linker_rids, {"rid-ghost"})

        self._step(scheduler, [], [])

        self.assertEqual(scheduler._deferred_linker_rids, set())

    def test_a_matched_rid_is_never_deferred(self):
        scheduler = self._scheduler()
        req = self._req("rid-a")

        self._step(scheduler, ["rid-a"], [req])

        self.assertIsInstance(req.to_finish, FINISH_ABORT)
        self.assertEqual(scheduler._deferred_linker_rids, set())


class TestLinkerLoadFailureIsDistinguishable(CustomTestCase):
    """The PD-prefill drop must fire for a linker failure and nothing else.

    A user abort reaches the decode node through its own AbortReq; a linker
    failure is an internal prefill-node decision that nothing else propagates,
    so this drop is all that stands between corrupt KV and the decode node.
    Gating on is_aborted() alone would change user-abort semantics too, so the
    verdict carries an err_type and is matched on it.
    """

    def _req(self, to_finish=None, finished_reason=None):
        req = MagicMock()
        req.to_finish = to_finish
        req.finished_reason = finished_reason
        return req

    def test_matches_a_linker_load_failure(self):
        req = self._req(
            to_finish=FINISH_ABORT(
                "Aborted: external KV cache load failed.",
                HTTPStatus.INTERNAL_SERVER_ERROR,
                err_type=EXTERNAL_KV_LOAD_ERR_TYPE,
            )
        )
        self.assertTrue(is_external_kv_load_failure(req))

    def test_matches_after_the_reason_is_promoted(self):
        # update_finish_state moves to_finish -> finished_reason mid-loop.
        req = self._req(
            finished_reason=FINISH_ABORT(
                "Aborted: external KV cache load failed.",
                HTTPStatus.INTERNAL_SERVER_ERROR,
                err_type=EXTERNAL_KV_LOAD_ERR_TYPE,
            )
        )
        self.assertTrue(is_external_kv_load_failure(req))

    def test_does_not_match_any_other_outcome(self):
        """A bare FINISH_ABORT is abort_request's "method 3"; a bootstrap
        failure carries no err_type; an unaborted request has neither."""
        bootstrap = FINISH_ABORT(
            "Prefill bootstrap failed", HTTPStatus.INTERNAL_SERVER_ERROR
        )
        for req in (
            self._req(FINISH_ABORT()),
            self._req(finished_reason=bootstrap),
            self._req(),
        ):
            self.assertFalse(is_external_kv_load_failure(req))

    def test_the_scheduler_tags_the_verdict_it_stages(self):
        """The mark hook must emit the err_type the PD drop matches on."""
        req = MagicMock()
        req.rid = "rid-a"
        req.finished.return_value = False
        req.finished_reason = None
        req.to_finish = None
        req.skip_radix_cache_insert = False

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.tree_cache = MagicMock()
        scheduler.tree_cache.drain_linker_loads.return_value = ["rid-a"]
        scheduler.ipc_channels = MagicMock()
        scheduler.running_batch = MagicMock()
        scheduler.running_batch.reqs = []
        scheduler.running_batch.is_empty.return_value = True
        scheduler.chunked_req = None
        scheduler._pending_chunked_abort_req = None
        scheduler._deferred_linker_rids = set()
        scheduler._release_aborted_request = lambda rid: None
        batch = MagicMock()
        batch.reqs = [req]

        with patch("sglang.srt.managers.scheduler.release_kv_cache"):
            scheduler._mark_failed_linker_loads(batch)

        self.assertTrue(is_external_kv_load_failure(req))


class TestFailedChainIsCutOutOfTheTree(CustomTestCase):
    """The cut, from the wrapper and from the tree core.

    ``match_prefix`` consults no per-node validity flag, so a merely flagged
    chain stayed reachable until the reclaim dropped it -- and the first
    request to match it gave the chain a device child, one of the conditions
    ``invalidate_external_load_chain`` declines on, pinning it for good.
    Cutting the top link does the same job without touching either walk.
    """

    def _chain(self, length):
        """anchor -> 1 -> ... -> length, registered in a bare core."""
        nodes = {}
        parent = None
        for node_id in range(length + 1):
            parent = FakeNode(node_id, parent)
            nodes[node_id] = parent
        return nodes, _bare_core(nodes)

    def test_the_whole_chain_is_filed_endpoint_first(self):
        """Deleting the endpoint does not cascade past the first ancestor that
        still holds a device value, and every node this load filled has one."""
        wrapper = _make_wrapper([(["rid-a"], False)], chain_len=4)
        wrapper._queue_load("rid-a", 3, ["transfer"], anchor=0)
        _drain(wrapper, 1)
        self.assertEqual(wrapper.take_failed_chain("rid-a"), [3, 2, 1])

        single = _make_wrapper([(["rid-b"], False)], chain_len=2)
        single._queue_load("rid-b", 1, ["transfer"], anchor=0)
        _drain(single, 1)
        self.assertEqual(single.take_failed_chain("rid-b"), [1])

    def test_the_chain_is_unreachable_but_keeps_its_own_links(self):
        wrapper = _make_wrapper([(["rid-a"], False)], chain_len=4)
        wrapper._queue_load("rid-a", 3, ["transfer"], anchor=0)
        _drain(wrapper, 1)

        nodes = wrapper.cache.nodes
        reachable = set()
        stack = [nodes[0]]
        while stack:
            node = stack.pop()
            reachable.add(node.id)
            stack.extend(node.children.values())
        self.assertEqual(reachable, {0}, "the chain is still walkable from the anchor")
        # Only the top link is cut; the reclaim still walks the chain.
        self.assertEqual(set(nodes[1].children), {2})
        self.assertEqual(set(nodes[2].children), {3})
        self.assertIs(nodes[1].parent, nodes[0])
        for node_id in (1, 2, 3):
            self.assertTrue(nodes[node_id].detached, node_id)
        self.assertFalse(nodes[0].detached, "anchor must stay")

    def test_a_successful_load_is_left_attached(self):
        wrapper = _make_wrapper([(["rid-a"], True)], chain_len=4)
        wrapper._queue_load("rid-a", 3, ["transfer"], anchor=0)
        _drain(wrapper, 1)
        self.assertEqual(set(wrapper.cache.nodes[0].children), {1})
        self.assertFalse(wrapper.cache.nodes[1].detached)

    def test_the_core_cuts_exactly_one_link(self):
        nodes, core = self._chain(3)
        self.assertEqual(core.detach_external_load_chain(3, 0), [3, 2, 1])
        self.assertEqual(nodes[0].children, {})
        self.assertEqual(set(nodes[1].children), {2})

    def test_the_top_is_recorded_as_a_detached_root(self):
        """_collect_all_nodes seeds from here, so sanity_check still sees it."""
        nodes, core = self._chain(3)
        core.detach_external_load_chain(3, 0)
        self.assertEqual(set(core._detached_roots), {1})
        self.assertIn(1, {n.id for n in core._collect_all_nodes()})

    def test_the_anchors_other_children_are_left_alone(self):
        nodes, core = self._chain(3)
        sibling = FakeNode(99, nodes[0])
        core._node_arena[99] = sibling

        core.detach_external_load_chain(3, 0)

        self.assertEqual(set(nodes[0].children), {99})
        self.assertFalse(sibling.detached)

    def test_an_empty_or_missing_chain_cuts_nothing(self):
        """The load adopted no node, or its endpoint is already gone."""
        nodes, core = self._chain(2)
        self.assertEqual(core.detach_external_load_chain(0, 0), [])
        self.assertEqual(core.detach_external_load_chain(404, 0), [])
        self.assertEqual(set(nodes[0].children), {1})
        self.assertEqual(core._detached_roots, {})


class TestOwnershipOfADetachedChain(CustomTestCase):
    """The two tree-core probes the reclaim and the sweep are built on."""

    def _core(self, nodes, root):
        core = UnifiedTreeCore.__new__(UnifiedTreeCore)
        core._node_arena = {n.id: n for n in nodes}
        core.root_node = root
        return core

    def _node(self, node_id, parent=None, detached=False):
        class _Node:
            pass

        node = _Node()
        node.id = node_id
        node.parent = parent
        node.detached = detached
        return node

    def test_a_node_on_the_chain_is_on_the_chain(self):
        root = self._node(0)
        top = self._node(1, parent=root, detached=True)
        core = self._core([root, top], root)

        self.assertTrue(core.is_on_detached_chain(1))

    def test_a_node_built_under_the_chain_is_on_it_too(self):
        # A request that matched the chain and then extended it owns a node
        # the detach never touched, hanging off one that it did. Its pages are
        # its own, but its prefix is the chain's.
        root = self._node(0)
        top = self._node(1, parent=root, detached=True)
        below = self._node(2, parent=top)
        core = self._core([root, top, below], root)

        self.assertTrue(core.is_on_detached_chain(2))

    def test_an_ordinary_node_the_root_and_a_missing_node_are_not(self):
        root = self._node(0)
        node = self._node(1, parent=root)
        core = self._core([root, node], root)

        self.assertFalse(core.is_on_detached_chain(1))
        self.assertFalse(core.is_on_detached_chain(0))
        self.assertFalse(core.is_on_detached_chain(99))

    def test_holds_detached_node_follows_the_arena(self):
        root = self._node(0)
        node = self._node(1, parent=root, detached=True)
        core = self._core([root, node], root)

        self.assertTrue(core.holds_detached_node(1))
        self.assertFalse(core.holds_detached_node(99))


class TestTombstoneCascadeLeavesNoStaleLeaf(CustomTestCase):
    """The cascade must drop a node from *both* leaf sets when it deletes it.

    The cascade's delete discarded only the host set. A node left in
    ``evictable_device_leaves`` after its arena entry is gone is what
    ``sanity_check`` reports as ``stale nodes in device_leaves``, fatal under
    the strict idle check -- seen on all 8 ranks of a DSv4-Pro run. The
    cascade reaches such a node by walking ``deleted.parent`` into a detached
    chain node.
    """

    def _core(self, cur, parent, root):
        core = UnifiedTreeCore.__new__(UnifiedTreeCore)
        core.root_node = root
        core._node_arena = {n.id: n for n in (cur, parent, root)}
        core._detached_roots = {}
        core.full_host_duplicates = {}
        core.components = ()
        core.components_by_type = {}
        core.page_size = 1
        core.evictable_device_leaves = {cur}
        core.evictable_host_leaves = set()
        core._update_evictable_leaf_sets = lambda node: None
        return core

    def _node(self, node_id, parent=None, detached=False):
        class _Key:
            def child_key(self, page_size):
                return node_id

        class _Data:
            value = None
            host_value = None
            lock_ref = 0
            host_lock_ref = 0

        class _Node:
            pass

        node = _Node()
        node.id = node_id
        node.parent = parent
        node.detached = detached
        node.key = _Key()
        node.children = {}
        node.component_data = (_Data(),)
        if parent is not None:
            parent.children[node_id] = node
        return node

    def test_freeing_a_detached_root_re_anchors_what_hung_off_it(self):
        """The chain must stay reachable while any of it survives.

        ``_detached_roots`` anchors a detached chain by its top node only; the
        rest hangs off it, and sanity_check's walk (root plus those roots) is
        the only thing that reaches them. The chain frees endpoint-first, from
        the bottom, so a node above can go while one below is still owned by a
        request that matched the chain before the load failed -- orphaning it
        in the arena and in the leaf sets, invisible to the walk. That is the
        ``D-leaf extra`` / ``stale nodes in device_leaves`` pair.
        """
        root = FakeNode(0)
        top = FakeNode(1, root)
        endpoint = FakeNode(2, top)
        top.detached = endpoint.detached = True
        core = _bare_core({0: root, 1: top, 2: endpoint})
        core._detached_roots = {top.id: top}

        core._remove_leaf_from_parent(top)

        self.assertIn(
            endpoint.id,
            core._detached_roots,
            "the node below the freed top is no longer reachable from any "
            "root, so the sanity walk cannot see it",
        )

    def test_removing_a_leaf_clears_it_from_both_leaf_sets(self):
        """The choke point, tested directly.

        Every deletion goes through ``_remove_leaf_from_parent``. Two callers
        had each independently discarded only the host set -- the tombstone
        cascade and ``_evict_host_leaf`` -- so the rule belongs here rather
        than in each caller, where it has now been forgotten twice.
        """
        parent = FakeNode(0)
        node = FakeNode(1, parent)
        node.detached = True
        core = _bare_core({0: parent, 1: node})
        core._detached_roots = {1: node}
        core.evictable_device_leaves = {node}
        core.evictable_host_leaves = {node}

        core._remove_leaf_from_parent(node)

        self.assertEqual(core.evictable_device_leaves, set())
        self.assertEqual(core.evictable_host_leaves, set())
        self.assertNotIn(1, core._node_arena)

    def test_a_cascade_deleted_node_does_not_stay_in_device_leaves(self):
        root = self._node(0)
        parent = self._node(1, parent=root)
        # `cur`: no value on either layer, so the cascade deletes it -- and it
        # is in the device set, which is the state the has_device branch and a
        # detached chain node produce together.
        cur = self._node(2, parent=parent, detached=True)
        deleted = self._node(3, parent=cur)
        cur.children.pop(3)
        core = self._core(cur, parent, root)

        core._iteratively_delete_tombstone_leaf(
            deleted, tracker={}, device_frees={}, host_frees={}
        )

        self.assertNotIn(
            2,
            [n.id for n in core.evictable_device_leaves],
            "the cascade unregistered the node but left the device-leaf set "
            "pointing at it -- sanity_check reports that as a stale entry",
        )
        self.assertNotIn(2, core._node_arena, "the node should have been deleted")


class TestFreeingADetachedChain(CustomTestCase):
    """A detached node is off the tree, so the free must not assume otherwise."""

    def _core(self):
        nodes = {}
        parent = None
        for node_id in range(3):
            parent = FakeNode(node_id, parent)
            nodes[node_id] = parent
        core = _bare_core(nodes)
        core.kv_events = MagicMock()
        core._release_all_component_layers = MagicMock()
        core.cascaded = []
        core._iteratively_delete_tombstone_leaf = (
            lambda node, tracker, device_frees, host_frees: core.cascaded.append(
                node.id
            )
        )
        return nodes, core

    def test_the_top_of_a_detached_chain_frees_without_asserting(self):
        nodes, core = self._core()
        core.detach_external_load_chain(2, 0)

        core._delete_unbacked_device_leaf(nodes[1], {}, {}, {})

        self.assertNotIn(1, core._node_arena)
        # Node 2 hung off the freed top and is still in the arena, so it takes
        # the top's place as the chain's anchor. Emptying _detached_roots here
        # -- the old behaviour -- left it reachable from nothing, which is the
        # stale device leaf sanity_check reports.
        self.assertEqual(set(core._detached_roots), {2})
        self.assertIn(2, core._node_arena)

    def test_freeing_it_never_evicts_a_node_that_took_its_place(self):
        """The anchor's child slot is reusable, and reuse must survive the reclaim.

        This is the whole point of detaching: a later request with the same
        tokens builds a fresh node under the anchor. Popping the key blind when
        the detached chain is finally freed would delete that request's live KV.
        """
        nodes, core = self._core()
        core.detach_external_load_chain(2, 0)
        replacement = FakeNode(1, nodes[0])  # same child key, new node
        core._node_arena[replacement.id] = replacement

        core._delete_unbacked_device_leaf(nodes[1], {}, {}, {})

        self.assertIs(nodes[0].children[1], replacement)

    def test_freeing_it_does_not_cascade_into_the_anchor(self):
        """The chain no longer hangs off the anchor, so walking up from it
        would delete a live node out from under its other children."""
        nodes, core = self._core()
        core.detach_external_load_chain(2, 0)

        core._delete_unbacked_device_leaf(nodes[2], {}, {}, {})

        self.assertEqual(core.cascaded, [])

    def test_an_attached_node_still_cascades(self):
        nodes, core = self._core()

        core._delete_unbacked_device_leaf(nodes[2], {}, {}, {})

        self.assertEqual(core.cascaded, [2])

    def test_an_attached_node_that_is_not_its_parents_child_still_asserts(self):
        """The tolerance is scoped to detached nodes; a real inconsistency
        must not be swallowed."""
        nodes, core = self._core()
        del nodes[0].children[1]

        with self.assertRaises(AssertionError):
            core._remove_leaf_from_parent(nodes[1])


class _LeafNode:
    """Enough node surface for the real leaf-set predicates."""

    def __init__(self, node_id, parent=None, device=True, host=False):
        self.id = node_id
        self.parent = parent
        self.detached = False
        self.backuped = host
        self.write_through_pending_id = None
        self.load_back_pending_id = None
        self.key = FakeKey(node_id)
        self.children = {}
        self.component_data = (
            SimpleNamespace(
                value=object() if device else None,
                host_value=object() if host else None,
                lock_ref=0,
                host_lock_ref=0,
            ),
        )
        if parent is not None:
            parent.children[node_id] = self

    @property
    def evicted(self):
        return self.component_data[0].value is None


class TestALeafSetNeverHoldsADeletedNode(unittest.TestCase):
    """A node the arena no longer holds must not be in a leaf set.

    The crash reads ``1 stale nodes in device_leaves: [N]``, made fatal by
    SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE. Deleting clears both sets;
    what puts the node back is ``_update_evictable_leaf_sets(parent)``, whose
    parent can already be gone -- a detached endpoint is a device leaf while a
    KV-less child hangs off it, so the endpoint frees first and that child's
    later deletion names a dead parent.
    """

    def _core(self):
        core = UnifiedTreeCore.__new__(UnifiedTreeCore)
        core.root_node = _LeafNode(-1, device=False)
        core._detached_roots = {}
        core.full_host_duplicates = {}
        core.components = ()
        core.components_by_type = {}
        core.component_types = ()
        core.page_size = 1
        core.is_write_back = False
        core.evictable_device_leaves = set()
        core.evictable_host_leaves = set()
        # anchor <- top <- endpoint <- a node built under the chain
        anchor = _LeafNode(184, core.root_node)
        top = _LeafNode(239, anchor)
        end = _LeafNode(238, top)
        child = _LeafNode(300, end, device=False)
        core._node_arena = {n.id: n for n in (core.root_node, anchor, top, end, child)}
        return core, anchor, top, end, child

    def test_the_endpoint_is_a_device_leaf_while_a_child_hangs_off_it(self):
        """The enabling condition, stated on its own so it cannot drift."""
        core, _anchor, _top, end, _child = self._core()

        self.assertTrue(core._is_device_leaf(end))

    def test_a_deleted_parent_is_not_put_back_when_its_child_goes(self):
        core, _anchor, _top, end, child = self._core()
        core._update_evictable_leaf_sets(end)
        self.assertIn(end, core.evictable_device_leaves)

        # The endpoint is freed while the child still hangs off it.
        core._remove_leaf_from_parent(end)
        self.assertNotIn(238, core._node_arena)
        # Now the child goes, and its parent is named to be re-evaluated.
        core._update_evictable_leaf_sets(child.parent)

        self.assertNotIn(end, core.evictable_device_leaves)
        self.assertNotIn(end, core.evictable_host_leaves)

    def test_a_node_still_in_the_tree_is_unaffected(self):
        """The guard keys on arena membership, not on the detached flag.

        The endpoint is detached and qualifies as a device leaf; for as long as
        the arena holds it, it belongs in the set.
        """
        core, _anchor, _top, end, _child = self._core()
        end.detached = True

        core._update_evictable_leaf_sets(end)

        self.assertIn(end, core.evictable_device_leaves)


class TestTheReclaimNeverStrandsAChild(unittest.TestCase):
    """The reclaim must not delete a node that still has a child.

    ``invalidate_external_load_chain``'s only child check was
    ``_is_device_leaf``, which rejects a child holding Full KV *on device* and
    nothing else, so a host-only or empty child left the node looking free.
    Deleting it strands the child: ``_collect_all_nodes`` walks from the root
    and the detached roots and never reaches it again, while a host-only child
    stays in ``evictable_host_leaves`` -- reported as ``stale nodes in
    host_leaves``. Found by enumerating operation interleavings, not by reading.
    """

    def _core(self, child_kind):
        core = UnifiedTreeCore.__new__(UnifiedTreeCore)
        core.root_node = _LeafNode(-1, device=False)
        core._detached_roots = {}
        core.full_host_duplicates = {}
        core.components = ()
        core.components_by_type = {}
        core.component_types = ()
        core.page_size = 1
        core.is_write_back = False
        core.evictable_device_leaves = set()
        core.evictable_host_leaves = set()
        core.kv_events = SimpleNamespace(record_remove=lambda *a, **k: None)

        anchor = _LeafNode(184, core.root_node)
        top = _LeafNode(239, anchor)
        end = _LeafNode(238, top)
        child = _LeafNode(300, end, device=False, host=(child_kind == "host"))
        core._node_arena = {n.id: n for n in (core.root_node, anchor, top, end, child)}
        # what a failed load leaves behind: the chain cut out and anchored
        for node in (top, end):
            node.detached = True
        anchor.children.pop(239)
        core._detached_roots[239] = top
        for node in (anchor, top, end, child):
            core._update_evictable_leaf_sets(node)
        return core, end, child

    def _reachable(self, core):
        stack = [core.root_node, *core._detached_roots.values()]
        seen = set()
        while stack:
            node = stack.pop()
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.extend(node.children.values())
        return seen

    def test_the_endpoint_looks_free_though_a_host_only_child_hangs_off_it(self):
        """The enabling condition, stated on its own so it cannot drift."""
        core, end, _child = self._core("host")

        self.assertTrue(core._is_device_leaf(end))

    def test_it_declines_rather_than_strand_a_host_only_child(self):
        core, _end, child = self._core("host")
        self.assertIn(child, core.evictable_host_leaves)

        result = core.invalidate_external_load_chain(238)

        self.assertFalse(result.is_dropped)
        self.assertIn(id(child), self._reachable(core))
        stale = [
            n for n in core.evictable_host_leaves if id(n) not in self._reachable(core)
        ]
        self.assertEqual(stale, [], "a leaf-set member the walk cannot reach")

    def test_it_declines_rather_than_strand_an_empty_child(self):
        """No leaf set names this child, so the only symptom is the leak."""
        core, _end, child = self._core("empty")

        result = core.invalidate_external_load_chain(238)

        self.assertFalse(result.is_dropped)
        self.assertIn(id(child), self._reachable(core))

    def test_it_still_drops_the_endpoint_once_the_child_is_gone(self):
        """Declining has to be a delay, not a refusal, or the chain leaks."""
        core, end, child = self._core("host")
        self.assertFalse(core.invalidate_external_load_chain(238).is_dropped)

        # the child's owner releases and eviction takes it
        core._remove_leaf_from_parent(child)

        self.assertTrue(core.invalidate_external_load_chain(238).is_dropped)
        self.assertNotIn(238, core._node_arena)


class TestTheArenaIsComparedAgainstTheWalk(unittest.TestCase):
    """`sanity_check` has to be able to see a stranded node at all.

    Its other checks all compare against `_collect_all_nodes`, so a node that
    has fallen out of that walk is invisible to every one of them unless it
    also sits in a leaf set. The host-only child in the class above was caught
    only by that accident; one holding device slots would have been dropped in
    silence, surfacing much later as a pool accounting mismatch.
    """

    def _core(self, *nodes):
        core = UnifiedTreeCore.__new__(UnifiedTreeCore)
        core.root_node = _LeafNode(-1, device=False)
        core._node_arena = {n.id: n for n in (core.root_node, *nodes)}
        core._detached_roots = {}
        return core

    def test_a_reachable_tree_reports_nothing(self):
        anchor = _LeafNode(184)
        core = self._core(anchor)
        walk = {core.root_node, anchor}

        self.assertEqual(core._nodes_out_of_the_walk(walk), [])

    def test_a_node_the_walk_misses_is_reported(self):
        anchor = _LeafNode(184)
        stranded = _LeafNode(300)
        core = self._core(anchor, stranded)
        walk = {core.root_node, anchor}

        self.assertEqual(core._nodes_out_of_the_walk(walk), [stranded])

    def test_the_root_is_never_reported(self):
        """The root anchors the walk; it is not a stranded node."""
        core = self._core()

        self.assertEqual(core._nodes_out_of_the_walk(set()), [])

    def test_reporting_a_stranded_node_does_not_itself_raise(self):
        """`sanity_check` calls `_describe_unreachable` on whatever this finds.

        That helper was written for leaf-set members, whose parent chain still
        reaches something the walk holds. A stranded node's parent may be gone
        entirely, and a reporter that throws on the state it is describing
        would turn a diagnosable violation into a crash inside the crash.
        """
        orphan = _LeafNode(300)
        orphan.parent = _LeafNode(238)  # a parent already deleted
        core = self._core(orphan)
        stranded = core._nodes_out_of_the_walk({core.root_node})

        described = core._describe_unreachable(stranded, {core.root_node})

        self.assertIn("node=300", described)
        self.assertIn("X", described, "the dead parent should be marked")


if __name__ == "__main__":
    unittest.main()
