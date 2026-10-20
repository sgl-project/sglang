"""Shared-host publication, native layer waits, and reader retirement."""

import threading
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.managers.cache_controller import (
    CacheOperation,
    HiCacheAck,
    HiCacheController,
    LayerDoneCounter,
)
from sglang.srt.mem_cache.l2_transfer import L2Transfer, TransferCompletion
from sglang.srt.mem_cache.mla_host_shared import SharedCPUHostDedupContext
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _Event:
    def __init__(self, actions, name="copy", ready=True):
        self.actions = actions
        self.name = name
        self.ready = ready

    def record(self):
        self.actions.append((self.name, "record"))

    def query(self):
        return self.ready

    def synchronize(self):
        self.actions.append((self.name, "finished"))
        self.ready = True


def _controller(actions, *, owner=True):
    controller = HiCacheController.__new__(HiCacheController)
    with (
        mock.patch.object(torch.distributed, "is_initialized", return_value=True),
        mock.patch.object(torch.distributed, "get_backend", return_value="gloo"),
        mock.patch.object(torch.distributed, "get_world_size", return_value=2),
    ):
        controller.mla_dedup = SharedCPUHostDedupContext(process_group=object())
    controller.tp_group = controller.mla_dedup.process_group
    controller.mem_pool_device_allocator = mock.Mock()
    controller.mem_pool_device_allocator.set_hicache_transfer_done_event.side_effect = (
        lambda key, event: actions.append((key[1], "lease_published"))
    )
    controller._num_tokens_by_pool = mock.Mock(return_value={})
    controller._transfer_num_bytes = mock.Mock(return_value=0)
    controller.write_queue = []
    controller.load_queue = []
    controller.ack_write_queue = []
    controller.ack_load_queue = []
    controller.l2_transfer_engine = mock.Mock()
    controller.l2_transfer_engine.device_to_host_stream.wait_stream.side_effect = (
        lambda stream: actions.append("producer_wait")
    )
    controller.l2_transfer_engine.host_to_device_stream.wait_stream.side_effect = (
        lambda stream: actions.append("old_forward_wait")
    )
    completion = TransferCompletion(
        _Event(actions, "start"), _Event(actions, "copy"), False
    )
    controller.l2_transfer_engine.submit_device_to_host.side_effect = lambda transfers: (
        actions.append(("d2h", len(transfers))) or completion
    )
    controller.l2_transfer_engine.submit_host_to_device.side_effect = (
        lambda transfers, **kwargs: (
            actions.append(("h2d", len(transfers))) or completion
        )
    )
    host_pool = SimpleNamespace(
        allocator=SimpleNamespace(
            is_shared=True,
            is_owner=owner,
            process_group=controller.mla_dedup.process_group,
        ),
        shared_memory_group=controller.mla_dedup.process_group,
    )
    transfer = L2Transfer(
        host_pool=host_pool,
        device_pool=object(),
        host_indices=torch.tensor([7, 2]),
        device_indices=torch.tensor([11, 23]),
    )
    op = CacheOperation(transfer.host_indices, transfer.device_indices, node_id=17)
    controller._l2_transfers = mock.Mock(return_value=[transfer])
    controller._l2_load_transfers = mock.Mock(return_value=[transfer])
    controller._move_write_operation = mock.Mock(
        return_value=(op.host_indices, op.device_indices, None)
    )
    controller._move_op_indices = mock.Mock(
        return_value=(op.host_indices, op.device_indices, None)
    )
    controller.load_fence_stream = None
    controller.transfer_layer_id_max = 3
    controller.layer_done_counter = SimpleNamespace(
        update_producer=lambda: 0,
        events=[
            SimpleNamespace(
                start_event=_Event(actions, "producer"), complete=mock.Mock()
            )
        ],
    )
    return controller, transfer, op, completion


def _tree(controller, actions):
    tree = UnifiedRadixCache.__new__(UnifiedRadixCache)
    tree.cache_controller = controller
    tree.pp_rank = 0
    tree.buffer_pipeline = None
    tree.metrics_collector = None
    tree.tree_core = SimpleNamespace(
        write_back_duplicate_reclaim_digest=0,
        finish_load_back=lambda node: actions.append("host_rows_reclaimable"),
    )
    tree.ongoing_load_back = {17: (object(), object(), object())}
    tree.dec_lock_ref = mock.Mock()
    tree.dec_host_lock_ref = mock.Mock()
    return tree


class TestHiCacheSharedCPU(CustomTestCase):
    def test_owner_publishes_only_after_forward_and_d2h(self):
        actions = []
        controller, _, op, completion = _controller(actions)
        producer = object()
        controller.set_mla_producer_stream(producer)
        controller.write_queue = [op]

        def publish(values, *, op, group):
            self.assertIs(group, controller.mla_dedup.process_group)
            if op == torch.distributed.ReduceOp.MIN:
                actions.append("host_row_consensus")
                return
            self.assertEqual(op, torch.distributed.ReduceOp.MAX)
            self.assertIn(("copy", "finished"), actions)
            actions.append("host_rows_published")

        with mock.patch.object(torch.distributed, "all_reduce", side_effect=publish):
            controller.start_writing()

        self.assertEqual(
            actions,
            [
                "producer_wait",
                "host_row_consensus",
                ("d2h", 1),
                ("copy", "finished"),
                "host_rows_published",
                ("write", "lease_published"),
            ],
        )
        self.assertIs(
            controller.mla_dedup.last_write_finish_event, completion.finish_event
        )
        self.assertIs(
            controller.ack_write_queue[0].finish_event, completion.finish_event
        )
        next_forward = mock.Mock()
        controller.wait_mla_write_on_stream(next_forward)
        next_forward.wait_event.assert_called_once_with(completion.finish_event)

    def test_peer_does_not_overwrite_shared_backup(self):
        actions = []
        controller, _, op, _ = _controller(actions, owner=False)
        controller.set_mla_producer_stream(object())
        controller.write_queue = [op]
        with mock.patch.object(torch.distributed, "all_reduce") as publication:
            controller.start_writing()
        controller.l2_transfer_engine.submit_device_to_host.assert_called_once_with([])
        self.assertEqual(publication.call_count, 2)
        self.assertEqual(len(controller.ack_write_queue), 1)

    def test_failed_backup_does_not_publish_ack_or_release_fence(self):
        controller, _, op, _ = _controller([])
        controller.write_queue = [op]
        controller.l2_transfer_engine.submit_device_to_host.side_effect = RuntimeError(
            "injected D2H submission failure"
        )

        def check_failure(values, **kwargs):
            if kwargs["op"] == torch.distributed.ReduceOp.MAX:
                self.assertEqual(int(values[0]), 1)

        def gather_errors(values, failure, **kwargs):
            values[:] = [failure, None]

        with (
            mock.patch.object(
                torch.distributed, "all_reduce", side_effect=check_failure
            ) as publication,
            mock.patch.object(torch.distributed, "get_world_size", return_value=2),
            mock.patch.object(
                torch.distributed, "all_gather_object", side_effect=gather_errors
            ),
            self.assertRaisesRegex(RuntimeError, "D2H publication failed"),
        ):
            controller.start_writing()
        self.assertEqual(publication.call_count, 2)
        self.assertEqual(controller.ack_write_queue, [])
        controller.mem_pool_device_allocator.set_hicache_transfer_done_event.assert_not_called()

    def test_each_rank_loads_local_rows_without_broadcast_or_whole_stream_wait(self):
        actions = []
        controller, transfer, op, completion = _controller(actions)
        controller.load_queue = [op]
        controller.load_fence_stream = object()
        controller._mla_layer_loader = mock.Mock(
            side_effect=AssertionError("Shared H2D must not enqueue a broadcast")
        )
        with mock.patch.object(torch.distributed, "all_reduce"):
            self.assertEqual(controller.start_loading(), 0)
        args, kwargs = controller.l2_transfer_engine.submit_host_to_device.call_args
        self.assertEqual(args, ([transfer],))
        self.assertEqual(transfer.device_indices.tolist(), [11, 23])
        self.assertIs(
            kwargs["on_layer_done"], controller.layer_done_counter.events[0].complete
        )
        self.assertEqual(kwargs["transfer_layer_id_max"], 3)
        self.assertNotIn("load_layer", kwargs)
        controller._mla_layer_loader.assert_not_called()
        controller.l2_transfer_engine.host_to_device_stream.synchronize.assert_not_called()
        self.assertIs(
            controller.ack_load_queue[0].finish_event, completion.finish_event
        )
        self.assertEqual(controller.ack_load_queue[0].consumer_index, -1)
        self.assertIn("old_forward_wait", actions)

    def test_divergent_host_row_order_aborts_before_copy(self):
        controller, _, op, _ = _controller([])
        controller.load_queue = [op]

        def mismatch(values, **kwargs):
            values[0] -= 1

        with (
            mock.patch.object(torch.distributed, "all_reduce", side_effect=mismatch),
            self.assertRaisesRegex(RuntimeError, "row order diverged"),
        ):
            controller.start_loading()
        controller.l2_transfer_engine.submit_host_to_device.assert_not_called()
        self.assertEqual(controller.ack_load_queue, [])

    def test_peer_copy_failure_is_not_a_successful_local_ack(self):
        controller, _, op, _ = _controller([])
        controller.load_queue = [op]

        def peer_failed(values, **kwargs):
            if kwargs["op"] == torch.distributed.ReduceOp.MAX:
                values[0] = 1

        def gather_errors(values, failure, **kwargs):
            self.assertIsNone(failure)
            values[:] = [None, "injected peer H2D failure"]

        with (
            mock.patch.object(torch.distributed, "all_reduce", side_effect=peer_failed),
            mock.patch.object(torch.distributed, "get_world_size", return_value=2),
            mock.patch.object(
                torch.distributed, "all_gather_object", side_effect=gather_errors
            ),
            self.assertRaisesRegex(RuntimeError, "peer H2D failure"),
        ):
            controller.start_loading()
        self.assertEqual(controller.ack_load_queue, [])
        controller.mem_pool_device_allocator.set_hicache_transfer_done_event.assert_not_called()

    def test_preparation_failure_reaches_rank_consensus_before_raising(self):
        for operation in ("load", "write"):
            controller, _, op, _ = _controller([])
            if operation == "load":
                controller.load_queue = [op]
                controller._l2_load_transfers.side_effect = RuntimeError(
                    "injected preparation failure"
                )
                invoke = controller.start_loading
            else:
                controller.write_queue = [op]
                controller.set_mla_producer_stream(object())
                controller.l2_transfer_engine.device_to_host_stream.wait_stream.side_effect = RuntimeError(
                    "injected preparation failure"
                )
                invoke = controller.start_writing

            def verdict(values, **kwargs):
                self.assertEqual(kwargs["op"], torch.distributed.ReduceOp.MIN)
                self.assertEqual(int(values[-1]), -1)

            def gather_errors(values, failure, **kwargs):
                values[:] = [failure, None]

            with (
                self.subTest(operation=operation),
                mock.patch.object(
                    torch.distributed, "all_reduce", side_effect=verdict
                ) as consensus,
                mock.patch.object(torch.distributed, "get_world_size", return_value=2),
                mock.patch.object(
                    torch.distributed, "all_gather_object", side_effect=gather_errors
                ),
                self.assertRaisesRegex(RuntimeError, "injected preparation failure"),
            ):
                invoke()
            consensus.assert_called_once()
            controller.l2_transfer_engine.submit_device_to_host.assert_not_called()
            controller.l2_transfer_engine.submit_host_to_device.assert_not_called()
            self.assertEqual(controller.ack_load_queue, [])
            self.assertEqual(controller.ack_write_queue, [])

    def test_native_counter_waits_only_for_requested_layer(self):
        stream = mock.Mock()
        events = []

        def event():
            value = _Event([], name=str(len(events)), ready=True)
            events.append(value)
            return value

        with (
            mock.patch(
                "sglang.srt.managers.cache_controller.device_module.Event",
                side_effect=event,
            ),
            mock.patch(
                "sglang.srt.managers.cache_controller.device_module.current_stream",
                return_value=stream,
            ),
        ):
            counter = LayerDoneCounter(3)
            producer = counter.update_producer()
            counter.set_consumer(producer)
            first, _, last = counter.events[producer].load_events
            last.ready = False
            counter.wait_until(0)
            stream.wait_event.assert_called_once_with(first)
            self.assertFalse(last.query())
            self.assertIsNone(counter.on_layer_ready)
            self.assertFalse(counter.needs_layerwise_restore())
            counter.wait_until(2)
            self.assertIs(stream.wait_event.call_args.args[0], last)

    def test_slow_peer_keeps_host_rows_pinned_until_tp_min_is_ready(self):
        actions = []
        controller, _, _, completion = _controller(actions)
        controller.ack_load_queue = [
            HiCacheAck(completion.start_event, completion.finish_event, [17])
        ]
        tree = _tree(controller, actions)
        peer_ready = 0

        def consensus(values, operation):
            self.assertEqual(operation, torch.distributed.ReduceOp.MIN)
            values[0] = min(int(values[0]), peer_ready)

        tree._all_reduce = consensus
        tree.loading_check()
        self.assertEqual(len(controller.ack_load_queue), 1)
        self.assertIn(17, tree.ongoing_load_back)
        tree.dec_host_lock_ref.assert_not_called()
        self.assertNotIn("host_rows_reclaimable", actions)
        peer_ready = 1
        tree.loading_check()
        self.assertEqual(controller.ack_load_queue, [])
        self.assertEqual(tree.ongoing_load_back, {})
        tree.dec_host_lock_ref.assert_called_once()
        self.assertEqual(actions[-1], "host_rows_reclaimable")

    def test_pending_local_copy_also_blocks_global_ack(self):
        actions = []
        controller, _, _, completion = _controller(actions)
        completion.finish_event.ready = False
        controller.ack_load_queue = [
            HiCacheAck(completion.start_event, completion.finish_event, [17])
        ]
        tree = _tree(controller, actions)
        tree._all_reduce = lambda values, operation: None
        tree.loading_check()
        self.assertEqual(len(controller.ack_load_queue), 1)
        tree.dec_host_lock_ref.assert_not_called()

    def test_cancel_reset_waits_for_readers_before_dropping_acks(self):
        actions = []
        controller, _, _, completion = _controller(actions)
        completion.finish_event.ready = False
        controller.ack_load_queue = [
            HiCacheAck(completion.start_event, completion.finish_event, [17])
        ]
        controller.enable_storage = False
        controller.storage_stop_event = threading.Event()
        controller._stop_storage_threads = mock.Mock()
        controller.l2_transfer_engine.host_to_device_stream.synchronize.side_effect = (
            lambda: actions.append("h2d_stream_finished")
        )
        controller.l2_transfer_engine.device_to_host_stream.synchronize.side_effect = (
            lambda: actions.append("d2h_stream_finished")
        )

        def reader_barrier(*args, **kwargs):
            self.assertTrue(completion.finish_event.query())
            self.assertEqual(len(controller.ack_load_queue), 1)
            self.assertIn("h2d_stream_finished", actions)
            self.assertIn("d2h_stream_finished", actions)
            actions.append("all_readers_finished")

        with (
            mock.patch.object(torch.distributed, "barrier", side_effect=reader_barrier),
            mock.patch.object(torch.distributed, "all_reduce"),
        ):
            controller.reset()
        self.assertIn("all_readers_finished", actions)
        self.assertEqual(controller.ack_load_queue, [])
        self.assertFalse(controller.has_inflight_device_transfers())

    def test_runtime_storage_attachment_rejected_before_backend_changes(self):
        controller, _, _, _ = _controller([])
        controller.enable_storage = False
        controller.storage_backend = None
        with self.assertRaisesRegex(ValueError, "L2 cache only"):
            controller.attach_storage_backend("file")
        self.assertFalse(controller.enable_storage)
        self.assertIsNone(controller.storage_backend)

    def test_side_pool_constructor_failure_destroys_registered_pools_in_reverse(self):
        from sglang.srt.mem_cache.hicache_storage import PoolName
        from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as assembler

        for built_side_pool in (False, True):
            actions = []
            root = SimpleNamespace(
                allocator=SimpleNamespace(is_shared=True),
                destroy=lambda: actions.append("root_destroyed"),
            )
            side = SimpleNamespace(destroy=lambda: actions.append("indexer_destroyed"))

            def configuration(name, *, layout_root=False, builder=None):
                return SimpleNamespace(
                    decl=SimpleNamespace(
                        pool_name=name,
                        device_pool=object(),
                        is_layout_root=layout_root,
                        is_primary=layout_root,
                        layout_source=PoolName.KV,
                        host_pool_builder=builder,
                    ),
                    packed_draft_device_pools=(),
                    layer_mapping={0: 0},
                    transfer_layer_id_max=1,
                )

            failing_builder = mock.Mock()
            failing_builder.build.side_effect = RuntimeError(
                "injected indexer constructor failure"
            )
            pools = [configuration(PoolName.KV, layout_root=True)]
            if built_side_pool:
                pools.append(
                    configuration(
                        PoolName.INDEXER,
                        builder=mock.Mock(build=mock.Mock(return_value=side)),
                    )
                )
            pools.append(
                configuration(
                    PoolName.DRAFT_INDEXER if built_side_pool else PoolName.INDEXER,
                    builder=failing_builder,
                )
            )
            with (
                self.subTest(built_side_pool=built_side_pool),
                mock.patch.object(
                    assembler, "_get_allocator_type", return_value="default"
                ),
                self.assertRaisesRegex(
                    RuntimeError, "injected indexer constructor failure"
                ),
            ):
                assembler._build_pool_entries(
                    config=SimpleNamespace(pools=pools), root_host_pool=root
                )
            self.assertEqual(
                actions,
                ["indexer_destroyed", "root_destroyed"]
                if built_side_pool
                else ["root_destroyed"],
            )


class TestHiCacheSharedCPUArguments(CustomTestCase):
    def _arguments(self, **overrides):
        from sglang.srt.server_args import ServerArgs

        fields = dict(
            model_path="dummy",
            enable_hierarchical_cache=True,
            enable_mla_hicache_host_dedup=True,
            mla_hicache_host_dedup_mode="shared",
            hicache_io_backend="direct",
            hicache_mem_layout="page_first_direct",
            hicache_ratio=2,
            tp_size=2,
        )
        fields.update(overrides)
        return ServerArgs(**fields)

    def test_shared_mode_preserves_native_waits_without_broadcast_flag(self):
        from sglang.srt.arg_groups.hicache_hook import handle_hicache
        from sglang.srt.server_args import ServerArgs

        self.assertEqual(
            ServerArgs(model_path="dummy").mla_hicache_host_dedup_mode, "broadcast"
        )
        arguments = self._arguments()
        handle_hicache(arguments)
        self.assertFalse(arguments.enable_mla_hicache_layerwise_overlap)

    def test_shared_mode_rejects_uncovered_combinations(self):
        from sglang.srt.arg_groups.hicache_hook import handle_hicache

        for fields in (
            {"enable_mla_hicache_host_dedup": False},
            {"enable_hierarchical_cache": False},
            {"nnodes": 2},
            {"tp_size": 1},
            {"pp_size": 2},
            {"dp_size": 2},
            {"attn_dp_size": 2},
            {"attn_cp_size": 2},
            {"dcp_size": 2},
            {"speculative_algorithm": "EAGLE"},
            {"disaggregation_mode": "decode"},
            {"hicache_storage_backend": "file"},
            {"hicache_host_memory_mode": "buffer_only"},
            {"enable_mla_hicache_layerwise_overlap": True},
            {"hicache_io_backend": "kernel", "hicache_mem_layout": "layer_first"},
        ):
            with self.subTest(**fields), self.assertRaises(ValueError):
                handle_hicache(self._arguments(**fields))


class TestHiCacheSharedCPUBudget(CustomTestCase):
    def test_layer_sharded_target_rejected_before_shared_context_creation(self):
        from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as assembler
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

        pool = mock.Mock(spec=MLATokenToKVPool, layer_shard_enabled=True)
        params = SimpleNamespace(
            token_to_kv_pool_allocator=mock.Mock(get_kvcache=lambda: pool),
            mtp_draft_device_pools=(),
        )
        with self.assertRaisesRegex(ValueError, "flat CUDA MLA/DSA"):
            assembler._validate_shared_mla_pool(pool, params, None)

    def test_shared_allocation_is_charged_to_owner_once(self):
        from sglang.srt.mem_cache import hicache_auto_size as sizing
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

        target = mock.Mock(
            spec=MLATokenToKVPool,
            host_capacity_bytes=100,
            get_kv_size_bytes=mock.Mock(return_value=100),
        )
        params = SimpleNamespace(
            token_to_kv_pool_allocator=mock.Mock(get_kvcache=lambda: target),
            req_to_token_pool=None,
            mtp_draft_device_pools=(),
        )
        demands = []
        for rank in (0, 1):
            with (
                mock.patch.object(
                    sizing,
                    "get_memory",
                    return_value=SimpleNamespace(enable_mla_hicache_host_dedup=True),
                ),
                mock.patch.object(
                    sizing, "mla_dedup_rank_and_size", return_value=(rank, 2)
                ),
                mock.patch.object(sizing, "mla_host_dedup_eligible", return_value=True),
            ):
                demands.append(
                    sizing._estimate_hicache_bytes(params, None, physical=True)
                )
        self.assertEqual(demands, [100, 0])

    def test_peer_maps_owner_budget_without_zero_allowance_or_ratio_clipping(self):
        from sglang.srt.mem_cache import hicache_auto_size as sizing
        from sglang.srt.mem_cache.pool_host import base

        for fraction in (None, 0.8):
            ratios = []
            for rank in (0, 1):
                memory = SimpleNamespace(
                    enable_mla_hicache_host_dedup=True,
                    mla_hicache_host_dedup_mode="shared",
                    hicache_host_memory_fraction=fraction,
                    hicache_ratio=4 if fraction is None else 8,
                    hicache_size=0,
                )
                context = mock.Mock()
                params = SimpleNamespace(tp_cache_group=object())

                def owner_allowance(value, *, src, group):
                    self.assertIs(group, params.tp_cache_group)
                    self.assertEqual(src, 4)
                    value[0] = 500

                with (
                    self.subTest(rank=rank, fraction=fraction),
                    mock.patch.object(sizing, "get_memory", return_value=memory),
                    mock.patch.object(sizing, "get_context", return_value=context),
                    mock.patch.object(
                        sizing,
                        "_estimate_hicache_bytes",
                        return_value=100 if rank == 0 else 0,
                    ),
                    mock.patch.object(
                        sizing, "host_memory_budget_bytes", return_value=125
                    ),
                    mock.patch.object(
                        sizing,
                        "_dedup_host_budget",
                        return_value=(5, 500 if rank == 0 else 0),
                    ),
                    mock.patch.object(
                        torch.distributed, "get_global_rank", return_value=4
                    ),
                    mock.patch.object(
                        torch.distributed, "broadcast", side_effect=owner_allowance
                    ),
                    mock.patch.object(
                        torch.distributed, "is_initialized", return_value=False
                    ),
                ):
                    with sizing.auto_size_hicache(params, None, enabled=True):
                        self.assertEqual(base.host_memory_budget_bytes(400), 500)
                        self.assertEqual(base._host_memory_budget.get(), 100)
                    if fraction is None:
                        context.override.assert_not_called()
                        ratios.append(memory.hicache_ratio)
                    else:
                        ratios.append(
                            context.override.call_args.kwargs["hicache_ratio"]
                        )
                self.assertIsNone(base._host_memory_budget.get())
            self.assertEqual(ratios[0], ratios[1])


if __name__ == "__main__":
    unittest.main()
