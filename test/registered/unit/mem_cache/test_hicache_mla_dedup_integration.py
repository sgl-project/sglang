"""Focused coverage for the opt-in MLA HiCache dedup integration."""

import unittest
from contextlib import contextmanager, nullcontext
from queue import Queue
from types import SimpleNamespace
from unittest import mock

import torch

import sglang.srt.managers.cache_controller as cache_controller_module
from sglang.srt.managers.cache_controller import CacheOperation, HiCacheController
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.l2_transfer import L2Transfer
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.speculative import base_spec_worker
from sglang.srt.speculative.base_spec_worker import (
    BaseSpecWorker,
    HiCacheDraftMode,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _nextn_worker():
    spec_algorithm = SimpleNamespace(
        is_eagle=lambda: True,
        is_eagle3=lambda: False,
        is_dspark=lambda: False,
        is_ngram=lambda: False,
        is_frozen_kv_mtp=lambda: False,
        is_dflash_family=lambda: False,
    )
    target_runner = SimpleNamespace(
        mtp_draft_device_pools=(),
        mtp_draft_mamba_pools=(),
        spec_algorithm=spec_algorithm,
        req_to_token_pool=SimpleNamespace(mamba_pool=None),
    )
    draft_pool = object()
    draft_runner = SimpleNamespace(
        token_to_kv_pool=draft_pool,
        req_to_token_pool=SimpleNamespace(mamba_pool=None),
        model_config=SimpleNamespace(num_nextn_predict_layers=1),
    )
    worker = SimpleNamespace(
        target_worker=SimpleNamespace(model_runner=target_runner),
        draft_worker=object(),
        _draft_model_runners=lambda: (draft_runner,),
    )
    return worker, target_runner, draft_pool


class TestHiCacheMLADedupIntegration(unittest.TestCase):
    def test_disabled_flag_keeps_original_write_path(self):
        op = CacheOperation(torch.arange(2), torch.arange(2), node_id=1)
        completion = SimpleNamespace(
            start_event=object(), finish_event=object(), timing_enabled=False
        )
        controller = HiCacheController.__new__(HiCacheController)
        controller.mla_dedup = None
        controller.write_queue = [op]
        controller.ack_write_queue = []
        controller._move_write_operation = mock.Mock(
            return_value=(op.host_indices, op.device_indices, None)
        )
        controller._move_mla_write_operation = mock.Mock()
        controller._l2_transfers = mock.Mock(return_value=[mock.sentinel.transfer])
        controller._mla_l2_transfers = mock.Mock()
        controller._num_tokens_by_pool = mock.Mock(return_value={})
        controller._transfer_num_bytes = mock.Mock(return_value=0)
        controller._mla_transfer_num_bytes = mock.Mock()
        controller.mem_pool_device_allocator = SimpleNamespace(
            set_hicache_transfer_done_event=mock.Mock()
        )
        controller.l2_transfer_engine = SimpleNamespace(
            submit_device_to_host=mock.Mock(return_value=completion)
        )

        controller.start_writing()

        controller._move_write_operation.assert_called_once_with(op)
        controller._l2_transfers.assert_called_once()
        controller._move_mla_write_operation.assert_not_called()
        controller._mla_l2_transfers.assert_not_called()

    def test_write_fences_precede_copy_with_and_without_dedup(self):
        for dedup in (False, True):
            for wait_for_forward in (False, True):
                with self.subTest(dedup=dedup, wait_for_forward=wait_for_forward):
                    operations = []
                    op = CacheOperation(torch.arange(2), torch.arange(2), node_id=1)
                    completion = SimpleNamespace(
                        start_event=object(),
                        finish_event=object(),
                        timing_enabled=False,
                    )
                    controller = HiCacheController.__new__(HiCacheController)
                    controller.mla_dedup = (
                        SimpleNamespace(producer_stream=mock.sentinel.producer)
                        if dedup
                        else None
                    )
                    controller.write_fence_stream = mock.sentinel.forward
                    controller.write_queue = [op]
                    controller.ack_write_queue = []
                    for name in ("_move_write_operation", "_move_mla_write_operation"):
                        setattr(
                            controller,
                            name,
                            mock.Mock(
                                return_value=(op.host_indices, op.device_indices, None)
                            ),
                        )
                    for name in ("_l2_transfers", "_mla_l2_transfers"):
                        setattr(
                            controller, name, mock.Mock(return_value=[mock.sentinel.kv])
                        )
                    controller._num_tokens_by_pool = mock.Mock(return_value={})
                    controller._transfer_num_bytes = mock.Mock(return_value=0)
                    controller._mla_transfer_num_bytes = mock.Mock(return_value=0)
                    controller.mem_pool_device_allocator = mock.Mock()

                    def submit(transfers):
                        operations.append("copy")
                        self.assertEqual(transfers, [mock.sentinel.kv])
                        return completion

                    controller.l2_transfer_engine = SimpleNamespace(
                        device_to_host_stream=SimpleNamespace(
                            wait_stream=operations.append
                        ),
                        submit_device_to_host=submit,
                    )

                    controller.start_writing(wait_for_forward=wait_for_forward)

                    expected = [mock.sentinel.forward] if wait_for_forward else []
                    if dedup:
                        expected.append(mock.sentinel.producer)
                    self.assertEqual(operations, [*expected, "copy"])
                    self.assertEqual(controller.write_queue, [])
                    self.assertEqual(len(controller.ack_write_queue), 1)

    def test_disabled_flag_keeps_original_load_path(self):
        op = CacheOperation(torch.arange(2), torch.arange(2), node_id=1)
        producer_event = SimpleNamespace(start_event=mock.Mock(), complete=mock.Mock())
        completion = SimpleNamespace(
            start_event=object(), finish_event=object(), timing_enabled=False
        )
        controller = HiCacheController.__new__(HiCacheController)
        controller.mla_dedup = None
        controller.load_queue = [op]
        controller.ack_load_queue = []
        controller.transfer_layer_id_max = 2
        controller.layer_done_counter = SimpleNamespace(
            update_producer=mock.Mock(return_value=0), events=[producer_event]
        )
        controller._move_op_indices = mock.Mock(
            return_value=(op.host_indices, op.device_indices, None)
        )
        controller._l2_load_transfers = mock.Mock(return_value=[mock.sentinel.transfer])
        controller._start_loading_mla = mock.Mock()
        controller._num_tokens_by_pool = mock.Mock(return_value={})
        controller._transfer_num_bytes = mock.Mock(return_value=0)
        controller.load_fence_stream = None
        controller.mem_pool_device_allocator = SimpleNamespace(
            set_hicache_transfer_done_event=mock.Mock()
        )
        controller.l2_transfer_engine = SimpleNamespace(
            submit_host_to_device=mock.Mock(return_value=completion)
        )

        self.assertEqual(controller.start_loading(), 0)
        controller._move_op_indices.assert_called_once_with(op)
        controller._l2_load_transfers.assert_called_once()
        controller._start_loading_mla.assert_not_called()

    def test_peer_write_keeps_only_rank_local_sidecars(self):
        target_host = SimpleNamespace(_is_dummy=True)
        draft_host = SimpleNamespace(_is_dummy=False)
        anchor = SimpleNamespace(
            host_pool=target_host,
            device_pool=object(),
            layer_mapper=lambda layer_id: layer_id,
        )
        draft_entry = SimpleNamespace(
            host_pool=draft_host,
            device_pool=object(),
            layer_mapper=lambda layer_id: layer_id,
        )
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.mem_pool_host = SimpleNamespace(
            anchor_entry=anchor,
            entry_map={PoolName.DRAFT: draft_entry},
        )
        indices = torch.arange(2)

        transfers = controller._mla_l2_transfers(
            indices,
            indices,
            [
                PoolTransfer(
                    name=PoolName.DRAFT,
                    host_indices=indices,
                    device_indices=indices,
                )
            ],
        )

        self.assertEqual(len(transfers), 1)
        self.assertIs(transfers[0].host_pool, draft_host)

    def test_source_and_rank_local_loads_are_separated(self):
        target_host, draft_host = object(), object()
        target = L2Transfer(target_host, object(), torch.arange(1), torch.arange(1))
        draft = L2Transfer(draft_host, object(), torch.arange(1), torch.arange(1))
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.mem_pool_host = SimpleNamespace(
            entry_map={
                PoolName.KV: SimpleNamespace(host_pool=target_host),
                PoolName.DRAFT: SimpleNamespace(host_pool=draft_host),
            }
        )
        controller._prepare_mla_load_transfers = mock.Mock(return_value=[target, draft])
        op = CacheOperation(torch.arange(1), torch.arange(1), node_id=1)

        self.assertEqual(controller._prepare_mla_source_load(op), [target])
        self.assertEqual(controller._prepare_draft_load(op), [draft])

    def test_source_load_and_broadcast_are_layerwise(self):
        operations = []

        class Event:
            def record(self):
                pass

            def wait(self, stream):
                pass

        class Stream:
            def synchronize(self):
                operations.append(("sync", None))

        completion = SimpleNamespace(
            start_event=object(), finish_event=object(), timing_enabled=False
        )

        class Engine:
            host_to_device_stream = Stream()

            @staticmethod
            def resolve_device_indices(transfer):
                return transfer.device_indices

            @contextmanager
            def host_to_device_submission(self, transfers, *, start_event=None):
                yield transfers, completion

        broadcaster = SimpleNamespace(
            is_src=True,
            layer_num=2,
            prepare_broadcast=lambda indices, stream: (indices, None),
            broadcast_loaded_layer=lambda layer_id, plan: operations.append(
                ("broadcast", layer_id)
            ),
        )
        producer_event = SimpleNamespace(
            start_event=Event(),
            complete=lambda layer_id: operations.append(("complete", layer_id)),
        )
        controller = HiCacheController.__new__(HiCacheController)
        controller.mla_dedup = SimpleNamespace(broadcaster=broadcaster)
        controller.transfer_layer_id_max = 2
        controller.layer_done_counter = SimpleNamespace(events=[producer_event])
        controller.l2_transfer_engine = Engine()
        controller.load_fence_stream = None
        controller.ack_load_queue = []
        controller.mem_pool_device_allocator = SimpleNamespace(
            set_hicache_transfer_done_event=mock.Mock()
        )
        controller._prepare_mla_source_load = mock.Mock(return_value=["target"])
        controller._prepare_draft_load = mock.Mock(return_value=["draft"])
        controller._mla_broadcast_transfer = mock.Mock(
            return_value=L2Transfer(
                object(), object(), torch.arange(2), torch.arange(2)
            )
        )
        controller._load_mla_transfers_layer = lambda transfers, layer_id: (
            operations.append((transfers[0], layer_id))
        )
        controller._num_tokens_by_pool = mock.Mock(return_value={})
        controller._mla_transfer_num_bytes = mock.Mock(return_value=0)
        op = CacheOperation(torch.arange(2), torch.arange(2), node_id=1)

        with mock.patch.object(
            cache_controller_module,
            "device_module",
            SimpleNamespace(stream=lambda stream: nullcontext()),
        ):
            controller._start_loading_mla(0, op)

        self.assertEqual(
            operations,
            [
                ("target", 0),
                ("broadcast", 0),
                ("draft", 0),
                ("complete", 0),
                ("target", 1),
                ("broadcast", 1),
                ("draft", 1),
                ("complete", 1),
                ("sync", None),
            ],
        )

    def test_dummy_rank_prefetch_does_not_touch_storage(self):
        controller = HiCacheController.__new__(HiCacheController)
        controller.mla_dedup = SimpleNamespace(is_dummy_rank=True)
        controller.page_size = 4
        controller.prefetch_sync_queue = Queue()
        operation = SimpleNamespace(
            hash_value=["a", "b", "c"],
            request_id="rid",
            is_terminated=lambda: False,
        )

        self.assertEqual(controller._page_transfer(operation), 3)
        self.assertEqual(controller.prefetch_sync_queue.qsize(), 1)
        ack = controller.prefetch_sync_queue.get_nowait()
        self.assertEqual(ack.completed_tokens, 12)

    def test_runtime_attach_rejects_non_file_backend(self):
        controller = HiCacheController.__new__(HiCacheController)
        controller.mla_dedup = SimpleNamespace(is_dummy_rank=False)
        controller.enable_storage = False

        with self.assertRaisesRegex(RuntimeError, "while MLA host dedup is active"):
            controller.attach_storage_backend("mooncake")

    def test_dedup_draft_sidecar_rejects_l3(self):
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.mla_dedup = SimpleNamespace(is_dummy_rank=False)

        with self.assertRaisesRegex(RuntimeError, "rank-local draft pools"):
            controller.attach_storage_backend(
                "file",
                host_pools=[SimpleNamespace(name=PoolName.DRAFT)],
            )

    def test_shutdown_releases_dedup_communicator(self):
        context = mock.Mock()
        controller = SimpleNamespace(mla_dedup=context)
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache._storage_attachment = None
        cache.cache_controller = controller

        cache.shutdown()

        context.destroy.assert_called_once()
        self.assertIsNone(controller.mla_dedup)

    def test_dedup_keeps_nextn_draft_rank_local(self):
        for enabled, expected_mode in (
            (False, HiCacheDraftMode.PACKED),
            (True, HiCacheDraftMode.SIDECAR),
        ):
            with (
                self.subTest(enable_dedup=enabled),
                mock.patch.object(
                    base_spec_worker,
                    "get_memory",
                    return_value=SimpleNamespace(
                        enable_hierarchical_cache=True,
                        enable_unified_cache_external_linker=False,
                        enable_mla_hicache_host_dedup=enabled,
                    ),
                ),
                mock.patch.object(
                    base_spec_worker,
                    "get_disagg",
                    return_value=SimpleNamespace(
                        disaggregation_decode_retraction_backup=None
                    ),
                ),
            ):
                worker, target_runner, draft_pool = _nextn_worker()
                plan = BaseSpecWorker._build_hicache_draft_plan(worker)
                self.assertEqual(plan.mode, expected_mode)
                expected_packed = () if enabled else (draft_pool,)
                self.assertEqual(target_runner.mtp_draft_device_pools, expected_packed)


if __name__ == "__main__":
    unittest.main()
