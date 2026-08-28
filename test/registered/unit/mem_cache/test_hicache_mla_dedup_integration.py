"""Unit tests for the MLA host-dedup controller integration."""

import unittest
from contextlib import nullcontext
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
from sglang.srt.speculative import base_spec_worker
from sglang.srt.speculative.base_spec_worker import (
    BaseSpecWorker,
    HiCacheDraftMode,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _nextn_worker(*, enable_dedup: bool):
    spec_algorithm = SimpleNamespace(
        is_eagle=lambda: True,
        is_eagle3=lambda: False,
        is_dspark=lambda: False,
    )
    target_runner = SimpleNamespace(
        mtp_draft_device_pools=(),
        spec_algorithm=spec_algorithm,
    )
    draft_pool = object()
    draft_runner = SimpleNamespace(
        token_to_kv_pool=draft_pool,
        model_config=SimpleNamespace(
            num_nextn_predict_layers=1,
            hf_config=SimpleNamespace(architectures=["Glm4MoeForCausalLM"]),
        ),
    )
    worker = SimpleNamespace(
        target_worker=SimpleNamespace(model_runner=target_runner),
        server_args=SimpleNamespace(
            enable_hierarchical_cache=True,
            enable_mla_hicache_host_dedup=enable_dedup,
        ),
        _draft_model_runners=lambda: (draft_runner,),
    )
    return worker, target_runner, draft_pool


class TestHiCacheMLADedupDraftPlan(unittest.TestCase):
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
        controller.l2_transfer_engine = SimpleNamespace(
            submit_device_to_host=mock.Mock(return_value=completion)
        )

        controller.start_writing()

        controller._move_write_operation.assert_called_once_with(op)
        controller._l2_transfers.assert_called_once()
        controller._move_mla_write_operation.assert_not_called()
        controller._mla_l2_transfers.assert_not_called()
        controller._mla_transfer_num_bytes.assert_not_called()

    def test_disabled_flag_keeps_original_load_path(self):
        op = CacheOperation(torch.arange(2), torch.arange(2), node_id=1)
        start_event = mock.Mock()
        producer_event = SimpleNamespace(start_event=start_event, complete=mock.Mock())
        completion = SimpleNamespace(
            start_event=object(), finish_event=object(), timing_enabled=False
        )
        controller = HiCacheController.__new__(HiCacheController)
        controller.mla_dedup = None
        controller.load_queue = [op]
        controller.ack_load_queue = []
        controller.layer_num = 2
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
        controller.l2_transfer_engine = SimpleNamespace(
            submit_host_to_device=mock.Mock(return_value=completion)
        )

        self.assertEqual(controller.start_loading(), 0)

        controller._move_op_indices.assert_called_once_with(op)
        controller._l2_load_transfers.assert_called_once()
        controller._start_loading_mla.assert_not_called()

    def test_peer_l2_transfer_keeps_only_rank_local_sidecars(self):
        target_host = SimpleNamespace(_is_dummy=True)
        draft_host = SimpleNamespace(_is_dummy=False)
        target_device = object()
        draft_device = object()
        anchor = SimpleNamespace(
            host_pool=target_host,
            device_pool=target_device,
            layer_mapper=lambda layer_id: layer_id,
        )
        draft_entry = SimpleNamespace(
            host_pool=draft_host,
            device_pool=draft_device,
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
        target_host = object()
        draft_host = object()
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

        broadcaster = SimpleNamespace(
            is_src=True,
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
        controller.layer_num = 2
        controller.layer_done_counter = SimpleNamespace(events=[producer_event])
        controller.l2_transfer_engine = SimpleNamespace(host_to_device_stream=Stream())
        controller.ack_load_queue = []
        controller._prepare_mla_source_load = mock.Mock(return_value=[object()])
        controller._prepare_draft_load = mock.Mock(return_value=[object()])
        controller._load_mla_source_layer = lambda state, layer_id: operations.append(
            ("target", layer_id)
        )
        controller._load_draft_layer = lambda state, layer_id: operations.append(
            ("draft", layer_id)
        )
        controller._record_mla_source_load = mock.Mock()
        controller._record_draft_load = mock.Mock()
        op = CacheOperation(torch.arange(2), torch.arange(2), node_id=1)

        fake_device_module = SimpleNamespace(
            stream=lambda stream: nullcontext(),
        )
        with (
            mock.patch.object(
                cache_controller_module, "device_module", fake_device_module
            ),
            mock.patch.object(
                cache_controller_module,
                "make_timing_event_pair",
                return_value=(Event(), Event(), False),
            ),
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

    def test_dedup_keeps_nextn_draft_rank_local(self):
        with mock.patch.object(
            base_spec_worker,
            "get_memory",
            return_value=SimpleNamespace(enable_hierarchical_cache=True),
        ):
            for enabled, expected_mode in (
                (False, HiCacheDraftMode.PACKED),
                (True, HiCacheDraftMode.SIDECAR),
            ):
                with self.subTest(enable_dedup=enabled):
                    worker, target_runner, draft_pool = _nextn_worker(
                        enable_dedup=enabled
                    )
                    plan = BaseSpecWorker._build_hicache_draft_plan(worker)
                    self.assertEqual(plan.mode, expected_mode)
                    expected_packed = () if enabled else (draft_pool,)
                    self.assertEqual(
                        target_runner.mtp_draft_device_pools, expected_packed
                    )


if __name__ == "__main__":
    unittest.main()
