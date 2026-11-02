"""DSA/MTP controller binding and load-stream ordering regressions."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.managers.cache_controller import CacheOperation, HiCacheController
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    StackBuildResult,
    _apply_stack_result,
)
from sglang.srt.mem_cache.page_interleave_pool import PageInterleaveDSATokenToKVPool
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestShardedDSAHiCache(CustomTestCase):
    def test_draft_gather_binds_same_final_controller_counter(self):
        target = PageInterleaveDSATokenToKVPool.__new__(PageInterleaveDSATokenToKVPool)
        draft = PageInterleaveDSATokenToKVPool.__new__(PageInterleaveDSATokenToKVPool)
        counter = object()
        result = StackBuildResult(
            host_pool_group=Mock(),
            cache_controller=SimpleNamespace(
                layer_done_counter=counter, transfer_layer_id_max=3
            ),
            component_host_pools={ComponentType.FULL: Mock()},
            pools_desc="KV + INDEXER",
        )
        cache = SimpleNamespace(components={ComponentType.FULL: SimpleNamespace()})
        _apply_stack_result(
            cache, target, SimpleNamespace(mtp_draft_device_pools=(draft,)), result
        )
        self.assertIs(target.layer_transfer_counter, counter)
        self.assertIs(draft.layer_transfer_counter, counter)

    def test_load_fences_resolved_indices_before_submission(self):
        events = []
        start = Mock()
        start.record.side_effect = lambda: events.append("fence")
        producer = SimpleNamespace(start_event=start, complete=Mock())
        indices = torch.arange(64)
        transfers = [object()]
        completion = SimpleNamespace(
            start_event=start, finish_event=object(), timing_enabled=False
        )

        def prepare(op):
            events.append("prepare")
            return op.host_indices[:16], op.device_indices[:16], None

        def describe(host, device, pool_transfers):
            events.append("describe")
            self.assertEqual((host.numel(), device.numel()), (16, 16))
            return transfers

        def submit(actual, **kwargs):
            events.append("submit")
            self.assertIs(actual, transfers)
            self.assertIs(kwargs["start_event"], start)
            return completion

        cc = SimpleNamespace(
            load_queue=[CacheOperation(indices, indices, node_id=7)],
            ack_load_queue=[],
            layer_done_counter=SimpleNamespace(
                update_producer=lambda: 0, events=[producer]
            ),
            _move_op_indices=prepare,
            _l2_load_transfers=describe,
            load_fence_stream=None,
            l2_transfer_engine=SimpleNamespace(submit_host_to_device=submit),
            transfer_layer_id_max=3,
            mem_pool_device_allocator=Mock(),
            _num_tokens_by_pool=lambda op: {"kv": 64},
            _transfer_num_bytes=lambda op: 0,
        )
        self.assertEqual(HiCacheController.start_loading(cc), 0)
        self.assertEqual(events, ["prepare", "fence", "describe", "submit"])
        self.assertEqual(cc.ack_load_queue[0].num_tokens, 64)
        self.assertIs(cc.ack_load_queue[0].finish_event, completion.finish_event)


if __name__ == "__main__":
    unittest.main()
