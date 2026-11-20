"""PD decode: the metadata rows cap the KV transfers in flight.

A decode request holds one metadata row from its preallocation until its KV has
landed. Rows below --disaggregation-decode-extra-slots leave the extra request
slots unusable: preallocation stops at the row count with request rows and KV free.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from test_decode_retraction_transfer_progress import SMALL_POOL, _DecodeNode

from sglang.srt.disaggregation.decode import DecodeReqToTokenPool
from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, enter_override

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDecodeMetadataRows(CustomTestCase):
    def setUp(self):
        enter_override(
            self,
            get_context().override_server_args(
                disaggregation_mode="decode", disable_radix_cache=True, page_size=1
            ),
        )
        # Built after the rows are sized; they only take the allocator and buffers.
        for queue in ("DecodeTransferQueue", "DecodePreallocQueue"):
            patcher = patch.object(scheduler_module, queue)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _decode_rows(self, *, running, extra_slots, architecture="LlamaForCausalLM"):
        """Metadata rows of a decode scheduler set up by init_disaggregation."""
        scheduler = SimpleNamespace(
            _abort_prefill_transfer=None,
            draft_worker=None,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(architectures=[architecture])
            ),
            req_to_token_pool=DecodeReqToTokenPool(
                size=running,
                max_context_len=4,
                device="cpu",
                enable_memory_saver=False,
                pre_alloc_size=extra_slots,
            ),
            server_args=SimpleNamespace(sampling_mask_max_tokens=0),
            token_to_kv_pool_allocator=SimpleNamespace(
                get_kvcache=lambda: SimpleNamespace(
                    maybe_get_custom_mem_pool=lambda: None
                )
            ),
            tree_cache=None,
            max_total_num_tokens=0,
        )
        Scheduler.init_disaggregation(scheduler)
        rows = scheduler.req_to_metadata_buffer_idx_allocator.available_size()
        self.assertEqual(scheduler.disagg_metadata_buffers.output_ids.shape[0], rows)
        return rows

    def test_every_extra_slot_can_hold_a_transfer(self):
        """One running request with 64 extra slots: 64 transfers may be in flight,
        not 2 x the running requests."""
        self.assertEqual(self._decode_rows(running=1, extra_slots=64), 64)
        self.assertEqual(self._decode_rows(running=16, extra_slots=64), 64)

    def test_preallocation_fills_the_extra_slots(self):
        """On the real decode queues sized so, one running request with 64 extra
        slots gives all 8 waiting arrivals KV destinations, not 2."""
        node = _DecodeNode(
            SMALL_POOL,
            rows=1,
            extra_slots=64,
            metadata_rows=self._decode_rows(running=1, extra_slots=64),
        )
        for i in range(8):
            node.arrive(f"r{i}", 100, 4, done_at=100)
        with patch("torch.distributed.get_world_size", return_value=1):
            node.step()
        self.assertEqual(len(node.transfer_rids()), 8)

    def test_rows_stay_stock_when_they_cover_the_extra_slots(self):
        # The extra-slot default for up to 32 running requests a worker is 2 x them.
        self.assertEqual(self._decode_rows(running=16, extra_slots=32), 32)
        self.assertEqual(self._decode_rows(running=36, extra_slots=64), 72)
        self.assertEqual(self._decode_rows(running=64, extra_slots=0), 128)
        self.assertEqual(
            self._decode_rows(
                running=4, extra_slots=8, architecture="MiniMaxM3SparseForCausalLM"
            ),
            32,
        )


if __name__ == "__main__":
    unittest.main()
