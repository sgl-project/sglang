import types
import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode  # noqa: E402

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def _make_req(req_pool_idx, origin_input_ids, output_ids):
    return types.SimpleNamespace(
        kv=types.SimpleNamespace(req_pool_idx=req_pool_idx),
        origin_input_ids=origin_input_ids,
        output_ids=output_ids,
        return_logprob=False,
        grammar=None,
        return_hidden_states=False,
        return_hidden_states_mode=CaptureHiddenMode.NULL,
        is_prefill_only=False,
    )


class TestHisparseDecodeBatchReqPoolCpu(unittest.TestCase):
    def test_build_hisparse_decode_batch_populates_req_pool_indices_cpu(self):
        # _build_hisparse_decode_batch builds a ScheduleBatch off the normal
        # extend path, so it must populate the req_pool_indices_cpu host mirror
        # in lockstep with the device tensor. A missing mirror crashes hisparse
        # decode bookkeeping (map_last_loc_to_buffer -> _grow_device_buffers
        # indexes req_pool_indices_cpu).
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.device = "cpu"
        scheduler.req_to_token_pool = types.SimpleNamespace(device="cpu")
        scheduler.token_to_kv_pool_allocator = None
        scheduler.tree_cache = None
        scheduler.model_config = types.SimpleNamespace(
            is_encoder_decoder=False, vocab_size=32
        )
        scheduler.enable_overlap = False
        scheduler.spec_algorithm = types.SimpleNamespace(is_none=lambda: True)
        scheduler.future_map = MagicMock()

        reqs = [
            _make_req(req_pool_idx=4, origin_input_ids=[1, 2, 3], output_ids=[7]),
            _make_req(req_pool_idx=9, origin_input_ids=[1, 2], output_ids=[8]),
        ]

        with patch(
            "sglang.srt.managers.scheduler.SamplingBatchInfo.from_schedule_batch",
            return_value=MagicMock(),
        ):
            batch = scheduler._build_hisparse_decode_batch(reqs)

        # Assert the invariant (cpu mirror == device tensor), not a hardcoded
        # copy of the input -- the latter would just restate the builder line.
        self.assertIsNotNone(batch.req_pool_indices_cpu)
        self.assertTrue(
            torch.equal(batch.req_pool_indices_cpu, batch.req_pool_indices.cpu())
        )


class TestFilterBatchReqsToExclude(unittest.TestCase):
    def _make_filter_req(self, rid: str, finished: bool = False):
        return types.SimpleNamespace(
            rid=rid,
            finished=lambda: finished,
            return_logprob=False,
            grammar=None,
            beam_group=None,
            return_hidden_states=False,
            return_hidden_states_mode=CaptureHiddenMode.NULL,
        )

    def test_filter_batch_excludes_specified_and_finished_reqs(self):
        from sglang.srt.managers.schedule_batch import ScheduleBatch
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        r0 = self._make_filter_req("r0")
        r1 = self._make_filter_req("r1")
        r2 = self._make_filter_req("r2", finished=True)
        r3 = self._make_filter_req("r3")
        batch = ScheduleBatch(
            reqs=[r0, r1, r2, r3],
            device="cpu",
            model_config=types.SimpleNamespace(is_encoder_decoder=False),
            sampling_info=MagicMock(),
            spec_info=None,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            multimodal_inputs=None,
            req_pool_indices=torch.arange(4, dtype=torch.int64),
            req_pool_indices_cpu=torch.arange(4, dtype=torch.int64),
            seq_lens=torch.ones(4, dtype=torch.int64),
            orig_seq_lens=torch.ones(4, dtype=torch.int64),
        )

        batch.filter_batch(reqs_to_exclude=[r1])

        self.assertEqual([r.rid for r in batch.reqs], ["r0", "r3"])
        self.assertTrue(
            torch.equal(
                batch.req_pool_indices_cpu, torch.tensor([0, 3], dtype=torch.int64)
            )
        )


class TestHf3fsRankMetadataSet(unittest.TestCase):
    def test_rank_metadata_release_and_delete_deduplicate_free_pages(self):
        from sglang.srt.mem_cache.storage.hf3fs.mini_3fs_metadata_server import (
            RankMetadata,
        )

        meta = RankMetadata(num_pages=4)
        alloc = meta.reserve_and_allocate_page_indices([("k0", ""), ("k1", "")])
        self.assertEqual(alloc, [(False, 3), (False, 2)])
        self.assertEqual(meta.free_pages, [0, 1])
        self.assertEqual(meta.free_pages_set, {0, 1})

        meta.confirm_write([("k0", 3)], [2, 2, 1])
        self.assertEqual(meta.free_pages, [0, 1, 2])
        self.assertEqual(meta.free_pages_set, {0, 1, 2})

        deleted = meta.delete_keys(["k0", "missing"])
        self.assertEqual(deleted, 1)
        self.assertEqual(meta.free_pages, [0, 1, 2, 3])
        self.assertEqual(meta.free_pages_set, {0, 1, 2, 3})


if __name__ == "__main__":
    unittest.main()
