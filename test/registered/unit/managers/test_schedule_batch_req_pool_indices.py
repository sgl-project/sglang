import types
import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _make_req(req_pool_idx, origin_input_ids, output_ids):
    return types.SimpleNamespace(
        req_pool_idx=req_pool_idx,
        origin_input_ids=origin_input_ids,
        output_ids=output_ids,
        return_logprob=False,
        grammar=None,
        return_hidden_states=False,
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


class TestHisparseCoordinatorReqPoolCpu(unittest.TestCase):
    def test_host_bookkeeping_requires_req_pool_indices_cpu(self):
        # Why the mirror must exist: hisparse host bookkeeping indexes
        # req_pool_indices_cpu element-wise (int(req_pool_indices_cpu[i]) in
        # _eager_backup_previous_token, the first thing map_last_loc_to_buffer
        # runs each decode step). A missing mirror (None) raises TypeError there
        # -- the exact nightly failure the scheduler-side fix prevents. Asserting
        # the crash directly keeps the "mirror is required" contract honest, with
        # no mocked attributes (the crash precedes any self access).
        coord = HiSparseCoordinator.__new__(HiSparseCoordinator)
        seq_lens = torch.tensor([10], dtype=torch.int64)
        seq_lens_cpu = torch.tensor([10], dtype=torch.int64)
        req_pool_indices = torch.tensor([0], dtype=torch.int64)
        out_cache_loc = torch.tensor([0], dtype=torch.int64)
        with self.assertRaises(TypeError):
            coord.map_last_loc_to_buffer(
                seq_lens,
                out_cache_loc,
                req_pool_indices,
                seq_lens_cpu,
                req_pool_indices_cpu=None,
            )


if __name__ == "__main__":
    unittest.main()
