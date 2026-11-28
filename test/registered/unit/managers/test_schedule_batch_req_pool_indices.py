import types
import unittest
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.model_executor.forward_mode import CaptureHiddenMode  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_req(req_pool_idx, origin_input_ids, output_ids, logprob=None):
    return types.SimpleNamespace(
        kv=types.SimpleNamespace(req_pool_idx=req_pool_idx),
        origin_input_ids=origin_input_ids,
        output_ids=output_ids,
        return_logprob=logprob is not None,
        logprob=logprob,
        grammar=None,
        return_hidden_states=False,
        return_hidden_states_mode=CaptureHiddenMode.NULL,
        is_prefill_only=False,
    )


def _make_scheduler():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.device = "cpu"
    scheduler.req_to_token_pool = types.SimpleNamespace(device="cpu")
    scheduler.token_to_kv_pool_allocator = None
    scheduler.tree_cache = None
    scheduler.model_config = types.SimpleNamespace(
        is_encoder_decoder=False, vocab_size=32, token_id_vocab_size=32
    )
    scheduler.enable_overlap = False
    scheduler.spec_algorithm = types.SimpleNamespace(is_none=lambda: True)
    scheduler.future_map = MagicMock()
    return scheduler


class TestHisparseDecodeBatchReqPoolCpu(unittest.TestCase):
    def test_build_hisparse_decode_batch_populates_req_pool_indices_cpu(self):
        # _build_hisparse_decode_batch builds a ScheduleBatch off the normal
        # extend path, so it must populate the req_pool_indices_cpu host mirror
        # in lockstep with the device tensor. A missing mirror crashes hisparse
        # decode bookkeeping (map_last_loc_to_buffer -> _grow_device_buffers
        # indexes req_pool_indices_cpu).
        scheduler = _make_scheduler()

        reqs = [
            _make_req(req_pool_idx=4, origin_input_ids=[1, 2, 3], output_ids=[7]),
            _make_req(req_pool_idx=9, origin_input_ids=[1, 2], output_ids=[8]),
        ]

        batch = scheduler._build_hisparse_decode_batch(reqs)

        # Assert the invariant (cpu mirror == device tensor), not a hardcoded
        # copy of the input -- the latter would just restate the builder line.
        self.assertIsNotNone(batch.req_pool_indices_cpu)
        self.assertTrue(
            torch.equal(batch.req_pool_indices_cpu, batch.req_pool_indices.cpu())
        )

    def test_build_hisparse_decode_batch_keeps_requested_token_ids_logprob(self):
        reqs = [
            _make_req(
                req_pool_idx=4,
                origin_input_ids=[1, 2, 3],
                output_ids=[7],
                logprob=types.SimpleNamespace(
                    top_logprobs_num=2, token_ids_logprob=[5, 11]
                ),
            ),
            _make_req(
                req_pool_idx=9,
                origin_input_ids=[1, 2],
                output_ids=[8],
                logprob=types.SimpleNamespace(
                    top_logprobs_num=0, token_ids_logprob=None
                ),
            ),
        ]

        batch = _make_scheduler()._build_hisparse_decode_batch(reqs)

        self.assertEqual(batch.top_logprobs_nums, [2, 0])
        self.assertEqual(batch.token_ids_logprobs, [[5, 11], None])


if __name__ == "__main__":
    unittest.main()
