import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.managers import data_parallel_controller
from sglang.srt.managers.data_parallel_controller import DataParallelController
from sglang.srt.managers.io_struct import (
    BatchTokenizedGenerateReqInput,
    wrap_as_pickle,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.observability.req_time_stats import (
    APIServerReqTimeStats,
    DPControllerReqTimeStats,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _request(dp_rank: int | None) -> SimpleNamespace:
    return SimpleNamespace(
        routed_dp_rank=dp_rank,
        time_stats=wrap_as_pickle(APIServerReqTimeStats()),
    )


class TestAtomicDPBatchRouting(CustomTestCase):
    def test_tokenizer_batches_only_a_single_explicit_dp_route(self) -> None:
        manager = object.__new__(TokenizerManager)
        manager.server_args = SimpleNamespace(
            enable_tokenizer_batch_encode=False,
            enable_dp_attention=True,
        )
        manager._batch_has_text = lambda _batch_size, _requests: False

        self.assertTrue(
            manager._should_use_batch_tokenization(
                4,
                [_request(1) for _ in range(4)],
            )
        )
        self.assertFalse(
            manager._should_use_batch_tokenization(
                4,
                [_request(0), _request(1), _request(0), _request(1)],
            )
        )
        self.assertFalse(
            manager._should_use_batch_tokenization(
                4,
                [_request(None) for _ in range(4)],
            )
        )

        manager.server_args.enable_dp_attention = False
        self.assertTrue(
            manager._should_use_batch_tokenization(
                4,
                [_request(None) for _ in range(4)],
            )
        )

    def test_controller_sends_single_rank_batch_as_one_message(self) -> None:
        controller = object.__new__(DataParallelController)
        target_worker = object()
        controller.workers = [object(), target_worker]
        controller._active_workers = [0, 1]
        controller.refresh_load_budget_on_dispatch = False
        fallback_requests = []
        controller.dispatching = fallback_requests.append
        batch = BatchTokenizedGenerateReqInput(batch=[_request(1) for _ in range(4)])

        with patch.object(data_parallel_controller, "sock_send") as sock_send:
            controller.dispatch_batch_generate(batch)

        sock_send.assert_called_once_with(target_worker, batch)
        self.assertEqual(fallback_requests, [])
        self.assertTrue(
            all(isinstance(req.time_stats, DPControllerReqTimeStats) for req in batch)
        )
        self.assertTrue(all(req.time_stats.dpc_dispatch_time > 0 for req in batch))
        self.assertTrue(
            all(req.time_stats.dpc_dispatch_finish_time > 0 for req in batch)
        )

    def test_controller_falls_back_for_mixed_rank_batch(self) -> None:
        controller = object.__new__(DataParallelController)
        controller.workers = [object(), object()]
        controller._active_workers = [0, 1]
        controller.refresh_load_budget_on_dispatch = False
        fallback_requests = []
        controller.dispatching = fallback_requests.append
        batch = BatchTokenizedGenerateReqInput(batch=[_request(0), _request(1)])

        with patch.object(data_parallel_controller, "sock_send") as sock_send:
            controller.dispatch_batch_generate(batch)

        sock_send.assert_not_called()
        self.assertEqual(fallback_requests, list(batch))


if __name__ == "__main__":
    unittest.main()
