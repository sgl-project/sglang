import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.new_token_ratio_tracker import (
    NewTokenRatioTracker,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

FLOOR = 0.098


def _survivors(n: int, output_len: int, max_new_tokens: int):
    return [
        SimpleNamespace(
            output_ids=[0] * output_len,
            sampling_params=SimpleNamespace(max_new_tokens=max_new_tokens),
        )
        for _ in range(n)
    ]


def _ratio_after_kv_full_retract(survivors, current: float) -> float:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.enable_skip_finishing_decode = False
    scheduler.result_queue = []
    scheduler.decode_offload_manager = None
    scheduler.token_to_kv_pool_allocator = MagicMock()
    scheduler.token_to_kv_pool_allocator.available_size.return_value = 0
    scheduler.tree_cache = MagicMock()
    scheduler.tree_cache.req_to_token_pool.mamba_allocator = None
    scheduler.metrics_reporter = MagicMock(enable_metrics=False)
    scheduler._add_request_to_queue = MagicMock()
    scheduler.new_token_ratio_tracker = NewTokenRatioTracker(
        init=0.7, min=FLOOR, decay=0.001, current=current
    )

    batch = MagicMock()
    batch.is_empty.return_value = False
    batch.batch_size.return_value = len(survivors)
    batch.check_decode_mem.return_value = False
    estimate = NewTokenRatioTracker.estimate_new_token_ratio_after_retract(survivors)
    batch.retract_decode.return_value = ([MagicMock()], estimate, [])

    scheduler.update_running_batch(batch)
    return scheduler.new_token_ratio_tracker.current


class TestKvFullRetractNewTokenRatio(CustomTestCase):
    """A KV-full retraction must not set the new-token ratio below its floor,
    even when the survivors' max_new_tokens is close to the context length."""

    def test_unset_max_tokens_stops_at_the_floor(self):
        # max_tokens unset: max_new_tokens is about context minus prompt.
        survivors = _survivors(38, output_len=1500, max_new_tokens=130070)
        self.assertEqual(_ratio_after_kv_full_retract(survivors, current=0.5), FLOOR)

    def test_a_higher_estimate_still_raises_the_ratio(self):
        survivors = _survivors(38, output_len=1500, max_new_tokens=2048)
        ratio = _ratio_after_kv_full_retract(survivors, current=FLOOR)
        self.assertAlmostEqual(ratio, (1500 + 20) / 2048, places=3)


if __name__ == "__main__":
    unittest.main()
