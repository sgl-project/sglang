"""Capacity resolution and flag validation for concurrent chunked prefill.

``--long-prefill-token-threshold F`` caps how many prompt tokens one request
takes per scheduled pass, so up to ``chunked_prefill_size // F`` requests can
be mid-prefill at once. ``Scheduler.init_chunked_prefill_concurrency``
resolves that capacity and pins it to 1 where the single-slot invariant is
still load-bearing (disagg prefill, PP > 1, dLLM).
"""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class TestLongPrefillTokenThresholdValidation(CustomTestCase):
    """The validation handler runs after model-specific defaults resolve
    ``chunked_prefill_size`` (a threshold without chunked prefill is only
    known-invalid then), which is past the dummy-model early return in
    ``__post_init__`` -- so these call the handler directly."""

    def _args(self, *, threshold, chunked_prefill_size) -> ServerArgs:
        args = ServerArgs(model_path="dummy")
        args.long_prefill_token_threshold = threshold
        args.chunked_prefill_size = chunked_prefill_size
        return args

    def test_negative_threshold_rejected(self):
        with self.assertRaises(ValueError):
            self._args(
                threshold=-1, chunked_prefill_size=8192
            )._handle_long_prefill_token_threshold()

    def test_threshold_requires_chunked_prefill(self):
        for chunk in (None, -1, 0):
            with self.subTest(chunked_prefill_size=chunk):
                with self.assertRaises(ValueError):
                    self._args(
                        threshold=2048, chunked_prefill_size=chunk
                    )._handle_long_prefill_token_threshold()

    def test_threshold_with_chunked_prefill_accepted(self):
        args = self._args(threshold=2048, chunked_prefill_size=8192)
        args._handle_long_prefill_token_threshold()
        self.assertEqual(args.long_prefill_token_threshold, 2048)

    def test_default_disabled(self):
        args = ServerArgs(model_path="dummy")
        self.assertEqual(args.long_prefill_token_threshold, 0)
        args._handle_long_prefill_token_threshold()


class TestInitChunkedPrefillConcurrency(CustomTestCase):
    def _resolve(
        self,
        *,
        threshold: int,
        chunked_prefill_size=8192,
        scheduler_chunk_size="__same__",
        disaggregation_mode=DisaggregationMode.NULL,
        pp_size: int = 1,
        dllm_config=None,
    ) -> int:
        set_global_server_args_for_scheduler(
            ServerArgs(
                model_path="dummy",
                chunked_prefill_size=chunked_prefill_size,
                long_prefill_token_threshold=threshold,
            )
        )
        s = Scheduler.__new__(Scheduler)
        s.chunked_prefill_size = (
            chunked_prefill_size
            if scheduler_chunk_size == "__same__"
            else scheduler_chunk_size
        )
        s.disaggregation_mode = disaggregation_mode
        s.ps = SimpleNamespace(pp_size=pp_size)
        s.dllm_config = dllm_config
        s.max_concurrent_chunked_reqs = 1
        Scheduler.init_chunked_prefill_concurrency(s)
        return s.max_concurrent_chunked_reqs

    def test_disabled_by_default(self):
        self.assertEqual(self._resolve(threshold=0), 1)

    def test_capacity_is_budget_over_threshold(self):
        self.assertEqual(self._resolve(threshold=2048), 4)

    def test_capacity_floors_at_one(self):
        # A threshold above the whole pool still allows one mid-prefill
        # request (identical to stock).
        self.assertEqual(self._resolve(threshold=16384), 1)

    def test_capacity_uses_floor_division(self):
        self.assertEqual(self._resolve(threshold=3000), 2)

    def test_disagg_prefill_stays_single_slot(self):
        self.assertEqual(
            self._resolve(
                threshold=2048, disaggregation_mode=DisaggregationMode.PREFILL
            ),
            1,
        )

    def test_disagg_decode_allows_concurrency(self):
        self.assertEqual(
            self._resolve(
                threshold=2048, disaggregation_mode=DisaggregationMode.DECODE
            ),
            4,
        )

    def test_pipeline_parallel_stays_single_slot(self):
        self.assertEqual(self._resolve(threshold=2048, pp_size=2), 1)

    def test_dllm_stays_single_slot(self):
        self.assertEqual(self._resolve(threshold=2048, dllm_config=object()), 1)

    def test_chunked_prefill_disabled_stays_single_slot(self):
        # Unreachable from real config (validation rejects threshold without
        # chunked prefill); the scheduler-side gate is defense in depth.
        self.assertEqual(self._resolve(threshold=2048, scheduler_chunk_size=None), 1)


if __name__ == "__main__":
    unittest.main()
