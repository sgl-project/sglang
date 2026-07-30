import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeSpecAlgorithm:
    def __init__(self, is_none: bool):
        self._is_none = is_none

    def is_none(self) -> bool:
        return self._is_none


def _make_processor() -> SchedulerBatchResultProcessor:
    return SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=None,
        enable_overlap=False,
        enable_overlap_mlx=False,
        server_args=SimpleNamespace(enable_metrics=False),
        model_config=SimpleNamespace(think_end_id=None),
        token_to_kv_pool_allocator=None,
        tree_cache=None,
        hisparse_coordinator=None,
        req_to_token_pool=None,
        decode_offload_manager=None,
        metrics_collector=None,
        metrics_reporter=SimpleNamespace(),
        draft_worker=None,
        model_worker=SimpleNamespace(),
        logprob_result_processor=None,
        output_streamer=SimpleNamespace(),
        abort_request=lambda *args, **kwargs: None,
    )


class TestMambaPingPongUpdate(CustomTestCase):
    def _run_update(self, *, spec_decode: bool):
        req = SimpleNamespace(
            mamba_ping_pong_track_buffer=[10, 11],
            mamba_next_track_idx=0,
            mamba_last_track_seqlen=None,
        )
        pool = SimpleNamespace(
            get_mamba_ping_pong_other_idx=Mock(return_value=1),
        )
        batch = SimpleNamespace(
            spec_algorithm=_FakeSpecAlgorithm(is_none=not spec_decode),
            req_to_token_pool=pool,
        )
        processor = _make_processor()

        with (
            patch(
                "sglang.srt.managers.scheduler_components."
                "batch_result_processor.get_server_args",
                return_value=SimpleNamespace(
                    enable_mamba_extra_buffer_lazy=lambda: False
                ),
            ),
            patch.object(
                SchedulerBatchResultProcessor,
                "_mamba_check_track_boundary",
                return_value=(True, 256),
            ),
        ):
            processor._mamba_prefix_cache_update(req, batch, result=None, i=0)

        return req, pool

    def test_spec_decode_does_not_flip_committed_slot_again(self):
        req, pool = self._run_update(spec_decode=True)

        self.assertEqual(req.mamba_next_track_idx, 0)
        self.assertEqual(req.mamba_last_track_seqlen, 256)
        pool.get_mamba_ping_pong_other_idx.assert_not_called()

    def test_non_spec_decode_still_flips_at_track_boundary(self):
        req, pool = self._run_update(spec_decode=False)

        self.assertEqual(req.mamba_next_track_idx, 1)
        self.assertEqual(req.mamba_last_track_seqlen, 256)
        pool.get_mamba_ping_pong_other_idx.assert_called_once_with(0)


if __name__ == "__main__":
    unittest.main()
