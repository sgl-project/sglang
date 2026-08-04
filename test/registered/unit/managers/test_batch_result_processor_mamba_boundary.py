import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _make_processor() -> SchedulerBatchResultProcessor:
    return SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=None,
        enable_overlap=False,
        enable_overlap_mlx=False,
        server_args=SimpleNamespace(),
        model_config=SimpleNamespace(think_end_ids=None),
        token_to_kv_pool_allocator=Mock(),
        tree_cache=None,
        hisparse_coordinator=None,
        req_to_token_pool=None,
        decode_offload_manager=None,
        metrics_collector=None,
        metrics_reporter=Mock(),
        draft_worker=None,
        model_worker=Mock(),
        logprob_result_processor=None,
        output_streamer=Mock(),
        abort_request=lambda *args, **kwargs: None,
    )


class TestMambaBoundaryMaskReuse(unittest.TestCase):
    def test_known_non_boundary_skips_mamba_update(self):
        processor = _make_processor()
        req = SimpleNamespace(finished=lambda: False)

        with (
            patch.object(
                SchedulerBatchResultProcessor,
                "_mamba_prefix_cache_update",
            ) as mamba_update,
            patch.object(
                SchedulerBatchResultProcessor,
                "_maybe_collect_customized_info",
            ),
            patch(
                "sglang.srt.managers.scheduler_components."
                "batch_result_processor.get_disagg",
                return_value=SimpleNamespace(
                    disaggregation_decode_enable_offload_kvcache=False
                ),
            ),
        ):
            processor._handle_finish_state_updated_req(
                req,
                SimpleNamespace(),
                SimpleNamespace(),
                0,
                SimpleNamespace(),
                known_mamba_boundary=False,
            )

        mamba_update.assert_not_called()

    def test_known_boundary_uses_result_length_during_overlap(self):
        processor = _make_processor()
        req = SimpleNamespace(
            mamba_ping_pong_track_buffer=torch.tensor([1, 2]),
            origin_input_ids=[1, 2, 3, 4, 5],
            output_ids=[6, 7, 8, 9],
            kv_committed_len=9,
            rid="req-0",
            mamba_last_track_seqlen=0,
            mamba_next_track_idx=0,
        )
        batch = SimpleNamespace(
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
            req_to_token_pool=SimpleNamespace(
                get_mamba_ping_pong_other_idx=lambda index: 1 - index
            ),
        )

        with (
            patch.object(
                SchedulerBatchResultProcessor,
                "_mamba_check_track_boundary",
                side_effect=AssertionError("boundary check should be reused"),
            ) as boundary_check,
            patch(
                "sglang.srt.managers.scheduler_components."
                "batch_result_processor.get_server_args",
                return_value=SimpleNamespace(
                    enable_mamba_extra_buffer_lazy=lambda: False
                ),
            ),
            patch(
                "sglang.srt.managers.scheduler_components."
                "batch_result_processor.get_exec",
                return_value=SimpleNamespace(
                    mamba=SimpleNamespace(mamba_track_interval=4)
                ),
            ),
        ):
            processor._mamba_prefix_cache_update(
                req,
                batch,
                SimpleNamespace(),
                0,
                known_boundary=True,
            )

        boundary_check.assert_not_called()
        self.assertEqual(req.mamba_last_track_seqlen, 8)
        self.assertEqual(req.mamba_next_track_idx, 1)


if __name__ == "__main__":
    unittest.main()
