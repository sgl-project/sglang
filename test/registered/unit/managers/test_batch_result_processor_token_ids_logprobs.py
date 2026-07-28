"""Unit tests for mixed host-list/tensor token_ids_logprobs normalization.

``LogitsProcessorOutput.next_token_token_ids_logprobs_val`` is declared as
``List[Union[List[float], torch.Tensor]]``: prefill-only requests can leave
already-copied host lists in the field (delayed-copy optimization). Both
normalization sites must tolerate the mix instead of crashing the scheduler
event loop.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _SpecNone:
    def is_none(self) -> bool:
        return True


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
        model_worker=SimpleNamespace(on_verify_complete_cpu=lambda *a, **k: None),
        logprob_result_processor=None,
        output_streamer=SimpleNamespace(),
        abort_request=lambda *a, **k: None,
    )


def _logits_output() -> SimpleNamespace:
    return SimpleNamespace(
        next_token_logprobs=torch.tensor([-0.5, -1.0]),
        input_token_logprobs=None,
        next_token_top_logprobs_val=[],
        next_token_top_logprobs_idx=[],
        next_token_token_ids_logprobs_val=[[-0.25, -2.0], torch.tensor([-0.75])],
        next_token_token_ids_logprobs_idx=[[7, 11], [3]],
    )


class TestTokenIdsLogprobsHostLists(CustomTestCase):
    def test_move_logprobs_to_cpu_tolerates_host_lists(self):
        processor = _make_processor()
        logits_output = _logits_output()
        batch = SimpleNamespace(return_logprob=True)

        processor.move_logprobs_to_cpu(batch=batch, logits_output=logits_output)

        self.assertEqual(
            logits_output.next_token_token_ids_logprobs_val,
            [[-0.25, -2.0], [-0.75]],
        )

    def test_normalize_decode_outputs_tolerates_host_lists(self):
        processor = _make_processor()
        logits_output = _logits_output()
        batch = SimpleNamespace(return_logprob=True, spec_algorithm=_SpecNone())

        next_token_ids, next_token_logprobs = processor._normalize_decode_outputs(
            batch=batch,
            result=SimpleNamespace(),
            logits_output=logits_output,
            next_token_ids=torch.tensor([5, 9]),
        )

        self.assertEqual(next_token_ids, [[5], [9]])
        self.assertEqual(next_token_logprobs, [-0.5, -1.0])
        self.assertEqual(
            logits_output.next_token_token_ids_logprobs_val,
            [[-0.25, -2.0], [-0.75]],
        )


if __name__ == "__main__":
    unittest.main()
