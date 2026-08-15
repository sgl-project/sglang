import types
import unittest
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import Req  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import (  # noqa: E402
    CaptureHiddenMode,
)
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (  # noqa: E402
    DSparkWorkerV2,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDSparkTargetLogprobs(unittest.TestCase):
    def test_target_scoring_request_is_prefill_only(self):
        req = Req.__new__(Req)
        req.sampling_params = types.SimpleNamespace(max_new_tokens=0)
        req.return_logprob = True

        spec = types.SimpleNamespace(speculative_algorithm="DSPARK")
        with patch("sglang.srt.managers.schedule_batch.get_spec", return_value=spec):
            self.assertTrue(req.is_prefill_only)

            req.return_logprob = False
            self.assertFalse(req.is_prefill_only)

            req.return_logprob = True
            req.sampling_params.max_new_tokens = 1
            self.assertFalse(req.is_prefill_only)

    def test_prefill_only_logprobs_bypass_dspark_draft(self):
        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        target_result = types.SimpleNamespace(new_seq_lens=None)
        target_worker = MagicMock()
        target_worker.forward_batch_generation.return_value = target_result
        worker._target_worker = target_worker

        seq_lens = object()
        batch = types.SimpleNamespace(
            return_logprob=True,
            is_prefill_only=True,
            seq_lens=seq_lens,
        )
        on_publish = MagicMock()

        result = worker.forward_batch_generation(batch, on_publish=on_publish)

        self.assertIs(result, target_result)
        target_worker.forward_batch_generation.assert_called_once_with(
            batch, capture_hidden_mode=CaptureHiddenMode.NULL
        )
        self.assertIs(result.new_seq_lens, seq_lens)
        on_publish.assert_called_once_with(seq_lens)

    def test_generated_logprobs_remain_rejected(self):
        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        batch = types.SimpleNamespace(
            return_logprob=True,
            is_prefill_only=False,
        )

        with self.assertRaisesRegex(ValueError, "does not support return_logprob"):
            worker.forward_batch_generation(batch)


if __name__ == "__main__":
    unittest.main()
