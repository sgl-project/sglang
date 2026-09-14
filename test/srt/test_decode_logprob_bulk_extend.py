import unittest
from types import SimpleNamespace

from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _req(top_logprobs_num: int = 0, token_ids_logprob=None):
    return SimpleNamespace(
        logprob=SimpleNamespace(
            output_token_logprobs_val=[],
            output_token_logprobs_idx=[],
            output_top_logprobs_val=[],
            output_top_logprobs_idx=[],
            output_token_ids_logprobs_val=[],
            output_token_ids_logprobs_idx=[],
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=token_ids_logprob,
        )
    )


def _apply(
    req, *, spec: bool, i: int, next_token_id, next_token_logprobs, logits_output
):
    batch = SimpleNamespace(
        spec_algorithm=SpeculativeAlgorithm.EAGLE if spec else SpeculativeAlgorithm.NONE
    )
    SchedulerBatchResultProcessor._apply_decode_logprobs(
        None,
        req=req,
        i=i,
        batch=batch,
        next_token_id=next_token_id,
        next_token_logprobs=next_token_logprobs,
        logits_output=logits_output,
    )


class TestDecodeLogprobBulkExtend(unittest.TestCase):
    """The sampled-token-only path extends the per-request lists once per step;
    it must record exactly what the per-token loop recorded, including when the
    per-step logprob row is padded past the accepted tokens."""

    def test_spec_padded_row_records_only_accepted_tokens(self):
        req = _req()
        # Row padded to the verify stride (4) while only 3 tokens were accepted.
        _apply(
            req,
            spec=True,
            i=1,
            next_token_id=[7, 8, 9],
            next_token_logprobs=[[0.0, 0.0, 0.0, 0.0], [-0.1, -0.2, -0.3, -9.0]],
            logits_output=SimpleNamespace(),
        )
        self.assertEqual(req.logprob.output_token_logprobs_val, [-0.1, -0.2, -0.3])
        self.assertEqual(req.logprob.output_token_logprobs_idx, [7, 8, 9])

    def test_non_spec_scalar_row(self):
        req = _req()
        _apply(
            req,
            spec=False,
            i=2,
            next_token_id=[11],
            next_token_logprobs=[-1.0, -2.0, -3.0],
            logits_output=SimpleNamespace(),
        )
        self.assertEqual(req.logprob.output_token_logprobs_val, [-3.0])
        self.assertEqual(req.logprob.output_token_logprobs_idx, [11])

    def test_top_logprobs_keep_the_per_token_path(self):
        req = _req(top_logprobs_num=2)
        logits_output = SimpleNamespace(
            next_token_top_logprobs_val=[[-0.5, -0.6], [-0.7, -0.8]],
            next_token_top_logprobs_idx=[[1, 2], [3, 4]],
        )
        _apply(
            req,
            spec=True,
            i=0,
            next_token_id=[5, 6],
            next_token_logprobs=[[-0.1, -0.2]],
            logits_output=logits_output,
        )
        self.assertEqual(req.logprob.output_token_logprobs_val, [-0.1, -0.2])
        self.assertEqual(req.logprob.output_token_logprobs_idx, [5, 6])
        self.assertEqual(
            req.logprob.output_top_logprobs_val, [[-0.5, -0.6], [-0.7, -0.8]]
        )
        self.assertEqual(req.logprob.output_top_logprobs_idx, [[1, 2], [3, 4]])


if __name__ == "__main__":
    unittest.main()
