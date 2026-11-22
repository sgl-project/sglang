import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.sampling.verify_accept import VerifyAccept
from sglang.srt.speculative import target_verify
from sglang.srt.speculative.target_verify import TargetVerify
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _batch():
    return SimpleNamespace(
        sampling_info=object(),
        return_logprob=True,
        forward_mode=SimpleNamespace(is_idle=lambda: False),
    )


class TestTargetVerifyCommit(CustomTestCase):
    def _commit(self, batch, accept):
        verify = TargetVerify(
            grammar_mask=None, watermark=object(), watermark_full_mask=True
        )
        logits_output = object()
        with (
            patch.object(target_verify, "commit_verify_accept") as commit,
            patch.object(target_verify, "compute_spec_logprobs") as logprobs,
        ):
            verify.commit(batch, logits_output=logits_output, accept=accept)
        commit.assert_called_once_with(
            batch.sampling_info, accept, watermark=verify.watermark
        )
        return logits_output, logprobs

    def test_tree_logprobs_gather_through_the_accept_index(self):
        accept = VerifyAccept(
            tokens=torch.arange(6),
            accept_lens=torch.tensor([2, 1]),
            accept_index=torch.tensor([[0, 1], [3, -1]]),
        )
        batch = _batch()
        logits_output, logprobs = self._commit(batch, accept)
        logprobs.assert_called_once_with(
            batch, logits_output, accept.tokens, accept_index=accept.accept_index
        )

    def test_chain_logprobs_read_rows_at_the_block_width(self):
        accept = VerifyAccept(
            tokens=torch.arange(8).view(2, 4), accept_lens=torch.tensor([2, 4])
        )
        batch = _batch()
        logits_output, logprobs = self._commit(batch, accept)
        args, kwargs = logprobs.call_args
        self.assertEqual(args[:2], (batch, logits_output))
        self.assertTrue(torch.equal(args[2], torch.arange(8)))
        self.assertEqual(kwargs, dict(chain_stride=4))


if __name__ == "__main__":
    unittest.main()
