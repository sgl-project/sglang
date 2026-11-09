import unittest
from types import SimpleNamespace

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

VOCAB_SIZE = 32000


class TestTopLogprobsNumValidation(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)
        self.manager = TokenizerManager.__new__(TokenizerManager)
        self.manager.context_len = 128
        self.manager.num_reserved_tokens = 0
        self.manager.allow_auto_truncate = False
        self.manager.validate_total_tokens = False
        self.manager.is_generation = True
        self.manager.model_config = SimpleNamespace(
            vocab_size=VOCAB_SIZE, joint_head_config=None, pointer_head=None
        )

    def _validate(self, top_logprobs_num):
        req = GenerateReqInput(
            input_ids=[1, 2, 3],
            return_logprob=True,
            top_logprobs_num=top_logprobs_num,
        )
        req.normalize_batch_and_arguments()
        self.manager._validate_one_request(req, req.input_ids)

    def test_rejects_top_logprobs_num_outside_vocab(self):
        """top_logprobs_num outside [0, vocab_size] feeds an out-of-range
        logprobs.topk(k) in the scheduler, which kills the server."""
        for k in (-1, VOCAB_SIZE + 1, 1_000_000):
            with self.subTest(k=k):
                with self.assertRaisesRegex(
                    ValueError, rf"top_logprobs_num must be in \[0, {VOCAB_SIZE}\]"
                ):
                    self._validate(k)
        for k in (0, 20, VOCAB_SIZE):
            with self.subTest(k=k):
                self._validate(k)


if __name__ == "__main__":
    unittest.main()
