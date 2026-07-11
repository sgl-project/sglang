"""Regression: a return_logprob request on the MLX backend must be rejected
with a clean per-request error, not crash the scheduler.

The MLX runner returns a LogitsProcessorOutput with no logprob tensors, so the
shared result processor dereferences None
(`logits_output.next_token_logprobs.tolist()`) and takes down the whole
scheduler process. `TokenizerManager._validate_one_request` now fails fast when
`use_mlx()` and the request asks for logprobs.

`use_mlx` is patched, so this is a pure-CPU test and runs on any platform.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_tm():
    """A TokenizerManager stub carrying only what _validate_one_request reads."""
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.context_len = 100_000
    tm.num_reserved_tokens = 0
    tm.validate_total_tokens = False
    tm.server_args = SimpleNamespace(
        allow_auto_truncate=False,
        enable_return_hidden_states=False,
        enable_custom_logit_processor=False,
    )
    return tm


def _make_req(return_logprob):
    obj = GenerateReqInput(text="hi", return_logprob=return_logprob)
    obj.sampling_params = {}  # normalized shape _validate_one_request expects
    return obj


def _patch_use_mlx(value):
    return mock.patch(
        "sglang.srt.managers.tokenizer_manager.use_mlx", return_value=value
    )


class TestMlxLogprobGuard(CustomTestCase):
    def test_return_logprob_rejected_under_mlx(self):
        """THE REGRESSION: logprobs + MLX -> clean ValueError, not a crash."""
        tm, obj = _make_tm(), _make_req(return_logprob=True)
        with _patch_use_mlx(True):
            with self.assertRaises(ValueError) as ctx:
                tm._validate_one_request(obj, [1, 2, 3])
        self.assertIn("MLX", str(ctx.exception))

    def test_return_logprob_allowed_when_not_mlx(self):
        """Non-MLX backends still accept logprobs (guard must not over-reject)."""
        tm, obj = _make_tm(), _make_req(return_logprob=True)
        with _patch_use_mlx(False):
            tm._validate_one_request(obj, [1, 2, 3])  # must not raise

    def test_no_logprob_allowed_under_mlx(self):
        """A normal (no-logprob) request on MLX is unaffected."""
        tm, obj = _make_tm(), _make_req(return_logprob=False)
        with _patch_use_mlx(True):
            tm._validate_one_request(obj, [1, 2, 3])  # must not raise


if __name__ == "__main__":
    unittest.main()
