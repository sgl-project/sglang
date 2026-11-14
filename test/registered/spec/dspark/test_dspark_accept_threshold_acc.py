"""DSpark rejection sampling must honor speculative_accept_threshold_acc.

The value is read from the live spec config on every verify step, including
updates from /set_internal_state, and forwarded into chain rejection sampling.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.speculative.dspark import dspark_accept
from sglang.srt.runtime_context import get_context, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestDsparkAcceptThresholdAcc(CustomTestCase):
    def setUp(self):
        reset_context()
        get_context().set_server_args(ServerArgs(model_path="dummy"))

    def tearDown(self):
        reset_context()

    def _forwarded_threshold(self) -> float:
        bs, slots, vocab = 2, 3, 8
        captured = {}

        def _fake_kernel(**kwargs):
            captured.update(kwargs)

        sampling_info = SimpleNamespace(
            need_top_k_sampling=False,
            need_top_p_sampling=False,
            temperatures=torch.ones(bs, dtype=torch.float32),
        )
        original = dspark_accept.chain_speculative_sampling_triton
        dspark_accept.chain_speculative_sampling_triton = _fake_kernel
        try:
            dspark_accept._accept_sampling_core(
                candidates=torch.zeros(bs, slots, dtype=torch.int64),
                target_logits=torch.zeros(bs * slots, vocab),
                draft_probs=torch.zeros(bs, slots, vocab),
                sampling_info=sampling_info,
                draft_input=None,
                gamma=slots - 1,
                verify_num_draft_tokens=slots,
                cutoff_verify_lens=None,
                block_verification=False,
            )
        finally:
            dspark_accept.chain_speculative_sampling_triton = original
        return captured["threshold_acc"]

    def test_default_threshold_is_one(self):
        self.assertEqual(self._forwarded_threshold(), 1.0)

    def test_runtime_override_is_forwarded(self):
        get_context().override(
            "update_server_args", speculative_accept_threshold_acc=0.95
        )
        self.assertAlmostEqual(self._forwarded_threshold(), 0.95)

    def test_nonpositive_threshold_is_clamped(self):
        get_context().override(
            "update_server_args", speculative_accept_threshold_acc=0.0
        )
        self.assertEqual(self._forwarded_threshold(), 1e-9)


if __name__ == "__main__":
    unittest.main()
