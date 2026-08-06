"""CPU unit tests for DFlash verify-time logit adjustments."""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.speculative.dflash_utils import (
    apply_dflash_verify_logits_adjustments,
)
from sglang.srt.speculative.dspark_components.dspark_verify import (
    verify_logits_adjustments_are_noop,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _SamplingInfo(SimpleNamespace):
    def __len__(self) -> int:
        return self.batch_size


def _make_sampling_info(*, batch_size: int, additive_penalties=None):
    return _SamplingInfo(
        batch_size=batch_size,
        has_custom_logit_processor=False,
        acc_additive_penalties=additive_penalties,
        penalizer_orchestrator=None,
        grammar_mask=None,
        logit_bias=None,
    )


class TestDFlashVerifyLogitsAdjustments(CustomTestCase):
    def test_broadcasts_additive_penalties_over_verify_block(self):
        additive_penalties = torch.tensor(
            [[0.0, -1.5, 2.0], [3.0, 0.5, -4.0]], dtype=torch.float32
        )
        sampling_info = _make_sampling_info(
            batch_size=2, additive_penalties=additive_penalties
        )
        logits = torch.zeros((6, 3), dtype=torch.float64)

        apply_dflash_verify_logits_adjustments(
            next_token_logits=logits,
            sampling_info=sampling_info,
            draft_token_num=3,
        )

        expected = (
            additive_penalties.to(dtype=logits.dtype)
            .unsqueeze(1)
            .expand(-1, 3, -1)
            .reshape_as(logits)
        )
        torch.testing.assert_close(logits, expected)

    def test_additive_penalties_disable_noop_fast_path(self):
        sampling_info = _make_sampling_info(
            batch_size=1,
            additive_penalties=torch.zeros((1, 4), dtype=torch.float32),
        )

        self.assertFalse(verify_logits_adjustments_are_noop(sampling_info))

    def test_missing_additive_penalties_keep_noop_fast_path(self):
        sampling_info = _make_sampling_info(batch_size=1)

        self.assertTrue(verify_logits_adjustments_are_noop(sampling_info))


if __name__ == "__main__":
    unittest.main()
