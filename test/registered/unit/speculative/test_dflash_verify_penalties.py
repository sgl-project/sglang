"""DFlash-family verify logits adjustments with per-row speculative penalties."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.sampling.penalties import (
    BatchedFrequencyPenalizer,
    BatchedMinNewTokensPenalizer,
    BatchedPenalizerOrchestrator,
    BatchedPresencePenalizer,
)
from sglang.srt.speculative import dflash_utils, spec_info
from sglang.srt.speculative.dflash_utils import (
    apply_dflash_verify_logits_adjustments,
    validate_dflash_request,
)
from sglang.srt.speculative.dspark_components.dspark_verify import (
    verify_logits_adjustments_are_noop,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils.log_utils import print_warning_once
from sglang.test.test_utils import CustomTestCase

VOCAB_SIZE = 32


def _req(freq, presence, min_new, stop_ids):
    params = SimpleNamespace(
        frequency_penalty=freq,
        presence_penalty=presence,
        min_new_tokens=min_new,
        stop_token_ids=set(stop_ids),
    )
    tokenizer = SimpleNamespace(additional_stop_token_ids=None, eos_token_id=2)
    return SimpleNamespace(
        sampling_params=params, eos_token_ids=None, tokenizer=tokenizer
    )


class _Batch:
    # The orchestrator holds its batch by weak reference.
    def __init__(self, reqs):
        self.reqs, self.device = reqs, "cpu"


class _Info(SimpleNamespace):
    def __len__(self):
        return self.acc_additive_penalties.shape[0]


class TestDFlashVerifyPenalties(CustomTestCase):
    def setUp(self):
        # The per-row path; a test of other devices patches this again.
        supported = patch.object(
            dflash_utils, "per_row_penalties_supported", return_value=True
        )
        supported.start()
        self.addCleanup(supported.stop)
        reqs = [_req(0.5, 0.25, 3, [7]), _req(-1.0, 0.0, 0, [])]
        self.batch = _Batch(reqs)
        self.orchestrator = BatchedPenalizerOrchestrator(
            VOCAB_SIZE,
            self.batch,
            {
                BatchedFrequencyPenalizer,
                BatchedMinNewTokensPenalizer,
                BatchedPresencePenalizer,
            },
        )
        self.orchestrator.cumulate_output_tokens(
            torch.tensor([3, 5]), torch.tensor([0, 1])
        )

    def _info(self, **fields):
        additive = torch.zeros(2, VOCAB_SIZE)
        self.orchestrator.accumulate_additive_penalties(additive)
        info = dict(
            has_custom_logit_processor=False,
            penalizer_orchestrator=None,
            acc_additive_penalties=additive,
            acc_scaling_penalties=None,
            speculative_penalties=self.orchestrator.speculative_penalties(
                additive, None
            ),
            grammar_mask=None,
            logit_bias=None,
        )
        info.update(fields)
        return _Info(**info)

    def test_rows_get_the_per_row_penalties_and_the_bias_in_one_pass(self):
        """Verify passes the drafts and the logit bias to the per-row penalties,
        which add the bias; a grammar mask then adds only -inf, and a custom
        logit processor sees the result, as in decoding one token at a time."""
        width = 4
        drafts = torch.tensor([[3, 7, 3], [5, 5, 9]])
        num_drafts = torch.tensor([3, 1], dtype=torch.int32)
        bias = torch.randn(2, VOCAB_SIZE)
        banned = torch.zeros(2, VOCAB_SIZE, dtype=torch.bool)
        banned[0, 4] = banned[1, 6] = True
        grammar_mask = SimpleNamespace(
            apply=lambda t: t.masked_fill_(banned, -torch.inf)
        )
        for mask in (None, grammar_mask):
            for custom in (False, True):
                penalties = MagicMock()
                penalties.apply_.side_effect = lambda logits, *_: logits.mul_(2)
                info = self._info(
                    speculative_penalties=penalties,
                    logit_bias=bias,
                    grammar_mask=mask,
                    has_custom_logit_processor=custom,
                )
                torch.manual_seed(0)
                logits = torch.randn(2 * width, VOCAB_SIZE)
                expected = logits.view(2, width, VOCAB_SIZE) * 2
                if mask is not None:
                    expected.masked_fill_(banned.unsqueeze(1), -torch.inf)
                if custom:
                    expected += 1

                with patch.object(
                    dflash_utils,
                    "apply_custom_logit_processor",
                    side_effect=lambda t, *_, **__: t.add_(1),
                ) as processor:
                    apply_dflash_verify_logits_adjustments(
                        next_token_logits=logits,
                        sampling_info=info,
                        draft_token_num=width,
                        draft_tokens=drafts,
                        num_drafts=num_drafts,
                    )
                self.assertEqual(processor.call_count, int(custom))
                penalties.overrides.assert_called_once_with(drafts, num_drafts)
                penalties.apply_.assert_called_once_with(
                    logits, penalties.overrides.return_value, bias
                )
                self.assertTrue(
                    torch.equal(logits.view(2, width, VOCAB_SIZE), expected)
                )

    def test_penalty_history_needs_the_per_row_kernels(self):
        for supported in (True, False):
            with patch.object(
                spec_info, "per_row_penalties_supported", return_value=supported
            ):
                for algorithm in (
                    SpeculativeAlgorithm.DFLASH,
                    SpeculativeAlgorithm.DSPARK,
                ):
                    self.assertEqual(
                        algorithm.tracks_speculative_penalty_history(), supported
                    )

    def test_penalties_without_per_row_values_are_a_bug(self):
        """Forward copies carry per-row penalties whenever a penalty is active."""
        info = self._info(speculative_penalties=None)
        with self.assertRaises(AssertionError):
            apply_dflash_verify_logits_adjustments(
                next_token_logits=torch.zeros(8, VOCAB_SIZE),
                sampling_info=info,
                draft_token_num=4,
            )

    def test_devices_without_the_per_row_kernels_keep_block_start_adjustments(self):
        """The custom logit processor first; then, with a grammar mask, the
        block-start apply_logits_bias in every row, and otherwise only the logit
        bias. Requests with penalties are accepted with a warning."""
        bias = torch.randn(2, VOCAB_SIZE)
        block_start = torch.randn(2, VOCAB_SIZE)
        for mask in (None, SimpleNamespace(apply=lambda t: None)):
            info = self._info(
                speculative_penalties=None,
                logit_bias=bias,
                grammar_mask=mask,
                has_custom_logit_processor=True,
                apply_logits_bias=lambda t: t.add_(block_start),
            )
            logits = torch.randn(8, VOCAB_SIZE)
            added = bias if mask is None else block_start
            expected = logits.view(2, 4, VOCAB_SIZE) * 2 + added.unsqueeze(1)
            with (
                patch.object(
                    dflash_utils, "per_row_penalties_supported", return_value=False
                ),
                patch.object(
                    dflash_utils,
                    "apply_custom_logit_processor",
                    side_effect=lambda t, *_, **__: t.mul_(2),
                ),
            ):
                apply_dflash_verify_logits_adjustments(
                    next_token_logits=logits, sampling_info=info, draft_token_num=4
                )
            self.assertTrue(torch.equal(logits.view(2, 4, VOCAB_SIZE), expected))

        request = SimpleNamespace(
            return_hidden_states=False,
            sampling_params=SimpleNamespace(
                frequency_penalty=0.0,
                presence_penalty=0.0,
                repetition_penalty=1.0,
                min_new_tokens=3,
            ),
        )
        print_warning_once.cache_clear()
        with (
            patch.object(
                dflash_utils, "per_row_penalties_supported", return_value=False
            ),
            self.assertLogs("sglang.srt.utils.log_utils", "WARNING") as logs,
        ):
            self.assertIsNone(validate_dflash_request(request, True))
        self.assertIn("exactly only on CUDA and ROCm", logs.output[0])

    def test_per_row_penalties_disable_the_noop_shortcut(self):
        self.assertFalse(verify_logits_adjustments_are_noop(self._info()))
        self.assertTrue(
            verify_logits_adjustments_are_noop(self._info(speculative_penalties=None))
        )


if __name__ == "__main__":
    unittest.main()
