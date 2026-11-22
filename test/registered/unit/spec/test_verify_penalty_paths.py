import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.sampling.penalties import PenaltyStatusPool, StepPenalties
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative import dflash_utils, eagle_utils
from sglang.srt.speculative.dflash_utils import apply_dflash_verify_logits_adjustments
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

VOCAB_SIZE = 16


class _StopAfterPenalties(Exception):
    pass


def _step(num_reqs):
    reqs = [
        SimpleNamespace(
            penalty_index=None,
            output_ids=[],
            sampling_params=SamplingParams(frequency_penalty=1.0),
        )
        for _ in range(num_reqs)
    ]
    pool = PenaltyStatusPool(VOCAB_SIZE, "cpu")
    pool.alloc(reqs)
    pool.reset_history(reqs)
    return StepPenalties.build(reqs, pool)


def _sampling_info(bs):
    return SamplingBatchInfo(
        temperatures=torch.ones(bs, 1),
        top_ps=torch.ones(bs),
        top_ks=torch.ones(bs, dtype=torch.int32),
        min_ps=torch.zeros(bs),
        is_all_greedy=True,
        is_any_greedy=True,
        need_top_p_sampling=False,
        need_top_k_sampling=False,
        need_min_p_sampling=False,
        vocab_size=VOCAB_SIZE,
        device="cpu",
    )


def _row_counts(logits, tokens):
    """Each row's count of `tokens`, read back from a frequency penalty of 1."""
    return (-logits[:, tokens]).round().long().tolist()


class TestVerifyPenaltyPaths(CustomTestCase):
    def test_eagle_sample_feeds_the_draft_tree(self):
        # Root 0 has children 1 and 2; node 3 is a child of 1.
        verify_input = SimpleNamespace(
            draft_token_num=4,
            draft_token=torch.tensor([9, 5, 6, 5]),
            retrieve_next_token=torch.tensor([[1, 3, -1, -1]]),
            retrieve_next_sibling=torch.tensor([[-1, 2, -1, -1]]),
        )
        step = _step(1)
        batch = SimpleNamespace(
            device="cpu",
            seq_lens=torch.tensor([4]),
            sampling_info=SimpleNamespace(step_penalties=step),
            forward_mode=SimpleNamespace(is_idle=lambda: False),
        )
        logits_output = SimpleNamespace(
            next_token_logits=torch.zeros(4, VOCAB_SIZE),
            next_token_logits_vocab_shard=None,
        )
        apply_verify = step.apply_verify

        def apply_then_stop(*args, **kwargs):
            apply_verify(*args, **kwargs)
            raise _StopAfterPenalties

        with patch.object(step, "apply_verify", side_effect=apply_then_stop):
            with self.assertRaises(_StopAfterPenalties):
                eagle_utils.eagle_sample(verify_input, batch, logits_output)

        self.assertEqual(
            _row_counts(logits_output.next_token_logits, [5, 6, 9]),
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0]],
        )

    def test_dflash_block_verify_applies_the_custom_processor_last(self):
        """Decoding's order: penalties, grammar mask, logit bias, custom processor."""
        bs, width = 2, 3
        verify_tokens = torch.tensor([[9, 5, 5], [9, 6, 5]])
        banned = torch.zeros(bs, VOCAB_SIZE, dtype=torch.bool)
        banned[0, 4] = banned[1, 6] = True
        info = _sampling_info(bs)
        info.step_penalties = _step(bs)
        info.grammar_mask = SimpleNamespace(
            apply=lambda t: t.masked_fill_(banned, float("-inf"))
        )
        generator = torch.Generator().manual_seed(0)
        info.logit_bias = torch.randn(bs, VOCAB_SIZE, generator=generator)
        info.has_custom_logit_processor = True
        logits = (torch.randn(bs * width, VOCAB_SIZE, generator=generator) * 4).to(
            torch.bfloat16
        )

        expected = logits.clone()
        info.step_penalties.apply_verify(expected, verify_tokens)
        by_request = expected.view(bs, width, VOCAB_SIZE)
        by_request.masked_fill_(banned[:, None, :], float("-inf"))
        by_request.add_(info.logit_bias[:, None, :])

        seen = []
        with patch.object(
            dflash_utils,
            "apply_custom_logit_processor",
            side_effect=lambda t, *_, **__: seen.append(t.clone()),
        ):
            apply_dflash_verify_logits_adjustments(
                next_token_logits=logits,
                sampling_info=info,
                draft_token_num=width,
                verify_tokens=verify_tokens,
            )
        self.assertEqual(len(seen), 1)
        self.assertTrue(torch.equal(seen[0], expected))


if __name__ == "__main__":
    unittest.main()
