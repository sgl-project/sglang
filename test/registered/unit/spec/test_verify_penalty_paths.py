import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.sampling.penalties import PenaltyStatusPool
from sglang.srt.sampling.sampling_batch_info import ProcessorEntry, SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.sampling.sampling_pipeline import VerifyRoute, VerifyStep
from sglang.srt.speculative import eagle_utils
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.sampling_test_utils import build_step_penalties
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

VOCAB_SIZE = 16


class _Stop(Exception):
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
    return build_step_penalties(reqs, pool)


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


def _verify(batch, grammar_mask=None):
    return VerifyStep(
        batch=batch,
        route=VerifyRoute(adjusts_logits=True, need_truncated_sampling=False),
        watermark=None,
        watermark_full_mask=True,
        grammar_mask=grammar_mask,
    )


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
            raise _Stop

        with patch.object(step, "apply_verify", side_effect=apply_then_stop):
            with self.assertRaises(_Stop):
                eagle_utils.eagle_sample(verify_input, _verify(batch), logits_output)

        self.assertEqual(
            _row_counts(logits_output.next_token_logits, [5, 6, 9]),
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0]],
        )

    def test_eagle_sample_adjusts_in_decoding_order_per_request_tree(self):
        bs, width = 2, 3
        verify_tokens = torch.tensor([[9, 5, 5], [9, 6, 5]])
        next_child = torch.tensor([[1, 2, -1]] * bs)
        next_sibling = torch.full((bs, width), -1)
        banned = torch.zeros(bs * width, VOCAB_SIZE, dtype=torch.bool)
        banned[:width, 4] = banned[width:, 6] = True
        grammar_mask = SimpleNamespace(
            apply=lambda t: t.masked_fill_(banned, float("-inf"))
        )
        info = _sampling_info(bs)
        info.step_penalties = _step(bs)
        generator = torch.Generator().manual_seed(0)
        info.logit_bias = torch.randn(bs, VOCAB_SIZE, generator=generator)
        logits = (torch.randn(bs * width, VOCAB_SIZE, generator=generator) * 4).to(
            torch.bfloat16
        )

        expected = logits.clone()
        info.step_penalties.apply_verify(
            expected, verify_tokens, next_child=next_child, next_sibling=next_sibling
        )
        expected.masked_fill_(banned, float("-inf"))
        expected.view(bs, width, VOCAB_SIZE).add_(info.logit_bias[:, None, :])

        def processor(logits, custom_params):
            seen.append((logits.clone(), custom_params))
            raise _Stop

        seen = []
        info.has_custom_logit_processor = True
        info.custom_params = [{"req": 0}, {"req": 1}]
        info.custom_logit_processor = {
            0: ProcessorEntry(processor=processor, rows=[1], indices=torch.tensor([1]))
        }
        verify_input = SimpleNamespace(
            draft_token_num=width,
            draft_token=verify_tokens.reshape(-1),
            retrieve_next_token=next_child,
            retrieve_next_sibling=next_sibling,
        )
        batch = SimpleNamespace(
            device="cpu",
            seq_lens=torch.tensor([4, 4]),
            sampling_info=info,
            forward_mode=SimpleNamespace(is_idle=lambda: False),
        )
        logits_output = SimpleNamespace(
            next_token_logits=logits, next_token_logits_vocab_shard=None
        )
        with self.assertRaises(_Stop):
            eagle_utils.eagle_sample(
                verify_input, _verify(batch, grammar_mask), logits_output
            )
        self.assertEqual(len(seen), 1)
        self.assertTrue(torch.equal(seen[0][0], expected[width:]))
        self.assertEqual(seen[0][1], [{"req": 1}] * width)


if __name__ == "__main__":
    unittest.main()
