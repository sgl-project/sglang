import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.dflash_info import (
    DFlashVerifyInput,
    _build_linear_grammar_vocab_mask,
)
from sglang.srt.speculative.dflash_utils import validate_dflash_request
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def _is_allowed(bitmask_row: torch.Tensor, token_id: int) -> bool:
    return (int(bitmask_row[token_id // 32].item()) & (1 << (token_id % 32))) != 0


class FakeGrammar:
    def __init__(self, allowed_by_state):
        self.allowed_by_state = allowed_by_state
        self.state = 0
        self.accept_calls = []
        self.fill_calls = []
        self.rollback_calls = []

    def allocate_vocab_mask(self, vocab_size: int, batch_size: int, device):
        width = (vocab_size + 31) // 32
        return torch.zeros((batch_size, width), dtype=torch.int32, device=device)

    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        self.fill_calls.append((self.state, idx))
        vocab_mask[idx].zero_()
        for token_id in self.allowed_by_state[self.state]:
            word = token_id // 32
            bit = 1 << (token_id % 32)
            vocab_mask[idx, word] = int(vocab_mask[idx, word].item()) | bit

    def accept_token(self, token_id: int) -> None:
        self.accept_calls.append(token_id)
        self.state += 1

    def rollback(self, num_tokens: int) -> None:
        self.rollback_calls.append(num_tokens)
        self.state -= num_tokens

    def is_terminated(self) -> bool:
        return False


class FakeSamplingInfo:
    is_all_greedy = False
    vocab_size = 16

    def __len__(self):
        return 1


class TestDFlashGrammarMask(unittest.TestCase):
    def test_validate_dflash_request_allows_greedy_grammar(self):
        req = SimpleNamespace(
            return_logprob=False,
            sampling_params=SimpleNamespace(
                json_schema=None,
                regex=r"^\d{3}$",
                ebnf=None,
                structural_tag=None,
                top_k=1,
            ),
        )

        self.assertIsNone(validate_dflash_request(req))

    def test_validate_dflash_request_rejects_non_greedy_grammar(self):
        req = SimpleNamespace(
            return_logprob=False,
            sampling_params=SimpleNamespace(
                json_schema=None,
                regex=r"^\d{3}$",
                ebnf=None,
                structural_tag=None,
                top_k=20,
            ),
        )

        self.assertIn("non-greedy", validate_dflash_request(req))

    def test_returns_none_without_grammar(self):
        batch = SimpleNamespace(reqs=[SimpleNamespace(grammar=None)])
        candidates = torch.tensor([[0, 5, 7]], dtype=torch.long)

        vocab_mask, apply_grammar = _build_linear_grammar_vocab_mask(
            batch=batch,
            candidates_cpu=candidates,
            vocab_size=16,
        )

        self.assertIsNone(vocab_mask)
        self.assertIsNone(apply_grammar)

    def test_advances_and_rolls_back_for_linear_chain(self):
        grammar = FakeGrammar([{5}, {7}, {1}])
        batch = SimpleNamespace(reqs=[SimpleNamespace(grammar=grammar)])
        candidates = torch.tensor([[0, 5, 7, 9]], dtype=torch.long)

        vocab_mask, apply_grammar = _build_linear_grammar_vocab_mask(
            batch=batch,
            candidates_cpu=candidates,
            vocab_size=16,
        )

        self.assertIs(apply_grammar, grammar)
        self.assertEqual(grammar.accept_calls, [5, 7])
        self.assertEqual(grammar.rollback_calls, [2])
        self.assertEqual(grammar.state, 0)
        self.assertEqual(grammar.fill_calls, [(0, 0), (1, 1), (2, 2)])

        self.assertTrue(_is_allowed(vocab_mask[0], 5))
        self.assertTrue(_is_allowed(vocab_mask[1], 7))
        self.assertTrue(_is_allowed(vocab_mask[2], 1))
        self.assertFalse(_is_allowed(vocab_mask[2], 9))
        self.assertEqual(vocab_mask[3].sum().item(), 0)

    def test_verify_rejects_non_greedy_grammar(self):
        verify_input = DFlashVerifyInput(
            draft_token=torch.tensor([0, 5], dtype=torch.long),
            positions=torch.tensor([0, 1], dtype=torch.long),
            draft_token_num=2,
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            device=torch.device("cpu"),
            sampling_info=FakeSamplingInfo(),
            has_grammar=True,
            reqs=[SimpleNamespace(grammar=FakeGrammar([{5}]))],
            batch_size=lambda: 1,
        )
        logits_output = LogitsProcessorOutput(
            next_token_logits=torch.zeros((2, 16), dtype=torch.float32),
            hidden_states=torch.zeros((2, 1), dtype=torch.float32),
        )

        with self.assertRaisesRegex(RuntimeError, "greedy requests only"):
            verify_input.verify(
                batch=batch,
                logits_output=logits_output,
                page_size=1,
            )


if __name__ == "__main__":
    unittest.main()
