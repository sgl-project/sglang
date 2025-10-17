from types import SimpleNamespace
from typing import List

import torch

from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject
from sglang.srt.constrained.reasoner_grammar_backend import ReasonerGrammarObject
from sglang.srt.managers.scheduler import Scheduler


class DummyGrammar(BaseGrammarObject):
    def accept_token(self, token: int) -> None:
        return None

    def allocate_vocab_mask(self, vocab_size: int, batch_size: int, device):
        return None

    def fill_vocab_mask(self, vocab_mask, idx: int) -> None:
        return None

    def move_vocab_mask(self, vocab_mask, device):
        return vocab_mask

    @property
    def apply_vocab_mask(self):
        return lambda logits, mask: None

    def copy(self) -> "DummyGrammar":
        return DummyGrammar()

    def try_jump_forward(self, tokenizer):
        return None

    def jump_forward_str_state(self, helper):
        raise NotImplementedError

    def jump_and_retokenize(self, old_output_ids, new_output_ids, next_state: int):
        return None


class TrackingGrammar(BaseGrammarObject):
    def __init__(self):
        self.recorded: List[int] = []

    def accept_token(self, token: int) -> None:
        self.recorded.append(token)

    def allocate_vocab_mask(self, vocab_size: int, batch_size: int, device):
        return None

    def fill_vocab_mask(self, vocab_mask, idx: int) -> None:
        return None

    def move_vocab_mask(self, vocab_mask, device):
        return vocab_mask

    @property
    def apply_vocab_mask(self):
        return lambda logits, mask: None

    def copy(self) -> "TrackingGrammar":
        cloned = TrackingGrammar()
        cloned.recorded = list(self.recorded)
        return cloned

    def try_jump_forward(self, tokenizer):
        return None

    def jump_forward_str_state(self, helper):
        raise NotImplementedError

    def jump_and_retokenize(self, old_output_ids, new_output_ids, next_state: int):
        return None


class MaskTrackingGrammar(BaseGrammarObject):
    def __init__(self, bitset: bool):
        super().__init__()
        self.bitset = bitset
        self.calls = 0

    def accept_token(self, token: int):
        return None

    def allocate_vocab_mask(self, vocab_size: int, batch_size: int, device):
        if self.bitset:
            width = (vocab_size + 31) // 32
            return torch.zeros(batch_size, width, dtype=torch.int32, device=device)
        return torch.zeros(batch_size, vocab_size, dtype=torch.bool, device=device)

    def fill_vocab_mask(self, vocab_mask, idx: int):
        self.calls += 1
        row = vocab_mask[idx]
        if self.bitset:
            row.fill_(0xAAAAAAAA)
        else:
            row.fill_(True)

    def move_vocab_mask(self, vocab_mask, device):
        return vocab_mask

    @property
    def apply_vocab_mask(self):
        return lambda logits, mask: None

    def copy(self) -> "MaskTrackingGrammar":
        return MaskTrackingGrammar(self.bitset)

    def try_jump_forward(self, tokenizer):
        return None

    def jump_forward_str_state(self, helper):
        raise NotImplementedError

    def jump_and_retokenize(self, old_output_ids, new_output_ids, next_state: int):
        return None


def _make_scheduler():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.tokenizer = SimpleNamespace(reasoning_initial_in_reasoning=False)
    return scheduler


def test_apply_reasoning_state_resets_without_custom_flag():
    scheduler = _make_scheduler()
    grammar = ReasonerGrammarObject(
        DummyGrammar(),
        think_end_ids=None,
        think_start_ids=None,
        initial_in_reasoning=False,
    )
    grammar.is_in_reasoning = True
    req = SimpleNamespace(
        sampling_params=SimpleNamespace(custom_params=None),
        grammar=grammar,
    )

    scheduler._apply_reasoning_initial_state(req)

    assert grammar.is_in_reasoning is False


def test_apply_reasoning_state_with_custom_flag():
    scheduler = _make_scheduler()
    grammar = ReasonerGrammarObject(
        DummyGrammar(),
        think_end_ids=None,
        think_start_ids=None,
        initial_in_reasoning=False,
    )
    req = SimpleNamespace(
        sampling_params=SimpleNamespace(
            custom_params={"reasoning_initial_in_reasoning": True}
        ),
        grammar=grammar,
    )

    scheduler._apply_reasoning_initial_state(req)

    assert grammar.is_in_reasoning is True


def test_multitoken_think_end_sequence_transitions_cleanly():
    tracking = TrackingGrammar()
    grammar = ReasonerGrammarObject(
        tracking,
        think_end_ids=[5, 6, 7],
        think_start_ids=[1, 2],
        initial_in_reasoning=False,
    )

    # Regular token before reasoning should be forwarded.
    grammar.accept_token(3)
    assert tracking.recorded == [3]
    assert grammar.is_in_reasoning is False

    # Enter reasoning via start sequence.
    grammar.accept_token(1)
    grammar.accept_token(2)
    assert tracking.recorded == [3]
    assert grammar.is_in_reasoning is True

    # Reasoning content should not reach the downstream grammar.
    grammar.accept_token(9)
    assert tracking.recorded == [3]

    # Multi-token end marker should switch back to normal mode without forwarding.
    grammar.accept_token(5)
    grammar.accept_token(6)
    assert grammar.is_in_reasoning is True
    grammar.accept_token(7)
    assert grammar.is_in_reasoning is False
    assert tracking.recorded == [3]

    # Subsequent tokens must be validated by the downstream grammar again.
    grammar.accept_token(8)
    assert tracking.recorded == [3, 8]


def test_fill_vocab_mask_allows_any_token_during_reasoning_bitset():
    base = MaskTrackingGrammar(bitset=True)
    grammar = ReasonerGrammarObject(
        base, think_end_ids=None, think_start_ids=None, initial_in_reasoning=True
    )
    mask = base.allocate_vocab_mask(vocab_size=64, batch_size=2, device="cpu")

    grammar.fill_vocab_mask(mask, 1)
    assert torch.all(mask[1] == -1)
    assert base.calls == 0

    mask[1].zero_()
    grammar.is_in_reasoning = False
    grammar.fill_vocab_mask(mask, 1)
    assert base.calls == 1
    expected = torch.full_like(mask[1], 0xAAAAAAAA)
    assert torch.equal(mask[1], expected)


def test_fill_vocab_mask_allows_any_token_during_reasoning_bool():
    base = MaskTrackingGrammar(bitset=False)
    grammar = ReasonerGrammarObject(
        base, think_end_ids=None, think_start_ids=None, initial_in_reasoning=True
    )
    mask = base.allocate_vocab_mask(vocab_size=8, batch_size=1, device="cpu")

    grammar.fill_vocab_mask(mask, 0)
    assert torch.all(mask[0] == 0)
    assert base.calls == 0

    mask[0].zero_()
    grammar.is_in_reasoning = False
    grammar.fill_vocab_mask(mask, 0)
    assert base.calls == 1
    assert torch.all(mask[0])
