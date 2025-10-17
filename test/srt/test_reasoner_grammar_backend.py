from types import SimpleNamespace

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


def _make_scheduler():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.tokenizer = SimpleNamespace(reasoning_initial_in_reasoning=False)
    return scheduler


def test_apply_reasoning_state_resets_without_custom_flag():
    scheduler = _make_scheduler()
    grammar = ReasonerGrammarObject(
        DummyGrammar(),
        think_end_id=None,
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
        think_end_id=None,
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
