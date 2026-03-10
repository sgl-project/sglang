import math

from sglang.srt.managers.tokenizer_manager import TokenizerManager


def test_detokenize_logprob_tokens_sanitizes_nonfinite() -> None:
    manager = TokenizerManager.__new__(TokenizerManager)
    manager.tokenizer = None

    result = manager.detokenize_logprob_tokens(
        [0.0, -math.inf, math.inf], [1, 2, 3], decode_to_text=False
    )

    assert result == [
        (0.0, 1, None),
        (None, 2, None),
        (None, 3, None),
    ]
