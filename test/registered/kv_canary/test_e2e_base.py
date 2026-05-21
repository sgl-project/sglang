import pytest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kv_canary.e2e_base import (
    _LONG_PROMPT_BODY,
    _UNIQUE_PROMPT_FIRST_CHARS,
    _make_unique_prompts,
)

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_make_unique_prompts_have_distinct_first_characters() -> None:
    prompts = _make_unique_prompts(8)

    assert len({prompt[0] for prompt in prompts}) == len(prompts)
    assert all(prompt.endswith(_LONG_PROMPT_BODY) for prompt in prompts)


def test_make_unique_prompts_rejects_more_prompts_than_distinct_first_characters() -> (
    None
):
    with pytest.raises(ValueError, match="unique prompt count"):
        _make_unique_prompts(len(_UNIQUE_PROMPT_FIRST_CHARS) + 1)
