import sys
from types import SimpleNamespace

import pytest

from sglang.srt.arg_groups.deepseek_v4_hook import (
    validate_deepseek_v4_attention_tp,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _server_args(*, dp_size: int, enable_dp_attention: bool):
    config = SimpleNamespace(num_attention_heads=128, o_groups=16)
    view = SimpleNamespace(
        enable_dp_attention=enable_dp_attention,
        attn_cp_size=1,
    )
    return SimpleNamespace(
        tp_size=24,
        dp_size=dp_size,
        get_model_config=lambda: SimpleNamespace(hf_config=config),
        _resolved=lambda: view,
    )


def test_deepseek_v4_tp24_requires_supported_attention_tp():
    validate_deepseek_v4_attention_tp(_server_args(dp_size=3, enable_dp_attention=True))
    # Leave non-integral layouts to the generic TP/DP/CP validation.
    validate_deepseek_v4_attention_tp(_server_args(dp_size=5, enable_dp_attention=True))

    with pytest.raises(ValueError, match=r"attn_tp_size=24.*data-parallel-size 3"):
        validate_deepseek_v4_attention_tp(
            _server_args(dp_size=1, enable_dp_attention=False)
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
