from types import SimpleNamespace

import pytest

from sglang.srt.layers.communicator import ScatterMode
from sglang.srt.models import deepseek_v2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Backend:
    def __init__(self, is_none: bool):
        self._is_none = is_none

    def is_none(self) -> bool:
        return self._is_none


@pytest.mark.parametrize(
    (
        "enable_dp_attention",
        "attn_dp_size",
        "a2a_is_none",
        "mlp_mode",
        "is_moe",
        "expected",
    ),
    [
        (True, 2, True, ScatterMode.FULL, True, True),
        (False, 2, True, ScatterMode.FULL, True, False),
        (True, 1, True, ScatterMode.FULL, True, False),
        (True, 2, False, ScatterMode.FULL, True, False),
        (True, 2, True, ScatterMode.SCATTERED, True, False),
        (True, 2, True, ScatterMode.FULL, False, False),
    ],
)
def test_moe_sees_dp_gathered_rows_only_for_no_a2a_partial_dpa_full(
    monkeypatch,
    enable_dp_attention,
    attn_dp_size,
    a2a_is_none,
    mlp_mode,
    is_moe,
    expected,
):
    monkeypatch.setattr(
        deepseek_v2,
        "get_parallel",
        lambda: SimpleNamespace(
            enable_dp_attention=enable_dp_attention, attn_dp_size=attn_dp_size
        ),
    )
    monkeypatch.setattr(
        deepseek_v2, "get_moe_a2a_backend", lambda: _Backend(a2a_is_none)
    )
    mlp = (
        deepseek_v2.DeepseekV2MoE.__new__(deepseek_v2.DeepseekV2MoE)
        if is_moe
        else object()
    )
    modes = SimpleNamespace(mlp_mode=mlp_mode)

    assert deepseek_v2._moe_sees_dp_gathered_rows(mlp, modes) is expected
