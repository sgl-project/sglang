from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.srt.arg_groups.speculative_hook import _handle_dflash
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_dflash_rejects_platform_without_capability():
    cfg = SimpleNamespace(device="custom")

    with (
        patch(
            "sglang.srt.arg_groups.speculative_hook.resolving_view",
            return_value=cfg,
        ),
        patch(
            "sglang.srt.arg_groups.speculative_hook.current_platform.supports_dflash",
            return_value=False,
        ),
        pytest.raises(ValueError, match="not supported on custom"),
    ):
        _handle_dflash(SimpleNamespace())


def test_dflash_accepts_platform_with_capability():
    cfg = SimpleNamespace(
        device="custom",
        pp_size=1,
        speculative_draft_model_path=None,
    )

    with (
        patch(
            "sglang.srt.arg_groups.speculative_hook.resolving_view",
            return_value=cfg,
        ),
        patch(
            "sglang.srt.arg_groups.speculative_hook.resolved_view",
            return_value=SimpleNamespace(enable_dp_attention=False),
        ),
        patch(
            "sglang.srt.arg_groups.speculative_hook.current_platform.supports_dflash",
            return_value=True,
        ),
        pytest.raises(ValueError, match="requires setting"),
    ):
        _handle_dflash(SimpleNamespace())
