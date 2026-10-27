"""EAGLE draft attention on XPU when intel_xpu lacks the draft's head size.

intel_xpu's paged kernels raise on head sizes outside 64/96/128/192/256/512, so a
draft that inherits it (e.g. head_dim 80) must resolve to triton instead.
"""

from unittest.mock import patch

import pytest
from transformers import PretrainedConfig

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.speculative_hook import (
    _fall_back_xpu_draft_attention_for_head_size,
)
from sglang.srt.runtime_context import override_platform
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _resolve(draft_config: PretrainedConfig, **overrides):
    args = ServerArgs(
        model_path="dummy",
        device="xpu",
        attention_backend="intel_xpu",
        speculative_algorithm="EAGLE3",
        speculative_draft_model_path="draft",
        **overrides,
    )
    with (
        override_platform(is_xpu=True),
        patch(
            "sglang.srt.utils.hf_transformers_utils.get_config",
            return_value=draft_config,
        ),
    ):
        _fall_back_xpu_draft_attention_for_head_size(args)
    return resolution_result(args, "speculative_draft_attention_backend")


@pytest.mark.parametrize(
    "draft_config",
    [
        PretrainedConfig(head_dim=80, hidden_size=5120, num_attention_heads=64),
        # No head_dim in the config: the head size is hidden_size / heads.
        PretrainedConfig(hidden_size=2560, num_attention_heads=32),
    ],
)
def test_unsupported_draft_head_size_falls_back_to_triton(draft_config):
    assert _resolve(draft_config) == "triton"


def test_supported_or_explicit_draft_backend_is_kept():
    supported = PretrainedConfig(head_dim=128, hidden_size=5120, num_attention_heads=64)
    assert _resolve(supported) is None
    unsupported = PretrainedConfig(
        head_dim=80, hidden_size=5120, num_attention_heads=64
    )
    explicit = _resolve(unsupported, speculative_draft_attention_backend="intel_xpu")
    assert explicit == "intel_xpu"
