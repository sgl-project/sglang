"""Architecture gate tests for Kimi-K3's fused attention residual."""

from unittest.mock import patch

import pytest

import sglang.srt.layers.attn_residual as attn_residual


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        ((10, 0), True),
        ((10, 3), True),
        ((11, 0), True),
        ((12, 0), False),
        ((9, 0), False),
    ],
)
def test_fast_attn_residual_arch_gate(capability, expected):
    with (
        patch.object(attn_residual, "_FAST_SUPPORTED", None),
        patch.object(attn_residual, "is_npu", return_value=False),
        patch.object(
            attn_residual.torch.cuda,
            "get_device_capability",
            return_value=capability,
        ),
    ):
        assert attn_residual._use_fast(7168) is expected


def test_fast_attn_residual_requires_kimi_hidden_size():
    with (
        patch.object(attn_residual, "_FAST_SUPPORTED", None),
        patch.object(attn_residual, "is_npu", return_value=False),
        patch.object(
            attn_residual.torch.cuda,
            "get_device_capability",
            return_value=(10, 0),
        ),
    ):
        assert not attn_residual._use_fast(4096)
