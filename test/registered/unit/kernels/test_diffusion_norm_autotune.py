"""Diffusion norm imports must not initialize an unavailable accelerator."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.kernels.ops.diffusion.norm.norm_triton import triton_autotune_configs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_unavailable_accelerator_does_not_query_device(monkeypatch):
    device_module = SimpleNamespace(
        is_available=lambda: False,
        current_device=Mock(side_effect=AssertionError("accelerator initialized")),
        get_device_properties=Mock(side_effect=AssertionError("device queried")),
    )
    monkeypatch.setattr(torch, "get_device_module", lambda: device_module)

    assert [config.num_warps for config in triton_autotune_configs()] == [
        1,
        2,
        4,
        8,
        16,
        32,
    ]
    device_module.current_device.assert_not_called()
    device_module.get_device_properties.assert_not_called()


def test_cpu_module_without_device_properties(monkeypatch):
    monkeypatch.setattr(torch, "get_device_module", lambda: torch.cpu)

    assert [config.num_warps for config in triton_autotune_configs()] == [
        1,
        2,
        4,
        8,
        16,
        32,
    ]


@pytest.mark.parametrize(
    "warp_size,expected_warps",
    [
        (32, [1, 2, 4, 8, 16, 32]),
        (64, [1, 2, 4, 8, 16]),
        (None, [1, 2, 4, 8, 16, 32]),
    ],
)
def test_accelerator_warp_count_respects_thread_limit(
    monkeypatch, warp_size, expected_warps
):
    device_module = SimpleNamespace(
        is_available=lambda: True,
        current_device=lambda: 2,
        get_device_properties=Mock(return_value=SimpleNamespace(warp_size=warp_size)),
    )
    monkeypatch.setattr(torch, "get_device_module", lambda: device_module)

    assert [config.num_warps for config in triton_autotune_configs()] == expected_warps
    device_module.get_device_properties.assert_called_once_with(2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
