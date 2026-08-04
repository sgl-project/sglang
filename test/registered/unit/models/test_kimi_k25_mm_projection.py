"""CPU coverage for the Kimi vision-projector packing fast path."""

import pytest
import torch
import torch.nn as nn

from sglang.srt.models.kimi_k25 import mm_projection_auto
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _FlattenProjector(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states.flatten(start_dim=1)


def test_mm_projection_auto_packs_variable_image_outputs_once():
    outputs = [
        torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3),
        torch.arange(3 * 4 * 3, dtype=torch.float32).reshape(3, 4, 3),
    ]
    expected = torch.cat([output.flatten(start_dim=1) for output in outputs], dim=0)

    actual = mm_projection_auto(_FlattenProjector(), outputs)

    torch.testing.assert_close(actual, expected)
    assert actual.shape == (5, 12)


def test_mm_projection_auto_accepts_already_packed_output():
    output = torch.randn(5, 4, 3)

    actual = mm_projection_auto(_FlattenProjector(), output)

    torch.testing.assert_close(actual, output.flatten(start_dim=1))


def test_mm_projection_auto_flattens_unprojected_3d_output():
    output = torch.randn(5, 4, 3)

    actual = mm_projection_auto(None, output)

    torch.testing.assert_close(actual, output.reshape(-1, output.shape[-1]))
    assert actual.shape == (20, 3)


def test_mm_projection_auto_single_item_avoids_cat_copy():
    output = torch.randn(5, 4, 3)

    actual = mm_projection_auto(_FlattenProjector(), [output])

    torch.testing.assert_close(actual, output.flatten(start_dim=1))
    assert actual.data_ptr() == output.data_ptr()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
