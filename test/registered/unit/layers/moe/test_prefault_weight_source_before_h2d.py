import pytest
import torch

from sglang.srt.layers.moe.fused_moe_triton import layer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.fixture(params=[True, False], ids=["host-page-tables", "plain"])
def gpu(request, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(layer, "_faults_pageable_copies", lambda index: request.param)


@pytest.mark.parametrize(
    "source",
    [
        torch.arange(64, dtype=torch.bfloat16).reshape(16, 4).narrow(0, 4, 8),
        torch.arange(64, dtype=torch.uint8).reshape(8, 8).narrow(1, 2, 4),
        torch.arange(3 * 4096, dtype=torch.float32),
        torch.ones(1),
        torch.empty(0),
    ],
    ids=["contiguous-view", "strided-view", "multi-page", "one-element", "empty"],
)
def test_returns_the_source_unchanged(gpu, source):
    expected = source.clone()
    assert layer._maybe_copy_weight_view_before_h2d(source) is source
    assert torch.equal(source, expected)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
