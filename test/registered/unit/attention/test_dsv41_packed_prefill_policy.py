"""Exercise dispatch without allocating GPU memory or importing a model."""

import pytest
import torch

from sglang.srt.layers.attention.dsv4.packed_prefill import PackedPrefillPolicy
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def tensors(rows, width):
    return (
        torch.empty(rows, 64, 512, device="meta", dtype=torch.bfloat16),
        torch.empty(rows, width, device="meta", dtype=torch.int32),
    )


@pytest.fixture
def policy():
    policy = PackedPrefillPolicy("meta")
    # Device qualification is tested separately; meta tensors exercise metadata.
    policy.supported_gpu = True
    return policy


@pytest.mark.parametrize("width,boundary", [(128, 512), (640, 4096)])
@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_thresholds(policy, width, boundary, offset):
    assert policy.can_use(*tensors(boundary + offset, width), real_heads=16) == (
        offset >= 0
    )


@pytest.mark.parametrize("heads", [8, 32, 64])
def test_real_heads_required(policy, heads):
    assert not policy.can_use(*tensors(4096, 640), real_heads=heads)


def test_configurable_threshold(policy):
    policy.mixed_min_rows = 8192
    assert not policy.can_use(*tensors(4096, 640), real_heads=16)
    assert policy.can_use(*tensors(8192, 640), real_heads=16)


@pytest.mark.parametrize(
    "shape", [(10, 0, 148), (10, 3, 148), (9, 0, 132), (10, 0, 132)]
)
def test_device_qualification(monkeypatch, shape):
    from types import SimpleNamespace

    major, minor, sms = shape
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(major=major, minor=minor, multi_processor_count=sms),
    )
    policy = PackedPrefillPolicy("cuda:0")
    assert policy.supported_gpu == (shape == (10, 0, 148))


def test_unsupported_inputs(policy):
    args = list(tensors(4096, 640))
    for index, value in (
        (0, args[0].float()),
        (0, args[0][:, :, ::2]),
        (0, args[0][:, :32]),
        (0, args[0].unsqueeze(1)),
        (1, args[1][:, ::2]),
        (1, args[1].unsqueeze(1)),
    ):
        modified = args.copy()
        modified[index] = value
        assert not policy.can_use(*modified, real_heads=16)
    policy.supported_gpu = False
    assert not policy.can_use(*args, real_heads=16)


def test_invalid_threshold():
    with pytest.raises(ValueError, match="positive"):
        PackedPrefillPolicy("cpu", mixed_min_rows=0)


def test_implicit_cuda_device(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(major=10, minor=0, multi_processor_count=148),
    )
    assert PackedPrefillPolicy("cuda").device == torch.device("cuda:2")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
