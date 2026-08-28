import sys
from types import SimpleNamespace

import pytest

from sglang.srt.layers.attention import vision
from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=300, suite="stage-a-test-1-gpu-xpu")


@pytest.mark.parametrize(
    ("server_backend", "passed_backend", "expected"),
    [
        (
            None,
            None,
            "xpu_attn",
        ),  # server backend is not set, expected to use xpu_attn as default
        (
            "xpu_attn",
            None,
            "xpu_attn",
        ),  # server backend is set to xpu_attn, expected to use xpu_attn
        (
            "xpu_attn",
            "triton_attn",
            "xpu_attn",
        ),  # server backend is set to xpu_attn, passed backend is triton_attn
        (
            "triton_attn",
            None,
            "triton_attn",
        ),  # server backend is set to triton_attn, expected to use triton_attn
    ],
)
def test_xpu_backend_selection_priority(
    monkeypatch,
    server_backend,  # specified by the server argument
    passed_backend,  # specified by the layer argument
    expected,
):
    monkeypatch.setattr(
        vision,
        "get_mm",
        lambda: SimpleNamespace(mm_attention_backend=server_backend),
    )

    backend = vision.VisionAttention._determine_attention_backend(None, passed_backend)

    assert backend == expected


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
