from types import SimpleNamespace

import pytest

from sglang.srt.layers.attention import vision
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.fixture
def npu_platform(monkeypatch):
    monkeypatch.setattr(vision, "is_cuda", lambda: False)
    monkeypatch.setattr(vision, "_is_npu", True)
    monkeypatch.setattr(vision, "_is_musa", False)
    monkeypatch.setattr(vision, "_is_hip", False)
    monkeypatch.setattr(vision, "_is_cpu", False)
    monkeypatch.setattr(vision, "_is_xpu", False)


@pytest.mark.parametrize(
    ("server_backend", "passed_backend", "expected"),
    [
        (None, None, "ascend_attn"),
        (None, "sdpa", "sdpa"),
        ("sdpa", None, "sdpa"),
        ("sdpa", "ascend_attn", "sdpa"),
    ],
)
def test_npu_backend_selection_priority(
    monkeypatch,
    npu_platform,
    server_backend,
    passed_backend,
    expected,
):
    monkeypatch.setattr(
        vision,
        "get_server_args",
        lambda: SimpleNamespace(mm_attention_backend=server_backend),
    )

    backend = vision.VisionAttention._determine_attention_backend(None, passed_backend)

    assert backend == expected


def test_vit_graph_runner_caches_resolved_backend_name():
    from sglang.srt.multimodal.vit_cuda_graph_runner import ViTCudaGraphRunner

    class Block:
        attn = SimpleNamespace(
            qkv_backend_name="fa3",
            qkv_backend=object(),
        )

        def forward(self, x, output_ws=None):
            return x

    vit = SimpleNamespace(blocks=[Block()])

    runner = ViTCudaGraphRunner(vit)

    assert runner._attn_backend == "fa3"


def test_internvl_graph_runner_caches_resolved_backend_name():
    from sglang.srt.multimodal.internvl_vit_cuda_graph_runner import (
        InternViTCudaGraphRunner,
    )

    attention = SimpleNamespace(
        qkv_backend_name="triton_attn",
        qkv_backend=object(),
    )
    layer = SimpleNamespace(attn=SimpleNamespace(attn=attention))
    encoder = SimpleNamespace(layers=[layer])

    runner = InternViTCudaGraphRunner(encoder)

    assert runner._attn_backend == "triton_attn"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
