import itertools
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

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


def test_xpu_attn_padded_batch_attends_within_each_image():
    """Images padded to a common seq_len with real lengths in cu_seqlens must each
    attend only over their own rows, and the padding rows must come back zero."""
    lens = [1036, 1014, 1014]
    bsz, seq_len, heads, head_dim = len(lens), max(lens), 16, 72
    cu_seqlens = torch.tensor(
        [0, *itertools.accumulate(lens)], dtype=torch.int32, device="xpu"
    )
    q, k, v = (
        torch.randn(bsz * seq_len, heads, head_dim, device="xpu", dtype=torch.bfloat16)
        for _ in range(3)
    )

    out = vision.VisionIntelXPUAttention().forward(
        q=q, k=k, v=v, cu_seqlens=cu_seqlens, bsz=bsz, seq_len=seq_len
    )

    for b, n in enumerate(lens):
        rows = slice(b * seq_len, b * seq_len + n)
        ref = F.scaled_dot_product_attention(
            *(t[rows].transpose(0, 1).float() for t in (q, k, v))
        ).transpose(0, 1)
        torch.testing.assert_close(out[rows].float(), ref, atol=2e-2, rtol=2e-2)
        assert (out[b * seq_len + n : (b + 1) * seq_len] == 0).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
