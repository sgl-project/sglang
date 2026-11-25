import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from sglang.srt.layers.attention import vision
from sglang.test.ci.ci_register import register_cpu_ci, register_npu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")
register_npu_ci(est_time=2, suite="extra-b-test-1-npu-a3")
register_npu_ci(est_time=2, suite="nightly-test-1-npu-a3", nightly=True)


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
        "get_mm",
        lambda: SimpleNamespace(mm_attention_backend=server_backend),
    )

    backend = vision.VisionAttention._determine_attention_backend(None, passed_backend)

    assert backend == expected


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        ((9, 0), "fa3"),
        ((10, 0), "fa4"),
        ((10, 3), "fa4"),
        ((12, 0), "triton_attn"),
    ],
)
def test_cuda_default_backend_by_capability(monkeypatch, capability, expected):
    monkeypatch.setattr(vision, "is_cuda", lambda: True)
    monkeypatch.setattr(vision, "get_device_capability", lambda: capability)
    monkeypatch.setattr(
        vision, "get_platform", lambda: SimpleNamespace(is_blackwell=False)
    )
    monkeypatch.setattr(
        vision, "get_mm", lambda: SimpleNamespace(mm_attention_backend=None)
    )

    backend = vision.VisionAttention._determine_attention_backend(None, None)

    assert backend == expected


@pytest.mark.parametrize("graph_env", ["0", "1"])
def test_fa4_seqlens_from_vit_cuda_graph_list(monkeypatch, graph_env):
    # ViT CUDA graph runners pass [cu_seqlens, max_seqlen]; tensors must keep
    # working even when SGLANG_VIT_ENABLE_CUDA_GRAPH is set for another model.
    monkeypatch.setenv("SGLANG_VIT_ENABLE_CUDA_GRAPH", graph_env)
    monkeypatch.setattr(vision, "_is_cuda", True)
    calls = []

    def fake_flash_attn_func(q, k, v, **kwargs):
        calls.append(kwargs)
        return q

    monkeypatch.setattr(vision, "flash_attn_func", fake_flash_attn_func, raising=False)
    q = torch.randn(6, 2, 8)
    cu_seqlens = torch.tensor([0, 4, 6], dtype=torch.int32)
    backend = vision.VisionFlash4Attention()

    backend(q, q, q, cu_seqlens=[cu_seqlens, 4], bsz=1, seq_len=6)
    backend(q, q, q, cu_seqlens=cu_seqlens, bsz=1, seq_len=6, max_seqlen=4)

    for kwargs in calls:
        assert kwargs["ver"] == 4
        assert torch.equal(kwargs["cu_seqlens_q"], cu_seqlens)
        assert kwargs["max_seqlen_q"] == kwargs["max_seqlen_k"] == 4
    assert calls[0]["cu_seqlens_q"] is cu_seqlens


def test_fa3_tensor_seqlens_with_vit_cuda_graph_env(monkeypatch):
    # A model without a ViT CUDA graph runner passes tensor cu_seqlens; setting
    # SGLANG_VIT_ENABLE_CUDA_GRAPH for another model must not make FA3 raise.
    monkeypatch.setenv("SGLANG_VIT_ENABLE_CUDA_GRAPH", "1")
    monkeypatch.setattr(vision, "_is_cuda", True)
    calls = []

    def fake_flash_attn_func(q, k, v, **kwargs):
        calls.append(kwargs)
        return q

    monkeypatch.setattr(vision, "flash_attn_func", fake_flash_attn_func, raising=False)
    q = torch.randn(6, 2, 8)
    cu_seqlens = torch.tensor([0, 4, 6], dtype=torch.int32)
    backend = vision.VisionFlash3Attention()

    backend(q, q, q, cu_seqlens=[cu_seqlens, 4], bsz=1, seq_len=6)
    backend(q, q, q, cu_seqlens=cu_seqlens, bsz=1, seq_len=6, max_seqlen=4)

    for kwargs in calls:
        assert torch.equal(kwargs["cu_seqlens_q"], cu_seqlens)
        assert kwargs["max_seqlen_q"] == kwargs["max_seqlen_k"] == 4
    assert calls[0]["cu_seqlens_q"] is cu_seqlens


def test_triton_tensor_seqlens_with_vit_cuda_graph_env(monkeypatch):
    # Same contract for the Triton backend: the list form (with its output
    # workspace) comes only from a graph runner, tensors take the eager path.
    monkeypatch.setenv("SGLANG_VIT_ENABLE_CUDA_GRAPH", "1")
    calls = []

    def fake_context_attention_fwd(
        q, k, v, o, cu_seqlens, seq_lens, max_seqlen, **kwargs
    ):
        calls.append((o, cu_seqlens, seq_lens, max_seqlen))

    monkeypatch.setattr(vision, "context_attention_fwd", fake_context_attention_fwd)
    q = torch.randn(6, 2, 8)
    cu_seqlens = torch.tensor([0, 4, 6], dtype=torch.int32)
    seq_lens = torch.tensor([4, 2], dtype=torch.int32)
    output_ws = torch.empty_like(q)
    backend = vision.VisionTritonAttention()

    out = backend(
        q,
        q,
        q,
        cu_seqlens=[cu_seqlens, seq_lens, 4],
        bsz=1,
        seq_len=6,
        output_ws=output_ws,
    )
    backend(
        q, q, q, cu_seqlens=cu_seqlens, bsz=1, seq_len=6, max_seqlen=4, output_ws=None
    )

    assert out is output_ws
    assert calls[0][1] is cu_seqlens
    for o, cu, lens, max_seqlen in calls:
        assert o.shape == q.shape
        assert torch.equal(cu, cu_seqlens)
        assert torch.equal(lens, seq_lens)
        assert max_seqlen == 4


def test_triton_graph_list_requires_output_workspace(monkeypatch):
    # VisionAttention forwards output_ws=None when its caller has no workspace;
    # the graph form must reject that instead of handing None to the kernel.
    monkeypatch.setenv("SGLANG_VIT_ENABLE_CUDA_GRAPH", "1")
    monkeypatch.setattr(vision, "context_attention_fwd", lambda *args, **kwargs: None)
    q = torch.randn(6, 2, 8)
    cu_seqlens = torch.tensor([0, 4, 6], dtype=torch.int32)
    seq_lens = torch.tensor([4, 2], dtype=torch.int32)
    backend = vision.VisionTritonAttention()

    with pytest.raises(RuntimeError, match="output_ws should be prepared"):
        backend(
            q,
            q,
            q,
            cu_seqlens=[cu_seqlens, seq_lens, 4],
            bsz=1,
            seq_len=6,
            output_ws=None,
        )


@pytest.mark.parametrize("missing", ["sequence_lengths", "max_seqlen"])
def test_flashinfer_rejects_seqlen_inputs_forwarded_as_none(monkeypatch, missing):
    # Same contract for the FlashInfer backend: VisionAttention forwards both
    # values as None when the encoder did not prepare them.
    monkeypatch.setattr(vision, "_is_cuda", True)
    q = torch.randn(6, 2, 8)
    inputs = dict(sequence_lengths=torch.tensor([4, 2]), max_seqlen=4)
    inputs[missing] = None
    backend = vision.VisionFlashInferAttention()

    with pytest.raises(RuntimeError, match=f"{missing} should be prepared"):
        backend(
            q,
            q,
            q,
            cu_seqlens=torch.tensor([0, 4, 6], dtype=torch.int32),
            bsz=1,
            seq_len=6,
            **inputs,
        )


def test_explicit_backend_without_published_mm_context(monkeypatch, npu_platform):
    monkeypatch.setattr(
        vision,
        "get_mm",
        Mock(side_effect=ValueError("config namespace 'mm' not published")),
    )
    monkeypatch.setattr(
        vision,
        "get_context",
        lambda: SimpleNamespace(is_config_namespace_published=lambda namespace: False),
    )

    backend = vision.VisionAttention._determine_attention_backend(None, "sdpa")

    assert backend == "sdpa"


def test_explicit_backend_keeps_published_context_errors(monkeypatch, npu_platform):
    monkeypatch.setattr(
        vision,
        "get_mm",
        Mock(side_effect=ValueError("mm namespace is not available for this role")),
    )
    monkeypatch.setattr(
        vision,
        "get_context",
        lambda: SimpleNamespace(is_config_namespace_published=lambda namespace: True),
    )

    with pytest.raises(ValueError, match="not available for this role"):
        vision.VisionAttention._determine_attention_backend(None, "sdpa")


def test_sdpa_preserves_flattened_batch_layout():
    torch.manual_seed(0)
    bsz, seq_len, num_heads, head_dim = 3, 5, 2, 8
    q, k, v = [torch.randn(bsz * seq_len, num_heads, head_dim) for _ in range(3)]
    backend = vision.VisionSdpaAttention(
        head_dim=head_dim,
        num_heads=num_heads,
        num_kv_heads=num_heads,
    )

    output = backend(q=q, k=k, v=v, bsz=bsz)
    q_ref, k_ref, v_ref = [
        x.reshape(bsz, seq_len, num_heads, head_dim).transpose(1, 2) for x in (q, k, v)
    ]
    expected = F.scaled_dot_product_attention(
        q_ref,
        k_ref,
        v_ref,
        scale=backend.scale,
    )
    expected = expected.transpose(1, 2).reshape(bsz * seq_len, num_heads, head_dim)

    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("mask_kind", ["causal", "padding"])
def test_ascend_attention_masked_inputs_fall_back_to_sdpa(
    monkeypatch,
    npu_platform,
    mask_kind,
):
    torch.manual_seed(0)
    bsz, seq_len, num_heads, head_dim = 2, 4, 2, 8
    softmax_scale = 0.37
    q, k, v = [torch.randn(bsz * seq_len, num_heads, head_dim) for _ in range(3)]
    mask = torch.zeros(bsz, 1, seq_len, seq_len)
    if mask_kind == "causal":
        masked_positions = torch.ones(seq_len, seq_len, dtype=torch.bool).triu(1)
        mask.masked_fill_(masked_positions, torch.finfo(mask.dtype).min)
    else:
        mask[:, :, :, -1] = torch.finfo(mask.dtype).min

    fused_attention = Mock(
        side_effect=AssertionError("masked inputs must not use Ascend fused attention")
    )
    monkeypatch.setattr(
        vision,
        "torch_npu",
        SimpleNamespace(npu_fused_infer_attention_score=fused_attention),
        raising=False,
    )
    backend = vision.VisionAscendAttention(
        head_dim=head_dim,
        num_heads=num_heads,
        num_kv_heads=num_heads,
        softmax_scale=softmax_scale,
    )

    output = backend(
        q=q,
        k=k,
        v=v,
        cu_seqlens=torch.arange(0, (bsz + 1) * seq_len, seq_len),
        bsz=bsz,
        seq_len=seq_len,
        attention_mask=mask,
    )
    q_ref, k_ref, v_ref = [
        rearrange(x, "(b s) h d -> b h s d", b=bsz) for x in (q, k, v)
    ]
    expected = F.scaled_dot_product_attention(
        q_ref,
        k_ref,
        v_ref,
        attn_mask=mask,
        scale=softmax_scale,
    )
    expected = rearrange(expected, "b h s d -> (b s) h d")

    torch.testing.assert_close(output, expected)
    fused_attention.assert_not_called()


def test_ascend_attention_unmasked_inputs_keep_fused_path(
    monkeypatch,
    npu_platform,
):
    bsz, seq_len, num_heads, head_dim = 1, 2, 2, 8
    q, k, v = [torch.randn(bsz * seq_len, num_heads, head_dim) for _ in range(3)]
    expected = torch.randn_like(q)
    fused_attention = Mock(return_value=(expected, None))
    monkeypatch.setattr(
        vision,
        "torch_npu",
        SimpleNamespace(npu_fused_infer_attention_score=fused_attention),
        raising=False,
    )
    backend = vision.VisionAscendAttention(
        head_dim=head_dim,
        num_heads=num_heads,
        num_kv_heads=num_heads,
    )
    sdpa_forward = Mock(
        side_effect=AssertionError("unmasked inputs must keep Ascend fused attention")
    )
    monkeypatch.setattr(backend.sdpa_fallback, "forward", sdpa_forward)

    output = backend(
        q=q,
        k=k,
        v=v,
        cu_seqlens=torch.tensor([0, seq_len], dtype=torch.int32),
        bsz=bsz,
        seq_len=seq_len,
    )

    torch.testing.assert_close(output, expected)
    fused_attention.assert_called_once()
    sdpa_forward.assert_not_called()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))


SDPA_NUM_HEADS = 2
SDPA_HEAD_DIM = 8


def _sdpa_qkv(s):
    torch.manual_seed(0)
    return [torch.randn(s, SDPA_NUM_HEADS, SDPA_HEAD_DIM) for _ in range(3)]


def _masked_forward(backend, q, k, v, cu_seqlens):
    """The original path: an explicit attention_mask bypasses the per-sequence split."""
    mask = backend.generate_patch_attention_mask(
        q.shape[0], cu_seqlens, flatten_batch=backend.flatten_batch
    )
    return backend(q=q, k=k, v=v, bsz=1, cu_seqlens=cu_seqlens, attention_mask=mask)


@pytest.mark.parametrize(
    ("flatten_batch", "cu"),
    [
        (False, [0, 12]),  # one image: the mask is all-True
        (True, [0, 12]),
        (True, [0, 5, 12, 20]),  # several images: block-diagonal mask
        (True, [0, 5, 5, 12]),  # zero-length segment
    ],
)
def test_per_sequence_sdpa_matches_masked(flatten_batch, cu):
    backend = vision.VisionSdpaAttention(
        head_dim=SDPA_HEAD_DIM,
        num_heads=SDPA_NUM_HEADS,
        num_kv_heads=SDPA_NUM_HEADS,
        flatten_batch=flatten_batch,
    )
    cu_seqlens = torch.tensor(cu, dtype=torch.int32)
    q, k, v = _sdpa_qkv(cu[-1])

    assert backend._maskless_seq_lens(s=cu[-1], bsz=1, cu_seqlens=cu_seqlens)
    output = backend(q=q, k=k, v=v, bsz=1, cu_seqlens=cu_seqlens)

    torch.testing.assert_close(output, _masked_forward(backend, q, k, v, cu_seqlens))


@pytest.mark.parametrize(
    ("flatten_batch", "cu", "s"),
    [
        (True, [0, 6, 10], 12),  # cu_seqlens does not cover the padded tail
        (False, [0, 6, 12], 12),  # non-flatten masks are padding masks, not blocks
    ],
)
def test_mask_kept_when_not_block_diagonal(flatten_batch, cu, s):
    backend = vision.VisionSdpaAttention(
        head_dim=SDPA_HEAD_DIM,
        num_heads=SDPA_NUM_HEADS,
        num_kv_heads=SDPA_NUM_HEADS,
        flatten_batch=flatten_batch,
    )
    cu_seqlens = torch.tensor(cu, dtype=torch.int32)

    assert backend._maskless_seq_lens(s=s, bsz=1, cu_seqlens=cu_seqlens) is None
