# SPDX-License-Identifier: Apache-2.0
"""fp8_fa_sm120 attention backend against cuDNN SDPA on MiniMax-H3 style inputs.

Needs an SM120 GPU; skipped elsewhere. Shapes are kept small (8 heads) so the
CuTe-DSL compile and the runs stay in seconds.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="fp8_fa_sm120 attention needs an SM120 GPU",
)

HEADS = 8
HEAD_DIM = 128
SOFTMAX_SCALE = HEAD_DIM**-0.5


def _make_impls(causal=False):
    from sglang.multimodal_gen.runtime.layers.attention.backends.fp8_fa_sm120_attn import (
        FP8FlashAttentionSM120Impl,
    )
    from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import (
        CudnnSDPAImpl,
    )

    ours = FP8FlashAttentionSM120Impl(
        num_heads=HEADS, head_size=HEAD_DIM, causal=causal, softmax_scale=SOFTMAX_SCALE
    )
    reference = CudnnSDPAImpl(
        num_heads=HEADS, head_size=HEAD_DIM, causal=causal, softmax_scale=SOFTMAX_SCALE
    )
    return ours, reference


def _fused_qkv_views(sequence, seed=0):
    """Q/K/V as views into one [S, 3*H*D] projection output, like the DiT hands them over."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    qkv = torch.randn(
        (sequence, 3 * HEADS * HEAD_DIM),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    q = qkv[:, 0 : HEADS * HEAD_DIM].view(sequence, HEADS, HEAD_DIM)
    k = qkv[:, HEADS * HEAD_DIM : 2 * HEADS * HEAD_DIM].view(sequence, HEADS, HEAD_DIM)
    v = qkv[:, 2 * HEADS * HEAD_DIM :].view(sequence, HEADS, HEAD_DIM)
    return q, k, v


def _error_metrics(candidate, reference):
    candidate = candidate.float()
    reference = reference.float()
    relative_rms = (candidate - reference).pow(2).mean().sqrt() / reference.pow(
        2
    ).mean().sqrt()
    cosine = torch.nn.functional.cosine_similarity(
        candidate.flatten(), reference.flatten(), dim=0
    )
    return float(relative_rms), float(cosine)


@pytest.mark.parametrize("sequence", [4096, 5980])
def test_forward_matches_cudnn(sequence):
    ours, reference = _make_impls()
    q, k, v = _fused_qkv_views(sequence)

    output = ours.forward(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), None)
    expected = reference.forward(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), None)

    assert output.shape == expected.shape
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output.float()).all()
    relative_rms, cosine = _error_metrics(output, expected)
    # Per-head E4M3 quantization of Q/K/V: ~5% relative RMS on normal inputs.
    assert relative_rms < 0.08, relative_rms
    assert cosine > 0.995, cosine


def _compiled_kernels():
    from sglang.kernels.ops.attention.fp8_fa_sm120.plan import _KERNEL_CACHE

    return len(_KERNEL_CACHE)


def test_kernel_is_reused_across_calls():
    ours, reference = _make_impls()
    q, k, v = _fused_qkv_views(4096, seed=1)
    first = ours.forward(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), None)
    kernels_after_first = _compiled_kernels()

    q2, k2, v2 = _fused_qkv_views(4096, seed=2)
    second = ours.forward(q2.unsqueeze(0), k2.unsqueeze(0), v2.unsqueeze(0), None)
    assert _compiled_kernels() == kernels_after_first

    # A second impl (another DiT layer) shares the same compiled kernel.
    other, _ = _make_impls()
    other.forward(q2.unsqueeze(0), k2.unsqueeze(0), v2.unsqueeze(0), None)
    assert _compiled_kernels() == kernels_after_first

    expected = reference.forward(
        q2.unsqueeze(0), k2.unsqueeze(0), v2.unsqueeze(0), None
    )
    relative_rms, _ = _error_metrics(second, expected)
    assert relative_rms < 0.08, relative_rms
    assert not torch.equal(first, second)


def test_varlen_trailing_padding_keeps_tail_zero():
    ours, _ = _make_impls()
    used = 5980
    total = 6016  # 64-aligned tail padding, as MiniMax-H3 packs it
    q, k, v = _fused_qkv_views(total)
    cu_seqlens = torch.tensor([0, used, total], device="cuda", dtype=torch.int32)

    output = ours.forward_varlen(
        q,
        k,
        v,
        cu_seqlens=cu_seqlens,
        max_seqlen=used,
        cu_seqlens_host=(0, used, total),
    )
    live = ours.forward(
        q[:used].unsqueeze(0), k[:used].unsqueeze(0), v[:used].unsqueeze(0), None
    )[0]

    assert output.shape == q.shape
    assert torch.equal(output[:used], live)
    assert torch.count_nonzero(output[used:]) == 0


def test_varlen_multiple_segments():
    ours, reference = _make_impls()
    bounds = (0, 1024, 3072, 4096)
    q, k, v = _fused_qkv_views(bounds[-1])
    cu_seqlens = torch.tensor(bounds, device="cuda", dtype=torch.int32)

    output = ours.forward_varlen(
        q, k, v, cu_seqlens=cu_seqlens, max_seqlen=2048, cu_seqlens_host=bounds
    )
    expected = reference.forward_varlen(
        q, k, v, cu_seqlens=cu_seqlens, max_seqlen=2048, cu_seqlens_host=bounds
    )

    for start, stop in zip(bounds[:-1], bounds[1:]):
        relative_rms, _ = _error_metrics(output[start:stop], expected[start:stop])
        assert relative_rms < 0.08, (start, stop, relative_rms)


def test_distinct_shapes_release_memory():
    """A call must not retain GPU buffers keyed by shape: each distinct sequence length
    used to keep its FP8/output buffers and the last Q/K/V alive, ~2.2 GiB at H3 size."""
    ours, _ = _make_impls()
    shapes = [_fused_qkv_views(1052), _fused_qkv_views(1088)]
    q, k, v = shapes[0]
    ours.forward(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), None)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()

    for q, k, v in shapes:
        output = ours.forward(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), None)
        assert torch.isfinite(output.float()).all()
        del output

    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == baseline


def test_prep_defines_every_padded_byte():
    """The FP8 buffers come from torch.empty, so the prep must write every padding
    position; an E4M3 NaN byte left in the V^T tail makes masked keys poison the row."""
    from sglang.kernels.ops.attention.fp8_fa_sm120.fused_prep import fused_prepare
    from sglang.kernels.ops.attention.fp8_fa_sm120.plan import _allocate_workspace

    q, k, v = _fused_qkv_views(1052)
    workspace = _allocate_workspace(q)
    for buffer in (workspace.q_fp8, workspace.k_fp8, workspace.v_fp8):
        buffer.view(torch.uint8).fill_(0xFF)

    fused_prepare(
        q=q.permute(1, 0, 2),
        k=k.permute(1, 0, 2),
        v=v.permute(1, 0, 2),
        workspace=workspace,
    )

    for buffer in (workspace.q_fp8, workspace.k_fp8, workspace.v_fp8):
        raw = buffer.view(torch.uint8)
        assert not ((raw == 0xFF) | (raw == 0x7F)).any()


def test_causal_falls_back_to_cudnn():
    ours, reference = _make_impls(causal=True)
    q, k, v = _fused_qkv_views(1024)
    kernels_before = _compiled_kernels()

    output = ours.forward(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), None)
    expected = reference.forward(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), None)

    assert torch.equal(output, expected)
    assert _compiled_kernels() == kernels_before


def test_backend_resolves_by_name():
    from sglang.multimodal_gen.runtime.platforms import (
        AttentionBackendEnum,
        current_platform,
    )

    class_path = current_platform.get_attn_backend_cls_str(
        AttentionBackendEnum.FP8_FA_SM120, HEAD_DIM, torch.bfloat16
    )
    assert class_path.endswith("fp8_fa_sm120_attn.FP8FlashAttentionSM120Backend")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
