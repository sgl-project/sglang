"""CUDA vision attention backends on padded image batches.

Idefics2 (MiniCPM-V/o) pads every image to one length and passes the real
lengths in ``cu_seqlens``. Each image must attend only over its own real rows,
the padding rows must come back as exact zeros, and no row may be left
uninitialized.
"""

import itertools
import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")

if not torch.cuda.is_available():
    pytest.skip("requires CUDA", allow_module_level=True)

from sglang.srt.layers.attention import vision  # noqa: E402

HEADS, HEAD_DIM = 16, 72  # SigLIP-400M, the MiniCPM-V vision tower

BACKENDS = [
    pytest.param(vision.VisionTritonAttention, id="triton_attn"),
    pytest.param(
        vision.VisionFlash3Attention,
        id="fa3",
        marks=pytest.mark.skipif(
            torch.cuda.get_device_capability()[0] != 9, reason="FA3 requires sm_90"
        ),
    ),
]


def _qkv(rows):
    return [
        torch.randn(rows, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    ]


def _cu_seqlens(lens):
    return torch.tensor(
        [0, *itertools.accumulate(lens)], dtype=torch.int32, device="cuda"
    )


def _reference(q, k, v):
    return F.scaled_dot_product_attention(
        *(t.transpose(0, 1).float() for t in (q, k, v))
    ).transpose(0, 1)


def _poison_allocator(like):
    # Hand NaN-filled blocks back to the caching allocator so any output row the
    # kernel never writes reads back as NaN.
    junk = [torch.full_like(like, float("nan")) for _ in range(4)]
    torch.cuda.synchronize()
    del junk


def _assert_padded_output(out, q, k, v, lens):
    seq_len = max(lens)
    assert not torch.isnan(out).any()
    for b, n in enumerate(lens):
        rows = slice(b * seq_len, b * seq_len + n)
        torch.testing.assert_close(
            out[rows].float(),
            _reference(q[rows], k[rows], v[rows]),
            atol=2e-2,
            rtol=2e-2,
        )
        assert (out[b * seq_len + n : (b + 1) * seq_len] == 0).all()


def _run_padded(backend, lens):
    torch.manual_seed(0)
    bsz, seq_len = len(lens), max(lens)
    q, k, v = _qkv(bsz * seq_len)
    _poison_allocator(q)
    out = backend().forward(
        q,
        k,
        v,
        cu_seqlens=_cu_seqlens(lens),
        bsz=bsz,
        seq_len=seq_len,
        softmax_scale=HEAD_DIM**-0.5,
    )
    _assert_padded_output(out, q, k, v, lens)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("lens", [[13, 20, 20], [1014, 1036, 1036]])
@pytest.mark.parametrize("use_metadata", [True, False])
def test_padded_batch_attends_within_each_image(backend, lens, use_metadata):
    torch.manual_seed(0)
    bsz, seq_len = len(lens), max(lens)
    q, k, v = _qkv(bsz * seq_len)
    cu_seqlens = _cu_seqlens(lens)
    metadata = (
        vision.prepare_vision_attention_metadata(cu_seqlens, device=q.device)
        if use_metadata
        else None
    )

    _poison_allocator(q)
    # The padded path must not read anything back to the host.
    torch.cuda.set_sync_debug_mode("error")
    try:
        out = backend().forward(
            q,
            k,
            v,
            cu_seqlens=cu_seqlens,
            bsz=bsz,
            seq_len=seq_len,
            softmax_scale=HEAD_DIM**-0.5,
            forward_metadata=metadata,
        )
    finally:
        torch.cuda.set_sync_debug_mode("default")

    _assert_padded_output(out, q, k, v, lens)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "lens, bsz",
    [
        pytest.param([13, 20, 20], 1, id="packed_bsz1"),
        pytest.param([20, 20, 20], 3, id="equal_length_bsz3"),
    ],
)
def test_unpadded_batches_are_unchanged(backend, lens, bsz):
    torch.manual_seed(0)
    total = sum(lens)
    q, k, v = _qkv(total)
    cu_seqlens = _cu_seqlens(lens)

    out = backend().forward(
        q,
        k,
        v,
        cu_seqlens=cu_seqlens,
        bsz=bsz,
        seq_len=total // bsz,
        softmax_scale=HEAD_DIM**-0.5,
        forward_metadata=vision.prepare_vision_attention_metadata(
            cu_seqlens, device=q.device
        ),
    )

    for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
        torch.testing.assert_close(
            out[start:end].float(),
            _reference(q[start:end], k[start:end], v[start:end]),
            atol=2e-2,
            rtol=2e-2,
        )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("use_metadata", [True, False])
def test_uniform_tiles_keep_packed_path(backend, use_metadata, monkeypatch):
    """InternVL passes the cached uniform offsets for its equal-length tiles."""
    torch.manual_seed(0)
    bsz, seq_len = 7, 1025
    q, k, v = _qkv(bsz * seq_len)
    cu_seqlens = vision._get_cu_seqlens_for_shape(bsz, seq_len, device=q.device)
    metadata = (
        vision.prepare_vision_attention_metadata(cu_seqlens, device=q.device)
        if use_metadata
        else None
    )
    assert vision._padded_batch_layout(cu_seqlens, bsz, seq_len, q.device) is None

    def fail(*args, **kwargs):
        raise AssertionError("uniform tiles must not allocate a zeroed output")

    monkeypatch.setattr(vision.torch, "zeros_like", fail)
    out = backend().forward(
        q,
        k,
        v,
        cu_seqlens=cu_seqlens,
        bsz=bsz,
        seq_len=seq_len,
        softmax_scale=HEAD_DIM**-0.5,
        forward_metadata=metadata,
    )
    monkeypatch.undo()

    for b in range(bsz):
        rows = slice(b * seq_len, (b + 1) * seq_len)
        torch.testing.assert_close(
            out[rows].float(),
            _reference(q[rows], k[rows], v[rows]),
            atol=2e-2,
            rtol=2e-2,
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_dense_batch_without_cu_seqlens_skips_padded_path(backend, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("dense batch must not take the padded path")

    monkeypatch.setattr(vision, "_padded_batch_layout", fail)
    torch.manual_seed(0)
    bsz, seq_len = 3, 20
    q, k, v = _qkv(bsz * seq_len)

    out = backend().forward(
        q, k, v, cu_seqlens=None, bsz=bsz, seq_len=seq_len, softmax_scale=HEAD_DIM**-0.5
    )

    for b in range(bsz):
        rows = slice(b * seq_len, (b + 1) * seq_len)
        torch.testing.assert_close(
            out[rows].float(),
            _reference(q[rows], k[rows], v[rows]),
            atol=2e-2,
            rtol=2e-2,
        )


def test_fa3_padded_batch_without_fa3_kernel_uses_triton(monkeypatch):
    """The FA2 fallback ignores seqused_q/k and out, so it must not be used."""

    def fail(*args, **kwargs):
        raise AssertionError("padded batch must not reach the FA2 fallback")

    monkeypatch.setattr(vision, "_is_fa3_supported", lambda: False)
    monkeypatch.setattr(vision, "flash_attn_func", fail)
    _run_padded(vision.VisionFlash3Attention, [1014, 1036, 1036])


@pytest.mark.skipif(
    torch.cuda.get_device_capability()[0] != 9, reason="FA3 requires sm_90"
)
def test_fa3_padded_batch_zeroes_padding_when_kernel_drops_out(monkeypatch):
    """A kernel without `out` allocates its own output; padding must still be 0."""
    flash_attn_func = vision.flash_attn_func
    calls = []

    def without_out(*args, out=None, **kwargs):
        result = flash_attn_func(*args, **kwargs)
        assert result.data_ptr() != out.data_ptr()
        # Stand in for uninitialized memory in the rows the kernel skipped.
        seq_len = kwargs["max_seqlen_q"]
        pad = torch.arange(seq_len, device=result.device) >= kwargs[
            "seqused_q"
        ].unsqueeze(1)
        result.unflatten(0, (-1, seq_len)).masked_fill_(
            pad[:, :, None, None], float("nan")
        )
        calls.append(True)
        return result

    monkeypatch.setattr(vision, "flash_attn_func", without_out)
    _run_padded(vision.VisionFlash3Attention, [1014, 1036, 1036])
    assert calls


def test_fa3_padded_batch_without_fa3_kernel_rejects_window(monkeypatch):
    """The Triton path has no window or sinks, so it must not silently drop them."""
    monkeypatch.setattr(vision, "_is_fa3_supported", lambda: False)
    lens = [13, 20, 20]
    bsz, seq_len = len(lens), max(lens)
    q, k, v = _qkv(bsz * seq_len)

    with pytest.raises(NotImplementedError):
        vision.VisionFlash3Attention().forward(
            q,
            k,
            v,
            cu_seqlens=_cu_seqlens(lens),
            bsz=bsz,
            seq_len=seq_len,
            softmax_scale=HEAD_DIM**-0.5,
            window_size=(8, 8),
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
