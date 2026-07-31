"""Validate the strided replicated-Q output consumed by the K3 FP8 prologue."""

from __future__ import annotations

import flashinfer
import pytest
import torch
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")

NUM_HEADS = 96
QK_NOPE_DIM = 512
QK_ROPE_DIM = 64


def _inputs(num_tokens: int):
    device = torch.device("cuda")
    generator = torch.Generator(device=device)
    generator.manual_seed(32541 + num_tokens)

    # torch.bmm(q_nope.transpose(0, 1), w_kc) produces [H,T,V].
    q_nope_storage = torch.randn(
        NUM_HEADS,
        num_tokens,
        QK_NOPE_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    q_nope_view = q_nope_storage.transpose(0, 1)
    q_rope = torch.randn(
        num_tokens,
        NUM_HEADS,
        QK_ROPE_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    k_nope = torch.randn(
        num_tokens,
        QK_NOPE_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    k_rope = torch.randn(
        num_tokens,
        QK_ROPE_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    cos_sin_cache = torch.randn(
        2048,
        QK_ROPE_DIM,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    positions = torch.arange(num_tokens, dtype=torch.int32, device=device)
    return (
        q_nope_storage,
        q_nope_view,
        q_rope,
        k_nope,
        k_rope,
        cos_sin_cache,
        positions,
    )


def _pack_query(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    enable_pdl: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_fp8 = torch.empty(
        (*q_nope.shape[:2], QK_NOPE_DIM + QK_ROPE_DIM),
        dtype=torch.float8_e4m3fn,
        device=q_nope.device,
    )
    k_fp8 = torch.empty(
        (q_nope.shape[0], QK_NOPE_DIM + QK_ROPE_DIM),
        dtype=torch.float8_e4m3fn,
        device=q_nope.device,
    )
    flashinfer.rope.mla_rope_quantize_fp8(
        q_rope=q_rope,
        k_rope=k_rope,
        q_nope=q_nope,
        k_nope=k_nope,
        cos_sin_cache=cos_sin_cache,
        pos_ids=positions,
        is_neox=False,
        quantize_dtype=torch.float8_e4m3fn,
        q_rope_out=q_fp8[..., QK_NOPE_DIM:],
        k_rope_out=k_fp8[..., QK_NOPE_DIM:],
        q_nope_out=q_fp8[..., :QK_NOPE_DIM],
        k_nope_out=k_fp8[..., :QK_NOPE_DIM],
        enable_pdl=enable_pdl,
    )
    return q_fp8, k_fp8


@pytest.mark.parametrize("num_tokens", [1, 17, 128])
@torch.inference_mode()
def test_qrep_strided_query_matches_contiguous_fp8_prologue(num_tokens: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required.")
    (
        _,
        q_nope_view,
        q_rope,
        k_nope,
        k_rope,
        cos_sin_cache,
        positions,
    ) = _inputs(num_tokens)
    assert not q_nope_view.is_contiguous() or num_tokens == 1
    assert q_nope_view.stride() == (
        QK_NOPE_DIM,
        num_tokens * QK_NOPE_DIM,
        1,
    )

    expected = _pack_query(
        q_nope_view.contiguous(),
        q_rope,
        k_nope,
        k_rope,
        cos_sin_cache,
        positions,
        enable_pdl=True,
    )
    actual = _pack_query(
        q_nope_view,
        q_rope,
        k_nope,
        k_rope,
        cos_sin_cache,
        positions,
        enable_pdl=True,
    )

    assert torch.equal(expected[0].view(torch.uint8), actual[0].view(torch.uint8))
    assert torch.equal(expected[1].view(torch.uint8), actual[1].view(torch.uint8))


@torch.inference_mode()
def test_qrep_strided_query_fp8_prologue_cuda_graph_replay() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required.")
    (
        q_nope_storage,
        q_nope_view,
        q_rope,
        k_nope,
        k_rope,
        cos_sin_cache,
        positions,
    ) = _inputs(17)

    # Warm all lazy FlashInfer state before capture.
    _pack_query(
        q_nope_view,
        q_rope,
        k_nope,
        k_rope,
        cos_sin_cache,
        positions,
        enable_pdl=True,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = _pack_query(
            q_nope_view,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            positions,
            enable_pdl=True,
        )

    for step in range(3):
        q_nope_storage.add_(step + 1)
        q_rope.add_(step + 1)
        k_nope.add_(step + 1)
        k_rope.add_(step + 1)
        graph.replay()
        expected = _pack_query(
            q_nope_view.contiguous(),
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            positions,
            enable_pdl=True,
        )
        assert torch.equal(
            expected[0].view(torch.uint8), graph_output[0].view(torch.uint8)
        )
        assert torch.equal(
            expected[1].view(torch.uint8), graph_output[1].view(torch.uint8)
        )
