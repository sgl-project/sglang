# SPDX-License-Identifier: Apache-2.0
"""Replay real DeepSeek-V4 Q8KV8 sparse-prefill kernel inputs dumped from runtime."""

import os
from pathlib import Path

import pytest
import torch

from sglang.jit_kernel.sparse_mla_q8kv8_prefill_sm90 import (
    sparse_mla_q8kv8_prefill_fwd,
)

def _load_replay_payload(path: str) -> dict:
    replay_path = Path(path)
    assert replay_path.exists(), f"Replay dump does not exist: {replay_path}"
    payload = torch.load(replay_path, map_location="cpu", weights_only=False)
    assert isinstance(payload, dict)
    assert "metadata" in payload
    return payload

def _to_cuda_contiguous(payload: dict) -> dict:
    device = torch.device("cuda")

    q = payload["q"].to(device=device).contiguous()
    kv = payload["kv"].to(device=device).contiguous()
    indices = payload["indices"].to(device=device).contiguous()
    topk_length = payload["topk_length"].to(device=device).contiguous()
    attn_sink = payload["attn_sink"].to(device=device).contiguous()
    q_scale = payload["q_scale"].to(device=device).contiguous()
    kv_scale = payload["kv_scale"].to(device=device).contiguous()

    if indices.ndim == 2:
        indices = indices.unsqueeze(1).contiguous()

    return {
        "q": q,
        "kv": kv,
        "indices": indices,
        "topk_length": topk_length,
        "attn_sink": attn_sink,
        "q_scale": q_scale,
        "kv_scale": kv_scale,
    }

def _assert_q8kv8_contract(tensors: dict) -> None:
    q = tensors["q"]
    kv = tensors["kv"]
    indices = tensors["indices"]
    topk_length = tensors["topk_length"]
    attn_sink = tensors["attn_sink"]
    q_scale = tensors["q_scale"]
    kv_scale = tensors["kv_scale"]

    assert q.is_cuda
    assert kv.is_cuda
    assert indices.is_cuda
    assert topk_length.is_cuda
    assert attn_sink.is_cuda
    assert q_scale.is_cuda
    assert kv_scale.is_cuda

    assert q.is_contiguous(), q.stride()
    assert kv.is_contiguous(), kv.stride()
    assert indices.is_contiguous(), indices.stride()
    assert topk_length.is_contiguous(), topk_length.stride()
    assert attn_sink.is_contiguous(), attn_sink.stride()

    assert q.ndim == 3
    assert q.shape[1] in (64, 128)
    assert q.shape[2] == 512

    assert kv.ndim == 3
    assert kv.shape[1] == 1
    assert kv.shape[2] == 512

    assert indices.ndim == 3
    assert indices.shape[0] == q.shape[0]
    assert indices.shape[1] == 1
    assert indices.dtype == torch.int32

    assert topk_length.shape == (q.shape[0],)
    assert topk_length.dtype == torch.int32

    assert attn_sink.shape == (q.shape[1],)
    assert attn_sink.dtype == torch.float32

    assert q_scale.numel() == 1
    assert kv_scale.numel() == 1

    topk_width = indices.shape[-1]
    sentinel = kv.shape[0] - 1

    assert int(topk_length.min().item()) >= 1
    assert int(topk_length.max().item()) <= topk_width
    assert int(indices.min().item()) >= 0
    assert int(indices.max().item()) <= sentinel

    cols = torch.arange(topk_width, device=indices.device).view(1, 1, topk_width)
    valid_mask = cols < topk_length.view(-1, 1, 1)

    valid_vals = indices[valid_mask]
    pad_vals = indices[~valid_mask]

    if valid_vals.numel() > 0:
        assert int(valid_vals.min().item()) >= 0
        assert int(valid_vals.max().item()) < sentinel

    if pad_vals.numel() > 0:
        assert torch.all(pad_vals == sentinel).item()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_q8kv8_sparse_prefill_replay_dump_returns():
    replay_file = os.getenv("SGLANG_DSV4_Q8KV8_REPLAY_FILE")
    assert replay_file, "Set SGLANG_DSV4_Q8KV8_REPLAY_FILE=/path/to/q8kv8_dump.pt"

    payload = _load_replay_payload(replay_file)
    tensors = _to_cuda_contiguous(payload)
    _assert_q8kv8_contract(tensors)

    metadata = payload["metadata"]
    sm_scale = float(metadata["sm_scale"])

    result = sparse_mla_q8kv8_prefill_fwd(
        q=tensors["q"],
        kv=tensors["kv"],
        indices=tensors["indices"],
        attn_sink=tensors["attn_sink"],
        q_scale=tensors["q_scale"],
        kv_scale=tensors["kv_scale"],
        sm_scale=sm_scale,
        d_v=512,
        topk_length=tensors["topk_length"],
    )

    torch.cuda.synchronize()

    assert isinstance(result, tuple)
    assert len(result) >= 1

    out = result[0]
    assert out.shape == (tensors["q"].shape[0], tensors["q"].shape[1], 512)
    assert out.is_cuda
    assert torch.isfinite(out.float()).all()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("topk_width", [128, 640])
@pytest.mark.parametrize("kv_len", [8, 64, 128, 256, 512, 768, 1024, 1202])
def test_q8kv8_sparse_prefill_synthetic_kv_len_sweep(kv_len: int, topk_width: int):
    replay_file = os.getenv("SGLANG_DSV4_Q8KV8_REPLAY_FILE")
    assert replay_file, "Set SGLANG_DSV4_Q8KV8_REPLAY_FILE=/path/to/q8kv8_dump.pt"

    payload = _load_replay_payload(replay_file)
    sm_scale = float(payload["metadata"]["sm_scale"])

    s_q = 6
    h_q = 64
    d = 512
    topk_len_max = min(7, topk_width)

    def make_tensors():
        device = torch.device("cuda")
        torch.manual_seed(1234)

        q = torch.randn(
            (s_q, h_q, d),
            dtype=torch.bfloat16,
            device=device,
        ).to(payload["q"].dtype).contiguous()

        kv = torch.randn(
            (kv_len, 1, d),
            dtype=torch.bfloat16,
            device=device,
        ).to(payload["kv"].dtype).contiguous()

        indices, topk_length = _make_synthetic_indices_like(
            s_q=s_q,
            topk_width=topk_width,
            topk_len_max=topk_len_max,
            kv_len=kv_len,
            device=device,
        )

        return {
            "q": q,
            "kv": kv,
            "indices": indices,
            "topk_length": topk_length,
            "attn_sink": torch.zeros((h_q,), dtype=torch.float32, device=device),
            "q_scale": torch.ones((), dtype=torch.float32, device=device),
            "kv_scale": torch.ones((), dtype=torch.float32, device=device),
        }

    tensors1 = make_tensors()
    tensors2 = make_tensors()

    _assert_q8kv8_contract(tensors1)
    _assert_q8kv8_contract(tensors2)

    out1 = _run_q8kv8_once(tensors1, sm_scale)
    out2 = _run_q8kv8_once(tensors2, sm_scale)

    diff = (out1 - out2).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    mismatch_ratio = float((diff > 1e-2).sum().item() / diff.numel())

    print(
        f"synthetic_kv_len_sweep kv_len={kv_len} topk_width={topk_width} "
        f"max_abs={max_abs} mean_abs={mean_abs} "
        f"mismatch_ratio_gt_1e_2={mismatch_ratio}",
        flush=True,
    )

    assert max_abs <= 1e-2
    assert mismatch_ratio == 0.0

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_q8kv8_sparse_prefill_replay_dump_is_numerically_stable():
    replay_file = os.getenv("SGLANG_DSV4_Q8KV8_REPLAY_FILE")
    assert replay_file, "Set SGLANG_DSV4_Q8KV8_REPLAY_FILE=/path/to/q8kv8_dump.pt"

    payload = _load_replay_payload(replay_file)
    sm_scale = float(payload["metadata"]["sm_scale"])

    tensors1 = _to_cuda_contiguous(payload)
    tensors2 = _to_cuda_contiguous(payload)

    out1 = sparse_mla_q8kv8_prefill_fwd(
        q=tensors1["q"],
        kv=tensors1["kv"],
        indices=tensors1["indices"],
        attn_sink=tensors1["attn_sink"],
        q_scale=tensors1["q_scale"],
        kv_scale=tensors1["kv_scale"],
        sm_scale=sm_scale,
        d_v=512,
        topk_length=tensors1["topk_length"],
    )[0]
    torch.cuda.synchronize()

    out2 = sparse_mla_q8kv8_prefill_fwd(
        q=tensors2["q"],
        kv=tensors2["kv"],
        indices=tensors2["indices"],
        attn_sink=tensors2["attn_sink"],
        q_scale=tensors2["q_scale"],
        kv_scale=tensors2["kv_scale"],
        sm_scale=sm_scale,
        d_v=512,
        topk_length=tensors2["topk_length"],
    )[0]
    torch.cuda.synchronize()

    diff = (out1.float() - out2.float()).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    mismatch_ratio = float((diff > 1e-2).sum().item() / diff.numel())

    print("Q8KV8 numerical stability diagnostics:")
    print(f"  max_abs={max_abs}")
    print(f"  mean_abs={mean_abs}")
    print(f"  mismatch_ratio_gt_1e_2={mismatch_ratio:.6f}")

    assert max_abs <= 1e-2
    assert mismatch_ratio == 0.0