from __future__ import annotations

import sys

import torch
import torch.nn.functional as F


def _is_sm89() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 9)


def _run_case(batch_size: int, max_seq_len: int, seq_len: int) -> None:
    from sglang.srt.layers.attention.dsv4.triton_paged_mqa_logits import (
        FP8_DTYPE,
        fp8_paged_mqa_logits_triton_sm89,
    )

    num_heads, head_dim, block_size = 64, 128, 64
    max_pages = (max_seq_len + block_size - 1) // block_size
    total_pages = batch_size * max_pages
    total_dim = block_size * (head_dim + 4)
    scale_offset = block_size * head_dim
    device = "cuda"

    raw = torch.zeros(total_pages, total_dim, dtype=torch.uint8, device=device)
    kv_source = (
        torch.randn(total_pages, block_size, head_dim, device=device) * 0.1
    ).to(FP8_DTYPE)
    scales = (
        torch.rand(total_pages, block_size, dtype=torch.float32, device=device) * 0.2
        + 0.9
    )
    raw[:, :scale_offset] = kv_source.reshape(total_pages, block_size * head_dim).view(
        torch.uint8
    )
    raw[:, scale_offset:] = (
        scales.reshape(total_pages, block_size)
        .view(torch.uint8)
        .reshape(total_pages, block_size * 4)
    )
    kvcache = raw.view(total_pages, block_size, 1, head_dim + 4)

    page_table = torch.arange(total_pages, dtype=torch.int32, device=device).view(
        batch_size, max_pages
    )
    q = (torch.randn(batch_size, 1, num_heads, head_dim, device=device) * 0.1).to(
        FP8_DTYPE
    )
    weight = torch.randn(batch_size, num_heads, dtype=torch.float32, device=device)
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)

    got = fp8_paged_mqa_logits_triton_sm89(
        q,
        kvcache,
        weight,
        seq_lens,
        page_table,
        deep_gemm_metadata=None,
        max_seq_len=max_seq_len,
        clean_logits=False,
    )
    torch.cuda.synchronize()

    flat = kvcache.view(total_pages, total_dim)
    gathered = flat[page_table]
    kv_val = (
        gathered[..., :scale_offset]
        .contiguous()
        .view(FP8_DTYPE)
        .to(torch.float32)
        .reshape(batch_size, max_pages * block_size, head_dim)
    )
    kv_sc = (
        gathered[..., scale_offset:]
        .contiguous()
        .view(torch.float32)
        .reshape(batch_size, max_pages * block_size)
    )
    q_f32 = q[:, 0].to(torch.float32)
    ref = torch.bmm(kv_val, q_f32.transpose(1, 2))
    ref = F.relu(ref)
    ref = ref * weight.unsqueeze(1)
    ref = ref.sum(dim=2) * kv_sc
    ref_logits = torch.full(
        (batch_size, max_seq_len),
        float("-inf"),
        dtype=torch.float32,
        device=device,
    )
    valid_len = min(seq_len, max_seq_len)
    ref_logits[:, :valid_len] = ref[:, :valid_len]

    finite = torch.isfinite(ref_logits)
    max_diff = (got[finite] - ref_logits[finite]).abs().max().item()
    tail_ok = True
    if valid_len < max_seq_len:
        tail_ok = torch.isneginf(got[:, valid_len:]).all().item()

    print(
        f"case batch={batch_size} max_seq={max_seq_len} seq={seq_len} "
        f"max_diff={max_diff:.6f} tail_ok={tail_ok}"
    )
    assert max_diff < 2e-2
    assert tail_ok


def main() -> int:
    if not _is_sm89():
        print("[skip] SM89/L20 CUDA device is required.")
        return 0

    torch.manual_seed(0)
    for batch_size, max_seq_len, seq_len in (
        (2, 256, 192),
        (4, 512, 448),
        (16, 1024, 960),
    ):
        _run_case(batch_size, max_seq_len, seq_len)
    print("SM89 paged MQA logits correctness OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
