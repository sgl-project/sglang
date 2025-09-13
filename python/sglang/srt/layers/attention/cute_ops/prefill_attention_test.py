"""Tests for prefill_attention."""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.cute_ops.prefill_attention import (
    flash_attn_varlen_func,
)


def _green(x: str) -> str:
    return f"\033[1;32m{x}\033[0m"


def _red(x: str) -> str:
    return f"\033[1;31m{x}\033[0m"


def _yellow(x: str) -> str:
    return f"\033[1;33m{x}\033[0m"


torch.set_printoptions(precision=3, sci_mode=False, linewidth=120)

np.set_printoptions(
    suppress=True, precision=3, linewidth=120, formatter={"float": "{:>8.3f}".format}
)


def _ref_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
) -> torch.Tensor:
    head_dim = q.shape[-1]
    softmax_scale = head_dim**-0.5 if softmax_scale is None else softmax_scale

    if k.shape[1] != q.shape[1]:
        k = torch.repeat_interleave(k, repeats=q.shape[1] // k.shape[1], dim=1)

    if v.shape[1] != q.shape[1]:
        v = torch.repeat_interleave(v, repeats=q.shape[1] // v.shape[1], dim=1)

    num_seqs = cu_seqlens_q.shape[0] - 1 if cu_seqlens_q is not None else 1

    out = torch.empty_like(q)

    for i in range(num_seqs):
        if cu_seqlens_q is not None:
            qo_start = cu_seqlens_q[i].item()
            qo_final = cu_seqlens_q[i + 1].item()
        else:
            qo_start = 0
            qo_final = q.shape[0]

        if cu_seqlens_k is not None:
            kv_start = cu_seqlens_k[i].item()
            kv_final = cu_seqlens_k[i + 1].item()
        else:
            kv_start = 0
            kv_final = k.shape[0]

        curr_q = q[qo_start:qo_final, :, :]
        curr_k = k[kv_start:kv_final, :, :]
        curr_v = v[kv_start:kv_final, :, :]

        qo_len = qo_final - qo_start
        kv_len = kv_final - kv_start

        logits = (
            torch.einsum("qhd,khd->qhk", curr_q, curr_k).to(torch.float32)
            * softmax_scale
        )

        if causal:
            mask = (
                torch.arange(
                    kv_len - qo_len, kv_len, dtype=torch.int32, device=logits.device
                )[:, None]
                >= torch.arange(kv_len, dtype=torch.int32, device=logits.device)[
                    None, :
                ]
            )

            logits = torch.where(
                mask[:, None, :],
                logits,
                torch.tensor(float("-inf"), dtype=torch.float32, device=logits.device),
            )

        scores = F.softmax(logits, dim=-1).to(curr_v.dtype)
        out[qo_start:qo_final, :, :] = torch.einsum("qhv,vhd->qhd", scores, curr_v)

    return out


def test_ragged(
    qo_lens: tuple[int, ...],
    kv_lens: tuple[int, ...],
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    init_range: float = 0.5,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 31415,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    qo_len = sum(qo_lens)
    kv_len = sum(kv_lens)
    seqlens_q = torch.tensor(list(qo_lens), dtype=torch.int32, device="cuda")
    seqlens_k = torch.tensor(list(qo_lens), dtype=torch.int32, device="cuda")
    cu_seqlens_q = F.pad(
        torch.cumsum(seqlens_q, dim=0, dtype=torch.int32),
        pad=(1, 0),
        mode="constant",
        value=0,
    )
    cu_seqlens_k = F.pad(
        torch.cumsum(seqlens_k, dim=0, dtype=torch.int32),
        pad=(1, 0),
        mode="constant",
        value=0,
    )

    q = torch.empty(
        size=(qo_len, num_qo_heads, head_dim), dtype=dtype, device="cuda"
    ).uniform_(-init_range, init_range)
    k = torch.empty(
        size=(kv_len, num_kv_heads, head_dim), dtype=dtype, device="cuda"
    ).uniform_(-init_range, init_range)
    v = torch.empty(
        size=(kv_len, num_kv_heads, head_dim), dtype=dtype, device="cuda"
    ).uniform_(-init_range, init_range)

    out = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        causal=causal,
    )[0]

    ref = _ref_impl(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        softmax_scale=softmax_scale,
    )
    diff = (out - ref).abs_().max().item()

    print(_green("max_diff: "), f"{diff:<.5f}", flush=True)


def test_paged(
    qo_lens: tuple[int, ...],
    kv_lens: tuple[int, ...],
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int = 128,
    num_pages: int = 32 * 1024,
    softmax_scale: Optional[float] = None,
    init_range: float = 0.5,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 31415,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    assert len(qo_lens) == len(kv_lens)
    assert all(qo_len <= kv_len for qo_len, kv_len in zip(qo_lens, kv_lens))

    num_seqs = len(qo_lens)
    qo_len = sum(qo_lens)
    max_num_pages = (max(kv_lens) + page_size - 1) // page_size

    seqlens_q = torch.tensor(list(qo_lens), dtype=torch.int32, device="cuda")
    seqlens_k = torch.tensor(list(kv_lens), dtype=torch.int32, device="cuda")
    cu_seqlens_q = F.pad(
        torch.cumsum(seqlens_q, dim=0, dtype=torch.int32),
        pad=(1, 0),
        mode="constant",
        value=0,
    )
    cu_seqlens_k = F.pad(
        torch.cumsum(seqlens_k, dim=0, dtype=torch.int32),
        pad=(1, 0),
        mode="constant",
        value=0,
    )

    q = torch.empty(
        size=(qo_len, num_qo_heads, head_dim), dtype=dtype, device="cuda"
    ).uniform_(-init_range, init_range)
    k_cache = torch.empty(
        size=(num_pages + 1, page_size, num_kv_heads, head_dim),
        dtype=dtype,
        device="cuda",
    ).uniform_(-init_range, init_range)
    v_cache = torch.empty(
        size=(num_pages + 1, page_size, num_kv_heads, head_dim),
        dtype=dtype,
        device="cuda",
    ).uniform_(-init_range, init_range)

    page_table = np.random.randint(
        size=[num_seqs, max_num_pages], low=1, high=num_pages + 1, dtype=np.int32
    )
    page_table = torch.tensor(page_table, dtype=torch.int32, device="cuda")
    page_table = torch.where(
        torch.arange(max_num_pages, dtype=torch.int32, device="cuda")[None, :]
        < (
            torch.tensor(list(kv_lens), dtype=torch.int32, device="cuda")[:, None]
            + page_size
            - 1
        )
        // page_size,
        page_table,
        0,
    )

    out = flash_attn_varlen_func(
        q=q,
        k=k_cache,
        v=v_cache,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=torch.tensor(list(seqlens_k), dtype=torch.int32, device="cuda"),
        page_table=page_table,
        causal=True,
    )[0]

    def _extract_kv(cache: torch.Tensor):
        out = []
        for i, kv_len in enumerate(kv_lens):
            out.append(
                cache[page_table[i], :, :, :].reshape(-1, num_kv_heads, head_dim)[
                    :kv_len
                ]
            )
        return torch.concat(out, axis=0)

    k = _extract_kv(k_cache)
    v = _extract_kv(v_cache)
    ref = _ref_impl(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        softmax_scale=softmax_scale,
        causal=True,
    )

    out = torch.split(out, list(qo_lens), dim=0)
    ref = torch.split(ref, list(qo_lens), dim=0)

    okay = True
    for i in range(len(qo_lens)):
        max_diff = (out[i] - ref[i]).abs_().max().item()
        print(_yellow(f"max_diff_{i}: "), f"{max_diff:<.5f}", flush=True)
        if max_diff > 0.02:
            okay = False

    assert okay


if __name__ == "__main__":
    test_ragged(
        qo_lens=(8,),
        kv_lens=(8,),
        num_qo_heads=1,
        num_kv_heads=1,
        head_dim=128,
        softmax_scale=None,
    )

    test_ragged(
        qo_lens=(11, 12, 32),
        kv_lens=(256, 128, 64),
        num_qo_heads=4,
        num_kv_heads=4,
        head_dim=128,
        softmax_scale=None,
    )

    test_paged(
        qo_lens=(8, 17),
        kv_lens=(11, 19),
        num_qo_heads=2,
        num_kv_heads=2,
        num_pages=32,
        page_size=128,
        head_dim=128,
        softmax_scale=None,
    )

    test_paged(
        qo_lens=(11, 43, 16, 71),
        kv_lens=(31, 70, 31, 81),
        num_qo_heads=16,
        num_kv_heads=2,
        num_pages=32,
        page_size=128,
        head_dim=64,
        softmax_scale=None,
    )
