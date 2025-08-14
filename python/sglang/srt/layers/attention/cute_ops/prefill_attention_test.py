"""Tests for prefill_attention."""

from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F

from prefill_attention import flash_attn_varlen_func


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

    qo_len = q.shape[0]
    kv_len = k.shape[0]
    logits = torch.einsum("qhd,khd->qhk", q, k).to(torch.float32) * softmax_scale
    # print(_yellow(f"logits: {logits.shape=}"), "\n", logits[:, 0, :], flush=True)

    if causal:
        mask = (
            torch.arange(qo_len, dtype=torch.int32, device=logits.device)[:, None]
            >= torch.arange(kv_len, dtype=torch.int32, device=logits.device)[None, :]
        )

        logits = torch.where(
            mask[:, None, :],
            logits,
            torch.tensor(float("-inf"), dtype=torch.float32, device=logits.device),
        )

        # print(_yellow(f"mask: {mask.shape=}"), "\n", mask.to(torch.float32), flush=True)
        # print(_yellow(f"logits: {logits.shape=}"), "\n", logits[:, 0, :], flush=True)

    scores = F.softmax(logits, dim=-1).to(v.dtype)
    # print(_yellow(f"scores: {scores.shape=}"), "\n", scores[:, 0, :], flush=True)

    out = torch.einsum("qhv,vhd->qhd", scores, v)
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
        torch.cumsum(seqlens_q, dim=0, dtype=torch.int32), pad=(1, 0), mode="constant", value=0)
    cu_seqlens_k = F.pad(
        torch.cumsum(seqlens_k, dim=0, dtype=torch.int32), pad=(1, 0), mode="constant", value=0)

    q = torch.empty(
        size=(qo_len, num_qo_heads, head_dim), dtype=dtype, device="cuda"
    ).uniform_(-init_range, init_range)
    k = torch.empty(
        size=(kv_len, num_kv_heads, head_dim), dtype=dtype, device="cuda"
    ).uniform_(-init_range, init_range)
    v = torch.empty(
        size=(kv_len, num_kv_heads, head_dim), dtype=dtype, device="cuda"
    ).uniform_(-init_range, init_range)

    out, lse, *rest = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        causal=causal,
    )

    # out = flash_attn_varlen_func(
    #     q=q,
    #     k=k,
    #     v=v,
    #     cu_seqlens_q=cu_seqlens_q,
    #     cu_seqlens_k=cu_seqlens_k,
    #     softmax_scale=softmax_scale,
    # )
    ref = _ref_impl(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        softmax_scale=softmax_scale,
    )
    diff = (out - ref).abs_().max().item()

    print(_green(f"--> {q.shape=} {k.shape=} {v.shape=}"), f"{ref.shape=}", f"{out.shape=}")
    print(_green("max_diff: "), f"{diff:<.5f}", flush=True)


if __name__ == "__main__":
    test_ragged(
        qo_lens=(8,),
        kv_lens=(8,),
        num_qo_heads=1,
        num_kv_heads=1,
        head_dim=128,
        softmax_scale=None,
    )

    # test_ragged(
    #     qo_lens=(128,),
    #     kv_lens=(1024,),
    #     num_qo_heads=4,
    #     num_kv_heads=4,
    #     head_dim=128,
    #     softmax_scale=None,
    # )

    # test_ragged(
    #     qo_lens=(1024,),
    #     kv_lens=(1024,),
    #     num_qo_heads=8,
    #     num_kv_heads=1,
    #     head_dim=128,
    #     softmax_scale=None,
    # )
