# Adapted from https://github.com/flashinfer-ai/flashinfer/blob/55576c626421b5ee7e7ebe74afd26465c8ae863f/flashinfer/triton/kernels/cascade.py

from typing import List

import pytest
import torch
import triton
import triton.language as tl
from sgl_kernel import merge_state


def check_input(x: torch.Tensor):
    assert x.is_cuda, f"{str(x)} must be a CUDA Tensor"
    assert x.is_contiguous(), f"{str(x)} must be contiguous"


def check_dim(d, x: torch.Tensor):
    assert x.dim() == d, f"{str(x)} must be a {d}D tensor"


def check_shape(a: torch.Tensor, b: torch.Tensor):
    assert a.dim() == b.dim(), "tensors should have same dim"
    for i in range(a.dim()):
        assert a.size(i) == b.size(
            i
        ), f"tensors shape mismatch, {a.size()} and {b.size()}"


def check_device(tensors: List[torch.Tensor]):
    device = tensors[0].device
    for t in tensors:
        assert (
            t.device == device
        ), f"All tensors should be on the same device, but got {device} and {t.device}"


@triton.jit
def state_merge(o, m, d, other_o, other_m, other_d):
    m_max = tl.maximum(m, other_m)
    d = d * tl.exp2(m - m_max) + other_d * tl.exp2(other_m - m_max)
    o = o * tl.exp2(m - m_max) + other_o * tl.exp2(other_m - m_max)
    return o, m_max, d


@triton.jit
def state_normalize(o, m, d):
    o = o / d
    return o, m, d


@triton.jit
def state_get_lse(o, m, d):
    return m + tl.log2(d)


@triton.jit
def merge_state_kernel(
    v_a_ptr,
    s_a_ptr,
    v_b_ptr,
    s_b_ptr,
    v_merged_ptr,
    s_merged_ptr,
    num_heads,
    head_dim,
    bdx: tl.constexpr,
    bdy: tl.constexpr,
):
    pos = tl.program_id(axis=0)
    for tx in tl.range(bdx):
        for head_idx in tl.range(bdy):
            s_a_val = tl.load(s_a_ptr + pos * num_heads + head_idx)
            s_b_val = tl.load(s_b_ptr + pos * num_heads + head_idx)

            offsets = (pos * num_heads + head_idx) * head_dim + tx
            v_a = tl.load(v_a_ptr + offsets)
            v_b = tl.load(v_b_ptr + offsets)

            v_merged, s_max, d = state_merge(
                o=v_a, m=s_a_val, d=1, other_o=v_b, other_m=s_b_val, other_d=1
            )
            v_merged, s_max, d = state_normalize(v_merged, s_max, d)
            v_merged_offset = (pos * num_heads + head_idx) * head_dim + tx
            tl.store(v_merged_ptr + v_merged_offset, v_merged)

            if s_merged_ptr:
                tl.store(
                    s_merged_ptr + pos * num_heads + head_idx,
                    tl.log2(d) + s_max,
                )


def merge_state_triton(
    v_a: torch.Tensor, s_a: torch.Tensor, v_b: torch.Tensor, s_b: torch.Tensor
):
    check_input(v_a)
    check_input(s_a)
    check_input(v_b)
    check_input(s_b)
    check_device([v_a, s_a, v_b, s_b])
    check_dim(3, v_a)
    check_dim(2, s_a)
    check_dim(3, v_b)
    check_dim(2, s_b)
    check_shape(v_a, v_b)
    check_shape(s_a, s_b)
    assert v_a.size(0) == s_a.size(0)
    assert v_a.size(1) == s_b.size(1)
    s_a = s_a.to(torch.float32)
    s_b = s_b.to(torch.float32)
    seq_len = v_a.size(0)
    num_heads = v_a.size(1)
    head_dim = v_a.size(2)
    v_merged = torch.empty_like(v_a).to(s_a.device)
    s_merged = torch.empty((seq_len, num_heads)).to(s_a.device)
    bdx = head_dim
    bdy = num_heads

    merge_state_kernel[lambda meta: (seq_len,)](
        v_a, s_a, v_b, s_b, v_merged, s_merged, num_heads, head_dim, bdx=bdx, bdy=bdy
    )

    return v_merged, s_merged


@pytest.mark.parametrize("seq_len", [2048])
@pytest.mark.parametrize("num_heads", [32])
@pytest.mark.parametrize("head_dim", [128])
def test_merge_state(seq_len, num_heads, head_dim):
    va = torch.randn(seq_len, num_heads, head_dim).half().to("cuda:0")
    sa = torch.randn(seq_len, num_heads, dtype=torch.float32).to("cuda:0")
    vb = torch.randn(seq_len, num_heads, head_dim).half().to("cuda:0")
    sb = torch.randn(seq_len, num_heads, dtype=torch.float32).to("cuda:0")
    v_merged, s_merged = merge_state_triton(va, sa, vb, sb)
    v_merged_std, s_merged_std = merge_state(va, sa, vb, sb)

    assert torch.allclose(v_merged, v_merged_std, atol=1e-2)
    assert torch.allclose(s_merged, s_merged_std, atol=1e-2)
