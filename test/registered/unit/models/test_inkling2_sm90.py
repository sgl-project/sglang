"""Hopper expert GEMMs and decode attention, and the MTP verify commit."""

import functools
import sys
import types

import pytest
import torch

from sglang.kernels.ops.attention.inkling2_decode_attention_sm90 import (
    decode_partials,
)
from sglang.kernels.ops.kvcache.mxfp8_quant_store_kv import quant_store_kv_mxfp8
from sglang.srt.models.inkling2_mtp import Inkling2ForConditionalGenerationMTP
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=65, stage="base-b-kernel-unit", runner_config="1-gpu-large")


EXPERTS, LOCAL, TOP_K = 64, 8, 8


def _fake_model(req_pool_size, hidden):
    model = types.SimpleNamespace(
        _f_rnn_buf=torch.full((req_pool_size + 1, hidden), -1.0, device="cuda")
    )

    def store_f_rnn(req_pool_indices, rows):
        model._f_rnn_buf.index_copy_(0, req_pool_indices.view(-1).long(), rows)

    model.store_f_rnn = store_f_rnn
    return model


@pytest.mark.parametrize("valid_mask", [None, "gated"])
def test_store_verify_commit_routes_invalid_rows_to_sink(valid_mask):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    req_pool_size, hidden, stride, bs, real_bs = 16, 8, 4, 6, 3
    model = _fake_model(req_pool_size, hidden)
    verify_hidden = torch.arange(
        bs * stride * hidden, dtype=torch.float32, device="cuda"
    ).view(bs * stride, hidden)
    req_pool_indices = torch.tensor([5, 2, 9, 0, 0, 0], device="cuda")
    commit_lens = torch.tensor([1, 4, 2, 0, 1, 3], dtype=torch.int32, device="cuda")
    valid = torch.arange(bs, device="cuda") < real_bs if valid_mask == "gated" else None
    num_rows = bs if valid is not None else real_bs

    Inkling2ForConditionalGenerationMTP.store_verify_commit(
        model,
        req_pool_indices=req_pool_indices[:num_rows],
        verify_hidden=verify_hidden[: num_rows * stride],
        commit_lens=commit_lens[:num_rows],
        stride=stride,
        valid=valid,
    )

    expected = torch.full((req_pool_size + 1, hidden), -1.0, device="cuda")
    rows = verify_hidden.view(bs, stride, hidden)
    for i in range(real_bs):
        expected[req_pool_indices[i]] = rows[i, commit_lens[i] - 1]
    buf = model._f_rnn_buf
    assert torch.equal(buf[:req_pool_size], expected[:req_pool_size])
    if valid is None:
        assert torch.equal(buf[req_pool_size], expected[req_pool_size])


# A global layer at a TP4 rank of Inkling 2 Small: 2 KV heads, 6 query heads each.
HEADS_K, GROUP, HEAD_DIM, REL_EXTENT, PAGE = 2, 6, 128, 1024, 128


def _random_kv(slots):
    # A random power-of-two magnitude per 32-element block varies every scale byte.
    x = torch.randn(slots, HEADS_K, HEAD_DIM // 32, 32, device="cuda")
    x *= 2.0 ** torch.randint(-2, 3, (*x.shape[:3], 1), device="cuda")
    return x.view(slots, HEADS_K, HEAD_DIM).bfloat16()


def _dequant_mxfp8(data, scale):
    # The pool stores the UE8M0 byte of slot t, elements 32j..32j+31,
    # at scale[page, head, t % 32, t // 32, j].
    pages, page, heads, dim = data.shape
    exp = scale.view(torch.uint8).permute(0, 3, 2, 1, 4)
    exp = exp.reshape(pages, page, heads, dim // 32, 1).float()
    out = data.float().view(pages, page, heads, dim // 32, 32) * torch.exp2(exp - 127)
    return out.view(data.shape).bfloat16()


@pytest.mark.parametrize("qlen", [1, 8])
def test_decode_partials_reads_mxfp8_cache_as_its_bf16_dequant(qlen):
    """The MXFP8 cache reads the same values as the BF16 cache it dequantizes to."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("needs SM90")
    torch.manual_seed(0)
    seq_lens = torch.tensor([2900, 1537, 130], dtype=torch.int32, device="cuda")
    seq_pages = [-(-n // PAGE) for n in seq_lens.tolist()]
    num_pages = 1 + sum(seq_pages)
    # Page 0 is the pool's reserved padding page. 64 columns (8192 keys) put
    # decode on 128-key tiles and verify on 32-key tiles.
    page_table = torch.zeros((len(seq_pages), 64), dtype=torch.int32, device="cuda")
    shuffled = (torch.randperm(num_pages - 1, device="cuda") + 1).int()
    for row, ids in zip(page_table, shuffled.split(seq_pages)):
        row[: len(ids)] = ids
    k8 = torch.zeros(
        (num_pages, PAGE, HEADS_K, HEAD_DIM), dtype=torch.float8_e4m3fn, device="cuda"
    )
    v8 = torch.zeros_like(k8)
    k_scale = torch.zeros(
        (num_pages, HEADS_K, 32, PAGE // 32, HEAD_DIM // 32),
        dtype=torch.float8_e8m0fnu,
        device="cuda",
    )
    v_scale = torch.zeros_like(k_scale)
    quant_store_kv_mxfp8(
        k=_random_kv(num_pages * PAGE),
        v=_random_kv(num_pages * PAGE),
        loc=torch.arange(num_pages * PAGE, device="cuda"),
        k_cache=k8.flatten(0, 1),
        v_cache=v8.flatten(0, 1),
        sfk=k_scale,
        sfv=v_scale,
        page_size=PAGE,
    )
    q = torch.randn(len(seq_pages) * qlen, HEADS_K * GROUP, HEAD_DIM, device="cuda")
    attend = functools.partial(
        decode_partials,
        q=q.bfloat16(),
        page_table=page_table,
        seqused_k=seq_lens,
        rel_bias=torch.randn(*q.shape[:2], REL_EXTENT, device="cuda").bfloat16(),
        softmax_scale=HEAD_DIM**-0.5,
    )
    mxfp8 = attend(k_cache=k8, v_cache=v8, k_scale=k_scale, v_scale=v_scale)
    bf16 = attend(
        k_cache=_dequant_mxfp8(k8, k_scale), v_cache=_dequant_mxfp8(v8, v_scale)
    )
    torch.testing.assert_close(mxfp8, bf16, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
