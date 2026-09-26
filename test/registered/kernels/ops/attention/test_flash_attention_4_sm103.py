"""SM103 FA4 tests for the softmax row max from tcgen05.ld.red.

The output and LSE must match the FMNMX path (SGLANG_FA4_TMEM_LOAD_RED_MAX=0)
bit for bit and stay within FA4's usual tolerance of the reference.
"""

import math
import sys

import pytest
import torch

from sglang.kernels.ops.attention.flash_attention import flash_attn_varlen_func
from sglang.kernels.ops.attention.flash_attn.cute.flash_fwd_sm100 import (
    FlashAttentionForwardSm100,
)
from sglang.kernels.ops.attention.flash_attn.cute.testing import (
    attention_ref,
    generate_qkv,
    generate_random_padding_mask,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=150, stage="base-c", runner_config="4-gpu-gb300")

if not (
    torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] == 10
    and torch.cuda.get_device_capability()[1] >= 3
):
    pytest.skip(
        "tcgen05.ld.red is only used on SM 10.3+ (B300 / GB300).",
        allow_module_level=True,
    )

RED_MAX_ENV = "SGLANG_FA4_TMEM_LOAD_RED_MAX"


def _bits(t: torch.Tensor) -> torch.Tensor:
    int_dtype = {1: torch.uint8, 2: torch.int16, 4: torch.int32}[t.element_size()]
    return t.contiguous().view(int_dtype)


def _run_fa4(monkeypatch, red_max: bool, **kwargs):
    # The flag is in the compile key, so both variants coexist.
    monkeypatch.setenv(RED_MAX_ENV, "1" if red_max else "0")
    return flash_attn_varlen_func(**kwargs, return_softmax_lse=True, ver=4)


def _check_red_max(
    monkeypatch,
    *,
    batch_size,
    seqlen_q,
    seqlen_k,
    nheads,
    nheads_kv,
    d,
    dv=None,
    dtype=torch.bfloat16,
    causal=False,
    window_size=(None, None),
    softcap=0.0,
    has_learnable_sink=False,
    pack_gqa=None,
    num_splits=1,
    page_size=None,
):
    dv = dv or d
    device = "cuda"
    local = window_size != (None, None)
    torch.random.manual_seed(
        seqlen_q + seqlen_k + d + dv + int(causal) * 2 + int(local)
    )
    dtype_ref = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
    q_ref = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype_ref)
    if softcap > 0.0:
        # Keep qk within the softcap range.
        q_ref = q_ref * softcap / 4
    q_ref = q_ref.to(dtype).to(dtype_ref)
    k_ref = (
        torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype_ref)
        .to(dtype)
        .to(dtype_ref)
    )
    v_ref = (
        torch.randn(batch_size, seqlen_k, nheads_kv, dv, device=device, dtype=dtype_ref)
        .to(dtype)
        .to(dtype_ref)
    )
    learnable_sink = (
        torch.randn(nheads, dtype=torch.bfloat16, device=device)
        if has_learnable_sink
        else None
    )
    if dtype == torch.float8_e4m3fn:
        q_descale, k_descale, v_descale = [
            torch.rand(batch_size, nheads_kv, device=device, dtype=torch.float32) * 2
            for _ in range(3)
        ]
    else:
        q_descale, k_descale, v_descale = None, None, None
    query_padding_mask = generate_random_padding_mask(
        seqlen_q, batch_size, device, mode="random"
    )
    key_padding_mask = generate_random_padding_mask(
        seqlen_k, batch_size, device, mode="random"
    )
    if (causal or local) and seqlen_q == seqlen_k:
        key_padding_mask = query_padding_mask

    qkv = generate_qkv(q_ref, k_ref, v_ref, query_padding_mask, key_padding_mask)
    q_unpad, k_unpad, v_unpad = [x.detach().to(dtype) for x in qkv[:3]]
    cu_seqlens_q, cu_seqlens_k, _, _, max_seqlen_q, max_seqlen_k = qkv[4:10]
    output_pad_fn = qkv[14]

    kwargs = dict(
        q=q_unpad,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        sinks=learnable_sink,
        pack_gqa=pack_gqa,
        num_splits=num_splits,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    if page_size is None:
        kwargs.update(
            k=k_unpad, v=v_unpad, cu_seqlens_k=cu_seqlens_k, max_seqlen_k=max_seqlen_k
        )
    else:
        # Contiguous pages per sequence.
        num_pages = math.ceil(seqlen_k / page_size)
        k_cache, v_cache = [
            torch.zeros(
                batch_size,
                num_pages * page_size,
                nheads_kv,
                x.shape[-1],
                device=device,
                dtype=dtype,
            )
            for x in (k_ref, v_ref)
        ]
        k_cache[:, :seqlen_k] = k_ref.to(dtype)
        v_cache[:, :seqlen_k] = v_ref.to(dtype)
        kwargs.update(
            k=k_cache.view(batch_size * num_pages, page_size, nheads_kv, d),
            v=v_cache.view(batch_size * num_pages, page_size, nheads_kv, dv),
            cu_seqlens_k=None,
            page_table=torch.arange(
                batch_size * num_pages, dtype=torch.int32, device=device
            ).view(batch_size, num_pages),
            seqused_k=key_padding_mask.sum(-1, dtype=torch.int32),
        )

    out_red, lse_red = _run_fa4(monkeypatch, True, **kwargs)
    out_fmnmx, lse_fmnmx = _run_fa4(monkeypatch, False, **kwargs)
    assert torch.equal(_bits(out_red), _bits(out_fmnmx)), "output differs from FMNMX"
    assert torch.equal(_bits(lse_red), _bits(lse_fmnmx)), "LSE differs from FMNMX"

    ref_kwargs = dict(
        causal=causal,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        window_size=window_size,
        learnable_sink=learnable_sink,
        softcap=softcap,
    )
    out_ref, _ = attention_ref(
        q_ref, k_ref, v_ref, query_padding_mask, key_padding_mask, **ref_kwargs
    )
    out_pt, _ = attention_ref(
        q_ref,
        k_ref,
        v_ref,
        query_padding_mask,
        key_padding_mask,
        upcast=False,
        reorder_ops=True,
        intermediate_dtype=dtype if dtype == torch.float8_e4m3fn else None,
        **ref_kwargs,
    )
    out = output_pad_fn(out_red.to(dtype_ref))
    err = (out - out_ref).abs()
    pt_err = (out_pt - out_ref).abs()
    print(f"Output max diff: {err.max().item()}, mean diff: {err.mean().item()}")
    print(f"Pytorch max diff: {pt_err.max().item()}, mean diff: {pt_err.mean().item()}")
    if dtype == torch.float8_e4m3fn:
        # FA4 fp8 max error is far above the emulated reference (also without
        # ld.red); its mean error is close, so bound the mean.
        assert err.mean().item() <= 3 * pt_err.mean().item()
    else:
        # Same bound as test_flash_attention_4.py.
        fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
        rtol = 2 if softcap == 0.0 else 3
        assert err.max().item() <= rtol * pt_err.max().item() + fwd_atol


def test_red_max_only_where_it_is_exact():
    def uses_red_max(**kwargs):
        return FlashAttentionForwardSm100(head_dim=128, **kwargs).use_tmem_load_red_max

    assert uses_red_max()
    assert not uses_red_max(tmem_load_red_max=False)
    # score_mod / bias modify S, so the max of the raw S is not usable.
    assert not uses_red_max(score_mod=lambda *args: args[0])
    assert not uses_red_max(has_bias=True)


@pytest.mark.parametrize("mha_type", ["mha", "gqa"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [64, 80, 128])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (256, 256),
        (307, 1000),
        (1023, 1024),
        (4096, 4096),
    ],
)
def test_red_max_varlen_output(monkeypatch, seqlen_q, seqlen_k, d, causal, mha_type):
    if causal:
        seqlen_k = seqlen_q
    _check_red_max(
        monkeypatch,
        batch_size=9 if seqlen_q <= 1024 else 2,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        nheads=6,
        nheads_kv=6 if mha_type == "mha" else 2,
        d=d,
        causal=causal,
    )


@pytest.mark.parametrize(
    "case",
    [
        dict(dtype=torch.float16, causal=True, nheads_kv=2),
        dict(dtype=torch.float8_e4m3fn, causal=True),
        dict(has_learnable_sink=True, causal=True),
        dict(window_size=(128, 0)),
        dict(softcap=15.0, causal=True),
        dict(seqlen_q=256, seqlen_k=2048, causal=True),
        dict(seqlen_q=64, seqlen_k=4096, num_splits=3),
        dict(seqlen_q=4, seqlen_k=2000, nheads=32, nheads_kv=4, pack_gqa=True),
        dict(seqlen_q=512, seqlen_k=2048, causal=True, page_size=128),
        dict(seqlen_q=512, seqlen_k=2048, causal=True, page_size=64),
        dict(d=96, causal=True),
        dict(d=192, dv=128, causal=True),
        dict(d=192, dv=128, seqlen_q=2048, seqlen_k=2048, batch_size=2),
    ],
    ids=[
        "fp16",
        "fp8_e4m3",
        "learnable_sink",
        "sliding_window",
        "softcap",
        "chunked_prefill",
        "split_kv",
        "pack_gqa_decode",
        "paged_kv_page128",
        "paged_kv_page64",
        "hdim96",
        "hdim192_128_causal",
        "hdim192_128",
    ],
)
def test_red_max_features(monkeypatch, case):
    kwargs = dict(
        batch_size=5,
        seqlen_q=1024,
        seqlen_k=1024,
        nheads=6,
        nheads_kv=6,
        d=128,
    )
    kwargs.update(case)
    _check_red_max(monkeypatch, **kwargs)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
