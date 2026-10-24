"""Regression test: MLA chunked-prefix MHA merge must use natural-log LSE.

DeepSeek MHA_CHUNKED_KV merges per-chunk attention with sgl_kernel.merge_state_v2 (natural log).
FlashInfer ragged prefill and trtllm_ragged_attention_deepseek return base-2 LSE, so the MLA
backends must convert exactly once (lse_base2_to_natural). This test pins:
  1. the raw kernel convention (base-2), so a library-side change can't silently double-convert;
  2. that SGLang's merge pattern with the backend conversion matches an FP64 single-pass reference.

Run: pytest -q test/srt/test_mla_chunked_prefix_lse.py
"""

import math

import pytest
import torch

flashinfer = pytest.importorskip("flashinfer")
from sgl_kernel import merge_state_v2  # noqa: E402

from sglang.srt.layers.attention.merge_state import lse_base2_to_natural  # noqa: E402

DEV = "cuda"
H, DQK, DV = 16, 192, 128
SCALE = 1.0 / math.sqrt(DQK)


@pytest.fixture(scope="module")
def ws():
    return torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=DEV)


def _ref(q, k, v, causal_offset=None):
    s = torch.einsum("qhd,khd->hqk", q.double(), k.double()) * SCALE
    if causal_offset is not None:
        nq, nk = q.shape[0], k.shape[0]
        mask = (
            torch.arange(nk, device=DEV)[None]
            > (causal_offset + torch.arange(nq, device=DEV))[:, None]
        )
        s = s.masked_fill(mask[None], -math.inf)
    lse = torch.logsumexp(s, -1).transpose(0, 1)
    return torch.einsum("hqk,khd->qhd", torch.softmax(s, -1), v.double()), lse


def _flashinfer(ws, q, k, v, causal):
    w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, "NHD")
    qi = torch.tensor([0, q.shape[0]], dtype=torch.int32, device=DEV)
    ki = torch.tensor([0, k.shape[0]], dtype=torch.int32, device=DEV)
    w.plan(
        qi,
        ki,
        H,
        H,
        DQK,
        head_dim_vo=DV,
        causal=causal,
        sm_scale=SCALE,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    return w.forward_return_lse(q, k, v, causal=causal, sm_scale=SCALE)


def _trtllm(ws, q, k, v, causal):
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("trtllm ragged MLA kernel needs SM100+")
    nq, nk = q.shape[0], k.shape[0]
    return flashinfer.prefill.trtllm_ragged_attention_deepseek(
        query=q,
        key=k,
        value=v,
        workspace_buffer=ws,
        seq_lens=torch.tensor([nk], dtype=torch.int32, device=DEV),
        max_q_len=nq,
        max_kv_len=nk,
        bmm1_scale=SCALE,
        bmm2_scale=1.0,
        o_sf_scale=-1.0,
        batch_size=1,
        window_left=-1,
        cum_seq_lens_q=torch.tensor([0, nq], dtype=torch.int32, device=DEV),
        cum_seq_lens_kv=torch.tensor([0, nk], dtype=torch.int32, device=DEV),
        enable_pdl=False,
        is_causal=causal,
        return_lse=True,
        out=torch.zeros(nq, H, DV, dtype=torch.bfloat16, device=DEV),
    )


KERNELS = {"flashinfer_ragged": _flashinfer, "trtllm_ragged": _trtllm}


@pytest.mark.parametrize("kernel", KERNELS)
def test_raw_kernel_lse_is_base2(ws, kernel):
    torch.manual_seed(0)
    q = torch.randn(128, H, DQK, device=DEV).bfloat16()
    k = torch.randn(2048, H, DQK, device=DEV).bfloat16()
    v = torch.randn(2048, H, DV, device=DEV).bfloat16()
    _, lse = KERNELS[kernel](ws, q, k, v, False)
    _, lse_ref = _ref(q, k, v)
    torch.testing.assert_close(lse.double() * math.log(2), lse_ref, atol=5e-3, rtol=0)


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("prefix", [4096, 16384])
def test_chunked_prefix_merge_matches_reference(ws, kernel, prefix):
    torch.manual_seed(prefix)
    n_ext, chunk = 256, 4096
    q = torch.randn(n_ext, H, DQK, device=DEV).bfloat16()
    k = torch.randn(prefix + n_ext, H, DQK, device=DEV).bfloat16()
    v = torch.randn(prefix + n_ext, H, DV, device=DEV).bfloat16()
    o_ref, _ = _ref(q, k, v, causal_offset=prefix)
    floor = (o_ref.bfloat16().double() - o_ref).abs().max().item()

    run = KERNELS[kernel]
    # Same order as forward_mha.py: causal extend part first, then fold in prefix chunks.
    acc_o, acc_lse = lse_base2_to_natural(run(ws, q, k[prefix:], v[prefix:], True))
    for c in range(0, prefix, chunk):
        o_c, lse_c = lse_base2_to_natural(
            run(ws, q, k[c : c + chunk], v[c : c + chunk], False)
        )
        tmp_o, tmp_lse = torch.empty_like(acc_o), torch.empty_like(acc_lse)
        merge_state_v2(
            o_c, lse_c.contiguous(), acc_o, acc_lse.contiguous(), tmp_o, tmp_lse
        )
        acc_o, acc_lse = tmp_o, tmp_lse
    err = (acc_o.double() - o_ref).abs().max().item()
    assert err <= 8 * floor, f"merged error {err:.2e} > 8x bf16 floor {floor:.2e}"
