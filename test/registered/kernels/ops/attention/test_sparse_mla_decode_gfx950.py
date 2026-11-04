"""gfx950 MXFP8 sparse MLA decode over fp8_ds_mla rows: output and LSE vs fp32 reference.

The reference uses the same MXFP8-quantized Q and dequantized KV; cases cover ragged
and empty per-row lengths (a DCP rank's top-k share) and every split count."""

import sys
from types import SimpleNamespace

import pytest
import torch

import sglang.kernels.ops.attention.dsa.sparse_mla_decode_gfx950 as mla_gfx950
from sglang.kernels.ops.quantization.dsa_quant_k_cache import quantize_k_cache
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci

# backend-specific: the gfx950 MXFP8 decode kernel (ported from aiter's opus), no CUDA counterpart.
register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

_DEVICE = "cuda"
_DV, _DR, _TOPK, _N = 512, 64, 2048, 16384


def _kv():
    torch.manual_seed(0)
    k = (torch.randn(_N, 1, _DV + _DR, device=_DEVICE) * 2).bfloat16()
    kv = quantize_k_cache(k.unsqueeze(1), pow2_scale=True).view(_N, 1, -1)
    raw = kv.view(torch.uint8).view(_N, 656)
    nope = raw[:, :512].view(torch.float8_e4m3fn).float().view(_N, 4, 128)
    scale = raw[:, 512:528].contiguous().view(torch.float32).view(_N, 4, 1)
    rope = raw[:, 528:].contiguous().view(torch.bfloat16).float()
    return kv, torch.cat([(nope * scale).view(_N, 512), rope], -1)


def _reference(qn, qr, idx, lens, kd, sm, mxfp8_q=True):
    R, H, _ = qn.shape
    if not mxfp8_q:
        return _attend(torch.cat([qn.float(), qr.float()], -1), idx, lens, kd, sm)
    q8 = torch.empty(R, H, 512, device=_DEVICE, dtype=torch.float8_e4m3fn)
    qs = torch.empty(R, H, 16, device=_DEVICE, dtype=torch.uint8)
    qrc = torch.empty(R, H, 64, device=_DEVICE, dtype=torch.bfloat16)
    mla_gfx950._mxfp8_quant_kernel[(R, H // 16)](
        qn,
        q8,
        qs,
        qr,
        qrc,
        qn.stride(0),
        qn.stride(1),
        qr.stride(0),
        qr.stride(1),
        H,
        HB=16,
    )
    scale = torch.exp2(qs.float() - 127).view(R, H, 16, 1)
    qd = torch.cat(
        [(q8.float().view(R, H, 16, 32) * scale).view(R, H, 512), qr.float()], -1
    )
    return _attend(qd, idx, lens, kd, sm)


def _attend(qd, idx, lens, kd, sm):
    R, H, _ = qd.shape
    o = torch.zeros(R, H, 512, device=_DEVICE)
    lse = torch.full((R, H), -float("inf"), device=_DEVICE)
    for r in range(R):
        n = int(lens[r])
        if n:
            K = kd[idx[r, :n].long()]
            s = qd[r] @ K.T * sm
            lse[r] = torch.logsumexp(s, -1)
            o[r] = torch.softmax(s, -1) @ K[:, :512]
    return o, lse


@pytest.mark.skipif(not is_gfx95_supported(), reason="gfx950 kernel")
@pytest.mark.parametrize("H", [16, 96, 128])
@pytest.mark.parametrize(
    "lens", [[256] * 8, [256, 0, 1, 63, 65, 2048, 300, 200], [100] * 10, [40] * 33]
)
@pytest.mark.parametrize("splits", [None, 1, 2, 8])
def test_matches_reference(H, lens, splits):
    kv, kd = _kv()
    R = len(lens)
    qkv = torch.randn(R, H, _DV + _DR, device=_DEVICE).bfloat16()
    qn, qr = qkv[..., :_DV], qkv[..., _DV:]
    lengths = torch.tensor(lens, dtype=torch.int32, device=_DEVICE)
    idx = torch.full((R, _TOPK), -1, dtype=torch.int32, device=_DEVICE)
    for r, n in enumerate(lens):
        idx[r, :n] = torch.randperm(_N, device=_DEVICE)[:n].int()
    sm = (_DV + _DR) ** -0.5
    o, lse = mla_gfx950.sparse_mla_decode_gfx950(
        qn, qr, kv, idx.unsqueeze(1), sm, topk_length=lengths, num_splits=splits
    )
    o_ref, lse_ref = _reference(qn, qr, idx, lengths, kd, sm)
    finite = torch.isfinite(lse_ref)
    assert torch.equal(torch.isfinite(lse), finite)
    torch.testing.assert_close(lse[finite], lse_ref[finite], rtol=0, atol=1e-3)
    torch.testing.assert_close(o[0].float(), o_ref, rtol=0, atol=3e-2)


@pytest.mark.skipif(not is_gfx95_supported(), reason="gfx950 kernel")
def test_decode_without_row_lengths_skips_the_padding():
    """Decode without DCP hands the backend -1-padded top-k rows and no lengths."""
    kv, kd = _kv()
    lens = [_TOPK, 300, 1, 1024]
    R, H = len(lens), 96
    qkv = torch.randn(R, H, _DV + _DR, device=_DEVICE).bfloat16()
    qn, qr = qkv[..., :_DV], qkv[..., _DV:]
    idx = torch.full((R, _TOPK), -1, dtype=torch.int32, device=_DEVICE)
    for r, n in enumerate(lens):
        idx[r, :n] = torch.randperm(_N, device=_DEVICE)[:n].int()
    sm = (_DV + _DR) ** -0.5
    backend = SimpleNamespace(_triton_sparse_mla_workspaces={}, attn_dcp_size=1)
    o = DeepseekSparseAttnBackend._forward_triton_decode(
        backend, qn, qr, kv, _DV, idx, sm
    )
    lengths = torch.tensor(lens, dtype=torch.int32, device=_DEVICE)
    # the Triton split-K kernel keeps q in BF16
    o_ref, _ = _reference(qn, qr, idx, lengths, kd, sm, mxfp8_q=False)
    torch.testing.assert_close(o.float(), o_ref, rtol=0, atol=3e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
