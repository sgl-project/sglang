"""Unit tests for MXFP4 KV quantize/dequantize kernels (sm86)."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
from sglang.srt.layers.jit_kernels.mxfp4_kv import (
    dequantize,
    dequantize_indices,
    quantize_and_store,
    reference_quantize,
)


def test_roundtrip():
    torch.manual_seed(42)
    t, h, d = 64, 8, 128
    x = (torch.randn(t, h, d, dtype=torch.bfloat16, device="cuda") * 0.5)
    loc = torch.arange(t, dtype=torch.int32, device="cuda")  # no duplicate slots
    S = 2048
    data = torch.zeros(S, h, d // 2, dtype=torch.uint8, device="cuda")
    scale = torch.zeros(S, h, d // 32, dtype=torch.uint8, device="cuda")
    quantize_and_store(x, loc, data, scale)
    out = torch.zeros(t, h, d, dtype=torch.bfloat16, device="cuda")
    dequantize_indices(data, scale, loc, out)
    xf = x.float()
    of = out.float()
    rel_mse = ((xf - of) ** 2).mean() / (xf**2).mean()
    max_abs = (xf - of).abs().max()
    print(f"[roundtrip] rel_mse={rel_mse:.6f} max_abs_err={max_abs:.4f}")
    assert rel_mse < 0.05, f"rel_mse too high: {rel_mse}"
    return x, loc, data, scale


def test_vs_torch_reference():
    x, loc, data, scale = test_roundtrip()
    packed_ref, scale_ref = reference_quantize(x)
    # gather scattered data back
    gathered = data[loc]  # [T, H, 64]
    gathered_scale = scale[loc]
    assert torch.equal(gathered, packed_ref), "packed data mismatch vs torch ref"
    assert torch.equal(gathered_scale, scale_ref), "scale mismatch vs torch ref"
    print("[vs_torch_ref] packed data + scale bit-exact OK")


def test_dequant_indices():
    torch.manual_seed(7)
    t, h, d = 32, 8, 128
    x = (torch.randn(t, h, d, dtype=torch.bfloat16, device="cuda") * 1.0)
    loc = torch.arange(t, dtype=torch.int32, device="cuda")
    S = 256
    data = torch.zeros(S, h, d // 2, dtype=torch.uint8, device="cuda")
    scale = torch.zeros(S, h, d // 32, dtype=torch.uint8, device="cuda")
    quantize_and_store(x, loc, data, scale)
    # reversed order gather
    idx = torch.arange(t - 1, -1, -1, dtype=torch.int32, device="cuda")
    out = torch.zeros(t, h, d, dtype=torch.bfloat16, device="cuda")
    dequantize_indices(data, scale, idx, out)
    packed_ref, scale_ref = reference_quantize(x)
    # dequant reference (unpack packed fp4 first)
    b, m, n = x.shape
    fp4_vals = torch.empty(b, m, n, dtype=torch.uint8, device="cuda")
    fp4_vals[..., 0::2] = packed_ref & 0x0F
    fp4_vals[..., 1::2] = (packed_ref >> 4) & 0x0F
    vals = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device="cuda")
    mag = fp4_vals & 0x07
    sign = (fp4_vals & 0x08) != 0
    fv = vals[mag.long()]
    fv = torch.where(sign, -fv, fv)
    scale_exp = scale_ref.float() - 127  # E8M0 bits -> exponent
    expected = (fv.view(b, m * n // 32, 32) * torch.exp2(scale_exp.view(b, m * n // 32, 1)))
    expected = expected.view(b, m, n)
    out_r = out.flip(0)
    err = (out_r.float() - expected.float()).abs().max()
    print(f"[dequant_indices] max err vs torch dequant ref: {err:.4f}")
    assert err < 0.01
    print("[dequant_indices] OK")


if __name__ == "__main__":
    test_vs_torch_reference()
    test_dequant_indices()
    print("\nALL TESTS PASSED")
