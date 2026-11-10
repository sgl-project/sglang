"""Tiny fake-table Engram gather + gate on SM70 (fp16, not bf16).

Host gather DoD: pin a few KB of CPU memory, pass data_ptr into engram_gather,
compare to torch dequant of the same rows. Never allocates the real 189 GiB table.
"""

from __future__ import annotations

import mmap
import unittest

import torch

from sglang.kernels.ops.embeddings.engram_gate import fused_engram_gate
from sglang.kernels.ops.embeddings.engram_gather import engram_gather
from sglang.srt.layers.attention.dsv4.torch_quant import FP8_BLOCK_SIZE
from sglang.srt.layers.engram import engram_gate, engram_lookup_dtype
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

torch.set_num_threads(2)

ROWS = 64
DIM = 32
BLK = FP8_BLOCK_SIZE
_CUDA = torch.cuda.is_available() and torch.version.cuda is not None


def _table(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    weight = (torch.randn(ROWS, DIM, generator=g) * 2).to(torch.float8_e4m3fn)
    scale = torch.randint(
        100, 140, (ROWS, DIM // BLK), dtype=torch.uint8, generator=g
    ).view(torch.float8_e8m0fnu)
    # Row 0: 1.0 * e8m0-zero (2**-127), flushes to 0 in fp16.
    weight.view(torch.uint8)[0, 0] = (
        torch.tensor(1.0).to(torch.float8_e4m3fn).view(torch.uint8).item()
    )
    scale.view(torch.uint8)[0, 0] = 0
    # Row 1: e4m3fn denorms (exp=0, man=1..7) with scale 1.0 (e8m0 bias 127).
    weight.view(torch.uint8)[1, :8] = torch.arange(8, dtype=torch.uint8)
    scale.view(torch.uint8)[1, 0] = 127
    return weight, scale


def _ref_dequant(weight, scale, ids, dtype):
    ids = ids.detach().cpu()
    rows = weight.float().cpu()[ids].unflatten(-1, (-1, BLK))
    return (rows * scale.float().cpu()[ids].unsqueeze(-1)).flatten(-2).to(dtype)


def _gate_ref(x, kv, qw, kw, eps, clamp):
    hc, dim = x.shape[-2:]
    key, value = kv.split([hc * dim, dim], dim=-1)
    h, key, value = x.float(), key.float().reshape_as(x), value.float()
    rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(
        key.square().mean(-1) + eps
    )
    dot = (h * (qw.float() * kw.float()) * key).sum(-1) * rstd * dim**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(clamp).sqrt(), dot))
    return (h + gate.unsqueeze(-1) * value.unsqueeze(-2)).to(x.dtype)


@unittest.skipUnless(_CUDA, "engram_gather needs CUDA (use CUDA_VISIBLE_DEVICES=4)")
class TestDsv41EngramGather(CustomTestCase):
    def setUp(self):
        self.dtype = engram_lookup_dtype("cuda")
        if torch.cuda.get_device_capability()[0] == 7:
            self.assertEqual(self.dtype, torch.float16, "SM70 gather must be fp16")
        else:
            self.assertEqual(self.dtype, torch.bfloat16, "Hopper gather stays bf16")

    def test_device_gather_matches_torch_dequant(self):
        weight, scale = _table()
        weight, scale = weight.cuda(), scale.cuda()
        ids = torch.tensor([0, 1, 5, 63, 7], device="cuda", dtype=torch.int64)
        out = torch.empty(ids.numel(), DIM, dtype=self.dtype, device="cuda")
        engram_gather(
            weight.data_ptr(),
            scale.data_ptr(),
            ids,
            out,
            DIM,
            BLK,
        )
        torch.cuda.synchronize()
        want = _ref_dequant(weight, scale, ids, self.dtype)
        torch.testing.assert_close(out.cpu().float(), want.float(), rtol=0, atol=0)
        self.assertEqual(out.dtype, self.dtype)
        # e8m0 byte 0 is 2**-127, below fp16 range; the kernel matches torch's flush.
        if self.dtype == torch.float16:
            self.assertEqual(out[0, 0].item(), 0.0)
        else:
            self.assertEqual(out[0, 0].item(), 2.0**-127)
        # Row 1 is denorm e4m3 with scale 1.0; those values survive fp16.
        self.assertTrue(torch.any(out[1, 1:8].float() != 0))

    def test_unowned_ids_are_zero(self):
        weight, scale = _table()
        weight, scale = weight.cuda(), scale.cuda()
        ids = torch.tensor([0, 100, 3], device="cuda", dtype=torch.int64)
        out = torch.empty(ids.numel(), DIM, dtype=self.dtype, device="cuda")
        engram_gather(
            weight.data_ptr(),
            scale.data_ptr(),
            ids,
            out,
            DIM,
            BLK,
            row_lo=0,
            row_hi=ROWS,
        )
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(out[1].cpu(), torch.zeros(DIM, dtype=self.dtype)))

    def test_host_pin_memory_pointer(self):
        weight, scale = _table()
        host_w = torch.empty(weight.shape, dtype=weight.dtype, pin_memory=True)
        host_s = torch.empty(scale.shape, dtype=scale.dtype, pin_memory=True)
        host_w.copy_(weight)
        host_s.copy_(scale)
        ids = torch.tensor([0, 2, 9, 63], device="cuda", dtype=torch.int64)
        out = torch.empty(ids.numel(), DIM, dtype=self.dtype, device="cuda")
        engram_gather(
            host_w.data_ptr(),
            host_s.data_ptr(),
            ids,
            out,
            DIM,
            BLK,
        )
        torch.cuda.synchronize()
        want = _ref_dequant(weight, scale, ids, self.dtype)
        torch.testing.assert_close(out.cpu().float(), want.float(), rtol=0, atol=0)
        self.assertLess(host_w.nbytes + host_s.nbytes, 64 * 1024)

    def test_host_mmap_mapped_pointer(self):
        weight, scale = _table()
        mm_w = mmap.mmap(-1, weight.nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        mm_s = mmap.mmap(-1, scale.nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        tw = torch.frombuffer(mm_w, dtype=torch.uint8)
        ts = torch.frombuffer(mm_s, dtype=torch.uint8)
        tw.copy_(weight.view(torch.uint8).reshape(-1))
        ts.copy_(scale.view(torch.uint8).reshape(-1))
        cudart = torch.cuda.cudart()
        self.assertEqual(int(cudart.cudaHostRegister(tw.data_ptr(), weight.nbytes, 2)), 0)
        self.assertEqual(int(cudart.cudaHostRegister(ts.data_ptr(), scale.nbytes, 2)), 0)
        try:
            ids = torch.tensor([0, 4, 15, 32], device="cuda", dtype=torch.int64)
            out = torch.empty(ids.numel(), DIM, dtype=self.dtype, device="cuda")
            engram_gather(
                tw.data_ptr(),
                ts.data_ptr(),
                ids,
                out,
                DIM,
                BLK,
            )
            torch.cuda.synchronize()
            want = _ref_dequant(weight, scale, ids, self.dtype)
            torch.testing.assert_close(out.cpu().float(), want.float(), rtol=0, atol=0)
        finally:
            cudart.cudaHostUnregister(tw.data_ptr())
            cudart.cudaHostUnregister(ts.data_ptr())


class TestDsv41EngramGate(CustomTestCase):
    def test_cpu_fp16_matches_torch(self):
        torch.manual_seed(3)
        x = torch.randn(4, 2, 32, dtype=torch.float16)
        kv = torch.randn(4, 3 * 32, dtype=torch.float16)
        qw = torch.randn(2, 32, dtype=torch.float16)
        kw = torch.randn_like(qw)
        got = engram_gate(x, kv, qw, kw, 1e-6, 1e-6)
        self.assertEqual(got.dtype, torch.float16)
        torch.testing.assert_close(
            got, _gate_ref(x, kv, qw, kw, 1e-6, 1e-6), rtol=8e-3, atol=1e-2
        )

    @unittest.skipUnless(_CUDA, "fused gate needs CUDA (use CUDA_VISIBLE_DEVICES=4)")
    def test_fused_fp16_matches_torch(self):
        torch.manual_seed(23)
        x = torch.randn(8, 4, 32, device="cuda", dtype=torch.float16)
        kv = torch.randn(8, 5 * 32, device="cuda", dtype=torch.float16)
        qw = torch.randn(4, 32, device="cuda", dtype=torch.float16)
        kw = torch.randn_like(qw)
        got = fused_engram_gate(x, kv, qw, kw, 1e-6, 1e-6)
        self.assertEqual(got.dtype, torch.float16)
        torch.testing.assert_close(
            got, _gate_ref(x, kv, qw, kw, 1e-6, 1e-6), rtol=8e-3, atol=1e-2
        )
        layer = engram_gate(x, kv, qw, kw, 1e-6, 1e-6)
        torch.testing.assert_close(layer, got, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
