# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Bit-exact unit test for the fused DCP verify mask+pack kernel.

``_forward_verify_dcp`` (trtllm_mla_backend) used to mask pass-1 with two
``torch.where``, ``.contiguous()`` the outputs, and let ``dcp_a2a_lse_reduce``
copy them into its packed a2a send buffer (~6 elementwise/copy kernels per
layer, 61 layers/step). The fused path
(``SGLANG_DCP_FUSED_PACK=1``, default) collapses all of that into ONE Triton
kernel (``kernels.dcp_mask_pack_triton``) and exchanges the pre-packed buffer
via ``comm.dcp_a2a_exchange_packed`` + ``comm.dcp_unpack_lse_combine``.

Tests (dtype grid bf16/fp16/fp32 — the verify path emits bf16; fp8 skipped as
CPU torch.where/copy support is incomplete):
  1. CPU: the torch reference pack (verbatim OLD-path ops) round-trips, and the
     fp32-alias slot math the Triton wrapper uses ((D*itemsize)//4) addresses
     the packed LSE.
  2. GPU: dcp_mask_pack_triton is BIT-identical to the CPU reference for
     random/none/all masks (masked rows poisoned with nan/inf) and for
     non-contiguous o1/lse1.
  3. GPU: full-pipeline equivalence on a simulated 4-rank a2a — per-rank packed
     buffers and the final combined (out, lse) match the OLD path
     (real ``dcp_a2a_lse_reduce``) bit-for-bit.

Usage:
    python -m pytest test_dcp_mask_pack.py -v
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="extra-a", runner_config="1-gpu-large")


def _lse_pack_dim(dtype: torch.dtype) -> int:
    # Mirror of sglang.kernels.ops.attention.dcp_kernels._lse_pack_dim, kept local so the
    # CPU-reference tests run without importing triton.
    return torch.finfo(torch.float32).bits // torch.finfo(dtype).bits


def _mask_pack_ref(
    o1: torch.Tensor,
    lse1: torch.Tensor,
    zero_mask: torch.Tensor,
    cp_world: int,
) -> torch.Tensor:
    """Torch reference: the OLD-path ops, verbatim.

    The ``torch.where`` pair from ``_forward_verify_dcp`` followed by the eager
    pack of ``comm.dcp_a2a_lse_reduce`` (``send_combined`` layout
    [N, B, H_per_rank, D+lpd] with the fp32 LSE reinterpreted as out-dtype
    elements in the [D:] tail).
    """
    bs, T, H_full, D = o1.shape
    o1m = torch.where(zero_mask.view(bs, 1, 1, 1), o1.new_zeros(()), o1)
    lse1m = torch.where(
        zero_mask.view(bs, 1, 1), lse1.new_full((), float("-inf")), lse1
    )
    B = bs * T
    out2d = o1m.reshape(B, H_full, D).contiguous()
    lse2d = lse1m.reshape(B, H_full).contiguous()
    N = cp_world
    H_pr = H_full // N
    lpd = _lse_pack_dim(o1.dtype)
    reshaped_out = out2d.view(B, N, H_pr, D).permute(1, 0, 2, 3)
    send_lse_contig = lse2d.view(B, N, H_pr).permute(1, 0, 2).contiguous()
    send = torch.empty(N, B, H_pr, D + lpd, dtype=o1.dtype, device=o1.device)
    send[:, :, :, :D].copy_(reshaped_out)
    send[:, :, :, D:].copy_(send_lse_contig.view(o1.dtype).view(N, B, H_pr, lpd))
    return send


def _bit_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Byte-level equality (NaN-safe, unlike torch.equal)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return torch.equal(
        a.contiguous().reshape(-1).view(torch.uint8),
        b.contiguous().reshape(-1).view(torch.uint8),
    )


def _masks(bs: int, device, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    rand = (torch.rand(bs, generator=g) < 0.4).to(device)
    return {
        "none_masked": torch.zeros(bs, dtype=torch.bool, device=device),
        "all_masked": torch.ones(bs, dtype=torch.bool, device=device),
        "random": rand,
    }


def _poison_masked_rows(o1, lse1, zero_mask):
    """Fill masked batch rows with nan/inf garbage: both paths must replace
    them (out -> 0, lse -> -inf), so garbage leaking through fails the test."""
    o1[zero_mask] = float("nan")
    lse1[zero_mask] = float("inf")


class TestMaskPackRefLayout(unittest.TestCase):
    """CPU-only: the reference pack layout is self-consistent."""

    BS, T, H, D, N = 3, 2, 8, 16, 4

    def _expected(self, o1, lse1, zero, dtype):
        bs, T, H, D, N = self.BS, self.T, self.H, self.D, self.N
        B, H_pr = bs * T, H // N
        exp_out = (
            torch.where(zero.view(bs, 1, 1, 1), o1.new_zeros(()), o1)
            .reshape(B, N, H_pr, D)
            .permute(1, 0, 2, 3)
        )
        exp_lse = (
            torch.where(zero.view(bs, 1, 1), lse1.new_full((), float("-inf")), lse1)
            .reshape(B, N, H_pr)
            .permute(1, 0, 2)
        )
        return exp_out, exp_lse

    def test_ref_pack_roundtrip(self):
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            bs, T, H, D, N = self.BS, self.T, self.H, self.D, self.N
            o1 = torch.randn(bs, T, H, D).to(dtype)
            lse1 = torch.randn(bs, T, H) * 4
            zero = torch.tensor([True, False, True])
            send = _mask_pack_ref(o1, lse1, zero, N)
            exp_out, exp_lse = self._expected(o1, lse1, zero, dtype)
            self.assertTrue(torch.equal(send[..., :D], exp_out), f"out {dtype}")
            unpacked_lse = send[..., D:].contiguous().view(torch.float32).squeeze(-1)
            self.assertTrue(torch.equal(unpacked_lse, exp_lse), f"lse {dtype}")

    def test_f32_alias_slot_math(self):
        """The Triton wrapper (and dcp_unpack_lse_combine) address the packed
        LSE through send.view(float32)[..., (D*itemsize)//4]."""
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            bs, T, H, D, N = self.BS, self.T, self.H, self.D, self.N
            o1 = torch.randn(bs, T, H, D).to(dtype)
            lse1 = torch.randn(bs, T, H) * 4
            zero = torch.tensor([False, True, False])
            send = _mask_pack_ref(o1, lse1, zero, N)
            _, exp_lse = self._expected(o1, lse1, zero, dtype)
            idx = (D * dtype.itemsize) // 4
            self.assertTrue(
                torch.equal(send.view(torch.float32)[..., idx], exp_lse),
                f"alias slot {dtype}",
            )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA (triton kernel)")
class TestMaskPackKernelVsRef(unittest.TestCase):
    """GPU kernel vs CPU reference, bit-exact."""

    CASES = [
        # (bs, T, H_full, D, cp_world, dtype)
        (5, 3, 8, 64, 4, torch.bfloat16),
        (2, 4, 16, 128, 8, torch.bfloat16),
        (3, 2, 8, 64, 4, torch.float16),
        (2, 2, 4, 32, 2, torch.float32),
        (2, 4, 128, 512, 8, torch.bfloat16),  # Kimi-K2 verify shape
    ]

    def _check(self, o1, lse1, zero, cp_world, tag):
        from sglang.srt.layers.dcp import dcp_mask_pack_triton

        send_gpu = dcp_mask_pack_triton(o1, lse1, zero, cp_world)
        ref = _mask_pack_ref(o1.cpu(), lse1.cpu(), zero.cpu(), cp_world)
        self.assertTrue(_bit_equal(send_gpu.cpu(), ref), tag)

    def test_kernel_matches_reference(self):
        torch.manual_seed(7)
        for i, (bs, T, H, D, N, dtype) in enumerate(self.CASES):
            for mask_name, zero in _masks(bs, "cuda", seed=i).items():
                o1 = torch.randn(bs, T, H, D, dtype=dtype, device="cuda")
                lse1 = torch.randn(bs, T, H, dtype=torch.float32, device="cuda") * 4
                _poison_masked_rows(o1, lse1, zero)
                self._check(o1, lse1, zero, N, f"case{i} {dtype} {mask_name}")

    def test_kernel_matches_reference_strided(self):
        """Non-contiguous o1/lse1 (the kernel takes any stride)."""
        torch.manual_seed(11)
        bs, T, H, D, N, dtype = self.CASES[0]
        zero = _masks(bs, "cuda", seed=3)["random"]
        # o1: last-dim slice of a wider buffer (row stride D+32, d stride 1)
        o1 = torch.randn(bs, T, H, D + 32, dtype=dtype, device="cuda")[
            :, :, :, 16 : 16 + D
        ]
        # lse1: [bs, H, T] storage permuted to [bs, T, H]
        lse1 = (torch.randn(bs, H, T, dtype=torch.float32, device="cuda") * 4).permute(
            0, 2, 1
        )
        self._check(o1, lse1, zero, N, "strided slice+permute")
        # o1: fully transposed storage [T, bs, H, D] -> [bs, T, H, D]
        o1_t = torch.randn(T, bs, H, D, dtype=dtype, device="cuda").permute(1, 0, 2, 3)
        self._check(o1_t, lse1, zero, N, "transposed o1")


class _FakeA2AGroup:
    """Single-process stand-in for the DCP GroupCoordinator's byte-level
    ``all_to_all_single`` (equal split along dim 0: this rank's output chunk n
    is rank n's input chunk r). ``record_only`` captures the internally-built
    send buffer of ``dcp_a2a_lse_reduce`` so the OLD path needs no code copy.
    """

    def __init__(self, world_size, rank, sends=None, record_only=False):
        self.world_size = world_size
        self.rank = rank
        self.sends = sends  # list of flat uint8 send buffers, one per rank
        self.record_only = record_only
        self.recorded = None

    def all_to_all_single(self, output, input):
        if self.record_only:
            self.recorded = input.clone()
            output.zero_()
            return
        out_chunks = list(output.chunk(self.world_size))
        for src in range(self.world_size):
            out_chunks[src].copy_(self.sends[src].chunk(self.world_size)[self.rank])


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA (triton kernel)")
class TestMaskPackE2EOldVsNew(unittest.TestCase):
    """Full pipeline on a simulated 4-rank a2a: OLD path (torch.where x2 +
    .contiguous x2 + real dcp_a2a_lse_reduce) vs NEW path (dcp_mask_pack_triton
    + dcp_a2a_exchange_packed + dcp_unpack_lse_combine), bit-for-bit."""

    N, BS, T, H_FULL, D = 4, 5, 3, 8, 64

    def test_e2e_old_vs_new(self):
        from sglang.srt.layers.dcp import dcp_mask_pack_triton
        from sglang.srt.layers.dcp.comm import (
            dcp_a2a_exchange_packed,
            dcp_a2a_lse_reduce,
            dcp_unpack_lse_combine,
        )

        torch.manual_seed(23)
        N, bs, T, H, D = self.N, self.BS, self.T, self.H_FULL, self.D
        B, dtype, dev = bs * T, torch.bfloat16, "cuda"

        o1s, lse1s, zeros = [], [], []
        for r in range(N):
            o1s.append(torch.randn(bs, T, H, D, dtype=dtype, device=dev))
            lse1s.append(torch.randn(bs, T, H, dtype=torch.float32, device=dev) * 4)
            zeros.append(torch.rand(bs, device=dev) < 0.4)
        # Per-rank mask coverage: rank 1 none masked, rank 2 ALL rows masked,
        # ranks 0/3 random. Rank 1 guarantees every row is owned by at least
        # one rank (a row masked on EVERY rank cannot happen in production —
        # prefix >= 1 token — and would 0/0-NaN the combine).
        zeros[1] = torch.zeros(bs, dtype=torch.bool, device=dev)
        zeros[2] = torch.ones(bs, dtype=torch.bool, device=dev)
        for r in range(N):
            _poison_masked_rows(o1s[r], lse1s[r], zeros[r])

        def old_masked_inputs(r):
            # Verbatim _forward_verify_dcp old-path masking + flatten.
            o1m = torch.where(zeros[r].view(bs, 1, 1, 1), o1s[r].new_zeros(()), o1s[r])
            lse1m = torch.where(
                zeros[r].view(bs, 1, 1),
                lse1s[r].new_full((), float("-inf")),
                lse1s[r],
            )
            return (
                o1m.reshape(B, H, D).contiguous(),
                lse1m.reshape(B, H).contiguous(),
            )

        # --- OLD path, phase A: record each rank's internally-built send. ---
        old_sends = []
        for r in range(N):
            rec = _FakeA2AGroup(N, r, record_only=True)
            o_flat, lse_flat = old_masked_inputs(r)
            dcp_a2a_lse_reduce(
                o_flat, lse_flat, rec, is_lse_base_on_e=False, return_lse=True
            )
            old_sends.append(rec.recorded)

        # --- NEW path: fused pack; packed buffers must match OLD's bytes. ---
        new_send_bufs = [
            dcp_mask_pack_triton(o1s[r], lse1s[r], zeros[r], N) for r in range(N)
        ]
        new_sends = [s.reshape(-1).view(torch.uint8) for s in new_send_bufs]
        for r in range(N):
            self.assertTrue(
                _bit_equal(new_sends[r], old_sends[r]), f"packed buffer rank {r}"
            )

        # --- Phase B: run both full pipelines and compare the combines. ---
        for r in range(N):
            o_flat, lse_flat = old_masked_inputs(r)
            out_old, lse_old = dcp_a2a_lse_reduce(
                o_flat,
                lse_flat,
                _FakeA2AGroup(N, r, sends=old_sends),
                is_lse_base_on_e=False,
                return_lse=True,
            )
            recv = torch.empty_like(new_send_bufs[r])
            dcp_a2a_exchange_packed(
                new_send_bufs[r], recv, _FakeA2AGroup(N, r, sends=new_sends)
            )
            out_new, lse_new = dcp_unpack_lse_combine(
                recv, D, is_lse_base_on_e=False, return_lse=True
            )
            self.assertTrue(_bit_equal(out_new, out_old), f"combined out rank {r}")
            self.assertTrue(_bit_equal(lse_new, lse_old), f"combined lse rank {r}")


if __name__ == "__main__":
    unittest.main()
