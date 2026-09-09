"""XCD swizzle in the Triton sparse-MLA prefill kernel must cover every token.

The swizzle hands each XCD a contiguous run of tokens so that consecutive
workgroups (which round-robin across XCDs) share KV locality. It is only a
valid remap if pid -> s_i stays a bijection onto [0, n_tok); when it is not,
some output rows are never stored and keep whatever torch.empty handed back.

N_XCD=1 degenerates to s_i == pid, so it is used here as the reference.
"""

import unittest

import torch

from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=120, suite="nightly-amd-kernel-1-gpu")

D_V = 512
D_TAIL = 64
DIM = D_V + D_TAIL
TOPK = 128
N_PAGES = 4096


@unittest.skipUnless(
    torch.cuda.is_available() and is_gfx95_supported(),
    "Triton sparse MLA prefill is gated to gfx950.",
)
class TestTritonSparseMLASwizzle(CustomTestCase):
    def setUp(self):
        super().setUp()
        from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz

        self.device = torch.device("cuda")
        self.fp8_dtype = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn

    def tearDown(self):
        torch.cuda.empty_cache()
        super().tearDown()

    def _inputs(self, seq: int, heads: int):
        gen = torch.Generator(device=self.device).manual_seed(seq * 131 + heads)

        def fp8(*shape):
            raw = torch.randn(*shape, device=self.device, generator=gen) * 0.3
            return raw.to(self.fp8_dtype)

        q_nope = fp8(seq, heads, D_V)
        q_rope = fp8(seq, heads, D_TAIL)
        kv = fp8(N_PAGES, DIM)
        indices = torch.randint(
            0,
            N_PAGES,
            (seq, 1, TOPK),
            device=self.device,
            dtype=torch.int32,
            generator=gen,
        )
        return q_nope, q_rope, kv, indices.contiguous()

    def _run(self, q_nope, q_rope, kv, indices, n_xcd: int) -> torch.Tensor:
        from sglang.kernels.ops.attention.dsa.triton_sparse_mla_prefill import (
            _launch_sparse_mla_prefill,
        )

        seq, heads, _ = q_nope.shape
        # NaN sentinel: any row the kernel skips stays NaN instead of silently
        # picking up plausible-looking garbage from a recycled allocation.
        out = torch.full(
            (seq, heads, D_V),
            float("nan"),
            device=self.device,
            dtype=torch.bfloat16,
        )
        _launch_sparse_mla_prefill(
            q_nope, q_rope, kv, indices, out, 1.0 / (DIM**0.5), D_V, n_xcd=n_xcd
        )
        torch.cuda.synchronize()
        return out

    def _check_seq(self, seq: int, heads: int):
        q_nope, q_rope, kv, indices = self._inputs(seq, heads)
        swizzled = self._run(q_nope, q_rope, kv, indices, n_xcd=8)
        reference = self._run(q_nope, q_rope, kv, indices, n_xcd=1)

        unwritten = torch.isnan(swizzled).any(dim=-1).any(dim=-1)
        if bool(unwritten.any()):
            rows = unwritten.nonzero().flatten().tolist()
            self.fail(
                f"seq={seq} heads={heads}: {len(rows)} token rows never written, "
                f"e.g. {rows[:8]}"
            )
        # Per-row math does not depend on which workgroup runs it, so the
        # swizzled result must match the identity mapping bit for bit.
        torch.testing.assert_close(swizzled, reference, rtol=0, atol=0)

    def test_seq_not_divisible_by_xcd_count(self):
        for seq in (513, 1000, 4095, 12289):
            with self.subTest(seq=seq):
                self._check_seq(seq, heads=16)

    def test_seq_divisible_by_xcd_count(self):
        for seq in (512, 4096):
            with self.subTest(seq=seq):
                self._check_seq(seq, heads=16)

    def test_eight_heads(self):
        self._check_seq(1023, heads=8)

    def test_strided_query_matches_contiguous(self):
        """Strided q halves from one [seq, heads, DIM] tensor match copied halves."""
        from sglang.kernels.ops.attention.dsa.triton_sparse_mla_prefill import (
            triton_sparse_mla_prefill_fwd,
        )

        seq, heads = 512, 16
        gen = torch.Generator(device=self.device).manual_seed(17)
        q_all = (torch.randn(seq, heads, DIM, device=self.device, generator=gen) * 0.3).to(
            self.fp8_dtype
        )
        kv = (torch.randn(N_PAGES, DIM, device=self.device, generator=gen) * 0.3).to(
            self.fp8_dtype
        )
        indices = torch.randint(
            0, N_PAGES, (seq, 1, TOPK), device=self.device, dtype=torch.int32, generator=gen
        )
        sm_scale = 1.0 / (DIM**0.5)
        strided = triton_sparse_mla_prefill_fwd(
            q_all[..., :D_V], q_all[..., D_V:], kv, indices, sm_scale, D_V
        ).squeeze(0)
        copied = triton_sparse_mla_prefill_fwd(
            q_all[..., :D_V].contiguous(),
            q_all[..., D_V:].contiguous(),
            kv,
            indices,
            sm_scale,
            D_V,
        ).squeeze(0)
        torch.cuda.synchronize()
        torch.testing.assert_close(copied, strided, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
