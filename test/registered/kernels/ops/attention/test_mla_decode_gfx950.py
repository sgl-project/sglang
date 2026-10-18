"""The gfx950 Kimi-K3 MLA decode/verify kernels against the op chains they replace."""

import unittest

import torch

from sglang.kernels.ops.attention.mla_decode_epilogue import mla_decode_epilogue
from sglang.kernels.ops.attention.mla_decode_prep import mla_qk_prep
from sglang.kernels.ops.attention.mla_split_reduce import mla_split_reduce
from sglang.kernels.ops.attention.mla_v_up_gate import mla_v_up_gate
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

# backend-specific: these kernels serve aiter's head-padded asm MLA path on gfx950.
register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

# Kimi-K3 TP8 MLA: 12 heads padded to the asm kernel's 16, latent 512 + rope 64, V 128.
H, HP, L, DR, V = 12, 16, 512, 64, 128


def _beyond_ulp(got: torch.Tensor, want: torch.Tensor) -> int:
    want = want.float()
    return ((got.float() - want).abs() > want.abs() * 2**-7 + 1e-3).sum().item()


def _partials(splits, max_q, uncovered=(), seed=0):
    """aiter persistent-decode partials: request r has splits[r] splits (0, or listed in
    uncovered: stage 1 finalized its rows in o); a tile covers a request's max_q rows."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    indptr, final_map, pmap, base = [0], [], [], 0
    for r, n in enumerate(splits):
        if n == 0 or r in uncovered:
            continue
        for _ in range(n):
            pmap.append(base)
            base += max_q
        indptr.append(len(pmap))
        final_map.append([r * max_q, (r + 1) * max_q])
    rows = len(splits) * max_q
    i32 = dict(device="cuda", dtype=torch.int32)
    return dict(
        partial=torch.randn(max(base, 1), 1, HP, L, generator=g, device="cuda"),
        lse=torch.randn(max(base, 1), 1, HP, 1, generator=g, device="cuda") * 3,
        indptr=torch.tensor(indptr, **i32),
        fmap=(
            torch.tensor(final_map, **i32) if final_map else torch.zeros(0, 2, **i32)
        ),
        pmap=torch.tensor(pmap or [0], **i32),
        o=torch.randn(rows, HP, L, generator=g, device="cuda").bfloat16(),
        w=(torch.randn(H, L, V, generator=g, device="cuda") * 0.05).bfloat16(),
        gate=torch.randn(rows, H * V, generator=g, device="cuda").bfloat16(),
    )


@unittest.skipIf(torch.version.hip is None, "gfx950 MLA decode kernels")
class TestMlaDecodeGfx950(CustomTestCase):
    @torch.inference_mode()
    def test_qk_prep_matches_cat_pad_and_kv_write(self):
        """The prep kernel must write exactly what the unfused chain writes: the
        cat + zero head padding of q (bf16, or a saturating unit-scale fp8 cast),
        the fp8 KV rows, zeros for a padding token, and a finite KV row 0."""
        g = torch.Generator(device="cuda").manual_seed(0)
        for tokens in (1, 4, 64):
            # q_nope is the transposed bmm output; q_pe and the k parts are slices.
            q_nope = torch.randn(H, tokens, L, generator=g, device="cuda").bfloat16()
            q_nope = q_nope.transpose(0, 1)
            q_pe = torch.randn(
                tokens, H, 128 + DR, generator=g, device="cuda"
            ).bfloat16()
            q_pe = q_pe[..., 128:]
            kv_a = (
                torch.randn(tokens, 1536 + L + DR, generator=g, device="cuda") * 3
            ).bfloat16()
            k_nope = kv_a[:, 1536 : 1536 + L].unsqueeze(1)
            k_pe = kv_a[:, 1536 + L :].unsqueeze(1)
            loc = torch.randperm(4095, generator=g, device="cuda")[:tokens] + 1
            loc[-1] = 0  # a CUDA-graph padding token
            kv_ref = torch.randn(4096, 1, L + DR, generator=g, device="cuda")
            kv_ref = kv_ref.to(torch.float8_e4m3fn)
            kv_ref[0] = float("nan")
            kv = kv_ref.clone()

            q_ref = torch.zeros(tokens, HP, L + DR, device="cuda", dtype=torch.bfloat16)
            q_ref[:, :H] = torch.cat([q_nope, q_pe], dim=-1)
            kv_ref[loc] = torch.cat([k_nope, k_pe], dim=-1).to(torch.float8_e4m3fn)
            kv_ref[0].zero_()
            q8_ref = q_ref.float().clamp(-448, 448).to(torch.float8_e4m3fn)

            q = mla_qk_prep(q_nope, q_pe, k_nope, k_pe, loc, kv, HP)
            q8 = mla_qk_prep(
                q_nope, q_pe, k_nope, k_pe, loc, kv, HP, q_dtype=torch.float8_e4m3fn
            )
            with self.subTest(tokens=tokens):
                self.assertTrue(torch.equal(q, q_ref))
                self.assertTrue(
                    torch.equal(q8.view(torch.uint8), q8_ref.view(torch.uint8))
                )
                self.assertTrue(
                    torch.equal(kv.view(torch.uint8), kv_ref.view(torch.uint8))
                )

    @torch.inference_mode()
    def test_split_reduce_matches_aiter(self):
        """The real-head split combine must match aiter.mla_reduce_v1 on every
        covered row, one chunk or several (more splits than its tile)."""
        try:
            import aiter
        except ImportError:
            self.skipTest("aiter is not installed")
        for splits, max_q in (([128], 4), ([64] * 4, 4), ([16] * 16, 4), ([300, 7], 1)):
            t = _partials(splits, max_q, seed=len(splits))
            want = t["o"].clone()
            got = t["o"].clone()
            aiter.mla_reduce_v1(
                t["partial"], t["lse"], t["indptr"], t["fmap"], t["pmap"], max_q,
                max(splits), want, None,
            )  # fmt: skip
            mla_split_reduce(
                t["partial"], t["lse"], t["indptr"], t["fmap"], t["pmap"], max_q, got, H
            )
            with self.subTest(splits=splits[:2], max_q=max_q):
                self.assertEqual(_beyond_ulp(got[:, :H], want[:, :H]), 0)

    @torch.inference_mode()
    def test_v_up_gate_matches_bmm_then_gate(self):
        """One launch keeps the bmm's and the gate's bf16 rounding points, with
        w_vc held as the transposed view the model stores."""
        g = torch.Generator(device="cuda").manual_seed(0)
        w_vc = (torch.randn(H, V, L, generator=g, device="cuda") / L**0.5).bfloat16()
        w_vc = w_vc.transpose(1, 2)
        for tokens in (1, 4, 16, 64):
            attn = torch.randn(tokens, HP, L, generator=g, device="cuda").bfloat16()[
                :, :H
            ]
            gate = torch.randn(tokens, H * V, generator=g, device="cuda").bfloat16()
            y = torch.bmm(attn.transpose(0, 1), w_vc).transpose(0, 1).flatten(1)
            want = (
                y.float() * torch.sigmoid(gate.float()).bfloat16().float()
            ).bfloat16()
            with self.subTest(tokens=tokens):
                self.assertEqual(_beyond_ulp(mla_v_up_gate(attn, w_vc, gate), want), 0)
                self.assertEqual(_beyond_ulp(mla_v_up_gate(attn, w_vc, None), y), 0)

    @torch.inference_mode()
    def test_epilogue_matches_split_reduce_then_v_up_gate(self):
        """One launch must reproduce the backend's split combine followed by the
        W_vc + gate kernel within one bf16 ulp: single-query and 4-row verify
        tiles, rows stage 1 finalized, and rows with more splits than the
        program's tile (combined in chunks)."""
        cases = (
            ([37, 0, 128, 5], 1, ()),
            ([128, 3], 4, ()),
            ([64, 128, 7, 1], 4, (1,)),
            ([128] * 16, 4, ()),
        )
        for splits, max_q, uncovered in cases:
            t = _partials(splits, max_q, uncovered, seed=len(splits) + max_q)
            for gate in (t["gate"], None):
                o_ref = t["o"].clone()
                mla_split_reduce(
                    t["partial"],
                    t["lse"],
                    t["indptr"],
                    t["fmap"],
                    t["pmap"],
                    max_q,
                    o_ref,
                    H,
                )
                want = mla_v_up_gate(o_ref[:, :H, :], t["w"], gate)
                got = mla_decode_epilogue(
                    t["partial"], t["lse"], t["indptr"], t["fmap"], t["pmap"], t["o"],
                    t["w"], gate, max_q=max_q,
                )  # fmt: skip
                with self.subTest(
                    splits=splits[:4], max_q=max_q, gated=gate is not None
                ):
                    self.assertLessEqual(
                        _beyond_ulp(got, want), max(got.numel() // 1000, 1)
                    )


if __name__ == "__main__":
    unittest.main()
