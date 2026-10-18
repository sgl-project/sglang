"""The gfx950 Kimi-K3 MLA cached-prefix kernels against reference attention and expansion."""

import unittest

import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.mla_absorbed_extend import (
    mla_absorbed_extend,
    plan_absorbed_extend,
)
from sglang.kernels.ops.attention.mla_prefix_kv_pack import (
    gather_prefix_latent,
    pack_prefix_kv_fp8,
)
from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

# backend-specific: both kernels feed aiter's gfx950 FP8 MLA prefill path.
register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

# Kimi-K3 TP8 MLA: 12 heads per rank (16 in the FP8 prefill kernel), 512-wide
# latent, 64-wide rope, 128-wide nope and value heads.
H, HP, R, P, DN, DV = 12, 16, 512, 64, 128, 128
FP8 = torch.float8_e4m3fn


def _attend_latent(q, cache, qo_indptr, kv_indptr, kv_indices, scale):
    out = torch.empty(q.shape[0], H, R, device="cuda")
    keys_all = cache.float()
    for b in range(qo_indptr.numel() - 1):
        q0, q1 = qo_indptr[b].item(), qo_indptr[b + 1].item()
        k0, k1 = kv_indptr[b].item(), kv_indptr[b + 1].item()
        keys = keys_all[kv_indices[k0:k1].long()]
        n_q, n_k = q1 - q0, k1 - k0
        s = torch.einsum("qhd,kd->hqk", q[q0:q1].float(), keys) * scale
        pos = torch.arange(n_k - n_q, n_k, device="cuda")[:, None]
        s = s.masked_fill(
            torch.arange(n_k, device="cuda")[None, :] > pos, float("-inf")
        )
        out[q0:q1] = torch.einsum("hqk,kd->qhd", torch.softmax(s, -1), keys[:, :R])
    return out


def _pad_heads(x, padded):
    reps = -(-padded // x.shape[1])
    return x.repeat(1, reps, 1)[:, :padded, :].contiguous()


@unittest.skipUnless(is_gfx95_supported(), "requires gfx950")
class TestMlaPrefillGfx950(CustomTestCase):
    @torch.inference_mode()
    def test_absorbed_extend_attends_prefix_and_earlier_positions(self):
        """Ragged extends after cached prefixes (and one without) attend every
        prefix key and their own earlier positions only."""
        torch.manual_seed(0)
        scale = 1.0 / 192**0.5
        for prefix_lens, ext_lens in (
            ([1000], [37]),
            ([4000, 0, 2500], [256, 128, 1]),
            ([300, 7000], [5, 200]),
        ):
            kv_lens = [p + e for p, e in zip(prefix_lens, ext_lens)]
            cache = (torch.randn(sum(kv_lens) + 1, R + P, device="cuda") * 0.5).to(FP8)
            kv_indices = (torch.randperm(sum(kv_lens), device="cuda") + 1).int()
            kv_indptr = torch.tensor(
                [0, *torch.tensor(kv_lens).cumsum(0).tolist()],
                device="cuda",
                dtype=torch.int32,
            )
            qo_indptr = torch.tensor(
                [0, *torch.tensor(ext_lens).cumsum(0).tolist()],
                device="cuda",
                dtype=torch.int32,
            )
            q = (torch.randn(sum(ext_lens), H, R + P, device="cuda") * 0.3).to(FP8)
            plan = plan_absorbed_extend(ext_lens, kv_lens, q.device)
            out = mla_absorbed_extend(
                q, cache, kv_indices, kv_indptr, qo_indptr, plan, scale, R
            )
            ref = _attend_latent(q, cache, qo_indptr, kv_indptr, kv_indices, scale)
            cos = F.cosine_similarity(out.float().flatten(), ref.flatten(), dim=0)
            with self.subTest(prefix_lens=prefix_lens, ext_lens=ext_lens):
                self.assertGreater(cos.item(), 0.9995)
                torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)

    @torch.inference_mode()
    def test_prefix_kv_pack_matches_unfused_expansion_bitwise(self):
        """The two-launch expansion must produce the exact fp8 K/V bytes the
        aiter backend's index_select / cast / cat / pad_heads / cast chain does:
        padding heads tiled as h % H, the rope part carried as cached bytes."""
        torch.manual_seed(0)
        slots = 4096
        cache = (torch.randn(slots, 1, R + P, device="cuda") * 2).to(FP8)
        w = (torch.randn(H * (DN + DV), R, device="cuda") * 0.05).to(torch.bfloat16)
        for num_rows in (1, 37, 3000):
            idx = torch.randperm(slots, device="cuda")[:num_rows].to(torch.int32)
            kvc_ref, k_pe = torch.split(
                torch.index_select(cache, 0, idx), [R, P], dim=-1
            )
            kvc_ref = kvc_ref.to(torch.bfloat16).squeeze(1)
            k_ref, v_ref = torch.split(
                F.linear(kvc_ref, w).view(-1, H, DN + DV), [DN, DV], dim=-1
            )
            k_pe_heads = torch.broadcast_to(k_pe.to(torch.bfloat16), (num_rows, H, P))
            k_ref = _pad_heads(torch.cat([k_ref, k_pe_heads], dim=-1), HP).to(FP8)
            v_ref = _pad_heads(v_ref, HP).to(FP8)

            kvc, pe = gather_prefix_latent(cache, idx, R)
            k, v = pack_prefix_kv_fp8(
                F.linear(kvc, w),
                pe,
                num_heads=H,
                num_heads_padded=HP,
                nope_dim=DN,
                v_dim=DV,
            )
            with self.subTest(num_rows=num_rows):
                self.assertTrue(torch.equal(kvc, kvc_ref))
                self.assertTrue(
                    torch.equal(k.view(torch.uint8), k_ref.view(torch.uint8))
                )
                self.assertTrue(
                    torch.equal(v.view(torch.uint8), v_ref.view(torch.uint8))
                )


if __name__ == "__main__":
    unittest.main()
