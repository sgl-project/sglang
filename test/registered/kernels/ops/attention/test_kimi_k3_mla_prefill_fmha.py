"""FP8 PS-ASM MHA prefill for Kimi-K3 MLA
(12 heads, d_qk 192, d_v 128) vs an fp32 reference and vs a reference on
E4M3-rounded Q/K/V/P (the error of TRT-LLM's FP8 prefill).

Runs AiterAttnBackend._k3_fp8_prefill (prefix + no-prefix branches) on varlen
batches. gfx950 only.
"""

import math
import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

H, DN, DR, DV, LORA = 12, 128, 64, 128, 512
SCALE = 1.0 / math.sqrt(DN + DR)


register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")


def _ref(q, k, v, qo_lens, kv_lens, p8=False):
    outs = []
    qo = ko = 0
    for a, n in zip(qo_lens, kv_lens):
        qq = q[qo : qo + a].float()
        kk = k[ko : ko + n].float()
        vv = v[ko : ko + n].float()
        s = torch.einsum("qhd,khd->hqk", qq, kk) * SCALE
        mask = torch.ones(a, n, dtype=torch.bool, device=q.device).tril(n - a)
        s = s.masked_fill(~mask, float("-inf"))
        if p8:
            e = torch.exp(s - s.amax(-1, keepdim=True))
            den = e.sum(-1, keepdim=True)
            o = torch.einsum("hqk,khd->qhd", e.to(torch.float8_e4m3fn).float(), vv)
            o = o / den.permute(1, 0, 2)
        else:
            o = torch.einsum("hqk,khd->qhd", torch.softmax(s, -1), vv)
        outs.append(o)
        qo += a
        ko += n
    return torch.cat(outs)


def _rel(a, b):
    return ((a.float() - b).pow(2).sum() / b.pow(2).sum()).sqrt().item()


@unittest.skipUnless(
    torch.cuda.is_available()
    and "gfx950" in torch.cuda.get_device_properties(0).gcnArchName,
    "gfx950 only",
)
class TestK3MlaPrefillFmha(CustomTestCase):
    def _run(self, qo_lens, prefix_lens, seed=0):
        from sglang.kernels.ops.attention import k3_mla_prefill_fp8 as K
        from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend

        torch.manual_seed(seed)
        dev = "cuda"
        fp8 = torch.float8_e4m3fn
        kv_lens = [a + p for a, p in zip(qo_lens, prefix_lens)]
        tq, tk = sum(qo_lens), sum(kv_lens)
        P = tk + 333
        cache = (
            torch.cat(
                [
                    torch.randn(P, LORA, device=dev) * 0.5,
                    torch.randn(P, DR, device=dev),
                ],
                1,
            )
            .to(fp8)
            .view(P, 1, LORA + DR)
        )
        kv_indices = torch.randperm(P, device=dev)[:tk].to(torch.int32)
        w = (
            torch.randn(H * (DN + DV), LORA, device=dev, dtype=torch.bfloat16)
            / math.sqrt(LORA)
            * 2
        )
        layer = SimpleNamespace(
            kv_b_proj=lambda x: (torch.nn.functional.linear(x, w),),
            tp_k_head_num=H,
            v_head_dim=DV,
            scaling=SCALE,
        )
        q = torch.randn(tq, H, DN + DR, device=dev, dtype=torch.bfloat16)
        plan = K.make_plan(qo_lens, kv_lens, H, torch.device(dev))
        no_prefix = not any(prefix_lens)
        # reference K/V exactly as the bf16 path builds them
        lat = cache.view(P, -1)[kv_indices.long()]
        kv = torch.nn.functional.linear(lat[:, :LORA].to(torch.bfloat16), w).view(
            -1, H, DN + DV
        )
        kref = torch.cat(
            [kv[..., :DN], lat[:, LORA:].to(torch.bfloat16)[:, None].expand(-1, H, DR)],
            -1,
        )
        vref = kv[..., DN:]
        if no_prefix:
            k_in, v_in = kref, vref  # the model's own k / v
        else:
            k_in = v_in = None
        out = AiterAttnBackend._k3_fp8_prefill(
            None,
            q,
            k_in,
            v_in,
            layer,
            cache,
            kv_indices,
            plan,
            no_prefix=no_prefix,
            kv_lora_rank=LORA,
            qk_rope_head_dim=DR,
            qk_nope_head_dim=DN,
        )
        torch.cuda.synchronize()
        self.assertEqual(tuple(out.shape), (tq, H, DV))
        self.assertTrue(torch.isfinite(out).all())
        ref = _ref(q, kref, vref, qo_lens, kv_lens)
        sim = _ref(
            q.to(fp8).to(torch.bfloat16),
            kref.to(fp8).to(torch.bfloat16),
            vref.to(fp8).to(torch.bfloat16),
            qo_lens,
            kv_lens,
            p8=True,
        )
        e_k, e_sim = _rel(out, ref), _rel(sim, ref)
        # the kernel must not be worse than ideal E4M3 Q/K/V/P attention (+20%)
        self.assertLess(e_k, 1.2 * e_sim + 1e-3)

    def test_prefix_varlen(self):
        self._run([300, 1, 777, 64], [5000, 4096, 0, 129])

    def test_prefix_single(self):
        self._run([2048], [6000])

    def test_no_prefix(self):
        self._run([1000, 37, 512], [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
