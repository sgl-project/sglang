"""`compact_attention_hip` against a torch softmax-with-sink over the dequantized pages."""

import itertools
import unittest

import torch

from sglang.kernels.ops.attention.dsv4.compact_attention_hip import (
    compact_attention_hip,
)
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=20, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")

PAGE, PAGES = 64, 8
E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]


def _cache(fp4: bool) -> torch.Tensor:
    """[pages, page, 1, bytes]: 512 fp8 + 16 ue8m0 per token (528 B), or 512 e2m1 + 32 e4m3 (288 B)."""
    n = PAGES * PAGE
    if fp4:
        data = torch.randint(0, 256, (PAGES, PAGE * 256), dtype=torch.uint8)
        scale = (torch.rand(PAGES, PAGE * 32) * 0.5 + 0.25).to(torch.float8_e4m3fn)
    else:
        data = torch.randn(PAGES, PAGE * 512).to(torch.float8_e4m3fn).view(torch.uint8)
        scale = torch.randint(120, 130, (PAGES, PAGE * 16), dtype=torch.uint8)
    flat = torch.cat([data, scale.view(torch.uint8)], dim=1)
    return flat.view(PAGES, PAGE, 1, flat.shape[1] // PAGE).cuda().contiguous()


def _dequant(cache: torch.Tensor) -> torch.Tensor:
    """[pages * page, 512] fp32 rows, rounded to bf16 as the kernel reads them."""
    flat = cache.view(PAGES, -1).cpu()
    if cache.shape[-1] == 288:
        packed = flat[:, : PAGE * 256].view(PAGES, PAGE, 256).long()
        lut = torch.tensor(E2M1)
        v = torch.stack([lut[packed & 15], lut[packed >> 4]], -1).view(PAGES, PAGE, 512)
        s = flat[:, PAGE * 256 :].view(PAGES, PAGE, 32).view(torch.float8_e4m3fn)
        v = v * s.float().repeat_interleave(16, -1)
    else:
        v = (
            flat[:, : PAGE * 512]
            .view(PAGES, PAGE, 512)
            .view(torch.float8_e4m3fn)
            .float()
        )
        e = flat[:, PAGE * 512 :].view(PAGES, PAGE, 16).float()
        v = v * torch.exp2(e - 127).repeat_interleave(32, -1)
    return v.to(torch.bfloat16).float().view(-1, 512).cuda()


@unittest.skipUnless(is_hip() and is_gfx95_supported(), "gfx950 gluon MFMA kernel")
class TestCompactAttentionHip(CustomTestCase):
    def test_matches_torch_over_both_caches_splits_and_empty_rows(self):
        """Both packed layouts, a second cache, -1 slots and a row with no keys (sink only)."""
        torch.manual_seed(0)
        n, h, nk, ne, scale = 5, 16, 256, 128, 512**-0.5
        for fp4, splits in itertools.product((False, True), (1, 4)):
            main, extra = _cache(fp4), _cache(not fp4)
            q = torch.randn(n, h, 512, device="cuda").bfloat16()
            idx = torch.randint(
                0, PAGES * PAGE, (n, nk), device="cuda", dtype=torch.int32
            )
            idx[:, ::7] = -1
            lens = torch.randint(1, nk + 1, (n,), device="cuda", dtype=torch.int32)
            eidx = torch.randint(
                0, PAGES * PAGE, (n, ne), device="cuda", dtype=torch.int32
            )
            elens = torch.randint(1, ne + 1, (n,), device="cuda", dtype=torch.int32)
            lens[0] = elens[0] = 0
            sink = torch.randn(h, device="cuda")

            out = compact_attention_hip(
                q,
                main,
                idx,
                lens,
                sink,
                extra_cache=extra,
                extra_indices=eidx,
                extra_lengths=elens,
                softmax_scale=scale,
                splits=splits,
            ).float()

            kd, ed = _dequant(main), _dequant(extra)
            for t in range(n):
                rows = [idx[t, : lens[t]], eidx[t, : elens[t]]]
                kv = torch.cat(
                    [kd[rows[0][rows[0] >= 0].long()], ed[rows[1][rows[1] >= 0].long()]]
                )
                s = torch.cat([q[t].float() @ kv.T * scale, sink[:, None]], 1)
                ref = torch.softmax(s, 1)[:, :-1] @ kv
                torch.testing.assert_close(
                    out[t], ref, rtol=2e-2, atol=1e-2, msg=f"{fp4=} {splits=} row {t}"
                )


if __name__ == "__main__":
    unittest.main()
