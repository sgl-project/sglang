"""FlashMLA sparse decode over the ratio 1/2 latent pools, against a torch oracle."""

import unittest

import torch
from sgl_kernel.flash_mla import FlashMLASchedMeta, flash_mla_with_kvcache

from sglang.kernels.ops.attention.dsv4.attn import fused_store_cache
from sglang.kernels.ops.attention.dsv4.dequant_k_cache import dequantize_k_cache_paged
from sglang.srt.layers.attention.dsv4.torch_quant import fake_quant_fp4

LAYOUT_BYTES = 584  # 448 fp8 nope + 64 bf16 rope + 8 scale bytes
HEAD_DIM = 512
WINDOW = 128
TOPK = 512


def sparse_attention_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    valid: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """q [T, H, D], k [T, N, D] (one latent is both key and value), valid [T, N],
    attn_sink [H] fp32 -> [T, H, D]. Probabilities round to bf16 before the value
    sum, as the reference kernel does."""
    T, H, _ = q.shape
    sink = attn_sink.float().view(1, H, 1)
    qs, ks = q.float(), k.float()
    scores = torch.einsum("bhd,bnd->bhn", qs, ks) * softmax_scale
    scores = scores.masked_fill(~valid[:, None, :], -torch.inf)
    row_max = scores.amax(dim=-1, keepdim=True).clamp_min(-1e30)
    probs = torch.exp(scores - row_max)
    denom = probs.sum(dim=-1, keepdim=True) + torch.exp(sink - row_max)
    out = torch.einsum("bhn,bnd->bhd", probs.to(q.dtype).float(), ks) / denom
    return out.to(q.dtype)


def make_cache(num_pages: int, page_size: int, device) -> torch.Tensor:
    per_page = -(-page_size * LAYOUT_BYTES // 576) * 576
    return torch.zeros(num_pages, per_page, dtype=torch.uint8, device=device)


def kernel_view(cache: torch.Tensor, page_size: int) -> torch.Tensor:
    return cache[:, : page_size * LAYOUT_BYTES].view(
        cache.shape[0], page_size, 1, LAYOUT_BYTES
    )


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
class TestFlashMLALowRatio(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.device = torch.device("cuda")

    def _fill(self, cache, page_size, rows):
        ids = torch.arange(rows.shape[0], device=self.device, dtype=torch.int32)
        fused_store_cache(rows, cache, ids, page_size=page_size, type="flashmla")

    def test_fp4_latent_roundtrip_is_exact(self):
        """The reference keeps latents on the fp4 grid; the fp8/ue8m0-64 layout must
        return those values bit for bit."""
        for page_size in (256, 128):
            rows = fake_quant_fp4(
                torch.randn(
                    8 * page_size, HEAD_DIM, device=self.device, dtype=torch.bfloat16
                )
            )
            cache = make_cache(8, page_size, self.device)
            self._fill(cache, page_size, rows)
            back = dequantize_k_cache_paged(
                cache, torch.arange(rows.shape[0], device=self.device), page_size
            ).view(-1, HEAD_DIM)
            self.assertTrue(torch.equal(back, rows), f"page_size={page_size}")

    def _run(self, ratio: int, num_tokens: int = 6, num_heads: int = 64):
        swa_ps, swa_pages = WINDOW, 16
        ex_ps, ex_pages = 256 // ratio, 8
        swa = make_cache(swa_pages, swa_ps, self.device)
        ex = make_cache(ex_pages, ex_ps, self.device)
        n_swa, n_ex = swa_pages * swa_ps, ex_pages * ex_ps
        self._fill(
            swa,
            swa_ps,
            torch.randn(n_swa, HEAD_DIM, device=self.device, dtype=torch.bfloat16),
        )
        self._fill(
            ex,
            ex_ps,
            fake_quant_fp4(
                torch.randn(n_ex, HEAD_DIM, device=self.device, dtype=torch.bfloat16)
            ),
        )

        q = torch.randn(
            num_tokens, 1, num_heads, HEAD_DIM, device=self.device, dtype=torch.bfloat16
        )
        sink = torch.randn(num_heads, device=self.device, dtype=torch.float32)
        win_idx = torch.full(
            (num_tokens, 1, WINDOW), -1, dtype=torch.int32, device=self.device
        )
        ex_idx = torch.full(
            (num_tokens, 1, TOPK), -1, dtype=torch.int32, device=self.device
        )
        win_len = torch.randint(
            1, WINDOW + 1, (num_tokens,), device=self.device, dtype=torch.int32
        )
        ex_len = torch.randint(
            1, TOPK + 1, (num_tokens,), device=self.device, dtype=torch.int32
        )
        ex_len[0] = 1  # a query with a single reachable compressed position
        ex_len[1] = TOPK
        for t in range(num_tokens):
            wl, el = int(win_len[t]), int(ex_len[t])
            win_idx[t, 0, :wl] = torch.randperm(n_swa, device=self.device)[:wl].to(
                torch.int32
            )
            ex_idx[t, 0, :el] = torch.randperm(n_ex, device=self.device)[:el].to(
                torch.int32
            )
        scale = HEAD_DIM**-0.5

        out = flash_mla_with_kvcache(
            q=q,
            k_cache=kernel_view(swa, swa_ps),
            head_dim_v=HEAD_DIM,
            block_table=None,
            cache_seqlens=None,
            tile_scheduler_metadata=FlashMLASchedMeta(),
            softmax_scale=scale,
            is_fp8_kvcache=True,
            indices=win_idx,
            topk_length=win_len,
            attn_sink=sink,
            extra_k_cache=kernel_view(ex, ex_ps),
            extra_indices_in_kvcache=ex_idx,
            extra_topk_length=ex_len,
        )[0].squeeze(1)

        win_k = dequantize_k_cache_paged(
            swa, win_idx.view(-1).clamp_min(0), swa_ps
        ).view(num_tokens, WINDOW, HEAD_DIM)
        ex_k = dequantize_k_cache_paged(ex, ex_idx.view(-1).clamp_min(0), ex_ps).view(
            num_tokens, TOPK, HEAD_DIM
        )
        k = torch.cat([win_k, ex_k], dim=1)
        valid = torch.cat(
            [win_idx.view(num_tokens, WINDOW) >= 0, ex_idx.view(num_tokens, TOPK) >= 0],
            1,
        )
        ref = sparse_attention_oracle(q.squeeze(1), k, valid, sink, scale)

        self.assertTrue(torch.isfinite(out).all())
        o, r = out.float(), ref.float()
        rel = ((o - r).norm() / r.norm()).item()
        cos = torch.nn.functional.cosine_similarity(
            o.flatten(), r.flatten(), dim=0
        ).item()
        print(f"ratio={ratio} extra_page={ex_ps}: rel_l2={rel:.3e} cos={cos:.6f}")
        self.assertLess(rel, 1e-2)
        self.assertGreater(cos, 0.9999)

    def test_ratio_1_extra_page_256(self):
        self._run(1)

    def test_ratio_2_extra_page_128(self):
        self._run(2)


if __name__ == "__main__":
    unittest.main()
