"""DSV4 TileLang FlashMLA must compile with the gfx942 tile config and decode the
e4m3fnuz KV cache that gfx942 writes (ROCM-33500)."""

import math
import unittest
from unittest import mock

import torch

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

# The kernel decodes FP8 in software, so any ROCm GPU can check the gfx942 format.
register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd-mi35x")

_PAGE = 64
_SLOT_BYTES = 576  # 448 fp8 nope + 64 bf16 rope
_BYTES_PER_PAGE = math.ceil(_PAGE * (_SLOT_BYTES + 8) / _SLOT_BYTES) * _SLOT_BYTES


def _write_cache(latent: torch.Tensor, fp8_dtype: torch.dtype) -> torch.Tensor:
    import sglang.kernels.ops.kvcache.triton_store_cache as store

    num_pages = latent.shape[0] // _PAGE
    cache = torch.zeros(
        num_pages, _BYTES_PER_PAGE, dtype=torch.uint8, device=latent.device
    )
    with mock.patch.multiple(
        store, _FP8_DTYPE=fp8_dtype, _FP8_INFO=torch.finfo(fp8_dtype)
    ):
        loc = torch.arange(latent.shape[0], device=latent.device)
        store.triton_fused_store_flashmla(latent, cache, loc, _PAGE)
    return cache[:, : _PAGE * _SLOT_BYTES].view(num_pages, _PAGE, 1, _SLOT_BYTES)


def _dequant(cache: torch.Tensor, fp8_dtype: torch.dtype, n: int) -> torch.Tensor:
    pages = cache.as_strided((cache.shape[0], _BYTES_PER_PAGE), (_BYTES_PER_PAGE, 1))
    slots = pages[:, : _PAGE * _SLOT_BYTES].reshape(-1, _SLOT_BYTES)[:n]
    scales = pages[:, _PAGE * _SLOT_BYTES : _PAGE * (_SLOT_BYTES + 8)].reshape(-1, 8)
    nope = slots[:, :448].contiguous().view(fp8_dtype).float().view(n, 7, 64)
    nope = nope * torch.exp2(scales[:n, :7].float() - 127.0)[:, :, None]
    rope = slots[:, 448:].contiguous().view(torch.bfloat16).float()
    return torch.cat([nope.reshape(n, 448), rope], dim=1)


def _ref_attention(q, kv, indices, topk_length, attn_sink, scale):
    out = torch.empty(*q.shape[:3], 512, device=q.device)
    for b in range(q.shape[0]):
        rows = indices[b, 0, : topk_length[b]].long()
        selected = kv[rows]
        logits = q[b, 0].float() @ selected.T * scale
        row_max = torch.maximum(logits.amax(-1), attn_sink)
        probs = torch.exp(logits - row_max[:, None])
        denom = probs.sum(-1) + torch.exp(attn_sink - row_max)
        out[b, 0] = probs @ selected / denom[:, None]
    return out


@unittest.skipUnless(is_hip() and torch.cuda.is_available(), "ROCm GPU required")
class TestDSV4TileLangFp8Fnuz(CustomTestCase):
    def _run(self, *, gfx95: bool, fnuz: bool):
        import sglang.kernels.ops.attention.dsa.tilelang_kernel as tk

        torch.manual_seed(0)
        num_heads = 64 if gfx95 else 16
        num_tokens, topk = 8 * _PAGE, 128
        latent = torch.randn(num_tokens, 512, device="cuda") * torch.exp(
            torch.randn(num_tokens, 1, device="cuda")
        )
        latent = latent.to(torch.bfloat16)
        q = (torch.randn(2, 1, num_heads, 512, device="cuda") * 2).to(torch.bfloat16)
        indices = torch.stack(
            [torch.randperm(num_tokens, device="cuda")[:topk] for _ in range(2)]
        ).view(2, 1, topk)
        indices = indices.to(torch.int32)
        topk_length = torch.tensor([topk, 37], device="cuda", dtype=torch.int32)
        attn_sink = torch.randn(num_heads, device="cuda")
        scale = 512**-0.5

        fp8_dtype = torch.float8_e4m3fnuz if fnuz else torch.float8_e4m3fn
        cache = _write_cache(latent, fp8_dtype)
        with mock.patch.multiple(tk, _is_gfx95_supported=gfx95, _is_fp8_fnuz=fnuz):
            out, _ = tk.dpsk_v4_fp8_attention_fwd(
                q=q,
                k_cache=cache,
                block_table=None,
                cache_seqlens=None,
                head_dim_v=512,
                tile_scheduler_metadata=None,
                softmax_scale=scale,
                is_fp8_kvcache=True,
                indices=indices,
                attn_sink=attn_sink,
                topk_length=topk_length,
            )
        expected = _ref_attention(
            q,
            _dequant(cache, fp8_dtype, num_tokens),
            indices,
            topk_length,
            attn_sink,
            scale,
        )
        torch.testing.assert_close(out.float(), expected, atol=2e-2, rtol=2e-2)

    def test_gfx942_tile_config_reads_fnuz_cache(self):
        """gfx942's tile config must compile, and a 2x misread of fnuz K/V must not pass."""
        self._run(gfx95=False, fnuz=True)

    def test_gfx950_tile_config_reads_fn_cache(self):
        """The OCP e4m3fn decode used on gfx950 must stay unchanged."""
        self._run(gfx95=True, fnuz=False)


if __name__ == "__main__":
    unittest.main()
