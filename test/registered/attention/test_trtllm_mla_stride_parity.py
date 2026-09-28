"""The paged MLA decode kernels must not depend on the KV cache's slot stride.

trtllm_mla, cutedsl_mla and tokenspeed_mla hand their decode kernels
`paged_row_view(k_cache, page_size).unsqueeze(1)`: `[pages, 1, page_size,
kv_cache_dim]` whose in-page token stride is the pool's SLOT stride. Under the
unified pool's token-major layout that stride spans every layer's latent row,
so a kernel that derives the page address from `page_size * kv_cache_dim`
instead of reading `stride()` reads the wrong rows with no error. Identical
latent rows go through a packed cache and through an over-strided view of the
same values; the outputs must match bit for bit.

Runs on SM100 only, where these kernels exist.

    python -m pytest test/registered/attention/test_trtllm_mla_stride_parity.py -v
"""

import unittest

import torch

from sglang.srt.mem_cache.layout.page_major import paged_row_view
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_sm100_supported, is_tokenspeed_mla_available
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="4-gpu-b200")

_PAGES = 4
_PAGE_SIZE = 64
_KV_LORA_RANK = 512
_QK_ROPE_HEAD_DIM = 64
_QK_NOPE_HEAD_DIM = 128
_KV_CACHE_DIM = _KV_LORA_RANK + _QK_ROPE_HEAD_DIM
_Q_HEADS = 16
_BS = 2
# Stand-in for the unified pool's entry: every layer's latent row shares the slot.
_LAYER_NUM = 61


def _packed_cache(dtype):
    slots = _PAGES * _PAGE_SIZE
    return torch.randn(slots, 1, _KV_CACHE_DIM, device="cuda").to(dtype)


def _strided_like(src):
    """The SAME rows in a buffer whose slot stride spans a whole entry."""
    slots = src.shape[0]
    entry = _LAYER_NUM * _KV_CACHE_DIM
    backing = torch.zeros(slots * entry, dtype=src.dtype, device=src.device)
    view = backing.as_strided(src.shape, (entry, _KV_CACHE_DIM, 1))
    view.copy_(src)
    assert view.stride(0) == entry != src.stride(0)
    assert torch.equal(view.view(torch.uint8), src.view(torch.uint8))
    return view


def _paged(k_cache):
    # What the MLA backends' forward_decode builds from the pool's key buffer.
    return paged_row_view(k_cache, _PAGE_SIZE).unsqueeze(1)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
@unittest.skipUnless(is_sm100_supported(), "the paged MLA kernels require SM100")
class TestPagedMLADecodeStrideParity(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.block_tables = (
            torch.arange(_PAGES, dtype=torch.int32, device="cuda")
            .repeat(_BS, 1)
            .contiguous()
        )
        self.seq_lens = torch.tensor(
            [_PAGES * _PAGE_SIZE - 7, _PAGES * _PAGE_SIZE],
            dtype=torch.int32,
            device="cuda",
        )

    def _assert_stride_independent(self, run, kv_dtype):
        packed = _packed_cache(kv_dtype)
        want = run(_paged(packed))
        got = run(_paged(_strided_like(packed)))
        # Bit-for-bit: a tolerance here would hide the bug.
        self.assertTrue(
            torch.equal(want, got),
            f"MLA decode depends on the KV slot stride: "
            f"max |diff| = {(want.float() - got.float()).abs().max().item()}",
        )

    def test_flashinfer_mla_decode_is_identical_across_slot_strides(self):
        import flashinfer

        workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        # trtllm-gen is trtllm_mla's kernel; cute-dsl is cutedsl_mla's.
        for backend, kv_dtype in (
            ("trtllm-gen", torch.bfloat16),
            ("trtllm-gen", torch.float8_e4m3fn),
            ("cute-dsl", torch.bfloat16),
        ):
            with self.subTest(backend=backend, kv_dtype=kv_dtype):
                q = torch.randn(_BS, 1, _Q_HEADS, _KV_CACHE_DIM, device="cuda").to(
                    kv_dtype
                )
                extra = {} if backend == "trtllm-gen" else {"backend": backend}

                def run(kv_cache):
                    return flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
                        query=q,
                        kv_cache=kv_cache,
                        workspace_buffer=workspace,
                        qk_nope_head_dim=_QK_NOPE_HEAD_DIM,
                        kv_lora_rank=_KV_LORA_RANK,
                        qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
                        block_tables=self.block_tables,
                        seq_lens=self.seq_lens,
                        max_seq_len=_PAGES * _PAGE_SIZE,
                        bmm1_scale=_KV_CACHE_DIM**-0.5,
                        **extra,
                    )

                self._assert_stride_independent(run, kv_dtype)

    def test_tokenspeed_mla_decode_is_identical_across_slot_strides(self):
        if not is_tokenspeed_mla_available():
            self.skipTest("tokenspeed_mla is not installed")
        import tokenspeed_mla

        from sglang.srt.layers.attention.tokenspeed_mla_backend import (
            _get_tokenspeed_workspace,
        )

        # The workspace is sized for the DCP-gathered head count.
        with get_parallel().override(attn_dcp_size=1):
            workspace = _get_tokenspeed_workspace(
                torch.device("cuda"), _Q_HEADS, _KV_LORA_RANK
            )
        kv_dtype = torch.float8_e4m3fn
        q = torch.randn(_BS, 1, _Q_HEADS, _KV_CACHE_DIM, device="cuda").to(kv_dtype)

        def run(kv_cache):
            return tokenspeed_mla.tokenspeed_mla_decode(
                query=q,
                kv_cache=kv_cache,
                workspace_buffer=workspace,
                kv_lora_rank=_KV_LORA_RANK,
                qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
                block_tables=self.block_tables,
                seq_lens=self.seq_lens,
                max_seq_len=_PAGES * _PAGE_SIZE,
                softmax_scale=_KV_CACHE_DIM**-0.5,
            )

        self._assert_stride_independent(run, kv_dtype)


if __name__ == "__main__":
    unittest.main()
