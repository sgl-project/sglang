"""trtllm_mha output must not depend on the KV cache's slot stride.

`_reshape_paged_kv_cache` hands the trtllm-gen kernels a view whose in-page
token stride is the pool's SLOT stride. On a contiguous per-layer buffer that
stride is `heads * head_dim`; under the unified pool's token-major layout the
slot carries every layer, so the same view has a stride wider by
`2 * layer_num`. Shape and dtype are identical either way, so a kernel that
derives the page address from `page_size * heads * head_dim` instead of
reading `stride()` reads the wrong rows and returns wrong numbers -- with no
exception and no shape mismatch to catch it.

This is the check the view-level tests cannot make
(`unit/layers/attention/test_trtllm_mha_paged_kv.py` proves the VIEW carries
the slot stride; it says nothing about what the kernel does with it). Identical
page contents go through a contiguous cache and through an over-strided view of
the same values, for both kernels the backend calls on SM100 (decode and
context) and for bf16 and fp8 KV; the outputs must match bit for bit.

Runs on SM100 only: the trtllm-gen cubins are datacenter-Blackwell kernels, and
on SM90 / SM120 the backend dispatches to other kernels.

    python -m pytest test/registered/attention/test_trtllm_mha_stride_parity.py -v
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.utils import is_sm100_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=9, stage="base-b", runner_config="4-gpu-b200")

_PAGES = 4
_PAGE_SIZE = 16
_HEADS = 4
_HEAD_DIM = 128
_BS = 2
_Q_HEADS = 8
# Stand-in for the unified pool's entry: every layer's K and V share the slot.
_LAYER_NUM = 8


def _contiguous_cache(dtype, device):
    """A plain per-layer buffer: slot stride == heads * head_dim."""
    slots = _PAGES * _PAGE_SIZE
    return torch.randn(slots, _HEADS, _HEAD_DIM, device=device).to(dtype)


def _strided_like(src):
    """The SAME values in a buffer whose slot stride spans a whole entry.

    `as_strided` over a wider backing reproduces exactly what a token-major
    per-layer view looks like: rows `2 * _LAYER_NUM` apart, not packed.
    """
    slots = src.shape[0]
    entry = 2 * _LAYER_NUM * _HEADS * _HEAD_DIM
    backing = torch.zeros(slots * entry, dtype=src.dtype, device=src.device)
    view = backing.as_strided((slots, _HEADS, _HEAD_DIM), (entry, _HEAD_DIM, 1))
    view.copy_(src)
    assert view.stride(0) == entry != src.stride(0)
    assert torch.equal(view.view(torch.uint8), src.view(torch.uint8))
    return view


def _reshape(k, v):
    from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend

    backend = TRTLLMHAAttnBackend.__new__(TRTLLMHAAttnBackend)
    backend.page_size = _PAGE_SIZE
    layer = SimpleNamespace(tp_k_head_num=_HEADS, tp_v_head_num=_HEADS)
    return backend._reshape_paged_kv_cache(k, v, layer, _HEAD_DIM)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
@unittest.skipUnless(is_sm100_supported(), "trtllm-gen kernels require SM100")
class TestTRTLLMHAStrideParity(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.workspace = torch.zeros(
            128 * 1024 * 1024, dtype=torch.uint8, device="cuda"
        )
        self.block_tables = (
            torch.arange(_PAGES, dtype=torch.int32, device="cuda")
            .repeat(_BS, 1)
            .contiguous()
        )

    def _assert_stride_independent(self, run, kv_dtype):
        k_contig = _contiguous_cache(kv_dtype, "cuda")
        v_contig = _contiguous_cache(kv_dtype, "cuda")
        want = run(_reshape(k_contig, v_contig))
        got = run(_reshape(_strided_like(k_contig), _strided_like(v_contig)))
        # Bit-for-bit: a tolerance here would hide the bug.
        self.assertTrue(
            torch.equal(want, got),
            f"trtllm_mha depends on the KV slot stride: "
            f"max |diff| = {(want.float() - got.float()).abs().max().item()}",
        )

    def test_decode_output_is_identical_across_slot_strides(self):
        import flashinfer

        for kv_dtype in (torch.bfloat16, torch.float8_e4m3fn):
            with self.subTest(kv_dtype=kv_dtype):
                q = torch.randn(_BS, _Q_HEADS, _HEAD_DIM, device="cuda").to(
                    torch.bfloat16
                )
                seq_lens = torch.tensor(
                    [_PAGES * _PAGE_SIZE - 5, _PAGES * _PAGE_SIZE],
                    dtype=torch.int32,
                    device="cuda",
                )

                def run(kv_cache):
                    return flashinfer.decode.trtllm_batch_decode_with_kv_cache(
                        query=q,
                        kv_cache=kv_cache,
                        workspace_buffer=self.workspace,
                        block_tables=self.block_tables,
                        seq_lens=seq_lens,
                        max_seq_len=_PAGES * _PAGE_SIZE,
                        bmm1_scale=_HEAD_DIM**-0.5,
                        bmm2_scale=1.0,
                        window_left=-1,
                        out_dtype=torch.bfloat16,
                    )

                self._assert_stride_independent(run, kv_dtype)

    def test_context_output_is_identical_across_slot_strides(self):
        import flashinfer

        q_lens = [5, 9]
        kv_lens = [_PAGES * _PAGE_SIZE - 3, _PAGES * _PAGE_SIZE]
        # The context kernel takes Q in the KV dtype.
        for kv_dtype in (torch.bfloat16, torch.float8_e4m3fn):
            with self.subTest(kv_dtype=kv_dtype):
                q = torch.randn(sum(q_lens), _Q_HEADS, _HEAD_DIM, device="cuda").to(
                    kv_dtype
                )
                seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
                cum_seq_lens_q = torch.tensor(
                    [0, q_lens[0], sum(q_lens)], dtype=torch.int32, device="cuda"
                )
                cum_seq_lens_kv = torch.tensor(
                    [0, kv_lens[0], sum(kv_lens)], dtype=torch.int32, device="cuda"
                )

                def run(kv_cache):
                    return flashinfer.prefill.trtllm_batch_context_with_kv_cache(
                        query=q,
                        kv_cache=kv_cache,
                        workspace_buffer=self.workspace,
                        block_tables=self.block_tables,
                        seq_lens=seq_lens,
                        max_q_len=max(q_lens),
                        max_kv_len=max(kv_lens),
                        bmm1_scale=_HEAD_DIM**-0.5,
                        bmm2_scale=1.0,
                        batch_size=_BS,
                        cum_seq_lens_q=cum_seq_lens_q,
                        cum_seq_lens_kv=cum_seq_lens_kv,
                        window_left=-1,
                        out_dtype=torch.bfloat16,
                    )

                self._assert_stride_independent(run, kv_dtype)


if __name__ == "__main__":
    unittest.main()
