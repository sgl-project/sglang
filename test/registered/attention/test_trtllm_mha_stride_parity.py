"""trtllm_mha decode output must not depend on the KV cache's slot stride.

`_reshape_paged_kv_cache` hands the trtllm-gen kernel a view whose in-page
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
the same values; the outputs must match bit for bit.

Skips below SM100: the trtllm-gen decode cubins are Blackwell-first, so on
SM90 this asserts nothing and the backend stays unvalidated there.

    python -m pytest test/registered/attention/test_trtllm_mha_stride_parity.py -v
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.utils import is_sm100_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

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
    return torch.randn(slots, _HEADS, _HEAD_DIM, dtype=dtype, device=device)


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
    assert torch.equal(view, src)
    return view


def _reshape(k, v):
    from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend

    backend = TRTLLMHAAttnBackend.__new__(TRTLLMHAAttnBackend)
    backend.page_size = _PAGE_SIZE
    layer = SimpleNamespace(tp_k_head_num=_HEADS, tp_v_head_num=_HEADS)
    return backend._reshape_paged_kv_cache(k, v, layer, _HEAD_DIM)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
@unittest.skipUnless(
    is_sm100_supported(), "trtllm-gen decode cubins require SM100 or newer"
)
class TestTRTLLMHAStrideParity(unittest.TestCase):
    def test_decode_output_is_identical_across_slot_strides(self):
        import flashinfer

        device, dtype = "cuda", torch.bfloat16
        torch.manual_seed(0)

        k_contig = _contiguous_cache(dtype, device)
        v_contig = _contiguous_cache(dtype, device)
        k_strided = _strided_like(k_contig)
        v_strided = _strided_like(v_contig)

        q = torch.randn(_BS, _Q_HEADS, _HEAD_DIM, dtype=dtype, device=device)
        block_tables = (
            torch.arange(_PAGES, dtype=torch.int32, device=device)
            .repeat(_BS, 1)
            .contiguous()
        )
        seq_lens = torch.full(
            (_BS,), _PAGES * _PAGE_SIZE, dtype=torch.int32, device=device
        )
        workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        scale = _HEAD_DIM**-0.5

        def run(k, v):
            return flashinfer.decode.trtllm_batch_decode_with_kv_cache(
                query=q,
                kv_cache=_reshape(k, v),
                workspace_buffer=workspace,
                block_tables=block_tables,
                seq_lens=seq_lens,
                max_seq_len=_PAGES * _PAGE_SIZE,
                bmm1_scale=scale,
                bmm2_scale=1.0,
                window_left=-1,
            )

        want = run(k_contig, v_contig)
        got = run(k_strided, v_strided)
        # Bit-for-bit: a tolerance here would hide the bug.
        self.assertTrue(
            torch.equal(want, got),
            f"trtllm_mha decode depends on the KV slot stride: "
            f"max |diff| = {(want.float() - got.float()).abs().max().item()}",
        )


if __name__ == "__main__":
    unittest.main()
