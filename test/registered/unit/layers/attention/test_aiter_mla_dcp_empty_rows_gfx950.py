"""DCP decode on gfx950: rows with no KV on this rank (real asm and Gluon kernels)."""

import itertools
import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
from sglang.srt.layers.attention.aiter_mla_gluon import _gluon_fn
from sglang.srt.utils import get_device, is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="base-b-test-1-gpu-small-amd-mi35x")

NUM_HEADS, KV_LORA, ROPE = 96, 512, 64
HEAD_DIM = KV_LORA + ROPE
SCALING = HEAD_DIM**-0.5


def _reference(q, kv, kv_lens, offsets):
    # fp32 attention over each row's local KV -> (out, natural-log lse).
    out = torch.zeros(len(kv_lens), NUM_HEADS, KV_LORA, device=q.device)
    lse = torch.full((len(kv_lens), NUM_HEADS), -1e30, device=q.device)
    for i, n in enumerate(kv_lens):
        if n == 0:
            continue
        k = kv[offsets[i] : offsets[i] + n].float()
        s = (q[i].float() @ k.T) * SCALING
        out[i] = torch.softmax(s, dim=-1) @ k[:, :KV_LORA]
        lse[i] = torch.logsumexp(s, dim=-1)
    return out, lse


def _poison_cached_blocks(bs, device):
    # Leave NaN in cached blocks of the sizes the kernels' uninitialized out/lse buffers use;
    # holding several pairs at once keeps a stray allocation from taking the only one.
    poison = [
        (
            torch.full(
                (bs, NUM_HEADS, KV_LORA),
                float("nan"),
                dtype=torch.bfloat16,
                device=device,
            ),
            torch.full((bs, 1, NUM_HEADS), float("nan"), device=device),
        )
        for _ in range(4)
    ]
    del poison


def _kv_lens(bs):
    return [0 if i % 5 == 2 else 3 + (i % 4) for i in range(bs)]


def _layer():
    return SimpleNamespace(
        tp_q_head_num=NUM_HEADS,
        qk_head_dim=HEAD_DIM,
        v_head_dim=KV_LORA,
        scaling=SCALING,
        logit_cap=0.0,
        _aiter_kv_scale_float=1.0,
    )


def _indptr(kv_lens, device):
    return torch.tensor(
        [0] + list(itertools.accumulate(kv_lens)), dtype=torch.int32, device=device
    )


@unittest.skipUnless(is_hip() and is_gfx95_supported(), "DCP decode needs gfx950")
class TestMlaDcpDecodeEmptyRowsGfx950(CustomTestCase):
    def _check(self, out, lse, q, kv, kv_lens, *, atol):
        torch.cuda.synchronize()
        device = q.device
        offsets = [0] + list(itertools.accumulate(kv_lens))
        ref_out, ref_lse = _reference(q, kv, kv_lens, offsets)
        empty = torch.tensor([n == 0 for n in kv_lens], device=device)
        self.assertTrue(torch.all(out[empty] == 0))
        self.assertTrue(torch.all(lse[empty] == -1e30))
        torch.testing.assert_close(
            out[~empty].float(), ref_out[~empty], atol=atol, rtol=atol
        )
        torch.testing.assert_close(lse[~empty], ref_lse[~empty], atol=1e-3, rtol=1e-3)

    @unittest.skipUnless(_gluon_fn() is not None, "needs Triton with Gluon cga_layout")
    def test_gluon_rows_without_local_kv_return_a_zero_partial_with_no_weight(self):
        """A DCP rank whose shard of a short request is empty must not hand the merge stale memory."""
        from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend

        device = get_device()
        torch.manual_seed(0)
        bs = 64  # bs * ceil(96 / 16) exceeds the CU count, so there is one KV split
        kv_lens = _kv_lens(bs)
        total = sum(kv_lens)
        kv = torch.randn(total, HEAD_DIM, dtype=torch.bfloat16, device=device)
        q = torch.randn(bs, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
        be = AiterAttnBackend.__new__(AiterAttnBackend)
        be.use_mla_dcp_asm = False
        be.forward_metadata = SimpleNamespace(
            kv_indptr=_indptr(kv_lens, device),
            kv_indices=torch.arange(total, dtype=torch.int32, device=device),
        )

        _poison_cached_blocks(bs, device)
        out, lse = be._forward_decode_dcp(q, kv, _layer(), k_descale=1.0)
        self._check(out, lse, q, kv, kv_lens, atol=2e-2)

    def test_asm_rows_without_local_kv_return_a_zero_partial_with_no_weight(self):
        """The asm DCP decode also stores nothing for a row with no KV on this rank."""
        from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend

        device = get_device()
        torch.manual_seed(0)
        bs = 8
        kv_lens = _kv_lens(bs)
        total = sum(kv_lens)
        # fp8 KV and Q at unit scale; the reference sees the same quantized values.
        kv = torch.randn(total, HEAD_DIM, device=device).to(fp8_dtype)
        q = torch.randn(bs, NUM_HEADS, HEAD_DIM, device=device).to(fp8_dtype)
        kv_indptr = _indptr(kv_lens, device)

        be = AiterAttnBackend.__new__(AiterAttnBackend)
        be.page_size = 1
        be.kv_cache_dtype = fp8_dtype
        be.mla_kernel_num_head_padded = NUM_HEADS
        be.max_split_per_batch = 64
        be.fix_max_split_per_batch = 64
        be.attn_dp_enabled = False
        be.use_mla_auto_kv_splits = False
        be.use_mla_dcp_asm = True
        be.dcp_world_size = 8
        be.input_dtype = torch.bfloat16
        qo_indptr = torch.arange(bs + 1, dtype=torch.int32, device=device)
        last_page_len = torch.ones(bs, dtype=torch.int32, device=device)
        meta = be.make_mla_decode_meta_data_buffer(
            1, bs, metadata_fast_mode=True, metadata_intra_batch_mode=False
        )
        be.make_mla_meta_data(
            qo_indptr,
            kv_indptr,
            last_page_len,
            meta[0],
            meta[2],
            meta[1],
            meta[3],
            meta[4],
            meta[5],
            1,
            fast_mode=True,
            max_split_per_batch=64,
            intra_batch_mode=False,
        )
        be.forward_metadata = SimpleNamespace(
            kv_indptr=kv_indptr,
            kv_indices=torch.arange(total, dtype=torch.int32, device=device),
            kv_last_page_len=last_page_len,
            qo_indptr=qo_indptr,
            max_q_len=1,
            work_metadata=meta[0],
            work_indptr=meta[1],
            work_info_set=meta[2],
            reduce_indptr=meta[3],
            reduce_final_map=meta[4],
            reduce_partial_map=meta[5],
            num_kv_splits=64,
        )

        _poison_cached_blocks(bs, device)
        out, lse = be._forward_decode_dcp(
            q, kv, _layer(), k_descale=torch.ones((), device=device)
        )
        self._check(out, lse, q.float(), kv.float(), kv_lens, atol=1.5e-1)


if __name__ == "__main__":
    unittest.main()
