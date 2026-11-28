"""fp8-Q ASM MLA on gfx950: the real quant and persistent ASM kernels against an fp32 reference."""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.arch import current_arch
from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="base-b-test-1-gpu-small-amd-mi35x")


def _on_gfx950():
    hip = current_arch().hip
    return hip is not None and hip.gcn_arch == "gfx950"


NUM_HEADS, NUM_HEADS_PADDED, KV_LORA, ROPE = 12, 16, 512, 64
HEAD_DIM = KV_LORA + ROPE
SCALING = HEAD_DIM**-0.5
KV_LENS = [37, 200, 1, 513]
# Measured max abs error on MI355X (aiter 0.1.29.dev157): 0.031 at decode, 0.050-0.060 at qlen 5-8;
# the kernel casts P to fp8, and 0.1 keeps about 1.7x margin on outputs up to ~4.
ATOL = 0.1


def _reference(q, kv, kv_lens, qlen):
    # fp32 causal attention on the fp8-rounded operands the kernel reads.
    out = torch.zeros(len(kv_lens) * qlen, NUM_HEADS, KV_LORA, device=q.device)
    offset = 0
    for i, n in enumerate(kv_lens):
        k = kv[offset : offset + n].float()
        for j in range(qlen):
            visible = n - qlen + j + 1
            s = (q[i * qlen + j].float() @ k[:visible].T) * SCALING
            out[i * qlen + j] = torch.softmax(s, dim=-1) @ k[:visible, :KV_LORA]
        offset += n
    return out


@unittest.skipUnless(_on_gfx950(), "fp8-Q ASM MLA needs gfx950 and aiter")
class TestFp8QAsmMlaNumerics(CustomTestCase):
    def _run(self, *, qlen, auto_kv_splits, q_scale_value=1.0, q_amp=1.0):
        from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend

        torch.manual_seed(0)
        device = "cuda"
        bs = len(KV_LENS)
        # Verify attends the draft tokens, which sit at the end of each request's KV.
        kv_lens = [n + qlen - 1 if qlen > 1 else n for n in KV_LENS]
        total = sum(kv_lens)
        # Row 0 is a sentinel the wrapper zeroes in place; real KV starts at row 1.
        kv_all = torch.randn(total + 1, HEAD_DIM, device=device).to(fp8_dtype)
        kv_before = kv_all[1:].clone()
        q = (
            torch.randn(bs * qlen, NUM_HEADS, HEAD_DIM, device=device) * q_amp
        ).bfloat16()

        be = AiterAttnBackend.__new__(AiterAttnBackend)
        be.num_head = NUM_HEADS
        be.num_head_padded = NUM_HEADS_PADDED
        be.mla_kernel_num_head_padded = NUM_HEADS_PADDED
        be.head_pad_mode = "zero"
        be.kv_cache_dtype = fp8_dtype
        be.dcp_world_size = 1
        be.input_dtype = torch.bfloat16
        be.use_mla_fp8_q_asm = True
        be.use_mla_auto_kv_splits = auto_kv_splits
        be.max_split_per_batch = -1 if auto_kv_splits else 32
        be.fix_max_split_per_batch = be.max_split_per_batch
        be.attn_dp_enabled = False
        be.page_size = 1
        be.q_scale = torch.tensor([q_scale_value], dtype=torch.float32, device=device)
        be.forward_metadata = SimpleNamespace(num_kv_splits=32)
        # The route plans with the PERSIST=1 modes whatever the module modes are.
        modes = be._mla_decode_metadata_modes()

        qo_indptr = torch.arange(
            0, (bs + 1) * qlen, qlen, dtype=torch.int32, device=device
        )
        kv_indptr = torch.tensor(
            [0, *torch.tensor(kv_lens).cumsum(0).tolist()],
            dtype=torch.int32,
            device=device,
        )
        kv_indices = torch.arange(1, total + 1, dtype=torch.int32, device=device)
        last_page = torch.ones(bs, dtype=torch.int32, device=device)
        bufs = be.make_mla_decode_meta_data_buffer(
            qlen,
            bs,
            metadata_fast_mode=modes[0],
            metadata_intra_batch_mode=modes[1],
        )
        work_metadata, work_indptr, work_info_set, *reduce_bufs = bufs
        be.make_mla_meta_data(
            qo_indptr,
            kv_indptr,
            last_page,
            work_metadata,
            work_info_set,
            work_indptr,
            *reduce_bufs,
            qlen,
            modes[0],
            be.max_split_per_batch,
            modes[1],
        )
        layer = SimpleNamespace(tp_q_head_num=NUM_HEADS, v_head_dim=KV_LORA)
        q_fp8_view = be._quantize_q_fp8(q.view(bs * qlen, NUM_HEADS, HEAD_DIM))
        q_ref = (
            q_fp8_view[0].view(bs * qlen, NUM_HEADS, HEAD_DIM).float() * q_fp8_view[1]
        )
        schedule = be._mla_decode_launch_schedule(default_num_kv_splits=None)
        out = be._mla_decode_fwd_with_head_pad(
            q,
            kv_all.view(-1, 1, 1, HEAD_DIM),
            layer,
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            kv_last_page_lens=last_page,
            max_seqlen_q=qlen,
            sm_scale=SCALING,
            logit_cap=0.0,
            work_meta_data=bufs[0],
            work_indptr=bufs[1],
            work_info_set=bufs[2],
            reduce_indptr=bufs[3],
            reduce_final_map=bufs[4],
            reduce_partial_map=bufs[5],
            q_scale=torch.ones((), dtype=torch.float32, device=device),
            kv_scale=torch.ones((), dtype=torch.float32, device=device),
            **schedule,
        )
        # The reference reads the KV as it was before the launch, and the launch must not change it.
        self.assertTrue(torch.equal(kv_all[1:].float(), kv_before.float()))
        ref = _reference(q_ref, kv_before, kv_lens, qlen)
        return out.float(), ref

    def test_decode_and_verify_windows_match_fp32(self):
        for qlen in (1, 5, 6, 7, 8):
            for auto_kv_splits in (False, True):
                with self.subTest(qlen=qlen, auto_kv_splits=auto_kv_splits):
                    out, ref = self._run(qlen=qlen, auto_kv_splits=auto_kv_splits)
                    self.assertTrue(torch.isfinite(out).all())
                    torch.testing.assert_close(out, ref, atol=ATOL, rtol=0)

    def test_out_of_range_query_saturates_instead_of_nan(self):
        out, _ = self._run(qlen=1, auto_kv_splits=False, q_amp=1e4)
        self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
