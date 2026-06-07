"""Correctness of the b12x CuTe-DSL GLM_NSA sparse MLA kernel vs the flashinfer
trtllm-gen sparse MLA ground truth (decode + extend), at topk=2048.

The b12x kernel is the candidate for the `cutedsl_sparse_mla` attention backend.
flashinfer `trtllm_batch_decode_with_kv_cache_mla(..., sparse_mla_top_k=...)` is
the reference. Both consume absorbed-space q [rows, heads, 576] (512 latent + 64
rope), latent KV (512 ckv + 64 kpe), and token-absolute top-k indices. trtllm is
fed the dequantized (unpack(pack)) KV so both see identical values.
"""

import sys
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, suite="base-b-test-4-gpu-b200")

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_NOPE_HEAD_DIM = 128
HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM
SM_SCALE = (QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM) ** -0.5
TOPK = 2048
NUM_HEADS = 128
WORKSPACE_BYTES = 150 * 1024 * 1024
_B12X_PATH = "/sgl-workspace/b12x"


def _load_b12x():
    if _B12X_PATH not in sys.path:
        sys.path.insert(0, _B12X_PATH)
    import b12x.cute.fp4 as fp4

    fp4.get_sm_version = lambda device=None: 120  # force unified_sm120 on sm10x
    from b12x.attention.mla.reference import (
        dense_mla_reference,
        pack_mla_kv_cache_reference,
        unpack_mla_kv_cache_reference,
    )
    from b12x.attention.workspace import B12XAttentionWorkspace
    from b12x.integration.mla import (
        sparse_mla_decode_forward,
        sparse_mla_extend_forward,
    )

    return dict(
        pack=pack_mla_kv_cache_reference,
        unpack=unpack_mla_kv_cache_reference,
        dense=dense_mla_reference,
        Workspace=B12XAttentionWorkspace,
        decode=sparse_mla_decode_forward,
        extend=sparse_mla_extend_forward,
    )


def _skip_reason():
    if not torch.cuda.is_available():
        return "CUDA required"
    if torch.cuda.get_device_capability()[0] < 10:
        return "requires SM100+ (Blackwell) for trtllm-gen sparse MLA"
    try:
        import flashinfer.decode as d

        if not hasattr(d, "trtllm_batch_decode_with_kv_cache_mla"):
            return "flashinfer trtllm MLA decode unavailable"
    except Exception as e:
        return f"flashinfer unavailable: {e}"
    try:
        _load_b12x()
    except Exception as e:
        return f"b12x unavailable: {e}"
    return None


def _cos(a, b):
    a_f = a.to(torch.float32).reshape(-1)
    b_f = b.to(torch.float32).reshape(-1)
    diff = (a_f - b_f).abs()
    return (
        torch.nn.functional.cosine_similarity(a_f, b_f, dim=0).item(),
        diff.max().item(),
    )


def _make_inputs(*, rows, cache_len, seed, device):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    k_nope = (torch.randn(cache_len, 1, KV_LORA_RANK, generator=gen) * 0.5).to(
        device, torch.bfloat16
    )
    k_rope = (torch.randn(cache_len, 1, QK_ROPE_HEAD_DIM, generator=gen) * 0.5).to(
        device, torch.bfloat16
    )
    q_all = (torch.randn(rows, NUM_HEADS, HEAD_DIM, generator=gen) * 0.5).to(
        device, torch.bfloat16
    )
    idx = [
        torch.sort(torch.randperm(cache_len, generator=gen)[:TOPK]).values
        for _ in range(rows)
    ]
    page_table = torch.stack(idx, 0).to(device, torch.int32)
    return k_nope, k_rope, q_all, page_table


def _trtllm(
    *, b12x, q_all, k_nope, k_rope, page_table, cache_len, batch, q_len, ws, device
):
    import flashinfer.decode as fi

    packed = b12x["pack"](k_nope, k_rope)
    unpacked = b12x["unpack"](packed).squeeze(1).to(torch.bfloat16)
    page_size = 64
    num_pages = (cache_len + page_size - 1) // page_size
    kv = torch.zeros(
        num_pages, page_size, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    tok = torch.arange(cache_len, device=device)
    kv[tok // page_size, tok % page_size] = unpacked
    out = fi.trtllm_batch_decode_with_kv_cache_mla(
        query=q_all.view(batch, q_len, NUM_HEADS, HEAD_DIM),
        kv_cache=kv,
        workspace_buffer=ws,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        block_tables=page_table.view(batch, q_len, TOPK),
        seq_lens=torch.full((batch,), cache_len, dtype=torch.int32, device=device),
        max_seq_len=cache_len,
        sparse_mla_top_k=TOPK,
        bmm1_scale=SM_SCALE,
        bmm2_scale=1.0,
        backend="trtllm-gen",
    )
    return out.view(batch * q_len, NUM_HEADS, KV_LORA_RANK)


def _b12x_run(*, b12x, mode, q_all, k_nope, k_rope, page_table, cache_len, device):
    packed = b12x["pack"](k_nope, k_rope)
    rows = page_table.shape[0]
    ws = b12x["Workspace"].for_fixed_capacity(
        mode=mode,
        device=device,
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        num_q_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        v_head_dim=KV_LORA_RANK,
        topk=TOPK,
        max_total_q=rows,
        max_batch=rows,
    )
    seqlens = torch.full((rows,), cache_len, dtype=torch.int32, device=device)
    nsa = torch.full((rows,), TOPK, dtype=torch.int32, device=device)
    if mode == "decode":
        return b12x["decode"](
            q_all=q_all,
            kv_cache=packed,
            page_table_1=page_table,
            cache_seqlens_int32=seqlens,
            nsa_cache_seqlens_int32=nsa,
            workspace=ws,
            sm_scale=SM_SCALE,
            v_head_dim=KV_LORA_RANK,
        )
    return b12x["extend"](
        q_all=q_all,
        kv_cache=packed,
        selected_token_offsets=page_table,
        cache_seqlens_int32=seqlens,
        nsa_cache_seqlens_int32=nsa,
        workspace=ws,
        sm_scale=SM_SCALE,
        v_head_dim=KV_LORA_RANK,
    )


class TestCuteDSLSparseMLA(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        reason = _skip_reason()
        if reason is not None:
            raise unittest.SkipTest(reason)
        cls.device = torch.device("cuda")
        cls.b12x = _load_b12x()
        cls.ws = torch.zeros(WORKSPACE_BYTES, dtype=torch.int8, device=cls.device)

    def _check(self, *, mode, batch, q_len, cache_len, seed):
        rows = batch * q_len
        k_nope, k_rope, q_all, pt = _make_inputs(
            rows=rows, cache_len=cache_len, seed=seed, device=self.device
        )
        out_trt = _trtllm(
            b12x=self.b12x,
            q_all=q_all,
            k_nope=k_nope,
            k_rope=k_rope,
            page_table=pt,
            cache_len=cache_len,
            batch=batch,
            q_len=q_len,
            ws=self.ws,
            device=self.device,
        )
        out_b12x = _b12x_run(
            b12x=self.b12x,
            mode=mode,
            q_all=q_all,
            k_nope=k_nope,
            k_rope=k_rope,
            page_table=pt,
            cache_len=cache_len,
            device=self.device,
        )
        torch.cuda.synchronize()
        cos, max_abs = _cos(out_b12x, out_trt)
        self.assertGreaterEqual(cos, 0.999, f"{mode} cos={cos:.6f}")
        self.assertLessEqual(max_abs, 0.05, f"{mode} max_abs={max_abs:.5f}")

    def test_decode(self):
        for i, (batch, cache_len) in enumerate([(4, 4096), (2, 8192)]):
            with self.subTest(batch=batch, cache_len=cache_len):
                self._check(
                    mode="decode",
                    batch=batch,
                    q_len=1,
                    cache_len=cache_len,
                    seed=100 + i,
                )

    def test_extend(self):
        for i, (batch, q_len, cache_len) in enumerate([(1, 4, 4096), (1, 16, 8192)]):
            with self.subTest(batch=batch, q_len=q_len, cache_len=cache_len):
                self._check(
                    mode="extend",
                    batch=batch,
                    q_len=q_len,
                    cache_len=cache_len,
                    seed=200 + i,
                )


if __name__ == "__main__":
    unittest.main(verbosity=3)
