"""Exercise the SGLang HD128 route with quantized KV and non-unit scales."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=45, suite="stage-b-test-1-gpu-small-amd")


@unittest.skipUnless(torch.version.hip, "ROCm ASM prefill")
class TestMiniMaxHD128AsmPrefill(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers.attention import aiter_backend
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        if "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
            raise unittest.SkipTest("HD128 ASM prefill requires gfx950")
        cls.ab = aiter_backend
        cls.extend_mode = ForwardMode.EXTEND

    def exercise(self, q_lens, k_lens, kv_heads=1):
        torch.manual_seed(4102)
        torch.backends.cuda.matmul.allow_tf32 = False
        device = "cuda"
        dtype = self.ab.fp8_dtype
        q_heads = kv_heads * 16
        slots = sum(k_lens) + 128
        token_ids = torch.randperm(slots, device=device)[: sum(k_lens)].to(torch.int32)
        k_cache = torch.randn(slots, kv_heads, 128, device=device).to(dtype)
        v_cache = torch.randn(slots, kv_heads, 128, device=device).to(dtype)
        q = torch.randn(sum(q_lens), q_heads, 128, device=device, dtype=torch.bfloat16)
        k_scale = torch.tensor([0.25], dtype=torch.float32, device=device)
        v_scale = torch.tensor([1.5], dtype=torch.float32, device=device)
        cu_q = torch.tensor(
            [0, *torch.tensor(q_lens).cumsum(0).tolist()],
            device=device,
            dtype=torch.int32,
        )
        cu_k = torch.tensor(
            [0, *torch.tensor(k_lens).cumsum(0).tolist()],
            device=device,
            dtype=torch.int32,
        )
        backend = self.ab.AiterAttnBackend.__new__(self.ab.AiterAttnBackend)
        backend.use_mla = False
        backend._use_unified_verify = False
        backend.kv_cache_dtype = dtype
        backend.kv_cache_is_vectorized_5d = False
        backend.input_dtype = torch.bfloat16
        backend.k_scale = torch.ones(1, device=device)
        backend.v_scale = torch.ones(1, device=device)
        backend.qo_indptr = cu_q
        backend.token_to_kv_pool = SimpleNamespace(
            get_kv_buffer=lambda _: (k_cache, v_cache)
        )
        backend.forward_metadata = self.ab.ForwardMetadata(
            kv_indptr=cu_k,
            kv_indices=token_ids,
            qo_indptr=cu_q,
            kv_last_page_len=None,
            max_q_len=max(q_lens),
            max_kv_len=max(k_lens),
        )
        batch = SimpleNamespace(
            batch_size=len(q_lens),
            forward_mode=self.extend_mode,
            attn_attend_prefix_cache=False,
            out_cache_loc=None,
            spec_info=None,
            seq_lens_cpu=torch.tensor(k_lens, dtype=torch.int32),
            seq_lens=torch.tensor(k_lens, dtype=torch.int32, device=device),
            extend_prefix_lens_cpu=[k - q for k, q in zip(k_lens, q_lens)],
        )
        layer = SimpleNamespace(
            layer_id=0,
            logit_cap=0.0,
            is_cross_attention=False,
            sliding_window_size=-1,
            qk_head_dim=128,
            v_head_dim=128,
            tp_q_head_num=q_heads,
            tp_k_head_num=kv_heads,
            tp_v_head_num=kv_heads,
            scaling=128**-0.5,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        # Deliberately different raw K/V: the route must read the quantized pool,
        # including on the first chunk, rather than cast these inputs again.
        raw_k = torch.zeros(
            sum(q_lens), kv_heads, 128, device=device, dtype=torch.bfloat16
        )
        raw_v = torch.zeros_like(raw_k)
        with patch.dict(os.environ):
            os.environ.pop("SGLANG_AITER_ASM_PREFILL_HD128", None)
            with patch.object(
                self.ab,
                "flash_attn_varlen_fp8_pertensor_func",
                wraps=self.ab.flash_attn_varlen_fp8_pertensor_func,
            ) as asm:
                output = backend.forward_extend(
                    q, raw_k, raw_v, layer, batch, save_kv_cache=False
                ).view_as(q)
                self.assertEqual(asm.call_count, 1)

        query_offset = key_offset = 0
        for q_len, k_len in zip(q_lens, k_lens):
            sample = sorted({0, q_len // 2, q_len - 1})
            physical = token_ids[key_offset : key_offset + k_len].long()
            keys = k_cache.view(torch.uint8)[physical].view(dtype).float() * k_scale
            values = v_cache.view(torch.uint8)[physical].view(dtype).float() * v_scale
            limit = k_len - q_len + torch.tensor(sample, device=device) + 1
            allowed = torch.arange(k_len, device=device)[None, :] < limit[:, None]
            for head in range(kv_heads):
                head_slice = slice(head * 16, (head + 1) * 16)
                query = (
                    q[query_offset : query_offset + q_len][sample, head_slice]
                    .to(dtype)
                    .float()
                )
                scores = (
                    torch.einsum("qhd,kd->hqk", query, keys[:, head]) * layer.scaling
                )
                scores.masked_fill_(~allowed[None, :, :], -float("inf"))
                reference = torch.einsum(
                    "hqk,kd->qhd", scores.softmax(-1), values[:, head]
                )
                actual = output[query_offset : query_offset + q_len][
                    sample, head_slice
                ].float()
                self.assertLess(
                    (actual - reference).abs().max().item(), 0.055 * v_scale.item()
                )
                self.assertLess(
                    ((actual - reference).norm() / reference.norm()).item(), 0.05
                )
            query_offset += q_len
            key_offset += k_len

    def test_first_chunk_reads_fp8_pool(self):
        self.exercise([7, 3], [7, 3])

    def test_ragged_context_and_two_kv_heads(self):
        self.exercise([4, 2], [129, 257], kv_heads=2)

    def test_one_million_token_context(self):
        self.exercise([4], [1048576])


if __name__ == "__main__":
    unittest.main()
