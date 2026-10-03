"""Regression test for https://github.com/sgl-project/sglang/issues/40903.

The DeepSeek chunked-prefix MHA path merges per-chunk attention states with
merge_state (merge_state_v2 / Triton), which expects natural-log LSE. The
flashinfer ragged prefill wrapper and trtllm_ragged_attention_deepseek return
base-2 LSE, so the MLA backends must convert before handing the LSE back.
"""

import math
import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")

# DeepSeek-V3 MHA prefill shapes on a TP8 slice.
H, DQK, DV = 16, 192, 128
SCALE = 1.0 / math.sqrt(DQK)


def _layer():
    return SimpleNamespace(
        tp_q_head_num=H,
        tp_k_head_num=H,
        tp_v_head_num=H,
        head_dim=DQK,
        v_head_dim=DV,
        scaling=SCALE,
        logit_cap=0.0,
    )


def _inputs(qlen, kvlen, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = (torch.randn(qlen, H, DQK, device="cuda", generator=g) * 0.5).bfloat16()
    k = (torch.randn(kvlen, H, DQK, device="cuda", generator=g) * 0.5).bfloat16()
    v = torch.randn(kvlen, H, DV, device="cuda", generator=g).bfloat16()
    return q, k, v


def _reference(q, k, v):
    s = torch.einsum("qhd,khd->qhk", q.float(), k.float()) * SCALE
    out = torch.einsum("qhk,khd->qhd", torch.softmax(s, dim=-1), v.float())
    return out, torch.logsumexp(s, dim=-1)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestChunkedPrefixLSEBase(CustomTestCase):
    def setUp(self):
        self.ws = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")

    def _flashinfer_chunk(self, q, k, v):
        from flashinfer import BatchPrefillWithRaggedKVCacheWrapper

        from sglang.srt.layers.attention.flashinfer_mla_backend import (
            FlashInferMhaChunkKVRunner,
        )

        wrapper = BatchPrefillWithRaggedKVCacheWrapper(self.ws, "NHD")
        wrapper.plan(
            torch.tensor([0, q.shape[0]], dtype=torch.int32, device="cuda"),
            torch.tensor([0, k.shape[0]], dtype=torch.int32, device="cuda"),
            H,
            H,
            DQK,
            head_dim_vo=DV,
            causal=False,
            sm_scale=SCALE,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )
        runner = SimpleNamespace(chunk_ragged_wrappers=[wrapper])
        forward_batch = SimpleNamespace(
            attn_attend_prefix_cache=True, prefix_chunk_idx=0, mha_return_lse=True
        )
        return FlashInferMhaChunkKVRunner.forward(
            runner, q, k, v, _layer(), forward_batch
        )

    def _trtllm_ragged(self, q, k, v):
        from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLABackend

        m, n = q.shape[0], k.shape[0]
        backend = SimpleNamespace(data_type=torch.bfloat16, workspace_buffer=self.ws)
        return TRTLLMMLABackend._run_prefill_kernel(
            backend,
            q=q,
            k=k,
            v=v,
            layer=_layer(),
            batch_size=1,
            cum_seq_lens_q=torch.tensor([0, m], dtype=torch.int32, device="cuda"),
            max_q_len=m,
            seq_lens_kv=torch.tensor([n], dtype=torch.int32, device="cuda"),
            cum_seq_lens_kv=torch.tensor([0, n], dtype=torch.int32, device="cuda"),
            max_kv_len=n,
            is_causal=False,
            return_lse=True,
            out_buffer=torch.empty(m, H, DV, dtype=torch.bfloat16, device="cuda"),
            o_sf_scale=-1.0,
        )

    def _check_lse_and_merge(self, attn):
        from sglang.srt.layers.attention.merge_state import merge_state

        q, k, v = _inputs(8, 512)
        _, lse = attn(q, k, v)
        _, ref_lse = _reference(q, k, v)
        torch.testing.assert_close(lse.float(), ref_lse, atol=1e-3, rtol=0)

        # Long cached prefix + short new chunk: the split where a base mismatch
        # hurts most (issue #40903 measured ~12% mean relative error).
        plen, slen = 3072, 64
        q, k, v = _inputs(8, plen + slen, seed=1)
        o_p, l_p = attn(q, k[:plen].contiguous(), v[:plen].contiguous())
        o_s, l_s = attn(q, k[plen:].contiguous(), v[plen:].contiguous())
        merged, _ = merge_state(o_p, l_p, o_s, l_s)
        ref, _ = _reference(q, k, v)
        rel = (merged.float() - ref).abs().mean() / ref.abs().mean()
        self.assertLess(rel.item(), 0.01)

    def test_flashinfer_chunk_runner_returns_natural_log_lse(self):
        self._check_lse_and_merge(self._flashinfer_chunk)

    @unittest.skipIf(
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 10,
        "trtllm_ragged_attention_deepseek requires SM100",
    )
    def test_trtllm_prefill_kernel_returns_natural_log_lse(self):
        self._check_lse_and_merge(self._trtllm_ragged)


if __name__ == "__main__":
    unittest.main()
