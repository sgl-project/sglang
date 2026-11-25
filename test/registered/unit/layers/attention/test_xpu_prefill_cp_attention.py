"""Zigzag prefill CP through the intel_xpu backend, with the kernel stubbed.

The real ZigzagCPStrategy drives XPUAttentionBackend._forward_extend_prefill_cp.
Each of the rank's two query halves must attend with that half's own query
offsets and KV lengths (not the batch's), against the full paged cache.
"""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

# Loaded before the patch.dict stub below, which unloads every module first
# imported inside it while their torch op registrations stay.
import sglang.srt.model_executor.forward_batch_info  # noqa: F401
from sglang.test.ci.ci_register import register_cpu_ci

with patch.dict(
    sys.modules,
    {
        module: MagicMock()
        for module in (
            "sgl_kernel",
            "sgl_kernel.flash_attn",
            "sgl_kernel.quantization",
            "sgl_kernel.scalar_type",
        )
    },
):
    from sglang.srt.layers.attention import xpu_backend
    from sglang.srt.layers.attention.xpu_backend import XPUAttentionBackend
    from sglang.srt.layers.cp.zigzag import ZigzagCPStrategy

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HQ, D = 2, 4


def _layer():
    return SimpleNamespace(
        tp_q_head_num=HQ, head_dim=D, v_head_dim=D, scaling=0.5, logit_cap=0.0
    )


def _forward_batch(prev_tokens, next_tokens):
    i32 = lambda values: torch.tensor(values, dtype=torch.int32)
    return SimpleNamespace(
        attn_cp_metadata=SimpleNamespace(
            total_q_prev_tokens=prev_tokens,
            total_q_next_tokens=next_tokens,
            cu_seqlens_q_prev_tensor=i32([0, prev_tokens]),
            kv_len_prev_tensor=i32([10]),
            max_seqlen_q_prev=prev_tokens,
            cu_seqlens_q_next_tensor=i32([0, next_tokens]),
            kv_len_next_tensor=i32([16]),
            max_seqlen_q_next=next_tokens,
        )
    )


class TestXpuPrefillCpAttention(unittest.TestCase):
    def setUp(self):
        self.backend = XPUAttentionBackend.__new__(XPUAttentionBackend)
        self.backend.device = torch.device("cpu")
        self.calls = []

        def fake_kernel(**kwargs):
            self.calls.append(kwargs)
            # Tag each half's output with its KV length to check the stitch order.
            return torch.full_like(kwargs["q"], float(kwargs["cache_seqlens"][0]))

        patcher = patch.multiple(
            xpu_backend,
            flash_attn_with_kvcache=fake_kernel,
            get_cp_strategy=lambda: ZigzagCPStrategy(cp_size=2),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _run(self, *, unsupported=False, causal=True, padded_rows=1):
        prev, nxt = 3, 2
        q = torch.randn(prev + nxt + padded_rows, HQ * D)
        page_table = torch.arange(8, dtype=torch.int32).view(1, 8)
        out = self.backend._forward_extend_prefill_cp(
            q=q,
            layer=_layer(),
            forward_batch=_forward_batch(prev, nxt),
            key_cache="k_cache",
            value_cache="v_cache",
            page_table=page_table,
            causal=causal,
            window_size=(-1, -1),
            unsupported=unsupported,
            kwargs={"num_splits": 1},
        )
        return q, page_table, out

    def test_each_half_attends_with_its_own_lengths(self):
        q, page_table, out = self._run()
        self.assertEqual(len(self.calls), 2)
        prev_call, next_call = self.calls
        self.assertEqual(prev_call["cu_seqlens_q"].tolist(), [0, 3])
        self.assertEqual(prev_call["cache_seqlens"].tolist(), [10])
        self.assertEqual(prev_call["max_seqlen_q"], 3)
        self.assertEqual(next_call["cu_seqlens_q"].tolist(), [0, 2])
        self.assertEqual(next_call["cache_seqlens"].tolist(), [16])
        self.assertEqual(next_call["max_seqlen_q"], 2)
        for call in self.calls:
            self.assertIs(call["page_table"], page_table)
            self.assertEqual(call["k_cache"], "k_cache")
            self.assertTrue(call["causal"])
            self.assertEqual(call["num_splits"], 1)
        # Halves stitched in order, padding rows zeroed, heads flattened back.
        self.assertEqual(out.shape, (6, HQ * D))
        self.assertEqual(out[:, 0].tolist(), [10.0] * 3 + [16.0] * 2 + [0.0])

    def test_unsupported_attention_kinds_are_refused(self):
        with self.assertRaisesRegex(NotImplementedError, "plain causal dense"):
            self._run(unsupported=True)
        self.assertEqual(self.calls, [])

    def test_non_causal_layers_are_refused(self):
        """Encoder-only and bidirectional layers are non-causal; the zigzag split
        bounds each half's KV to its causal prefix, so they must not run under CP."""
        with self.assertRaisesRegex(NotImplementedError, "plain causal dense"):
            self._run(causal=False)
        self.assertEqual(self.calls, [])

    def test_model_splitting_heads_by_full_tp_is_refused(self):
        """Llama splits KV heads by the full TP size, so under CP each rank's K/V
        has fewer heads than the attention-TP-sized pool and the store crashed
        with a bare reshape error."""
        self.backend._cp_pool_kv_heads = 8
        self.backend._check_cp_kv_heads(SimpleNamespace(layer_id=0, tp_k_head_num=8))
        with self.assertRaisesRegex(ValueError, "splits attention heads by the full"):
            self.backend._check_cp_kv_heads(
                SimpleNamespace(layer_id=3, tp_k_head_num=4)
            )


if __name__ == "__main__":
    unittest.main()
