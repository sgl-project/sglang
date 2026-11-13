import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

# Loaded before the patch.dict stub below, which unloads every module first
# imported inside it while their torch op registrations stay.
import sglang.srt.model_executor.forward_batch_info  # noqa: F401
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_mode import ForwardMode
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

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestBidirectionalSlidingWindow(unittest.TestCase):
    """The flash-attention window is (left, right) even when causal=False.

    DiffusionGemma's denoising passes run its sliding-window layers as
    DECODER_BIDIRECTIONAL. A right bound of 0 there hid every later canvas
    token, so the layers computed causal attention.
    """

    HQ, HK, D, PAGE, WINDOW = 4, 2, 8, 4, 1023

    def _run_forward_extend(self, attn_type, forward_mode):
        backend = XPUAttentionBackend.__new__(XPUAttentionBackend)
        key_cache = torch.zeros(2 * self.PAGE, self.HK, self.D)
        backend.__dict__.update(
            use_mla=False,
            dcp_size=1,
            attention_chunk_size=None,
            topk=1,
            page_size=self.PAGE,
            num_splits=0,
            is_encoder_decoder=False,
            use_sliding_window_kv_pool=False,
            bidirectional_block_ids=None,
            bidirectional_sliding_layers_only=False,
            token_to_kv_pool=SimpleNamespace(
                layer_router=SimpleNamespace(
                    read=lambda _: SimpleNamespace(kv=lambda: (key_cache, key_cache))
                )
            ),
            forward_metadata=SimpleNamespace(
                local_attn_metadata=None,
                swa_spec_metadata=None,
                page_table=torch.zeros(1, 2, dtype=torch.int32),
                cu_seqlens_q=torch.tensor([0, 4], dtype=torch.int32),
                cache_seqlens_int32=torch.tensor([6], dtype=torch.int32),
                max_seq_len_q=4,
                max_seq_len_k=6,
                cu_seqlens_k=torch.tensor([0, 6], dtype=torch.int32),
            ),
        )
        layer = SimpleNamespace(
            layer_id=0,
            attn_type=attn_type,
            is_cross_attention=False,
            sliding_window_size=self.WINDOW,
            tp_q_head_num=self.HQ,
            tp_k_head_num=self.HK,
            tp_v_head_num=self.HK,
            head_dim=self.D,
            v_head_dim=self.D,
            scaling=1.0,
            logit_cap=0.0,
        )
        forward_batch = SimpleNamespace(
            forward_mode=forward_mode,
            encoder_lens=None,
            extend_prefix_lens_cpu=[2],
            _attn_output=None,
        )
        captured = {}

        def fake_kvcache(**kw):
            captured.update(kw)
            return kw["q"].new_zeros(kw["q"].shape)

        with patch.object(xpu_backend, "flash_attn_with_kvcache", fake_kvcache):
            backend.forward_extend(
                torch.zeros(4, self.HQ * self.D),
                None,
                None,
                layer,
                forward_batch,
                save_kv_cache=False,
            )
        return captured

    def test_bidirectional_sliding_layer_sees_both_sides(self):
        captured = self._run_forward_extend(
            AttentionType.DECODER_BIDIRECTIONAL, ForwardMode.DLLM_EXTEND
        )
        self.assertFalse(captured["causal"])
        self.assertEqual(captured["window_size"], (self.WINDOW, self.WINDOW))

    def test_causal_sliding_layer_keeps_a_left_only_window(self):
        captured = self._run_forward_extend(AttentionType.DECODER, ForwardMode.EXTEND)
        self.assertTrue(captured["causal"])
        self.assertEqual(captured["window_size"], (self.WINDOW, 0))


if __name__ == "__main__":
    unittest.main()
