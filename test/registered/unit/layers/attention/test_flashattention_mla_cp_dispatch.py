import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.flashattention_backend import (
    FlashAttentionBackend,
    FlashAttentionMetadata,
)
from sglang.srt.layers.cp.zigzag import ZigzagCPStrategy
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MODULE = "sglang.srt.layers.attention.flashattention_backend"


class TestMLACPDispatch(unittest.TestCase):
    def test_zigzag_cp_prefill_is_one_launch_per_layer(self):
        """FA MLA under zigzag CP calls the kernel once per layer with the
        combined geometry and the duplicated page table."""
        strategy = ZigzagCPStrategy(cp_size=2)
        with get_parallel().override(attn_cp_rank=0):
            cp_meta = strategy.build_metadata(8, [8], [8])
        layer = SimpleNamespace(
            layer_id=0,
            head_dim=5,
            v_head_dim=3,
            tp_q_head_num=2,
            tp_k_head_num=1,
            sliding_window_size=-1,
            is_cross_attention=False,
            attn_type=None,
            scaling=0.5,
            logit_cap=0.0,
        )
        page_table = torch.arange(8, dtype=torch.int32).view(1, 8)
        backend = FlashAttentionBackend.__new__(FlashAttentionBackend)
        backend.__dict__.update(
            use_mla=True,
            fa_impl_ver=3,
            device="cpu",
            page_size=1,
            num_splits=0,
            topk=0,
            local_attn_builder=None,
            use_sliding_window_kv_pool=False,
            kv_cache_is_mxfp8=False,
            token_to_kv_pool=SimpleNamespace(
                get_key_buffer=lambda _: torch.zeros(16, 1, 5)
            ),
            prepare_paged_mha_query=lambda q, qr, kr, *a, **k: (q, qr, kr, None, None),
            forward_metadata=FlashAttentionMetadata(
                page_table=page_table, zigzag_page_table=page_table.repeat(2, 1)
            ),
        )
        batch = SimpleNamespace(
            batch_size=1,
            forward_mode=ForwardMode.EXTEND,
            attn_attend_prefix_cache=None,
            attn_cp_metadata=cp_meta,
        )
        q = torch.arange(4 * 2 * 3, dtype=torch.float32).view(4, 2, 3)
        calls = []

        def attention(**kwargs):
            calls.append(kwargs)
            return kwargs["qv"] + 1

        with (
            patch(f"{MODULE}.is_cp_active", return_value=True),
            patch(f"{MODULE}.get_cp_strategy", return_value=strategy),
            patch(f"{MODULE}.flash_attn_with_kvcache", side_effect=attention),
        ):
            for _ in range(2):  # two layers
                out = backend.forward_extend(
                    q, None, None, layer, batch, q_rope=torch.zeros(4, 2, 2)
                )
                torch.testing.assert_close(out, (q + 1).view(4, -1))
        self.assertEqual(len(calls), 2)
        for call in calls:
            self.assertIs(
                call["page_table"], backend.forward_metadata.zigzag_page_table
            )
            self.assertIs(call["cu_seqlens_q"], cp_meta.cu_seqlens_q_combined_tensor)
            self.assertIs(call["cache_seqlens"], cp_meta.kv_len_combined_tensor)
            self.assertNotIn("cu_seqlens_k_new", call)


if __name__ == "__main__":
    unittest.main()
