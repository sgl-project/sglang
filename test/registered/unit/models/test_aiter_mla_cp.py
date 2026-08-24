import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.attn_parallel import AttnParallelMode
from sglang.srt.layers.attention.aiter_backend import (
    AiterAttnBackend,
    AiterMlaCPLaunchMetadata,
    AiterMlaCPMetadata,
    ForwardMetadata,
)
from sglang.srt.layers.cp.base import CPAttentionBackendKind
from sglang.srt.layers.cp.cp_decode_attn_tp import (
    CP_DECODE_ATTN_TP_SUPPORTED_ARCHS,
    CpDecodeAttnTpContext,
)
from sglang.srt.layers.cp.zigzag import ZigzagCPStrategy
from sglang.srt.models.deepseek_common.attention_backend_handler import (
    handle_attention_aiter,
)
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_methods import (
    AttnForwardMethod,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _FakeTokenPool:
    def __init__(self, head_dim):
        self.buffer = torch.zeros(8, head_dim)

    def get_key_buffer(self, _layer_id):
        return self.buffer


class _FakeCPStrategy:
    def run_attention(self, q, forward_batch, _device, attn_fn, *, attention_backend):
        assert attention_backend is CPAttentionBackendKind.AITER
        metadata = forward_batch.attn_cp_metadata
        prev = attn_fn(
            q[:2],
            torch.tensor([0, 2], dtype=torch.int32),
            metadata.kv_len_prev_tensor,
            2,
        )
        next_ = attn_fn(
            q[2:],
            torch.tensor([0, 2], dtype=torch.int32),
            metadata.kv_len_next_tensor,
            2,
        )
        return torch.cat([prev, next_], dim=0)


class TestAiterMlaContextParallel(unittest.TestCase):
    def test_dcp_backend_dispatch_reads_batch_stamp(self):
        backend = object.__new__(AiterAttnBackend)
        backend.use_mla = True
        mode = SimpleNamespace(
            is_decode=lambda: True,
            is_target_verify=lambda: False,
            is_draft_extend_v2=lambda: False,
        )
        self.assertFalse(
            backend._is_dcp_batch(
                SimpleNamespace(
                    forward_mode=mode,
                    attn_parallel_mode=AttnParallelMode.TP,
                )
            )
        )
        self.assertTrue(
            backend._is_dcp_batch(
                SimpleNamespace(
                    forward_mode=mode,
                    attn_parallel_mode=AttnParallelMode.DCP,
                )
            )
        )

    def test_deepseek_v3_runtime_tp_is_enabled(self):
        self.assertIn("DeepseekV3ForCausalLM", CP_DECODE_ATTN_TP_SUPPORTED_ARCHS)

        ctx = object.__new__(CpDecodeAttnTpContext)
        ctx.decode_tp_rank = 0
        ctx.decode_tp_size = 8
        forward_mode = SimpleNamespace(
            is_target_verify=lambda: False,
            is_draft_extend_v2=lambda: False,
        )
        with patch(
            "sglang.srt.layers.cp.cp_decode_attn_tp.dsa_use_prefill_cp",
            return_value=False,
        ):
            self.assertTrue(
                ctx.should_use_attn_tp(
                    SimpleNamespace(
                        forward_mode=forward_mode,
                        attn_parallel_mode=0,
                    )
                )
            )
            self.assertFalse(
                ctx.should_use_attn_tp(
                    SimpleNamespace(
                        forward_mode=forward_mode,
                        attn_parallel_mode=1,
                    )
                )
            )

    def test_aiter_is_supported_by_zigzag_strategy(self):
        self.assertEqual(
            CPAttentionBackendKind.from_string("aiter"),
            CPAttentionBackendKind.AITER,
        )
        strategy = object.__new__(ZigzagCPStrategy)
        self.assertIn(
            CPAttentionBackendKind.AITER,
            strategy.get_supported_attention_backend(),
        )

    @patch(
        "sglang.srt.models.deepseek_common.attention_backend_handler.mla_use_prefill_cp",
        return_value=True,
    )
    def test_aiter_dispatches_cp_prefill_to_mla(self, _mock_cp):
        method = handle_attention_aiter(
            SimpleNamespace(), SimpleNamespace(forward_mode=SimpleNamespace())
        )
        self.assertEqual(method, AttnForwardMethod.MLA)

    @patch(
        "sglang.srt.models.deepseek_common.attention_backend_handler.mla_use_prefill_cp",
        return_value=False,
    )
    @patch(
        "sglang.srt.layers.cp.cp_decode_attn_tp.get_cp_decode_attn_tp_ctx",
        return_value=SimpleNamespace(should_use_attn_tp=lambda _batch: True),
    )
    def test_aiter_dispatches_runtime_tp_prefill_to_mla(
        self, _mock_runtime_tp, _mock_cp
    ):
        method = handle_attention_aiter(
            SimpleNamespace(), SimpleNamespace(forward_mode=SimpleNamespace())
        )
        self.assertEqual(method, AttnForwardMethod.MLA)

    def test_mla_cp_forward_dispatches_both_zigzag_halves(self):
        backend = object.__new__(AiterAttnBackend)
        backend.input_dtype = torch.bfloat16
        backend.device = torch.device("cpu")
        backend.token_to_kv_pool = _FakeTokenPool(head_dim=6)

        prev_lens = torch.tensor([2], dtype=torch.int32)
        next_lens = torch.tensor([3], dtype=torch.int32)
        prev = AiterMlaCPLaunchMetadata(
            kv_indptr=torch.tensor([0, 2], dtype=torch.int32),
            kv_indices=torch.tensor([0, 1], dtype=torch.int32),
            kv_last_page_len=torch.ones(1, dtype=torch.int32),
        )
        next_ = AiterMlaCPLaunchMetadata(
            kv_indptr=torch.tensor([0, 3], dtype=torch.int32),
            kv_indices=torch.tensor([0, 1, 2], dtype=torch.int32),
            kv_last_page_len=torch.ones(1, dtype=torch.int32),
        )
        backend.forward_metadata = ForwardMetadata(
            kv_indptr=prev.kv_indptr,
            kv_indices=prev.kv_indices,
            qo_indptr=torch.tensor([0, 2], dtype=torch.int32),
            kv_last_page_len=prev.kv_last_page_len,
            max_q_len=2,
            max_kv_len=3,
            mla_cp_metadata=AiterMlaCPMetadata(prev=prev, next=next_),
        )
        forward_batch = SimpleNamespace(
            attn_cp_metadata=SimpleNamespace(
                kv_len_prev_tensor=prev_lens,
                kv_len_next_tensor=next_lens,
            )
        )
        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=2,
            qk_head_dim=6,
            v_head_dim=4,
            scaling=0.5,
            logit_cap=0.0,
        )
        q = torch.zeros(4, 2, 6)
        launches = []

        def fake_mla_prefill(
            _q,
            _k,
            output,
            _qo_indptr,
            _kv_indptr,
            kv_indices,
            _last_page_len,
            _max_q_len,
            _scaling,
            _logit_cap,
        ):
            launches.append(kv_indices.clone())
            output.fill_(float(len(launches)))

        with (
            patch(
                "sglang.srt.layers.attention.aiter_backend.get_cp_strategy",
                return_value=_FakeCPStrategy(),
            ),
            patch(
                "sglang.srt.layers.attention.aiter_backend.mla_prefill_fwd",
                side_effect=fake_mla_prefill,
                create=True,
            ),
        ):
            output = backend._forward_mla_cp(q, layer, forward_batch)

        self.assertEqual(output.shape, (4, 2, 4))
        self.assertTrue(torch.equal(output[:2], torch.ones_like(output[:2])))
        self.assertTrue(torch.equal(output[2:], torch.full_like(output[2:], 2)))
        self.assertEqual([indices.numel() for indices in launches], [2, 3])


if __name__ == "__main__":
    unittest.main()
