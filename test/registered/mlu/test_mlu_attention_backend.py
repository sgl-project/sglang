import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch_mlu  # noqa: F401

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_mlu_ci
from sglang.test.test_utils import CustomTestCase

register_mlu_ci(est_time=60, suite="pr-test-mlu")


def _fake_mlu_ops():
    return SimpleNamespace(
        reshape_paged_cache=MagicMock(),
        flash_attention=MagicMock(),
        single_query_cached_kv_attn=MagicMock(),
    )


class TestMLUAttentionMetadata(CustomTestCase):
    def setUp(self):
        torch.mlu.set_device(0)
        self.device = torch.device("mlu", 0)

    def _make_backend(self):
        attn_mod = importlib.import_module(
            "sglang.srt.hardware_backend.mlu.attention.mlu_backend"
        )
        MLUAttnBackend = attn_mod.MLUAttnBackend

        runner = SimpleNamespace(
            device=self.device,
            page_size=16,
            model_config=SimpleNamespace(context_len=64),
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.arange(
                    4 * 64, dtype=torch.int32, device=self.device
                ).reshape(4, 64)
            ),
            token_to_kv_pool=MagicMock(),
        )
        backend = MLUAttnBackend(runner)
        return backend

    def test_extend_metadata_tracks_uncached_prefill(self):
        backend = self._make_backend()
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64, device=self.device),
            seq_lens=torch.tensor([4, 5], dtype=torch.int32, device=self.device),
            extend_seq_lens=torch.tensor([4, 5], dtype=torch.int32, device=self.device),
            extend_seq_lens_cpu=[4, 5],
            extend_prefix_lens=torch.tensor([0, 0], dtype=torch.int32, device=self.device),
            batch_size=2,
        )

        backend.init_forward_metadata(forward_batch)
        meta = backend.forward_metadata

        self.assertTrue(meta.is_uncached_prefill_only)
        self.assertEqual(meta.max_seq_len_q, 5)
        self.assertEqual(meta.max_seq_len_kv, 5)
        self.assertEqual(meta.cu_seqlens_q.cpu().tolist(), [0, 4, 9])
        self.assertEqual(meta.cu_seqlens_kv.cpu().tolist(), [0, 4, 9])
        self.assertEqual(tuple(meta.block_tables.shape), (2, 4))
        self.assertEqual(meta.block_tables.device.type, "mlu")

    def test_decode_metadata_uses_single_token_queries(self):
        backend = self._make_backend()
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64, device=self.device),
            seq_lens=torch.tensor([7, 9], dtype=torch.int32, device=self.device),
            batch_size=2,
        )

        backend.init_forward_metadata(forward_batch)
        meta = backend.forward_metadata

        self.assertEqual(meta.max_seq_len_q, 1)
        self.assertEqual(meta.max_seq_len_kv, 9)
        self.assertEqual(meta.cu_seqlens_q.cpu().tolist(), [0, 1, 2])
        self.assertEqual(meta.cu_seqlens_kv.cpu().tolist(), [0, 7, 16])

    def test_mixed_metadata_uses_tensor_boundary(self):
        backend = self._make_backend()
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.MIXED,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64, device=self.device),
            seq_lens=torch.tensor([8, 9], dtype=torch.int32, device=self.device),
            seq_lens_cpu=torch.tensor([8, 9], dtype=torch.int32),
            extend_seq_lens=torch.tensor([1], dtype=torch.int32, device=self.device),
            extend_seq_lens_cpu=[1],
            batch_size=2,
            mix_running_indices=torch.tensor([1], dtype=torch.int64, device=self.device),
        )

        backend.init_forward_metadata(forward_batch)
        meta = backend.forward_metadata

        self.assertEqual(meta.prefill_bs, 1)
        self.assertEqual(meta.decode_bs, 1)
        self.assertEqual(meta.mixed_num_prefill_tokens, 1)
        self.assertEqual(meta.mixed_expected_tokens, 2)
        self.assertEqual(tuple(meta.block_tables.shape), (2, 4))

    def test_mixed_forward_builds_split_prefill_and_decode_boundaries(self):
        backend = self._make_backend()
        fake_ops = _fake_mlu_ops()
        attn_mod = importlib.import_module(
            "sglang.srt.hardware_backend.mlu.attention.mlu_backend"
        )
        backend.token_to_kv_pool.get_key_buffer.return_value = torch.empty(
            4, 1, 16, 4, device=self.device
        )
        backend.token_to_kv_pool.get_value_buffer.return_value = torch.empty(
            4, 1, 16, 4, device=self.device
        )

        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.MIXED,
            req_pool_indices=torch.tensor([0, 1, 2], dtype=torch.int64, device=self.device),
            seq_lens=torch.tensor([5, 7, 9], dtype=torch.int32, device=self.device),
            seq_lens_cpu=torch.tensor([5, 7, 9], dtype=torch.int32),
            extend_seq_lens=torch.tensor([2, 3], dtype=torch.int32, device=self.device),
            extend_seq_lens_cpu=[2, 3],
            batch_size=3,
            mix_running_indices=torch.tensor([2], dtype=torch.int64, device=self.device),
            out_cache_loc=torch.arange(6, dtype=torch.int64, device=self.device),
        )
        layer = SimpleNamespace(
            layer_id=0,
            qk_head_dim=4,
            v_head_dim=4,
            tp_q_head_num=1,
            tp_k_head_num=1,
            tp_v_head_num=1,
            scaling=1.0,
        )

        backend.init_forward_metadata(forward_batch)
        with patch.object(attn_mod, "torch_mlu_ops", fake_ops):
            out = backend.forward_mixed(
                q=torch.empty(6, 4, device=self.device),
                k=torch.empty(6, 4, device=self.device),
                v=torch.empty(6, 4, device=self.device),
                layer=layer,
                forward_batch=forward_batch,
            )

        fake_ops.flash_attention.assert_called_once()
        fake_ops.single_query_cached_kv_attn.assert_called_once()
        kwargs = fake_ops.flash_attention.call_args.kwargs
        self.assertEqual(kwargs["cu_seq_lens_q"].cpu().tolist(), [0, 2, 5])
        self.assertEqual(kwargs["cu_seq_lens_kv"].cpu().tolist(), [0, 5, 12])
        self.assertEqual(tuple(kwargs["block_tables"].shape), (2, 4))
        self.assertEqual(kwargs["max_seq_len_q"], 3)
        self.assertEqual(kwargs["max_seq_len_kv"], 7)
        self.assertEqual(tuple(out.shape), (6, 4))
        self.assertEqual(out.device.type, "mlu")

    def test_mla_attention_is_rejected(self):
        backend = self._make_backend()
        layer = SimpleNamespace(qk_head_dim=6, v_head_dim=4, tp_q_head_num=1)
        forward_batch = SimpleNamespace(forward_mode=ForwardMode.IDLE)

        with self.assertRaisesRegex(RuntimeError, "MLA models are not supported"):
            backend.forward(
                q=torch.empty(0, 6, device=self.device),
                k=torch.empty(0, 4, device=self.device),
                v=torch.empty(0, 4, device=self.device),
                layer=layer,
                forward_batch=forward_batch,
                k_rope=torch.empty(0, 2, device=self.device),
            )


class TestMLUAttentionRegistry(CustomTestCase):
    def test_mlu_backend_is_registered_lazily(self):
        from sglang.srt.layers.attention import attention_registry

        self.assertIn("mlu", attention_registry.ATTENTION_BACKENDS)

    def test_mlu_backend_rejects_mla_before_import(self):
        from sglang.srt.layers.attention import attention_registry

        runner = SimpleNamespace(use_mla_backend=True)
        with self.assertRaisesRegex(ValueError, "MLA models are not supported"):
            attention_registry.ATTENTION_BACKENDS["mlu"](runner)


if __name__ == "__main__":
    unittest.main()
