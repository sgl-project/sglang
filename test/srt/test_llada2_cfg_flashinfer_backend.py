# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.layers.attention.llada2_attention_utils import (
    build_llada_image_custom_mask,
)
from sglang.srt.layers.attention.llada2_cfg_flashinfer_backend import (
    LLaDA2CFGFlashInferAttnBackend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode


class _FakeRaggedWrapper:
    def __init__(self):
        self.args = None
        self.kwargs = None
        self.forward_args = None
        self.forward_kwargs = None

    def begin_forward(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def forward(self, *args, **kwargs):
        self.forward_args = args
        self.forward_kwargs = kwargs
        return torch.zeros_like(args[0])


class _FakeTokenToKVPool:
    def __init__(self):
        self.set_args = None

    def set_kv_buffer(self, *args):
        self.set_args = args


class _FakePrefillUpdater:
    def __init__(self):
        self.num_qo_heads = 8
        self.num_kv_heads = 8
        self.head_dim = 64
        self.q_data_type = torch.bfloat16
        self.data_type = torch.bfloat16


def _make_backend():
    backend = object.__new__(LLaDA2CFGFlashInferAttnBackend)
    backend._llada_image_conditioning_mask_active = False
    backend._cfg_prefill_wrapper_ragged = _FakeRaggedWrapper()
    backend.num_wrappers = 1
    backend.prefill_wrappers_paged = [object()]
    backend.indices_updater_prefill = _FakePrefillUpdater()
    backend.prefill_split_tile_size = None
    return backend


def _make_forward_batch(prefix_lens=(0, 0)):
    seq_lens = torch.tensor([4, 3], dtype=torch.int32)
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        seq_lens=seq_lens,
        extend_prefix_lens=torch.tensor(prefix_lens, dtype=torch.int32),
        llada_image_conditioning_text_lens_cpu=[2, 1],
    )


class TestLLaDA2CFGFlashInferBackend(unittest.TestCase):
    def test_delegates_batches_without_llada_image_metadata(self):
        backend = _make_backend()
        backend._cfg_prefill_wrapper_ragged = SimpleNamespace()
        forward_batch = _make_forward_batch()
        del forward_batch.llada_image_conditioning_text_lens_cpu

        with patch.object(FlashInferAttnBackend, "init_forward_metadata") as stock_init:
            backend.init_forward_metadata(forward_batch)

        stock_init.assert_called_once_with(forward_batch)
        self.assertFalse(backend.conditioning_mask_active)
        from sglang.multimodal_gen.runtime.pipelines_core.stages.llada_image_conditioning import (
            ensure_conditioning_mask_active,
        )

        with self.assertRaisesRegex(RuntimeError, "without its block attention mask"):
            ensure_conditioning_mask_active(backend)

    def test_delegates_legacy_dllm_padding_metadata(self):
        backend = _make_backend()
        forward_batch = _make_forward_batch()
        del forward_batch.llada_image_conditioning_text_lens_cpu
        forward_batch.forward_mode = ForwardMode.DLLM_EXTEND
        forward_batch.dllm_left_pad_lens = torch.tensor([1, 1], dtype=torch.int32)

        with patch.object(FlashInferAttnBackend, "init_forward_metadata") as stock_init:
            backend.init_forward_metadata(forward_batch)

        stock_init.assert_called_once_with(forward_batch)

    def test_consumes_llada_image_text_lengths_as_custom_mask(self):
        backend = _make_backend()
        forward_batch = _make_forward_batch()

        with patch.object(FlashInferAttnBackend, "init_forward_metadata") as stock_init:
            backend.init_forward_metadata(forward_batch)

        stock_init.assert_not_called()
        wrapper = backend._cfg_prefill_wrapper_ragged
        self.assertEqual(wrapper.args[0].tolist(), [0, 4, 7])
        self.assertEqual(wrapper.args[1].tolist(), [0, 4, 7])
        self.assertFalse(wrapper.kwargs["causal"])
        torch.testing.assert_close(
            wrapper.kwargs["custom_mask"],
            build_llada_image_custom_mask([2, 1], [4, 3], "cpu"),
        )
        self.assertTrue(backend._llada_image_conditioning_mask_active)
        self.assertTrue(backend.conditioning_mask_active)
        self.assertTrue(backend.forward_metadata.use_ragged)

    def test_masked_forward_uses_ragged_kernel_without_saving_kv(self):
        backend = _make_backend()
        forward_batch = _make_forward_batch()
        forward_batch.token_to_kv_pool = _FakeTokenToKVPool()
        backend.init_forward_metadata(forward_batch)
        layer = SimpleNamespace(
            tp_q_head_num=2,
            tp_k_head_num=2,
            tp_v_head_num=2,
            head_dim=4,
            scaling=0.5,
            logit_cap=0.0,
            is_cross_attention=False,
        )
        q = torch.randn(7, 8)
        k = torch.randn(7, 8)
        v = torch.randn(7, 8)

        output = backend.forward_extend(q, k, v, layer, forward_batch)

        self.assertEqual(output.shape, q.shape)
        wrapper = backend._cfg_prefill_wrapper_ragged
        torch.testing.assert_close(wrapper.forward_args[0], q.view(7, 2, 4))
        self.assertFalse(wrapper.forward_kwargs["causal"])
        self.assertIsNone(forward_batch.token_to_kv_pool.set_args)

    def test_rejects_cached_prefix_for_one_shot_conditioning(self):
        backend = _make_backend()
        forward_batch = _make_forward_batch(prefix_lens=(1, 0))

        with self.assertRaisesRegex(RuntimeError, "does not support cached prefixes"):
            backend.init_forward_metadata(forward_batch)


if __name__ == "__main__":
    unittest.main()
