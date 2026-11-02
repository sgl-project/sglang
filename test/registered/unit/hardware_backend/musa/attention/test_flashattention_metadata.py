import inspect
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

with patch.dict(
    sys.modules,
    {
        module: MagicMock()
        for module in (
            "compressed_tensors",
            "compressed_tensors.config",
            "compressed_tensors.quantization",
            "sglang.srt.configs.model_config",
            "sglang.srt.layers.quantization",
            "sglang.srt.layers.quantization.fp4_kv_cache_quant_method",
            "sglang.srt.speculative.spec_utils",
            "sgl_kernel",
            "sgl_kernel.quantization",
            "sgl_kernel.scalar_type",
        )
    },
):
    from sglang.test.ci.ci_register import register_cpu_ci

    with patch.dict(
        sys.modules,
        {"flash_attn_interface": MagicMock(), "mate": MagicMock()},
    ):
        from sglang.srt.hardware_backend.musa.attention import (
            flashattention_backend as fa_backend,
        )


register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMusaFlashAttentionMetadata(unittest.TestCase):
    def test_parent_scheduler_callback_stays_disabled(self):
        model_runner = MagicMock()
        model_runner.server_args.flash_attention_num_splits = 1
        model_runner.model_config.hf_config.model_type = "test"
        model_runner.model_config.hf_config.architectures = []
        model_runner.model_config.is_hybrid_swa_compress = False
        model_runner.model.modules.return_value = []

        def initialize_parent(backend, *_args, **_kwargs):
            backend._get_scheduler_metadata = MagicMock()
            backend.use_mla = False
            backend.num_splits = 1
            backend.num_attention_heads = 16

        with patch.object(
            fa_backend.FlashAttentionBackend,
            "__init__",
            autospec=True,
            side_effect=initialize_parent,
        ):
            backend = fa_backend.MusaFlashAttentionBackend(model_runner)

        self.assertIsNone(backend._get_scheduler_metadata)

    def test_parent_metadata_is_converted_to_typed_musa_metadata(self):
        metadata = fa_backend.FlashAttentionMetadata()
        metadata.local_attn_metadata = fa_backend.LocalAttentionMetadata()
        metadata.swa_spec_metadata = fa_backend.FlashAttentionMetadata()
        mla_scheduler_metadata = MagicMock()
        metadata.scheduler_metadata = mla_scheduler_metadata

        converted = fa_backend.MusaFlashAttentionBackend._as_metadata(metadata)

        self.assertIsInstance(converted, fa_backend.MusaFlashAttentionMetadata)
        self.assertIsInstance(
            converted.local_attn_metadata, fa_backend.MusaLocalAttentionMetadata
        )
        self.assertIsInstance(
            converted.swa_spec_metadata, fa_backend.MusaFlashAttentionMetadata
        )
        self.assertIs(converted.scheduler_metadata, mla_scheduler_metadata)

    def test_public_signatures_match_parent_backend(self):
        for method_name in ("__init__", "forward_extend", "forward_decode"):
            with self.subTest(method=method_name):
                parent_signature = inspect.signature(
                    getattr(fa_backend.FlashAttentionBackend, method_name)
                )
                musa_signature = inspect.signature(
                    getattr(fa_backend.MusaFlashAttentionBackend, method_name)
                )
                self.assertEqual(musa_signature, parent_signature)

    def test_musa_scheduler_helper_does_not_override_parent_hook(self):
        self.assertNotIn(
            "_compute_scheduler_metadata",
            fa_backend.MusaFlashAttentionBackend.__dict__,
        )
        self.assertIn(
            "_build_scheduler_metadata",
            fa_backend.MusaFlashAttentionBackend.__dict__,
        )

    def test_removed_legacy_cuda_graph_metadata_hooks(self):
        self.assertNotIn(
            "init_forward_metadata_capture_cuda_graph",
            fa_backend.MusaFlashAttentionBackend.__dict__,
        )
        self.assertNotIn(
            "init_forward_metadata_replay_cuda_graph",
            fa_backend.MusaFlashAttentionBackend.__dict__,
        )

    @staticmethod
    def _forward_batch(
        *,
        batch_size=1,
        has_prefix=False,
        decode=False,
        target_verify=False,
        draft_extend=False,
    ):
        forward_mode = Mock()
        forward_mode.is_decode_or_idle.return_value = decode
        forward_mode.is_target_verify.return_value = target_verify
        forward_mode.is_draft_extend_v2.return_value = draft_extend
        forward_mode.is_extend_or_draft_extend_or_mixed.return_value = has_prefix
        return SimpleNamespace(
            batch_size=batch_size,
            extend_prefix_lens_cpu=[1] if has_prefix else [0],
            forward_mode=forward_mode,
        )

    @staticmethod
    def _backend():
        backend = object.__new__(fa_backend.MusaFlashAttentionBackend)
        backend._is_gemma4 = False
        backend._is_minimax_m2 = False
        backend.num_splits = 1
        backend.full_attention_num_splits = 1
        backend._get_scheduler_metadata = None
        backend.kv_cache_dtype = MagicMock()
        backend.has_softcap = False
        return backend

    def test_mla_metadata_uses_mate_scheduler_api(self):
        backend = self._backend()
        backend._mla_attention_layout = (16, 1, 64, 512)
        backend.page_size = 64
        backend.num_splits = 1
        metadata = fa_backend.MusaFlashAttentionMetadata()
        metadata.max_seq_len_k = 32
        metadata.cache_seqlens_int32 = MagicMock()
        metadata.cu_seqlens_q = MagicMock()
        metadata.cu_seqlens_k = None
        metadata.max_seq_len_q = 1
        backend.forward_metadata = metadata
        backend.forward_metadata_spec_decode_expand = None
        expected = MagicMock()

        with patch.object(
            fa_backend, "get_scheduler_metadata", return_value=expected
        ) as get_metadata:
            backend._init_mla_scheduler_metadata()

        self.assertIs(metadata.scheduler_metadata, expected)
        self.assertNotIn("has_qv", get_metadata.call_args.kwargs)
        self.assertIn("cache_seqlens", get_metadata.call_args.kwargs)
        self.assertEqual(get_metadata.call_args.kwargs["num_heads_q"], 16)
        self.assertEqual(get_metadata.call_args.kwargs["num_heads_kv"], 1)
        self.assertEqual(get_metadata.call_args.kwargs["headdim"], 64)
        self.assertEqual(get_metadata.call_args.kwargs["headdim_v"], 512)
        self.assertFalse(hasattr(backend, "_mate_mla_workspace_buffer"))

    def test_cross_attention_always_needs_kvcache_metadata(self):
        backend = self._backend()
        forward_batch = self._forward_batch(has_prefix=False)

        self.assertTrue(
            backend._needs_no_mla_kvcache_scheduler_metadata(
                forward_batch, is_cross_attention=True
            )
        )

    def test_cp_side_lengths_preserve_batched_metadata(self):
        cp_metadata = SimpleNamespace(
            actual_seq_q_prev_tensor=torch.tensor([4, 2], dtype=torch.int32),
            kv_len_prev_tensor=torch.tensor([8, 5], dtype=torch.int32),
        )

        q_total, k_max, k_lens, q_max = (
            fa_backend.MusaFlashAttentionBackend._cp_side_lengths(cp_metadata, "prev")
        )

        self.assertEqual((q_total, k_max, q_max), (6, 8, 4))
        self.assertTrue(torch.equal(k_lens, cp_metadata.kv_len_prev_tensor))

    def test_multistep_backend_initializes_parent_glue_state(self):
        runner = SimpleNamespace(kv_index_translator=object())
        child_backends = [MagicMock(), MagicMock()]

        with patch.object(
            fa_backend,
            "MusaFlashAttentionBackend",
            side_effect=child_backends,
        ):
            backend = fa_backend.MusaFlashAttentionMultiStepBackend(
                runner,
                topk=4,
                speculative_num_steps=3,
            )

        self.assertIs(backend.attn_backend_list, backend.attn_backends)
        self.assertEqual(backend.attn_backends, child_backends)
        self.assertIsNone(backend.forward_metadata)
        self.assertIs(backend.kv_index_translator, runner.kv_index_translator)

    def test_skip_kv_cache_extend_never_touches_pool(self):
        backend = self._backend()
        backend.fa_skip_kv_cache = True
        backend.use_cp = False
        backend.attn_cp_size = 1
        backend.topk = 1
        backend.has_local_attention = False
        backend.attention_chunk_size = None
        backend.kv_cache_dtype = None
        backend.kv_cache_dtype_str = "auto"
        backend.token_to_kv_pool = MagicMock()
        metadata = fa_backend.MusaFlashAttentionMetadata()
        metadata.cu_seqlens_q = MagicMock()
        metadata.max_seq_len_q = 8
        metadata.extend_seq_lens_cpu_int = [8]
        backend.forward_metadata = metadata
        output = MagicMock()
        expected = object()
        output.view.return_value = expected
        backend.flash_attn_varlen_func = Mock(return_value=output)
        backend.flash_attn_with_kvcache = Mock()
        layer = SimpleNamespace(
            is_cross_attention=False,
            sliding_window_size=16,
            attn_type=fa_backend.AttentionType.DECODER_BIDIRECTIONAL,
            tp_q_head_num=8,
            tp_k_head_num=2,
            tp_v_head_num=2,
            qk_head_dim=64,
            head_dim=64,
            v_head_dim=64,
            scaling=0.125,
            logit_cap=0.0,
        )

        result = backend.forward_extend(
            MagicMock(),
            MagicMock(),
            MagicMock(),
            layer,
            SimpleNamespace(
                forward_mode=MagicMock(),
                attn_cp_metadata=None,
                mha_return_lse=False,
            ),
        )

        self.assertIs(result, expected)
        backend.token_to_kv_pool.set_kv_buffer.assert_not_called()
        backend.flash_attn_varlen_func.assert_called_once()
        backend.flash_attn_with_kvcache.assert_not_called()
        self.assertFalse(backend.flash_attn_varlen_func.call_args.kwargs["causal"])
        self.assertEqual(
            backend.flash_attn_varlen_func.call_args.kwargs["window_size"],
            (16, 16),
        )

    def test_cross_attention_extend_uses_encoder_scheduler_metadata(self):
        backend = self._backend()
        backend.fa_skip_kv_cache = False
        backend.use_mla = False
        backend.use_cp = False
        backend.attn_cp_size = 1
        backend.topk = 1
        backend.has_local_attention = False
        backend.attention_chunk_size = None
        backend.kv_cache_dtype = None
        backend.kv_cache_dtype_str = "auto"
        backend.kv_cache_is_mxfp8 = False
        backend.has_softcap = False
        backend.use_sliding_window_kv_pool = False
        backend.page_size = 64
        backend.req_to_token_pool = MagicMock()
        backend.token_to_kv_pool = MagicMock()
        backend.token_to_kv_pool.get_kv_buffer.return_value = (
            MagicMock(),
            MagicMock(),
        )
        metadata = fa_backend.MusaFlashAttentionMetadata()
        metadata.cu_seqlens_q = MagicMock()
        metadata.max_seq_len_q = 4
        metadata.encoder_lens_int32 = MagicMock()
        metadata.encoder_scheduler_metadata = object()
        metadata.swa_out_cache_loc = object()
        backend.forward_metadata = metadata
        output = MagicMock()
        expected = object()
        output.view.return_value = expected
        backend.flash_attn_varlen_func = Mock()
        backend.flash_attn_with_kvcache = Mock(return_value=output)
        layer = SimpleNamespace(
            is_cross_attention=True,
            sliding_window_size=-1,
            attn_type=fa_backend.AttentionType.DECODER,
            layer_id=0,
            tp_q_head_num=8,
            tp_k_head_num=2,
            tp_v_head_num=2,
            qk_head_dim=64,
            head_dim=64,
            v_head_dim=64,
            scaling=0.125,
            logit_cap=0.0,
            k_scale=1.0,
            v_scale=1.0,
        )
        forward_batch = SimpleNamespace(
            req_pool_indices=MagicMock(),
            encoder_cached=MagicMock(),
            forward_mode=MagicMock(),
            attn_cp_metadata=None,
            mha_return_lse=False,
            extend_prefix_lens_cpu=[],
        )

        write_loc = object()
        with (
            patch.object(
                fa_backend.KVWriteLoc, "for_layer", return_value=write_loc
            ) as for_layer,
            patch.object(
                fa_backend,
                "get_exec",
                return_value=SimpleNamespace(
                    deterministic=SimpleNamespace(enable_deterministic_inference=False)
                ),
            ),
        ):
            result = backend.forward_extend(
                MagicMock(),
                MagicMock(),
                MagicMock(),
                layer,
                forward_batch,
            )

        self.assertIs(result, expected)
        for_layer.assert_called_once_with(
            forward_batch,
            layer,
            swa_loc=metadata.swa_out_cache_loc,
        )
        self.assertIs(
            backend.token_to_kv_pool.set_kv_buffer.call_args.args[1], write_loc
        )
        backend.flash_attn_varlen_func.assert_not_called()
        self.assertIs(
            backend.flash_attn_with_kvcache.call_args.kwargs["scheduler_metadata"],
            metadata.encoder_scheduler_metadata,
        )

    def test_kvcache_metadata_modes(self):
        backend = self._backend()

        self.assertFalse(
            backend._needs_no_mla_kvcache_scheduler_metadata(self._forward_batch())
        )
        for forward_batch in (
            self._forward_batch(has_prefix=True),
            self._forward_batch(decode=True),
            self._forward_batch(target_verify=True),
            self._forward_batch(draft_extend=True),
        ):
            with self.subTest(forward_batch=forward_batch):
                self.assertTrue(
                    backend._needs_no_mla_kvcache_scheduler_metadata(forward_batch)
                )

    def test_gemma4_split_policy(self):
        backend = self._backend()
        backend._is_gemma4 = True

        self.assertEqual(
            backend._num_splits_for_no_mla_call(
                self._forward_batch(batch_size=4), is_swa_layer=True
            ),
            0,
        )
        self.assertEqual(
            backend._num_splits_for_no_mla_call(
                self._forward_batch(batch_size=8), is_swa_layer=True
            ),
            1,
        )
        self.assertEqual(
            backend._num_splits_for_no_mla_call(
                self._forward_batch(batch_size=4), is_swa_layer=False
            ),
            1,
        )

    def test_kvcache_wrapper_forwards_explicit_metadata(self):
        q = MagicMock()
        k_cache = MagicMock()
        v_cache = MagicMock()
        metadata = MagicMock()
        expected = MagicMock()

        with patch.object(
            fa_backend, "mate_flash_attn_with_kvcache", return_value=expected
        ) as kernel:
            result = fa_backend.flash_attn_with_kvcache(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                scheduler_metadata=metadata,
            )

        self.assertIs(result, expected)
        self.assertIs(kernel.call_args.kwargs["scheduler_metadata"], metadata)

    def test_kvcache_wrapper_rejects_missing_metadata(self):
        with self.assertRaisesRegex(AssertionError, "requires scheduler_metadata"):
            fa_backend.flash_attn_with_kvcache(
                q=MagicMock(),
                k_cache=MagicMock(),
                v_cache=MagicMock(),
            )


if __name__ == "__main__":
    unittest.main()
