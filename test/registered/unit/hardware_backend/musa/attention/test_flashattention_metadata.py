import importlib.machinery
import inspect
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch


def _stub_module(name, **attrs):
    """A stub module with a real ``__spec__``.

    Importers now call ``importlib.util.find_spec`` on optional packages (e.g.
    transformers probes ``flash_attn_interface``), which rejects a bare
    ``MagicMock`` in ``sys.modules``. Give the stub a valid spec and the
    attributes the backend imports.
    """
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, None)
    for attr_name, value in attrs.items():
        setattr(module, attr_name, value)
    return module


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
        {
            "flash_attn_interface": _stub_module(
                "flash_attn_interface",
                flash_attn_varlen_func=MagicMock(),
                flash_attn_with_kvcache=MagicMock(),
                get_scheduler_metadata=MagicMock(),
            ),
            "mate": _stub_module("mate"),
        },
    ):
        from sglang.srt.hardware_backend.musa.attention import (
            flashattention_backend as fa_backend,
        )


register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMusaFlashAttentionMetadata(unittest.TestCase):
    def test_parent_scheduler_callback_stays_disabled(self):
        model_runner = MagicMock()
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

    def test_fa4_upgrade_is_pinned_off_on_musa(self):
        backend = self._backend()
        self.assertFalse(backend.fa3_upgrades_to_fa4)
        # The parent initializer assigns the CUDA-derived value; the MUSA
        # backend must ignore it and stay on the MATE FA3 kernels.
        backend.fa3_upgrades_to_fa4 = True
        self.assertFalse(backend.fa3_upgrades_to_fa4)

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

    @staticmethod
    def _forward_batch(
        *,
        batch_size=1,
        has_prefix=False,
        decode=False,
        target_verify=False,
        draft_extend=False,
        spec_info=None,
        context_parallel_extend=False,
    ):
        forward_mode = Mock()
        forward_mode.is_decode_or_idle.return_value = decode
        forward_mode.is_target_verify.return_value = target_verify
        forward_mode.is_draft_extend_v2.return_value = draft_extend
        forward_mode.is_extend_or_draft_extend_or_mixed.return_value = has_prefix
        forward_mode.is_context_parallel_extend.return_value = context_parallel_extend
        return SimpleNamespace(
            batch_size=batch_size,
            extend_prefix_lens_cpu=[1] if has_prefix else [0],
            forward_mode=forward_mode,
            spec_info=spec_info,
        )

    @staticmethod
    def _backend():
        backend = object.__new__(fa_backend.MusaFlashAttentionBackend)
        backend.num_splits = 1
        backend.topk = 1
        backend.page_size = 1
        # A non-translating backend: window_write_locs keys the write ids by the
        # runner's window storage.
        backend.runner_ids = MagicMock()
        backend._get_scheduler_metadata = None
        backend.kv_cache_dtype = MagicMock()
        backend.has_softcap = False
        backend._attention_layouts = fa_backend.MusaAttentionLayouts(
            full=(8, 2, 64, 64),
            sliding_window=None,
            sliding_window_size=None,
            has_full_attention=True,
            has_local_attention=False,
            has_cross_attention=False,
            has_sliding_window_local_attention=False,
        )
        backend._mla_attention_layout = None
        backend.forward_metadata = None
        backend.forward_metadata_spec_decode_expand = None
        backend._captured_graph_metadata = {}
        return backend

    @staticmethod
    def _mla_backend():
        backend = TestMusaFlashAttentionMetadata._backend()
        backend._mla_attention_layout = (16, 1, 64, 512)
        backend.page_size = 64
        backend.num_splits = 2
        metadata = fa_backend.MusaFlashAttentionMetadata()
        metadata.max_seq_len_k = 32
        metadata.cache_seqlens_int32 = MagicMock()
        metadata.cu_seqlens_q = MagicMock()
        metadata.cu_seqlens_k = None
        metadata.max_seq_len_q = 1
        backend.forward_metadata = metadata
        return backend

    def test_mla_metadata_uses_mate_scheduler_api(self):
        backend = self._mla_backend()
        expected = MagicMock()

        with patch.object(
            fa_backend, "get_scheduler_metadata", return_value=expected
        ) as get_metadata:
            backend._init_mla_scheduler_metadata(self._forward_batch(decode=True))

        self.assertIs(backend.forward_metadata.scheduler_metadata, expected)
        self.assertNotIn("has_qv", get_metadata.call_args.kwargs)
        self.assertIn("cache_seqlens", get_metadata.call_args.kwargs)
        self.assertEqual(get_metadata.call_args.kwargs["num_heads_q"], 16)
        self.assertEqual(get_metadata.call_args.kwargs["num_heads_kv"], 1)
        self.assertEqual(get_metadata.call_args.kwargs["headdim"], 64)
        self.assertEqual(get_metadata.call_args.kwargs["headdim_v"], 512)
        self.assertEqual(get_metadata.call_args.kwargs["num_splits"], 2)
        self.assertTrue(get_metadata.call_args.kwargs["causal"])

    def test_mla_metadata_causal_matches_kvcache_call_sites(self):
        # Decoder prefill/extend and target-verify cascade use different causal
        # flags at the kvcache call sites; the metadata plan must match them.
        backend = self._mla_backend()
        for forward_batch, expected_causal in (
            (self._forward_batch(has_prefix=True), True),
            (self._forward_batch(batch_size=2), True),
        ):
            with self.subTest(forward_batch=forward_batch):
                with patch.object(
                    fa_backend, "get_scheduler_metadata", return_value=MagicMock()
                ) as get_metadata:
                    backend._init_mla_scheduler_metadata(forward_batch)
                self.assertEqual(
                    get_metadata.call_args.kwargs["causal"], expected_causal
                )

        backend.topk = 2
        with patch.object(
            fa_backend, "get_scheduler_metadata", return_value=MagicMock()
        ) as get_metadata:
            backend._init_mla_scheduler_metadata(
                self._forward_batch(target_verify=True)
            )
        self.assertFalse(get_metadata.call_args.kwargs["causal"])

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

    def test_cp_side_lengths_prefer_cpu_metadata(self):
        # CPU mirrors must win over device tensors: reading them is what keeps
        # CP metadata init free of per-side D2H synchronizations.
        cp_metadata = SimpleNamespace(
            actual_seq_q_prev_tensor=torch.tensor([4, 2], dtype=torch.int32),
            kv_len_prev_tensor=torch.tensor([8, 5], dtype=torch.int32),
            total_q_prev_tokens=999,
            max_seqlen_q_prev=999,
            kv_len_prev_list=[100],
        )

        q_total, k_max, _k_lens, q_max = (
            fa_backend.MusaFlashAttentionBackend._cp_side_lengths(cp_metadata, "prev")
        )

        self.assertEqual((q_total, k_max, q_max), (999, 100, 999))

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
                extend_num_tokens=None,
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
        for_layer.assert_called_once()
        assert for_layer.call_args.args == (forward_batch, layer)
        window_locs = for_layer.call_args.kwargs["window_locs"]
        self.assertEqual(len(window_locs), 1)
        self.assertIs(next(iter(window_locs.values())), metadata.swa_out_cache_loc)
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

    def test_metadata_uses_parent_num_splits_policy(self):
        # No model-specific split tuning: metadata and call sites must both come
        # from the parent backend's `num_splits` policy.
        backend = self._backend()
        backend.num_splits = 0
        metadata = fa_backend.MusaFlashAttentionMetadata()
        metadata.cache_seqlens_int32 = MagicMock()
        metadata.cu_seqlens_q = MagicMock()
        metadata.max_seq_len_q = 1
        metadata.max_seq_len_k = 8

        with patch.object(
            fa_backend, "get_scheduler_metadata", return_value=MagicMock()
        ) as get_metadata:
            backend._init_full_attention_metadata(
                metadata,
                needs_kvcache_metadata=True,
                cu_seqlens_k_new=None,
                max_seqlen_k=8,
                causal=True,
                num_splits=backend.num_splits,
            )

        self.assertEqual(get_metadata.call_args.kwargs["num_splits"], 0)

    def test_graph_replay_refreshes_captured_metadata(self):
        backend = self._backend()
        forward_mode = self._forward_batch().forward_mode
        captured = torch.zeros(4, dtype=torch.int32)
        backend.forward_metadata = fa_backend.MusaFlashAttentionMetadata()
        backend.forward_metadata.scheduler_metadata = captured

        backend._capture_graph_scheduler_metadata(2, forward_mode, None)
        backend.forward_metadata.scheduler_metadata = torch.ones(4, dtype=torch.int32)

        backend._copy_fresh_metadata_to_graph_tensors(2, forward_mode, None)

        self.assertTrue(torch.equal(captured, torch.ones(4, dtype=torch.int32)))
        self.assertIs(backend.forward_metadata.scheduler_metadata, captured)

    def test_graph_replay_unknown_key_raises(self):
        backend = self._backend()
        backend.forward_metadata = fa_backend.MusaFlashAttentionMetadata()
        backend.forward_metadata.scheduler_metadata = torch.zeros(4, dtype=torch.int32)

        with self.assertRaisesRegex(RuntimeError, "no captured scheduler metadata"):
            backend._copy_fresh_metadata_to_graph_tensors(
                2, self._forward_batch().forward_mode, None
            )

    def test_graph_replay_slot_mismatch_raises(self):
        backend = self._backend()
        forward_mode = self._forward_batch().forward_mode
        backend.forward_metadata = fa_backend.MusaFlashAttentionMetadata()
        backend.forward_metadata.scheduler_metadata = torch.zeros(4, dtype=torch.int32)
        backend._capture_graph_scheduler_metadata(2, forward_mode, None)
        backend.forward_metadata.swa_scheduler_metadata = torch.zeros(
            4, dtype=torch.int32
        )

        with self.assertRaisesRegex(RuntimeError, "slots changed"):
            backend._copy_fresh_metadata_to_graph_tensors(2, forward_mode, None)

    def test_graph_replay_self_copy_raises(self):
        backend = self._backend()
        forward_mode = self._forward_batch().forward_mode
        captured = torch.zeros(4, dtype=torch.int32)
        backend.forward_metadata = fa_backend.MusaFlashAttentionMetadata()
        backend.forward_metadata.scheduler_metadata = captured
        backend._capture_graph_scheduler_metadata(2, forward_mode, None)
        backend.forward_metadata.scheduler_metadata = captured

        with self.assertRaisesRegex(RuntimeError, "did not produce a fresh tensor"):
            backend._copy_fresh_metadata_to_graph_tensors(2, forward_mode, None)

    @staticmethod
    def _cp_batch():
        cp_meta = SimpleNamespace(
            kv_len_prev_tensor=torch.zeros(2, dtype=torch.int32),
            kv_len_next_tensor=torch.ones(2, dtype=torch.int32),
        )
        return cp_meta, SimpleNamespace(attn_cp_metadata=cp_meta)

    def test_cp_scheduler_metadata_resolves_published_side(self):
        backend = self._backend()
        metadata = fa_backend.MusaFlashAttentionMetadata()
        prev = torch.zeros(4, dtype=torch.int32)
        nxt = torch.ones(4, dtype=torch.int32)
        metadata.cp_scheduler_metadata_prev = prev
        metadata.cp_scheduler_metadata_next = nxt
        cp_meta, forward_batch = self._cp_batch()

        self.assertIs(
            backend._cp_scheduler_metadata_for_kvcache(
                metadata,
                forward_batch,
                cp_meta.kv_len_prev_tensor,
                is_swa_layer=False,
                use_local_attn=False,
            ),
            prev,
        )
        self.assertIs(
            backend._cp_scheduler_metadata_for_kvcache(
                metadata,
                forward_batch,
                cp_meta.kv_len_next_tensor,
                is_swa_layer=False,
                use_local_attn=False,
            ),
            nxt,
        )
        # An unknown tensor must fail instead of silently selecting `next`.
        with self.assertRaisesRegex(RuntimeError, "neither"):
            backend._cp_scheduler_metadata_for_kvcache(
                metadata,
                forward_batch,
                torch.zeros(2, dtype=torch.int32),
                is_swa_layer=False,
                use_local_attn=False,
            )

    def test_cp_cu_seqlens_k_new_resolves_published_side(self):
        backend = self._backend()
        metadata = fa_backend.MusaFlashAttentionMetadata()
        prev = torch.zeros(2, dtype=torch.int32)
        metadata.cp_cu_seqlens_k_new_prev = prev
        cp_meta, forward_batch = self._cp_batch()

        self.assertIs(
            backend._cp_cu_seqlens_k_new_for_kvcache(
                metadata, forward_batch, cp_meta.kv_len_prev_tensor
            ),
            prev,
        )
        with self.assertRaisesRegex(RuntimeError, "requires precomputed"):
            backend._cp_cu_seqlens_k_new_for_kvcache(
                metadata, forward_batch, cp_meta.kv_len_next_tensor
            )

    def test_cp_metadata_attrs_single_source_of_truth(self):
        expected = {
            f"{prefix}_scheduler_metadata_{side}"
            for side in ("prev", "next")
            for prefix in ("cp", "cp_swa", "cp_local", "cp_local_swa")
        } | {f"cp_cu_seqlens_k_new_{side}" for side in ("prev", "next")}
        actual = {
            attr
            for side in fa_backend._CP_METADATA_SIDES
            for attr in fa_backend._cp_metadata_attrs(side)
        }
        self.assertEqual(actual, expected)

        backend = self._backend()
        metadata = fa_backend.MusaFlashAttentionMetadata()
        for attr in expected:
            setattr(metadata, attr, torch.zeros(1, dtype=torch.int32))
        backend._clear_cp_scheduler_metadata(metadata)
        for attr in expected:
            self.assertIsNone(getattr(metadata, attr))

    def test_gather_paged_kv_is_eager_only(self):
        backend = self._backend()

        with patch.object(fa_backend, "get_is_capture_mode", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "eager-only"):
                backend._gather_paged_kv_for_varlen(
                    MagicMock(), MagicMock(), MagicMock(), [1]
                )

    def test_empty_dp_rank_skips_scheduler_metadata(self):
        backend = self._backend()
        forward_batch = SimpleNamespace(input_ids=torch.empty(0, dtype=torch.long))

        with (
            patch.object(
                fa_backend.FlashAttentionBackend, "init_forward_metadata"
            ) as parent_init,
            patch.object(
                backend, "_init_scheduler_metadata_for_batch"
            ) as init_metadata,
        ):
            backend.init_forward_metadata(forward_batch)

        parent_init.assert_called_once_with(forward_batch)
        init_metadata.assert_not_called()
        self.assertIsNone(backend.forward_metadata)

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
        with self.assertRaisesRegex(RuntimeError, "requires scheduler_metadata"):
            fa_backend.flash_attn_with_kvcache(
                q=MagicMock(),
                k_cache=MagicMock(),
                v_cache=MagicMock(),
            )


if __name__ == "__main__":
    unittest.main()
