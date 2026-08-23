import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestDSV4CutedslH16Dispatch(CustomTestCase):
    @staticmethod
    def _select(
        *,
        backend: str = "cutedsl_h16",
        forward_mode: ForwardMode = ForwardMode.EXTEND,
        original_forward_mode: ForwardMode | None = None,
        num_heads: int = 16,
        q: object | None = None,
        in_breakable_graph: bool = False,
        in_piecewise_graph: bool = False,
        in_full_graph: bool = False,
        current_stream_capture: bool = False,
        tbo_parent_token_range: object | None = None,
    ) -> bool:
        from sglang.srt.layers.attention import deepseek_v4_backend

        if q is None:
            q = torch.empty((2, 1, 16, 512), dtype=torch.bfloat16)
        forward_batch = SimpleNamespace(
            forward_mode=forward_mode,
            _original_forward_mode=original_forward_mode,
            tbo_parent_token_range=tbo_parent_token_range,
        )
        piecewise_context = SimpleNamespace(full_graph=True) if in_full_graph else None
        with (
            mock.patch.object(
                deepseek_v4_backend,
                "is_in_breakable_cuda_graph",
                return_value=in_breakable_graph,
            ),
            mock.patch.object(
                deepseek_v4_backend,
                "is_in_tc_piecewise_cuda_graph",
                return_value=in_piecewise_graph,
            ),
            mock.patch.object(
                deepseek_v4_backend,
                "get_tc_piecewise_forward_context",
                return_value=piecewise_context,
            ),
            mock.patch.object(
                torch.cuda,
                "is_current_stream_capturing",
                return_value=current_stream_capture,
            ),
        ):
            return deepseek_v4_backend._use_dsv4_cutedsl_h16_sparse_prefill(
                backend,
                q,
                forward_batch,
                num_heads,
            )

    def test_explicit_backend_selects_supported_eager_extend(self):
        self.assertTrue(self._select())

    def test_production_h64_padded_query_is_supported(self):
        q = torch.empty((2, 1, 64, 512), dtype=torch.bfloat16)
        self.assertTrue(self._select(q=q))

    def test_default_backends_keep_existing_path(self):
        for backend in ("auto", "flashmla_sparse", "flashmla_sparse_q8"):
            with self.subTest(backend=backend):
                self.assertFalse(self._select(backend=backend))

    def test_explicit_backend_takes_precedence_over_q8_debug_override(self):
        from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
            DSV4_Q8KV8_PREFILL_ENV,
            use_dsv4_q8kv8_sparse_prefill,
        )

        with mock.patch.dict(os.environ, {DSV4_Q8KV8_PREFILL_ENV: "1"}):
            self.assertFalse(use_dsv4_q8kv8_sparse_prefill("cutedsl_h16"))
            self.assertTrue(use_dsv4_q8kv8_sparse_prefill("auto"))

    def test_non_cuda_dsv4_backends_reject_explicit_cutedsl_h16(self):
        from sglang.srt.layers.attention import attention_registry

        runner = SimpleNamespace(
            server_args=SimpleNamespace(dsv4_prefill_backend="cutedsl_h16")
        )
        for platform_flag, platform_name in (("_is_hip", "HIP"), ("_is_npu", "NPU")):
            flags = {"_is_hip": False, "_is_npu": False, platform_flag: True}
            with (
                self.subTest(platform=platform_name),
                mock.patch.multiple(attention_registry, **flags),
                self.assertRaisesRegex(ValueError, f"CUDA SM90.*{platform_name}"),
            ):
                attention_registry.create_dsv4_backend(runner)

    def test_speculative_and_cuda_graph_paths_fall_back(self):
        self.assertFalse(self._select(forward_mode=ForwardMode.DRAFT_EXTEND_V2))
        self.assertFalse(self._select(in_breakable_graph=True))
        self.assertFalse(self._select(in_piecewise_graph=True))
        self.assertFalse(self._select(in_full_graph=True))

        cuda_q = SimpleNamespace(is_cuda=True)
        self.assertFalse(self._select(q=cuda_q, current_stream_capture=True))

    def test_max_len_rewritten_extend_paths_fall_back(self):
        for original_mode in (
            ForwardMode.TARGET_VERIFY,
            ForwardMode.DRAFT_EXTEND_V2,
            ForwardMode.IDLE,
        ):
            with self.subTest(original_mode=original_mode):
                self.assertFalse(self._select(original_forward_mode=original_mode))

    def test_tbo_split_extend_falls_back(self):
        self.assertFalse(self._select(tbo_parent_token_range=(0, 16)))

    def test_selected_backend_rejects_unsupported_eager_contracts(self):
        cases = (
            ("heads", {"num_heads": 32}),
            (
                "dtype",
                {"q": torch.empty((2, 1, 16, 512), dtype=torch.float16)},
            ),
            (
                "dimension",
                {"q": torch.empty((2, 1, 16, 256), dtype=torch.bfloat16)},
            ),
            (
                "storage_heads",
                {"q": torch.empty((2, 1, 32, 512), dtype=torch.bfloat16)},
            ),
        )
        for name, kwargs in cases:
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(ValueError, "cutedsl_h16"),
            ):
                self._select(**kwargs)

    def test_lazy_adapter_preserves_tensor_api_and_2d_indices(self):
        from sglang.srt.layers.attention import deepseek_v4_backend

        module_name = "sglang.kernels.ops.attention.dsv4.cute_sparse_mla_h16"
        fake_module = types.ModuleType(module_name)
        expected = torch.empty((2, 16, 512), dtype=torch.bfloat16)
        fake_kernel = mock.Mock(return_value=expected)
        setattr(fake_module, "cute_sparse_mla_h16_fwd", fake_kernel)

        q = torch.empty((2, 64, 512), dtype=torch.bfloat16)
        kv = torch.empty((128, 512), dtype=torch.bfloat16)
        indices = torch.zeros((2, 128), dtype=torch.int32)
        topk_length = torch.full((2,), 128, dtype=torch.int32)
        attn_sink = torch.zeros(64, dtype=torch.float32)

        with mock.patch.dict(sys.modules, {module_name: fake_module}):
            actual = deepseek_v4_backend._run_dsv4_cutedsl_h16_sparse_prefill(
                q,
                kv,
                indices,
                topk_length,
                attn_sink,
                512**-0.5,
            )

        self.assertIs(actual, expected)
        kwargs = fake_kernel.call_args.kwargs
        self.assertIs(kwargs["kv"], kv)
        self.assertEqual(kwargs["kv"].ndim, 2)
        self.assertIs(kwargs["indices"], indices)
        self.assertEqual(kwargs["indices"].ndim, 2)
        self.assertNotIn("d_v", kwargs)
        self.assertNotIn("trusted_prefix", kwargs)

    def test_c128_fixed_width_validation_and_chunk_cache(self):
        from sglang.kernels.ops.attention.dsv4.cutedsl_h16_contract import (
            DSV4_CUTEDSL_H16_MAX_TOPK,
            DSV4_CUTEDSL_H16_TOPK_ALIGNMENT,
        )
        from sglang.srt.layers.attention import deepseek_v4_backend
        from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
            SPARSE_PREFILL_TOPK_ALIGNMENT,
            SparsePrefillChunkCache,
        )

        self.assertEqual(DSV4_CUTEDSL_H16_MAX_TOPK, 8192)
        self.assertEqual(
            DSV4_CUTEDSL_H16_TOPK_ALIGNMENT,
            SPARSE_PREFILL_TOPK_ALIGNMENT,
        )
        self.assertEqual(
            deepseek_v4_backend._validate_dsv4_cutedsl_h16_combined_topk(
                max_context_len=31744,
                c4_topk=512,
            ),
            384,
        )
        self.assertEqual(
            deepseek_v4_backend._validate_dsv4_cutedsl_h16_combined_topk(
                max_context_len=40960,
                c4_topk=512,
            ),
            512,
        )
        self.assertEqual(
            deepseek_v4_backend._validate_dsv4_cutedsl_h16_combined_topk(
                max_context_len=8064 * 128,
                c4_topk=8064,
            ),
            DSV4_CUTEDSL_H16_MAX_TOPK,
        )
        with self.assertRaisesRegex(ValueError, "8192"):
            deepseek_v4_backend._validate_dsv4_cutedsl_h16_combined_topk(
                max_context_len=8065 * 128,
                c4_topk=512,
            )
        with self.assertRaisesRegex(ValueError, "8192"):
            deepseek_v4_backend._validate_dsv4_cutedsl_h16_combined_topk(
                max_context_len=40960,
                c4_topk=8065,
            )

        cache = object.__new__(SparsePrefillChunkCache)
        cache.num_qo_tokens = 2
        cache.c128_combined_indices = torch.arange(
            2 * 384,
            dtype=torch.int32,
        ).reshape(2, 384)
        cache.c128_combined_lens = torch.tensor([321, 384], dtype=torch.int32)
        cache.c128_padded_combined_indices = {}

        base_indices, base_lens = cache.get_c128_combined()
        self.assertIs(base_indices, cache.c128_combined_indices)
        self.assertIs(base_lens, cache.c128_combined_lens)

        padded, padded_lens = cache.get_c128_combined(512)
        self.assertEqual(tuple(padded.shape), (2, 512))
        torch.testing.assert_close(padded[:, :384], cache.c128_combined_indices)
        self.assertTrue(torch.all(padded[:, 384:] == -1))
        self.assertIs(padded_lens, cache.c128_combined_lens)
        cached_again, _ = cache.get_c128_combined(512)
        self.assertIs(cached_again, padded)

        with self.assertRaisesRegex(ValueError, "smaller than live width"):
            cache.get_c128_combined(256)
        with self.assertRaisesRegex(ValueError, "aligned"):
            cache.get_c128_combined(450)

    def test_unbound_c128_sparse_prefill_uses_fixed_cached_bucket(self):
        from sglang.srt.layers.attention import deepseek_v4_backend
        from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
            SparsePrefillChunkCache,
        )

        num_queries = 2
        num_compressed_tokens = 3
        num_swa_tokens = 2
        q = torch.empty(
            (num_queries, 1, 64, 512),
            dtype=torch.bfloat16,
        )
        workspace = torch.empty(
            (num_compressed_tokens + num_swa_tokens, 1, 512),
            dtype=torch.bfloat16,
        )
        live_indices = torch.zeros((num_queries, 384), dtype=torch.int32)
        lengths = torch.tensor([321, 384], dtype=torch.int32)
        sink = torch.arange(64, dtype=torch.float32)
        expected = torch.empty(
            (num_queries, 16, 512),
            dtype=torch.bfloat16,
        )

        cache = object.__new__(SparsePrefillChunkCache)
        cache.num_qo_tokens = num_queries
        cache.swa_token_ids = torch.arange(num_swa_tokens, dtype=torch.int32)
        cache.swa_page_size = 256
        cache.c128_flat_token_ids = torch.arange(
            num_compressed_tokens,
            dtype=torch.int32,
        )
        cache.c128_combined_indices = live_indices
        cache.c128_combined_lens = lengths
        cache.c128_padded_combined_indices = {}

        workspace_get = mock.Mock(return_value=workspace)
        extra_buffer = object()
        swa_buffer = object()
        token_to_kv_pool = SimpleNamespace(
            get_extra_key_page_size=mock.Mock(return_value=16),
            get_extra_key_buffer=mock.Mock(return_value=extra_buffer),
            get_swa_key_buffer_radix=mock.Mock(return_value=swa_buffer),
        )
        backend = SimpleNamespace(
            forward_metadata=SimpleNamespace(sparse_prefill_cache=cache),
            sparse_prefill_workspace=SimpleNamespace(get=workspace_get),
            softmax_scale=512**-0.5,
            _cutedsl_h16_c128_combined_topk=512,
            _cutedsl_h16_prefill_log_emitted=True,
        )
        core_attn_metadata = SimpleNamespace(
            c128_page_indices=torch.zeros((num_queries, 1), dtype=torch.int32),
        )

        with (
            mock.patch.object(
                deepseek_v4_backend,
                "dequantize_k_cache_paged",
            ) as dequantize,
            mock.patch.object(
                deepseek_v4_backend,
                "_run_dsv4_cutedsl_h16_sparse_prefill",
                return_value=expected,
            ) as run_cutedsl,
        ):
            first = deepseek_v4_backend.DeepseekV4AttnBackend._forward_prefill_sparse(
                backend,
                q=q,
                layer_id=3,
                compress_ratio=128,
                forward_batch=SimpleNamespace(),
                token_to_kv_pool=token_to_kv_pool,
                core_attn_metadata=core_attn_metadata,
                attn_sink=sink,
                use_cutedsl_h16=True,
            )
            first_indices = run_cutedsl.call_args.kwargs["indices"]
            second = deepseek_v4_backend.DeepseekV4AttnBackend._forward_prefill_sparse(
                backend,
                q=q,
                layer_id=3,
                compress_ratio=128,
                forward_batch=SimpleNamespace(),
                token_to_kv_pool=token_to_kv_pool,
                core_attn_metadata=core_attn_metadata,
                attn_sink=sink,
                use_cutedsl_h16=True,
            )
            second_kwargs = run_cutedsl.call_args.kwargs

        self.assertIs(first, expected)
        self.assertIs(second, expected)
        workspace_get.assert_has_calls([mock.call(5), mock.call(5)])
        self.assertEqual(dequantize.call_count, 4)
        self.assertEqual(tuple(first_indices.shape), (num_queries, 512))
        torch.testing.assert_close(first_indices[:, :384], live_indices)
        self.assertTrue(torch.all(first_indices[:, 384:] == -1))
        self.assertIs(first_indices, second_kwargs["indices"])
        self.assertIs(
            first_indices,
            cache.c128_padded_combined_indices[512],
        )
        self.assertEqual(tuple(second_kwargs["q"].shape), (num_queries, 64, 512))
        self.assertEqual(second_kwargs["q"].data_ptr(), q.data_ptr())
        self.assertEqual(tuple(second_kwargs["kv"].shape), (5, 512))
        self.assertEqual(second_kwargs["kv"].data_ptr(), workspace.data_ptr())
        self.assertIs(second_kwargs["topk_length"], lengths)
        self.assertIs(second_kwargs["attn_sink"], sink)
        self.assertEqual(second_kwargs["sm_scale"], 512**-0.5)

        compressed_call, swa_call = dequantize.call_args_list[:2]
        self.assertIs(compressed_call.args[0], extra_buffer)
        self.assertIs(compressed_call.args[1], cache.c128_flat_token_ids)
        self.assertEqual(compressed_call.kwargs["page_size"], 16)
        self.assertEqual(
            tuple(compressed_call.kwargs["out"].shape),
            (num_compressed_tokens, 1, 512),
        )
        self.assertIs(swa_call.args[0], swa_buffer)
        self.assertIs(swa_call.args[1], cache.swa_token_ids)
        self.assertEqual(swa_call.kwargs["page_size"], 256)
        self.assertEqual(
            tuple(swa_call.kwargs["out"].shape),
            (num_swa_tokens, 1, 512),
        )

    def test_unbound_sparse_prefill_dispatch_preserves_cutedsl_contract(self):
        from sglang.srt.layers.attention import deepseek_v4_backend

        num_queries = 2
        num_workspace_tokens = 3
        q = torch.empty(
            (num_queries, 1, 64, 512),
            dtype=torch.bfloat16,
        )
        workspace = torch.empty(
            (num_workspace_tokens, 1, 512),
            dtype=torch.bfloat16,
        )
        indices = torch.zeros((num_queries, 128), dtype=torch.int32)
        lengths = torch.tensor([127, 128], dtype=torch.int32)
        sink = torch.arange(64, dtype=torch.float32)
        expected = torch.empty(
            (num_queries, 16, 512),
            dtype=torch.bfloat16,
        )
        cache = SimpleNamespace(
            swa_token_ids=torch.arange(num_workspace_tokens, dtype=torch.int32),
            swa_page_size=256,
            c0_combined_indices=indices,
            c0_combined_lens=lengths,
        )
        workspace_get = mock.Mock(return_value=workspace)
        swa_buffer = object()
        get_swa_buffer = mock.Mock(return_value=swa_buffer)
        backend = SimpleNamespace(
            forward_metadata=SimpleNamespace(sparse_prefill_cache=cache),
            sparse_prefill_workspace=SimpleNamespace(get=workspace_get),
            softmax_scale=512**-0.5,
            _cutedsl_h16_prefill_log_emitted=True,
        )
        token_to_kv_pool = SimpleNamespace(
            get_swa_key_buffer_radix=get_swa_buffer,
        )

        with (
            mock.patch.object(
                deepseek_v4_backend,
                "dequantize_k_cache_paged",
            ) as dequantize,
            mock.patch.object(
                deepseek_v4_backend,
                "_run_dsv4_cutedsl_h16_sparse_prefill",
                return_value=expected,
            ) as run_cutedsl,
        ):
            actual = deepseek_v4_backend.DeepseekV4AttnBackend._forward_prefill_sparse(
                backend,
                q=q,
                layer_id=3,
                compress_ratio=0,
                forward_batch=SimpleNamespace(),
                token_to_kv_pool=token_to_kv_pool,
                core_attn_metadata=SimpleNamespace(),
                attn_sink=sink,
                use_cutedsl_h16=True,
            )

        self.assertIs(actual, expected)
        self.assertEqual(tuple(actual.shape), (num_queries, 16, 512))
        workspace_get.assert_called_once_with(num_workspace_tokens)
        get_swa_buffer.assert_called_once_with(3)
        dequantize.assert_called_once_with(
            swa_buffer,
            cache.swa_token_ids,
            page_size=256,
            out=workspace,
        )
        kwargs = run_cutedsl.call_args.kwargs
        self.assertEqual(tuple(kwargs["q"].shape), (num_queries, 64, 512))
        self.assertEqual(kwargs["q"].data_ptr(), q.data_ptr())
        self.assertEqual(tuple(kwargs["kv"].shape), (num_workspace_tokens, 512))
        self.assertEqual(kwargs["kv"].data_ptr(), workspace.data_ptr())
        self.assertIs(kwargs["indices"], indices)
        self.assertEqual(kwargs["indices"].ndim, 2)
        self.assertIs(kwargs["topk_length"], lengths)
        self.assertIs(kwargs["attn_sink"], sink)
        self.assertEqual(kwargs["sm_scale"], 512**-0.5)


if __name__ == "__main__":
    unittest.main()
