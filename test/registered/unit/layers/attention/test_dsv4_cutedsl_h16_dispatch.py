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
            mock.patch.object(deepseek_v4_backend, "_is_cuda", False),
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
                mock.patch.object(
                    attention_registry,
                    "get_exec",
                    return_value=SimpleNamespace(kernel=runner.server_args),
                ),
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
            CompressedGather,
            SparsePrefillChunkCache,
        )

        self.assertEqual(DSV4_CUTEDSL_H16_MAX_TOPK, 8192)
        self.assertEqual(
            DSV4_CUTEDSL_H16_TOPK_ALIGNMENT,
            SPARSE_PREFILL_TOPK_ALIGNMENT,
        )
        validate = deepseek_v4_backend._validate_dsv4_cutedsl_h16_combined_topk
        for context, topk, expected in (
            (31744, 512, 384),
            (40960, 512, 512),
            (8064 * 128, 8064, DSV4_CUTEDSL_H16_MAX_TOPK),
        ):
            with self.subTest(context=context, topk=topk):
                self.assertEqual(
                    validate(max_context_len=context, c4_topk=topk), expected
                )
        for context, topk in ((8065 * 128, 512), (40960, 8065)):
            with self.subTest(context=context, topk=topk):
                with self.assertRaisesRegex(ValueError, "8192"):
                    validate(max_context_len=context, c4_topk=topk)

        cache = object.__new__(SparsePrefillChunkCache)
        cache.num_qo_tokens = 2
        indices = torch.arange(
            2 * 384,
            dtype=torch.int32,
        ).reshape(2, 384)
        lengths = torch.tensor([321, 384], dtype=torch.int32)
        cache.compressed = {
            128: CompressedGather(
                flat_token_ids=torch.empty(0, dtype=torch.int32),
                compressed_base=torch.empty(0, dtype=torch.int32),
                swa_base=torch.empty(0, dtype=torch.int32),
                combined_indices=indices,
                combined_lens=lengths,
            )
        }
        cache.c128_padded_combined_indices = {}

        base_indices, base_lens = cache.get_c128_combined()
        self.assertIs(base_indices, cache.compressed[128].combined_indices)
        self.assertIs(base_lens, cache.compressed[128].combined_lens)

        padded, padded_lens = cache.get_c128_combined(512)
        self.assertEqual(tuple(padded.shape), (2, 512))
        torch.testing.assert_close(
            padded[:, :384], cache.compressed[128].combined_indices
        )
        self.assertTrue(torch.all(padded[:, 384:] == -1))
        self.assertIs(padded_lens, cache.compressed[128].combined_lens)
        cached_again, _ = cache.get_c128_combined(512)
        self.assertIs(cached_again, padded)

        with self.assertRaisesRegex(ValueError, "smaller than live width"):
            cache.get_c128_combined(256)
        with self.assertRaisesRegex(ValueError, "aligned"):
            cache.get_c128_combined(450)

    def test_unbound_c128_sparse_prefill_uses_fixed_cached_bucket(self):
        self._check_sparse_prefill_dispatch(compress_ratio=128)

    def test_unbound_sparse_prefill_dispatch_preserves_cutedsl_contract(self):
        self._check_sparse_prefill_dispatch(compress_ratio=0)

    def _check_sparse_prefill_dispatch(self, compress_ratio):
        from sglang.srt.layers.attention import deepseek_v4_backend
        from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
            CompressedGather,
            SparsePrefillChunkCache,
        )

        compressed = compress_ratio == 128
        n_compressed, n_swa = (3, 2) if compressed else (0, 3)
        n_tokens = n_compressed + n_swa
        width = 384 if compressed else 128
        q = torch.empty((2, 1, 64, 512), dtype=torch.bfloat16)
        workspace = torch.empty((n_tokens, 1, 512), dtype=torch.bfloat16)
        indices = torch.zeros((2, width), dtype=torch.int32)
        lengths = torch.tensor(
            [321, 384] if compressed else [127, 128], dtype=torch.int32
        )
        sink = torch.arange(64, dtype=torch.float32)
        expected = torch.empty((2, 16, 512), dtype=torch.bfloat16)

        cache = object.__new__(SparsePrefillChunkCache)
        cache.num_qo_tokens = 2
        cache.swa_token_ids = torch.arange(n_swa, dtype=torch.int32)
        cache.swa_page_size = 256
        cache.c0_combined_indices, cache.c0_combined_lens = indices, lengths
        cache.c128_padded_combined_indices = {}
        cache.compressed = {
            128: CompressedGather(
                flat_token_ids=torch.arange(n_compressed, dtype=torch.int32),
                compressed_base=torch.zeros(1, dtype=torch.int32),
                swa_base=torch.full((1,), n_compressed, dtype=torch.int32),
                combined_indices=indices,
                combined_lens=lengths,
            )
        }
        workspace_get = mock.Mock(return_value=workspace)
        if compressed:
            workspace_get.side_effect = [(workspace, False), (workspace, True)]
        extra_buffer, swa_buffer = torch.empty(1), object()
        pool = SimpleNamespace(
            get_extra_key_page_size=mock.Mock(return_value=16),
            get_extra_key_buffer=mock.Mock(return_value=extra_buffer),
            get_swa_key_buffer_radix=mock.Mock(return_value=swa_buffer),
            get_extra_key_layout=mock.Mock(return_value="compressed_layout"),
            get_swa_key_layout=mock.Mock(return_value="swa_layout"),
        )
        backend = SimpleNamespace(
            forward_metadata=SimpleNamespace(sparse_prefill_cache=cache),
            sparse_prefill_kv_buffer=SimpleNamespace(
                get=workspace_get, get_with_head=workspace_get
            ),
            softmax_scale=512**-0.5,
            _cutedsl_h16_c128_combined_topk=512,
            _cutedsl_h16_prefill_log_emitted=True,
        )
        metadata = SimpleNamespace(
            sparse_page_indices=mock.Mock(
                return_value=torch.zeros((2, 1), dtype=torch.int32)
            )
        )
        with (
            mock.patch.object(
                deepseek_v4_backend, "dequantize_k_cache_paged_dsv4"
            ) as dequantize,
            mock.patch.object(
                deepseek_v4_backend,
                "_run_dsv4_cutedsl_h16_sparse_prefill",
                return_value=expected,
            ) as run,
        ):
            # Repeat C128 to cover padded-index and shared-dequant reuse.
            for _ in range(2 if compressed else 1):
                actual = (
                    deepseek_v4_backend.DeepseekV4AttnBackend._forward_prefill_sparse(
                        backend,
                        q=q,
                        layer_id=3,
                        compress_ratio=compress_ratio,
                        forward_batch=SimpleNamespace(),
                        token_to_kv_pool=pool,
                        core_attn_metadata=metadata,
                        attn_sink=sink,
                        use_cutedsl_h16=True,
                    )
                )
                self.assertIs(actual, expected)
                self.assertEqual(tuple(actual.shape), (2, 16, 512))
                kwargs = run.call_args.kwargs
                self.assertEqual(tuple(kwargs["q"].shape), (2, 64, 512))
                self.assertEqual(kwargs["q"].data_ptr(), q.data_ptr())
                self.assertEqual(tuple(kwargs["kv"].shape), (n_tokens, 512))
                self.assertEqual(kwargs["kv"].data_ptr(), workspace.data_ptr())
                self.assertIs(kwargs["topk_length"], lengths)
                self.assertIs(kwargs["attn_sink"], sink)
                self.assertEqual(kwargs["sm_scale"], 512**-0.5)
                if compressed:
                    self.assertIs(
                        kwargs["indices"], cache.c128_padded_combined_indices[512]
                    )
                    self.assertEqual(tuple(kwargs["indices"].shape), (2, 512))
                    torch.testing.assert_close(kwargs["indices"][:, :384], indices)
                    self.assertTrue(torch.all(kwargs["indices"][:, 384:] == -1))
                else:
                    self.assertIs(kwargs["indices"], indices)
                    self.assertEqual(kwargs["indices"].ndim, 2)

        if compressed:
            self.assertIs(run.call_args_list[0].kwargs["indices"], kwargs["indices"])
            head_source = (
                extra_buffer.data_ptr(),
                cache.compressed[128].flat_token_ids.data_ptr(),
                3,
            )
            workspace_get.assert_has_calls(
                [mock.call(n_tokens, head_source=head_source)] * 2
            )
            # A shared compressed source is dequantized once; SWA stays per-layer.
            self.assertEqual(dequantize.call_count, 3)
            compressed_call = dequantize.call_args_list[0]
            self.assertIs(compressed_call.args[0], extra_buffer)
            self.assertIs(compressed_call.args[1], cache.compressed[128].flat_token_ids)
            self.assertEqual(compressed_call.kwargs["page_size"], 16)
            self.assertEqual(compressed_call.kwargs["layout"], "compressed_layout")
            self.assertEqual(
                tuple(compressed_call.kwargs["out"].shape), (n_compressed, 1, 512)
            )
            swa_calls = dequantize.call_args_list[1:]
        else:
            workspace_get.assert_called_once_with(n_tokens)
            pool.get_swa_key_buffer_radix.assert_called_once_with(3)
            dequantize.assert_called_once_with(
                swa_buffer,
                cache.swa_token_ids,
                page_size=256,
                out=workspace,
                layout="swa_layout",
            )
            swa_calls = dequantize.call_args_list
        for call in swa_calls:
            self.assertIs(call.args[0], swa_buffer)
            self.assertIs(call.args[1], cache.swa_token_ids)
            self.assertEqual(call.kwargs["page_size"], 256)
            self.assertEqual(call.kwargs["layout"], "swa_layout")
            self.assertEqual(tuple(call.kwargs["out"].shape), (n_swa, 1, 512))

    def test_shared_sparse_gate_preserves_current_main_exclusions(self):
        from sglang.srt.layers.attention import deepseek_v4_backend as backend_module

        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            _original_forward_mode=None,
            tbo_parent_token_range=None,
        )
        backend = SimpleNamespace(
            trtllm_attn=False,
            dsv4_prefill_backend="cutedsl_h16",
            forward_metadata=SimpleNamespace(late_layer_tail=None),
        )

        def selected():
            return backend_module.DeepseekV4AttnBackend._use_sparse_prefill(
                backend, batch, num_qo_tokens=1
            )

        with (
            mock.patch.object(backend_module, "_is_cuda", False),
            mock.patch.object(backend_module, "is_cp_active", return_value=False) as cp,
            mock.patch.object(
                backend_module,
                "get_platform",
                return_value=SimpleNamespace(is_sm120=False),
            ),
            mock.patch.object(
                backend_module, "is_in_breakable_cuda_graph", return_value=False
            ) as graph,
            mock.patch.object(
                backend_module, "is_in_tc_piecewise_cuda_graph", return_value=False
            ),
            mock.patch.object(
                backend_module, "get_tc_piecewise_forward_context", return_value=None
            ),
            mock.patch.object(
                backend_module.envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL,
                "get",
                return_value=False,
            ),
        ):
            self.assertTrue(selected())
            backend.dsv4_prefill_backend = "auto"
            self.assertFalse(selected())
            backend.dsv4_prefill_backend = "cutedsl_h16"
            cp.return_value = True
            self.assertFalse(selected())
            cp.return_value = False
            backend.forward_metadata.late_layer_tail = object()
            self.assertFalse(selected())
            backend.forward_metadata.late_layer_tail = None
            backend.trtllm_attn = True
            self.assertFalse(selected())
            backend.trtllm_attn = False
            graph.return_value = True
            self.assertFalse(selected())
            graph.return_value = False
            batch.tbo_parent_token_range = (0, 1)
            self.assertFalse(selected())

    def test_raw_cuda_capture_does_not_force_sparse_snapshot(self):
        from sglang.srt.layers.attention import deepseek_v4_backend as backend_module

        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            _original_forward_mode=None,
            tbo_parent_token_range=None,
        )
        with (
            mock.patch.object(backend_module, "_is_cuda", True),
            mock.patch.object(
                backend_module, "is_in_breakable_cuda_graph", return_value=False
            ),
            mock.patch.object(
                backend_module, "is_in_tc_piecewise_cuda_graph", return_value=False
            ),
            mock.patch.object(
                backend_module, "get_tc_piecewise_forward_context", return_value=None
            ),
            mock.patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=True
            ),
        ):
            self.assertFalse(
                backend_module._use_dsv4_cutedsl_h16_forward("cutedsl_h16", batch)
            )


if __name__ == "__main__":
    unittest.main()
