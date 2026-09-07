import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups.overrides import _nvfp4_speculative_attention_mode
from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _config(**overrides):
    values = dict(
        kv_cache_dtype="nvfp4",
        speculative_algorithm="EAGLE",
        speculative_attention_mode="prefill",
        attention_backend=None,
        prefill_attention_backend="flashinfer",
        decode_attention_backend="trtllm_mha",
        speculative_draft_attention_backend=None,
        speculative_draft_kv_cache_dtype=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class TestNVFP4SpecConfig(CustomTestCase):
    def test_resolves_speculative_attention_to_decode(self):
        self.assertEqual(
            _nvfp4_speculative_attention_mode(_config()),
            {"speculative_attention_mode": "decode"},
        )

    def test_requires_nvfp4_split_backends(self):
        for prefill, decode in (
            ("flashinfer", "flashinfer"),
            ("trtllm_mha", "trtllm_mha"),
            ("triton", "trtllm_mha"),
        ):
            with self.subTest(prefill=prefill, decode=decode):
                with self.assertRaisesRegex(
                    ValueError, "requires.*flashinfer.*trtllm_mha"
                ):
                    _nvfp4_speculative_attention_mode(
                        _config(
                            prefill_attention_backend=prefill,
                            decode_attention_backend=decode,
                        )
                    )

    def test_rejects_incompatible_explicit_draft_backend(self):
        with self.assertRaisesRegex(
            ValueError, "speculative-draft-attention-backend trtllm_mha"
        ):
            _nvfp4_speculative_attention_mode(
                _config(speculative_draft_attention_backend="flashinfer")
            )

    def test_accepts_explicit_trtllm_mha_draft_backend(self):
        self.assertEqual(
            _nvfp4_speculative_attention_mode(
                _config(speculative_draft_attention_backend="trtllm_mha")
            ),
            {"speculative_attention_mode": "decode"},
        )

    def test_accepts_flashinfer_draft_with_explicit_non_nvfp4_cache(self):
        for draft_kv_dtype in (
            "auto",
            "bf16",
            "bfloat16",
            "fp8_e4m3",
            "fp8_e5m2",
        ):
            with self.subTest(draft_kv_dtype=draft_kv_dtype):
                self.assertEqual(
                    _nvfp4_speculative_attention_mode(
                        _config(
                            speculative_draft_attention_backend="flashinfer",
                            speculative_draft_kv_cache_dtype=draft_kv_dtype,
                        )
                    ),
                    {"speculative_attention_mode": "decode"},
                )

    def test_ignores_non_speculative_or_non_nvfp4_config(self):
        self.assertEqual(
            _nvfp4_speculative_attention_mode(_config(speculative_algorithm=None)),
            {},
        )
        self.assertEqual(
            _nvfp4_speculative_attention_mode(_config(kv_cache_dtype="bfloat16")),
            {},
        )


class TestNVFP4SpecBackendRouting(CustomTestCase):
    def setUp(self):
        self.prefill = object()
        self.decode = object()
        self.backend = HybridAttnBackend.__new__(HybridAttnBackend)
        self.backend.prefill_backend = self.prefill
        self.backend.decode_backend = self.decode

    def test_decode_mode_routes_both_speculative_extend_modes_to_decode(self):
        self.backend.spec_attn_is_decode = True
        for mode in (ForwardMode.TARGET_VERIFY, ForwardMode.DRAFT_EXTEND_V2):
            with self.subTest(mode=mode):
                self.assertIs(self.backend._select_backend(mode), self.decode)
        self.assertIs(self.backend._select_backend(ForwardMode.EXTEND), self.prefill)

    def test_prefill_mode_keeps_speculative_extend_on_prefill(self):
        self.backend.spec_attn_is_decode = False
        for mode in (ForwardMode.TARGET_VERIFY, ForwardMode.DRAFT_EXTEND_V2):
            with self.subTest(mode=mode):
                self.assertIs(self.backend._select_backend(mode), self.prefill)


class TestFlashInferNVFP4SpecGuard(CustomTestCase):
    def setUp(self):
        self.backend = FlashInferAttnBackend.__new__(FlashInferAttnBackend)
        self.backend.prefill_uses_dequant_workspace = True
        self.backend.dq_page_table = object()
        self.backend.dq_paged_kernel_lens = object()
        self.backend.cpu_req_pool_indices = object()

    def test_reset_clears_previous_prefill_metadata(self):
        self.backend._reset_dequant_workspace_metadata()
        self.assertIsNone(self.backend.dq_page_table)
        self.assertIsNone(self.backend.dq_paged_kernel_lens)
        self.assertIsNone(self.backend.cpu_req_pool_indices)

    def test_rejects_speculative_workspace_use_with_clear_error(self):
        for mode in (ForwardMode.TARGET_VERIFY, ForwardMode.DRAFT_EXTEND_V2):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(RuntimeError, "dequant workspace"):
                    self.backend._reject_dequant_workspace_speculative_forward(mode)

        self.backend._reject_dequant_workspace_speculative_forward(ForwardMode.EXTEND)

    def test_pool_rejects_missing_host_lengths_before_indexing(self):
        pool = SimpleNamespace(
            get_raw_kv_buffer=lambda layer_id: (None, None, None, None),
            get_dequant_workspace=lambda: (None, None),
        )
        with self.assertRaisesRegex(RuntimeError, "requires CPU request"):
            MHATokenToKVPool._prepare_dequant_extend_workspace(
                pool,
                layer_id=0,
                global_layer_id=0,
                req_to_token=torch.empty(0),
                req_pool_indices_cpu=None,
                extend_prefix_lens_cpu=None,
                extend_seq_lens_cpu=None,
                page_size=64,
            )


class _FakeKVPool:
    def __init__(self):
        self.write = None

    def set_kv_buffer(self, *args):
        self.write = args


class TestTRTLLMMHANVFP4SpecExtend(CustomTestCase):
    def _make_backend(self, calls):
        backend = TRTLLMHAAttnBackend.__new__(TRTLLMHAAttnBackend)
        backend.decode_uses_native_fp4 = True
        backend.is_nvfp4_kvcache = True
        backend.is_xqa_impl = False
        backend.use_fmha_v2 = False
        backend.data_type = torch.uint8
        backend.q_data_type = torch.bfloat16
        backend.workspace_buffer = torch.empty(0, dtype=torch.uint8)
        backend.max_context_len = 128
        backend._multi_ctas_kv_counter_buffer = None
        backend.token_to_kv_pool = _FakeKVPool()
        backend.forward_metadata = SimpleNamespace(
            swa_out_cache_loc=None,
            is_ragged_verify=False,
            max_seq_len_q=3,
            cache_seqlens_int32=torch.tensor([8, 9], dtype=torch.int32),
            cu_seqlens_q=torch.tensor([0, 3, 6], dtype=torch.int32),
            xqa_mask=TRTLLMHAAttnBackend._build_xqa_causal_mask(
                num_tokens=6, max_q_len=3, device="cpu"
            ),
        )
        backend._should_use_fused_fp8_path = lambda save, k, batch: False
        backend._kv_write_scales = lambda layer: (None, None)
        backend._get_nvfp4_decode_kv_cache = lambda layer: (
            "packed-kv",
            "block-scales",
        )
        backend._get_nvfp4_bmm_scales = lambda layer: (2.0, 3.0)
        backend._get_layer_page_table = lambda layer, batch: "page-table"

        def run_fixed(q, kv_cache, page_table, seq_lens, **kwargs):
            calls.append(
                dict(
                    q=q,
                    kv_cache=kv_cache,
                    page_table=page_table,
                    seq_lens=seq_lens,
                    **kwargs,
                )
            )
            return q.float()

        backend._run_fixed_q_len_decode = run_fixed
        return backend

    def test_plain_extend_still_rejects_native_fp4(self):
        backend = TRTLLMHAAttnBackend.__new__(TRTLLMHAAttnBackend)
        backend.decode_uses_native_fp4 = True
        batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
        with self.assertRaisesRegex(RuntimeError, "supports decode.*only"):
            backend.forward_extend(None, None, None, None, batch)

    def test_xqa_causal_mask_is_uint16_aligned_for_uniform_and_ragged_q(self):
        uniform = TRTLLMHAAttnBackend._build_xqa_causal_mask(
            num_tokens=6, max_q_len=3, device="cpu"
        )
        self.assertEqual(uniform.dtype, torch.uint16)
        self.assertEqual(tuple(uniform.shape), (6, 2))
        self.assertEqual(
            uniform.to(torch.int32).tolist(),
            [[1, 0], [3, 0], [7, 0], [1, 0], [3, 0], [7, 0]],
        )

        ragged = TRTLLMHAAttnBackend._build_xqa_causal_mask(
            num_tokens=5,
            max_q_len=3,
            device="cpu",
            cu_seqlens_q=torch.tensor([0, 2, 5], dtype=torch.int32),
        )
        self.assertEqual(
            ragged.to(torch.int32).tolist(),
            [[1, 0], [3, 0], [1, 0], [3, 0], [7, 0]],
        )

    def test_speculative_extend_uses_native_fp4_decode_inputs(self):
        layer = SimpleNamespace(
            tp_q_head_num=2,
            head_dim=4,
            scaling=0.5,
            sliding_window_size=-1,
            attn_type=None,
        )
        q = torch.randn(6, 8)
        k = torch.randn(6, 8)
        v = torch.randn(6, 8)

        for mode in (ForwardMode.TARGET_VERIFY, ForwardMode.DRAFT_EXTEND_V2):
            with self.subTest(mode=mode):
                calls = []
                backend = self._make_backend(calls)
                batch = SimpleNamespace(
                    forward_mode=mode,
                    out_cache_loc=torch.arange(6),
                )
                with patch(
                    "sglang.srt.layers.attention.trtllm_mha_backend.is_cp_v2_active",
                    return_value=False,
                ):
                    output = backend.forward_extend(q, k, v, layer, batch)

                self.assertEqual(len(calls), 1)
                call = calls[0]
                self.assertEqual(call["kv_cache"], "packed-kv")
                self.assertEqual(call["page_table"], "page-table")
                self.assertEqual(call["kv_cache_sf"], "block-scales")
                self.assertEqual(call["q_len_per_req"], 3)
                self.assertEqual(call["mask"].dtype, torch.uint16)
                self.assertEqual(tuple(call["mask"].shape), (6, 2))
                self.assertEqual(call["bmm1_scale"], 1.0)
                self.assertEqual(call["bmm2_scale"], 3.0)
                self.assertEqual(output.dtype, torch.bfloat16)
                self.assertIsNotNone(backend.token_to_kv_pool.write)
                self.assertEqual(backend.token_to_kv_pool.write[-2:], (None, None))

    def test_legacy_draft_extend_v1_uses_native_fp4_decode(self):
        calls = []
        backend = self._make_backend(calls)
        layer = SimpleNamespace(
            tp_q_head_num=2,
            head_dim=4,
            scaling=0.5,
            sliding_window_size=-1,
            attn_type=None,
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            out_cache_loc=torch.arange(6),
            spec_info=SimpleNamespace(is_draft_input=lambda: True),
        )
        backend.decode_uses_native_fp4 = False
        self.assertFalse(backend._uses_spec_decode_kernel(batch))
        backend.decode_uses_native_fp4 = True
        self.assertTrue(backend._uses_spec_decode_kernel(batch))
        q = torch.randn(6, 8)
        with patch(
            "sglang.srt.layers.attention.trtllm_mha_backend.is_cp_v2_active",
            return_value=False,
        ):
            output = backend.forward_extend(q, q, q, layer, batch)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["q_len_per_req"], 3)
        self.assertIs(calls[0]["mask"], backend.forward_metadata.xqa_mask)
        self.assertEqual(output.dtype, torch.bfloat16)

    def test_xqa_ragged_verify_uses_direct_api_and_packed_q(self):
        calls = []
        backend = self._make_backend([])
        backend.is_xqa_impl = True
        backend.forward_metadata.is_ragged_verify = True
        backend.forward_metadata.max_seq_len_q = 3
        backend.forward_metadata.cu_seqlens_q = torch.tensor(
            [0, 2, 5], dtype=torch.int32
        )
        backend.forward_metadata.xqa_mask = TRTLLMHAAttnBackend._build_xqa_causal_mask(
            num_tokens=5,
            max_q_len=3,
            device="cpu",
            cu_seqlens_q=backend.forward_metadata.cu_seqlens_q,
        )

        def run_ragged(**kwargs):
            calls.append(kwargs)
            return kwargs["query"].float()

        fake_flashinfer = SimpleNamespace(
            decode=SimpleNamespace(
                xqa_batch_decode_with_kv_cache=run_ragged,
            )
        )
        layer = SimpleNamespace(
            tp_q_head_num=2,
            head_dim=4,
            scaling=0.5,
            sliding_window_size=-1,
            attn_type=None,
        )
        q = torch.randn(5, 2, 8)[:, 0]
        self.assertFalse(q.is_contiguous())
        batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            out_cache_loc=torch.arange(5),
        )
        with (
            patch(
                "sglang.srt.layers.attention.trtllm_mha_backend.is_cp_v2_active",
                return_value=False,
            ),
            patch(
                "sglang.srt.layers.attention.trtllm_mha_backend.flashinfer",
                fake_flashinfer,
                create=True,
            ),
        ):
            output = backend.forward_extend(q, q, q, layer, batch)

        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0]["q_cu_seq_lens"], backend.forward_metadata.cu_seqlens_q)
        self.assertEqual(calls[0]["q_len_per_req"], 3)
        self.assertIs(calls[0]["mask"], backend.forward_metadata.xqa_mask)
        self.assertEqual(calls[0]["kv_layout"], "HND")
        self.assertTrue(calls[0]["query"].is_contiguous())
        self.assertNotIn("cum_seq_lens_q", calls[0])
        self.assertNotIn("max_q_len", calls[0])
        self.assertNotIn("out_dtype", calls[0])
        self.assertEqual(output.dtype, torch.bfloat16)

    def test_trtllm_gen_ragged_verify_uses_wrapper_varlen_api(self):
        calls = []
        backend = self._make_backend([])
        backend.is_xqa_impl = False
        backend.forward_metadata.is_ragged_verify = True
        backend.forward_metadata.max_seq_len_q = 3
        backend.forward_metadata.cu_seqlens_q = torch.tensor(
            [0, 2, 5], dtype=torch.int32
        )
        backend.forward_metadata.xqa_mask = None

        def run_ragged(**kwargs):
            calls.append(kwargs)
            return kwargs["query"].float()

        fake_flashinfer = SimpleNamespace(
            decode=SimpleNamespace(
                trtllm_batch_decode_with_kv_cache=run_ragged,
            )
        )
        layer = SimpleNamespace(
            tp_q_head_num=2,
            head_dim=4,
            scaling=0.5,
            sliding_window_size=-1,
            attn_type=None,
        )
        q = torch.randn(5, 8)
        batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            out_cache_loc=torch.arange(5),
        )
        with (
            patch(
                "sglang.srt.layers.attention.trtllm_mha_backend.is_cp_v2_active",
                return_value=False,
            ),
            patch(
                "sglang.srt.layers.attention.trtllm_mha_backend.flashinfer",
                fake_flashinfer,
                create=True,
            ),
        ):
            output = backend.forward_extend(q, q, q, layer, batch)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["backend"], "trtllm-gen")
        self.assertIsNone(calls[0]["q_len_per_req"])
        self.assertEqual(calls[0]["max_q_len"], 3)
        self.assertIs(calls[0]["cum_seq_lens_q"], backend.forward_metadata.cu_seqlens_q)
        self.assertNotIn("q_cu_seq_lens", calls[0])
        self.assertNotIn("mask", calls[0])
        self.assertEqual(output.dtype, torch.bfloat16)

    def test_fixed_q_xqa_makes_strided_batch_query_contiguous(self):
        calls = []
        backend = self._make_backend([])
        del backend._run_fixed_q_len_decode
        backend.is_xqa_impl = True
        backend.decode_seq_len_splits = 1

        def run_decode(**kwargs):
            calls.append(kwargs)
            return torch.zeros_like(kwargs["query"], dtype=backend.q_data_type)

        fake_flashinfer = SimpleNamespace(
            decode=SimpleNamespace(
                trtllm_batch_decode_with_kv_cache=run_decode,
            )
        )
        backing = torch.randn(4, 3, 2, 4)
        query = backing[:, 0]
        self.assertFalse(query.is_contiguous())

        with patch(
            "sglang.srt.layers.attention.trtllm_mha_backend.flashinfer",
            fake_flashinfer,
            create=True,
        ):
            output = backend._run_fixed_q_len_decode(
                query,
                "packed-kv",
                torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
                torch.tensor([19, 11], dtype=torch.int32),
                bmm1_scale=0.5,
                bmm2_scale=1.0,
                window_left=-1,
                sinks=None,
                q_len_per_req=2,
                kv_cache_sf="block-scales",
            )

        self.assertEqual(len(calls), 1)
        self.assertTrue(all(call["query"].is_contiguous() for call in calls))
        self.assertEqual(tuple(output.shape), (4, 2, 4))


if __name__ == "__main__":
    unittest.main()
