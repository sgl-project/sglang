import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.attention import trtllm_mla_backend as backend_module
from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLABackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestTRTLLMMLAExplicitBackend(CustomTestCase):
    """Verify ``TRTLLMMLABackend._run_decode_kernel`` passes ``backend='trtllm-gen'``
    explicitly on SM10x supported head counts while preserving ``auto`` fallbacks."""

    def _make_backend(self, backend_name: str = "trtllm-gen"):
        backend = object.__new__(TRTLLMMLABackend)
        backend.backend = backend_name
        backend.data_type = torch.bfloat16
        backend.qk_nope_head_dim = 128
        backend.kv_lora_rank = 512
        backend.qk_rope_head_dim = 64
        backend.workspace_buffer = torch.zeros(16, dtype=torch.uint8)
        backend._multi_ctas_kv_counter_buffer = torch.zeros(8, dtype=torch.int32)
        return backend

    def _invoke_decode(
        self,
        backend: TRTLLMMLABackend,
        *,
        query: torch.Tensor,
        sm100_supported: bool,
        page_size: int = 64,
        return_lse: bool = False,
    ) -> dict:
        captured: list[dict] = []
        sentinel = (torch.empty(0), torch.empty(0)) if return_lse else torch.empty(0)

        def fake_decode(**kw):
            backend_arg = kw.get("backend", "auto")
            trtllm_eligible = (
                sm100_supported
                and not (64 < kw["query"].size(-2) < 128)
                and kw["kv_cache"].size(-2) in (32, 64)
            )
            if kw.get("multi_ctas_kv_counter_buffer") is not None:
                if backend_arg not in ("auto", "trtllm-gen"):
                    raise ValueError(
                        "multi_ctas_kv_counter_buffer is only supported by the trtllm-gen backend"
                    )
                if backend_arg == "auto" and not trtllm_eligible:
                    raise ValueError(
                        "multi_ctas_kv_counter_buffer is only supported when a trtllm-gen runner is selected"
                    )
            captured.append(kw)
            return sentinel

        flashinfer_stub = SimpleNamespace(
            decode=SimpleNamespace(trtllm_batch_decode_with_kv_cache_mla=fake_decode)
        )
        parallel = SimpleNamespace(dcp_enabled=False, dcp_size=1, dcp_rank=0)
        layer = SimpleNamespace(scaling=0.125, k_scale_float=None)

        with (
            mock.patch.object(
                backend_module, "is_sm100_supported", return_value=sm100_supported
            ),
            mock.patch.object(backend_module, "get_parallel", return_value=parallel),
            mock.patch.object(
                backend_module, "flashinfer", flashinfer_stub, create=True
            ),
        ):
            out = backend._run_decode_kernel(
                query=query,
                kv_cache=torch.zeros((4, 1, page_size, 576), dtype=torch.bfloat16),
                block_tables=torch.zeros((2, 4), dtype=torch.int32),
                seq_lens=torch.tensor([32, 64], dtype=torch.int64),
                max_seq_len=64,
                layer=layer,
                return_lse=return_lse,
            )

        self.assertIs(out, sentinel)
        self.assertEqual(len(captured), 1)
        return captured[0]

    def test_sm100_supported_head_counts_pass_explicit_trtllm_gen(self):
        backend = self._make_backend("trtllm-gen")
        for num_heads in (16, 64, 128):
            for page_size in (32, 64):
                for q_len in (1, 4):
                    for return_lse in (False, True):
                        with self.subTest(
                            num_heads=num_heads,
                            page_size=page_size,
                            q_len=q_len,
                            return_lse=return_lse,
                        ):
                            query = torch.zeros(
                                (2, q_len, num_heads, 576), dtype=torch.bfloat16
                            )
                            kw = self._invoke_decode(
                                backend,
                                query=query,
                                sm100_supported=True,
                                page_size=page_size,
                                return_lse=return_lse,
                            )
                            self.assertEqual(kw.get("backend"), "trtllm-gen")
                            self.assertIs(
                                kw.get("multi_ctas_kv_counter_buffer"),
                                backend._multi_ctas_kv_counter_buffer,
                            )
                            self.assertEqual(kw["seq_lens"].dtype, torch.int32)
                            self.assertEqual(kw["return_lse"], return_lse)

    def test_sm100_three_dim_query_passes_explicit_trtllm_gen(self):
        backend = self._make_backend("trtllm-gen")
        query = torch.zeros((2, 128, 576), dtype=torch.bfloat16)
        kw = self._invoke_decode(backend, query=query, sm100_supported=True)
        self.assertEqual(kw.get("backend"), "trtllm-gen")
        self.assertIs(
            kw.get("multi_ctas_kv_counter_buffer"),
            backend._multi_ctas_kv_counter_buffer,
        )

    def test_sm100_head_count_hole_falls_back_to_auto_without_counter_buffer(self):
        backend = self._make_backend("trtllm-gen")
        for num_heads in (65, 96, 127):
            with self.subTest(num_heads=num_heads):
                query = torch.zeros((2, 1, num_heads, 576), dtype=torch.bfloat16)
                kw = self._invoke_decode(backend, query=query, sm100_supported=True)
                self.assertNotIn("backend", kw)
                self.assertNotIn("multi_ctas_kv_counter_buffer", kw)

    def test_sm100_non_trtllm_page_size_falls_back_to_auto_without_counter_buffer(self):
        backend = self._make_backend("trtllm-gen")
        for page_size in (16, 128):
            with self.subTest(page_size=page_size):
                query = torch.zeros((2, 1, 128, 576), dtype=torch.bfloat16)
                kw = self._invoke_decode(
                    backend,
                    query=query,
                    sm100_supported=True,
                    page_size=page_size,
                )
                self.assertNotIn("backend", kw)
                self.assertNotIn("multi_ctas_kv_counter_buffer", kw)

    def test_non_sm100_falls_back_to_auto_for_sm12x_xqa_without_counter_buffer(self):
        backend = self._make_backend("trtllm-gen")
        query = torch.zeros((2, 1, 128, 576), dtype=torch.bfloat16)
        kw = self._invoke_decode(backend, query=query, sm100_supported=False)
        self.assertNotIn("backend", kw)
        self.assertNotIn("multi_ctas_kv_counter_buffer", kw)

    def test_non_trtllm_gen_backend_forwards_explicit_backend_without_counter_buffer(
        self,
    ):
        backend = self._make_backend("cute-dsl")
        for sm100_supported in (False, True):
            with self.subTest(sm100_supported=sm100_supported):
                query = torch.zeros((2, 1, 128, 576), dtype=torch.bfloat16)
                kw = self._invoke_decode(
                    backend, query=query, sm100_supported=sm100_supported
                )
                self.assertEqual(kw.get("backend"), "cute-dsl")
                self.assertNotIn("multi_ctas_kv_counter_buffer", kw)


if __name__ == "__main__":
    unittest.main()
