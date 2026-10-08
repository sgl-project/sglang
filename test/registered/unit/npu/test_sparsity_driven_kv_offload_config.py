import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.ascend.sparse_pd import is_sparse_pd_decode_enabled
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.config import (
    SparseKVOffloadMode,
    get_sparsity_driven_kv_offload_cell_size,
    get_sparsity_driven_kv_offload_device_cache_capacity,
    get_sparsity_driven_kv_offload_fixed_memory_size,
    get_sparsity_driven_kv_offload_sparse_context_len,
    resolve_sparse_kv_offload_mode,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _make_glm51_model_config():
    hf_config = SimpleNamespace(
        architectures=["GlmMoeDsaForCausalLM"],
        index_head_dim=128,
        index_topk=1536,
    )
    hf_config.get_text_config = lambda: hf_config
    return SimpleNamespace(
        hf_config=hf_config,
        index_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )


class TestSparsityDrivenKVOffloadConfig(unittest.TestCase):
    def test_device_cache_capacity_defaults_to_two_windows(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_NPU_SPARSE_KV_DEVICE_CACHE_FACTOR", None)
            self.assertEqual(
                get_sparsity_driven_kv_offload_device_cache_capacity(
                    sparse_context_len=2048, enable_lru=True
                ),
                4096,
            )

    def test_device_cache_capacity_accepts_factors_one_to_four(self):
        for factor in range(1, 5):
            with (
                self.subTest(factor=factor),
                patch.dict(
                    os.environ,
                    {"SGLANG_NPU_SPARSE_KV_DEVICE_CACHE_FACTOR": str(factor)},
                ),
            ):
                self.assertEqual(
                    get_sparsity_driven_kv_offload_device_cache_capacity(
                        sparse_context_len=2048, enable_lru=True
                    ),
                    factor * 2048,
                )

    def test_device_cache_capacity_rejects_invalid_factors(self):
        for factor in ("0", "5", "1.5", "invalid"):
            with (
                self.subTest(factor=factor),
                patch.dict(
                    os.environ,
                    {"SGLANG_NPU_SPARSE_KV_DEVICE_CACHE_FACTOR": factor},
                ),
            ):
                with self.assertRaisesRegex(
                    ValueError, "must be an integer in \\[1, 4\\]"
                ):
                    get_sparsity_driven_kv_offload_device_cache_capacity(
                        sparse_context_len=2048, enable_lru=True
                    )

    def setUp(self):
        self.model_config = _make_glm51_model_config()
        self.disagg = SimpleNamespace(
            disaggregation_mode="null",
            disaggregation_transfer_backend="ascend",
            disaggregation_decode_extra_slots=0,
        )
        for patcher in (
            patch.dict(
                os.environ,
                {
                    "SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD": "1",
                    "SGLANG_NPU_SPARSE_KV_ENABLE_LRU": "0",
                },
            ),
            patch(
                "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.config.is_npu",
                return_value=True,
            ),
            patch(
                "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.config.attention_backends",
                return_value=("ascend", "ascend"),
            ),
            patch(
                "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.config.get_schedule",
                return_value=SimpleNamespace(max_running_requests=8),
            ),
            patch(
                "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.config.get_disagg",
                return_value=self.disagg,
            ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def resolve(self):
        return resolve_sparse_kv_offload_mode(
            model_config=self.model_config,
            use_mla_backend=True,
        )

    def cell_size(self):
        return get_sparsity_driven_kv_offload_cell_size(
            model_config=self.model_config,
            use_mla_backend=True,
            num_layers=2,
            element_size=2,
        )

    def fixed_memory_size(self, request_capacity=8):
        return get_sparsity_driven_kv_offload_fixed_memory_size(
            model_config=self.model_config,
            use_mla_backend=True,
            num_layers=2,
            element_size=2,
            max_running_requests_per_worker=request_capacity,
        )

    def test_fixed_memory_size_by_role_and_device_capacity(self):
        self.model_config.hf_config.index_topk = 2048
        for role in ("null", "prefill", "decode"):
            self.disagg.disaggregation_mode = role
            for factor in range(1, 5):
                with (
                    self.subTest(role=role, factor=factor),
                    patch.dict(
                        os.environ,
                        {
                            "SGLANG_NPU_SPARSE_KV_DEVICE_CACHE_FACTOR": str(factor),
                            "SGLANG_NPU_SPARSE_KV_ENABLE_LRU": "1",
                        },
                    ),
                ):
                    expected = (
                        None
                        if role == "prefill"
                        else (8 + 1) * factor * 2048 * (512 + 64) * 2 * 2
                    )
                    self.assertEqual(self.fixed_memory_size(), expected)

    def test_fixed_memory_size_rejects_empty_worker_capacity(self):
        for capacity in (0, -1):
            with (
                self.subTest(capacity=capacity),
                self.assertRaisesRegex(
                    ValueError, "positive per-worker max_running_requests"
                ),
            ):
                self.fixed_memory_size(capacity)

    def test_decode_budget_includes_preallocated_transfer_rows(self):
        self.disagg.disaggregation_mode = "decode"
        self.disagg.disaggregation_decode_extra_slots = 4
        self.assertEqual(self.fixed_memory_size(), (8 + 4 + 1) * 1536 * 576 * 2 * 2)

    def test_dynamic_topk_with_lru_disabled(self):
        with patch.dict(
            os.environ, {"SGLANG_NPU_SPARSE_KV_DEVICE_CACHE_FACTOR": "unused"}
        ):
            for topk in (512, 1536, 2048, 3072):
                with self.subTest(topk=topk):
                    self.model_config.hf_config.index_topk = topk
                    self.assertEqual(
                        self.fixed_memory_size(), (8 + 1) * topk * 576 * 2 * 2
                    )

    def test_lru_rejects_non_2048_topk_during_pool_sizing(self):
        with patch.dict(os.environ, {"SGLANG_NPU_SPARSE_KV_ENABLE_LRU": "1"}):
            for role in ("null", "decode"):
                with self.subTest(role=role):
                    self.disagg.disaggregation_mode = role
                    with self.assertRaisesRegex(ValueError, "index_topk=2048"):
                        self.fixed_memory_size()

    def test_native_prefill_does_not_validate_unused_device_cache(self):
        self.disagg.disaggregation_mode = "prefill"
        with patch.dict(
            os.environ,
            {
                "SGLANG_NPU_SPARSE_KV_DEVICE_CACHE_FACTOR": "invalid",
                "SGLANG_NPU_SPARSE_KV_ENABLE_LRU": "1",
            },
        ):
            self.assertIsNone(self.fixed_memory_size())

    def test_mode_and_device_capacity_by_role(self):
        for role, expected_mode, expected_cell_size in (
            ("null", SparseKVOffloadMode.LOCAL_OFFLOAD, 512),
            ("prefill", SparseKVOffloadMode.PD_PREFILL_NATIVE, None),
            ("decode", SparseKVOffloadMode.PD_DECODE_OFFLOAD, 512),
        ):
            with self.subTest(role=role):
                self.disagg.disaggregation_mode = role
                mode = self.resolve()
                self.assertIs(mode, expected_mode)
                self.assertEqual(self.cell_size(), expected_cell_size)
                self.assertEqual(mode.uses_host_kv_offload, role in ("null", "decode"))
                self.assertEqual(mode.uses_pd_decode_staging, role == "decode")

        self.assertEqual(
            get_sparsity_driven_kv_offload_sparse_context_len(
                model_config=self.model_config
            ),
            1536,
        )

    def test_disabled_mode_does_not_read_runtime_configuration(self):
        with (
            patch.dict(os.environ, {"SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD": "0"}),
            patch(
                "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.config.get_disagg",
                side_effect=AssertionError("runtime config should not be read"),
            ),
        ):
            self.assertIs(self.resolve(), SparseKVOffloadMode.DISABLED)
            self.assertIsNone(self.cell_size())
            self.assertIsNone(self.fixed_memory_size())

    def test_pool_can_resolve_from_published_process_config(self):
        with (
            patch(
                "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.config.process_model_config",
                return_value=self.model_config,
            ),
            patch(
                "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.config.uses_mla_backend",
                return_value=True,
            ),
        ):
            self.assertIs(
                resolve_sparse_kv_offload_mode(),
                SparseKVOffloadMode.LOCAL_OFFLOAD,
            )

    def test_sparse_pd_connector_uses_resolved_mode(self):
        for mode in SparseKVOffloadMode:
            with (
                self.subTest(mode=mode),
                patch(
                    "sglang.srt.disaggregation.ascend.sparse_pd.resolve_sparse_kv_offload_mode",
                    return_value=mode,
                ),
            ):
                self.assertEqual(
                    is_sparse_pd_decode_enabled(object()),
                    mode is SparseKVOffloadMode.PD_DECODE_OFFLOAD,
                )

        with (
            patch(
                "sglang.srt.disaggregation.ascend.sparse_pd.get_sparse_pd_manager",
                return_value=None,
            ),
            patch(
                "sglang.srt.disaggregation.ascend.sparse_pd.resolve_sparse_kv_offload_mode",
                side_effect=AssertionError("mode should not be resolved"),
            ),
        ):
            self.assertFalse(is_sparse_pd_decode_enabled())

    def test_pd_requires_ascend_transfer_backend(self):
        self.disagg.disaggregation_transfer_backend = "mooncake"
        for role in ("prefill", "decode"):
            with self.subTest(role=role):
                self.disagg.disaggregation_mode = role
                with self.assertRaisesRegex(
                    ValueError, "disaggregation_transfer_backend='ascend'"
                ):
                    self.resolve()
                with self.assertRaisesRegex(
                    ValueError, "disaggregation_transfer_backend='ascend'"
                ):
                    self.cell_size()
                with self.assertRaisesRegex(
                    ValueError, "disaggregation_transfer_backend='ascend'"
                ):
                    self.fixed_memory_size()

    def test_split_attention_backend_rejects_sparse_kv_offload(self):
        with patch(
            "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.config.attention_backends",
            return_value=("ascend", "torch_native"),
        ):
            with self.assertRaisesRegex(ValueError, "Ascend MLA attention backend"):
                self.resolve()

    def test_missing_request_capacity_rejects_sparse_kv_offload(self):
        with patch(
            "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.config.get_schedule",
            return_value=SimpleNamespace(max_running_requests=None),
        ):
            with self.assertRaisesRegex(ValueError, "max_running_requests"):
                self.resolve()

    def test_non_npu_and_non_mla_reject_sparse_kv_offload(self):
        with patch(
            "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.config.is_npu",
            return_value=False,
        ):
            with self.assertRaisesRegex(ValueError, "NPU DSA-family MLA"):
                self.resolve()
        with self.assertRaisesRegex(ValueError, "NPU DSA-family MLA"):
            resolve_sparse_kv_offload_mode(
                model_config=self.model_config,
                use_mla_backend=False,
            )
        self.model_config.hf_config.architectures = ["LlamaForCausalLM"]
        with self.assertRaisesRegex(ValueError, "NPU DSA-family MLA"):
            self.resolve()


if __name__ == "__main__":
    unittest.main()
