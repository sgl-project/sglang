"""Target-only offload policy under process-wide MTP environment flags."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload import config


class TestDraftOffloadPolicy(unittest.TestCase):
    def test_draft_bypasses_target_validation(self):
        with (
            envs.SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD.override(True),
            patch.object(
                config, "get_sparse_kv_indexer_impl", side_effect=AssertionError
            ),
        ):
            self.assertFalse(
                config.is_sparsity_driven_kv_offload_enabled(
                    model_config=None, use_mla_backend=True, is_draft_worker=True
                )
            )

    def test_draft_retains_native_cell_price(self):
        with envs.SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD.override(True):
            self.assertIsNone(
                config.get_sparsity_driven_kv_offload_cell_size(
                    model_config=None,
                    use_mla_backend=True,
                    num_layers=1,
                    element_size=2,
                    is_draft_worker=True,
                )
            )

    def test_target_offload_remains_enabled(self):
        with (
            envs.SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD.override(True),
            patch.object(config, "is_npu", return_value=True),
            patch.object(config, "is_deepseek_dsa", return_value=True),
            patch.object(config, "get_sparse_kv_indexer_impl", return_value="legacy"),
            patch.object(
                config, "attention_backends", return_value=("ascend", "ascend")
            ),
            patch.object(
                config,
                "get_schedule",
                return_value=SimpleNamespace(max_running_requests=16),
            ),
            patch.object(
                config,
                "get_spec",
                return_value=SimpleNamespace(speculative_algorithm="NONE"),
            ),
        ):
            self.assertTrue(
                config.is_sparsity_driven_kv_offload_enabled(
                    model_config=SimpleNamespace(hf_config=object()),
                    use_mla_backend=True,
                    is_draft_worker=False,
                )
            )

    def test_target_still_validates_unsupported_backend(self):
        with (
            envs.SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD.override(True),
            patch.object(config, "get_sparse_kv_indexer_impl", return_value="legacy"),
            patch.object(config, "attention_backends", return_value=("torch", "torch")),
        ):
            with self.assertRaisesRegex(ValueError, "Ascend MLA"):
                config.is_sparsity_driven_kv_offload_enabled(
                    model_config=SimpleNamespace(hf_config=object()),
                    use_mla_backend=True,
                    is_draft_worker=False,
                )


if __name__ == "__main__":
    unittest.main()
