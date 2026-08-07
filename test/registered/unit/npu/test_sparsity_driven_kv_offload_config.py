import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.config import (
    get_sparsity_driven_kv_offload_cell_size,
    get_sparsity_driven_kv_offload_sparse_context_len,
    is_sparsity_driven_kv_offload_enabled,
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
    )


class TestSparsityDrivenKVOffloadConfig(unittest.TestCase):
    def test_glm_dsa_model_enables_sparse_kv_offload(self):
        server_args = SimpleNamespace(
            attention_backend="ascend",
            max_running_requests=8,
        )

        with (
            patch.dict(
                os.environ,
                {"SGLANG_ENABLE_SPARSITY_DRIVEN_KV_OFFLOAD": "1"},
            ),
            patch(
                "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.config.is_npu",
                return_value=True,
            ),
        ):
            model_config = _make_glm51_model_config()

            self.assertTrue(
                is_sparsity_driven_kv_offload_enabled(
                    model_config=model_config,
                    server_args=server_args,
                    use_mla_backend=True,
                )
            )
            self.assertEqual(
                get_sparsity_driven_kv_offload_sparse_context_len(
                    model_config=model_config
                ),
                1536,
            )
            self.assertEqual(
                get_sparsity_driven_kv_offload_cell_size(
                    model_config=model_config,
                    server_args=server_args,
                    use_mla_backend=True,
                    num_layers=2,
                    element_size=2,
                ),
                512,
            )


if __name__ == "__main__":
    unittest.main()
