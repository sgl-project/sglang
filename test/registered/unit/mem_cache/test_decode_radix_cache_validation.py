"""Decode cache retains unified KV and admits only validated native profiles.

python -m pytest test/registered/unit/mem_cache/test_decode_radix_cache_validation.py -v
"""

import json
import unittest
from types import SimpleNamespace

from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.srt.environ import envs
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.kv_cache_builder import (
    swa_gates_tree_prefix_match,
    validate_decode_radix_cache,
)
from sglang.srt.runtime_context import (
    get_context,
    get_disagg,
    get_memory,
    get_parallel,
    get_spec,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_DSV4 = SimpleNamespace(is_compressed_kv_arch=True, is_hybrid_swa_compress=False)


def _validate(token_to_kv_pool):
    validate_decode_radix_cache(
        model_config=_DSV4,
        token_to_kv_pool=token_to_kv_pool,
        is_hybrid_swa=True,
        enable_hierarchical_cache=False,
    )


class TestValidateDecodeRadixCacheDSV4(CustomTestCase):
    def test_unified_kv_bf16_is_admitted(self):
        _validate(SimpleNamespace(_unified_kv=True, _unified_kv_fp8=False))

    def test_unified_kv_fp8_is_admitted(self):
        _validate(SimpleNamespace(_unified_kv=True, _unified_kv_fp8=True))

    def test_paged_layout_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unified KV layout"):
            _validate(SimpleNamespace(_unified_kv=False, _unified_kv_fp8=False))


class TestNativeDecodeCacheAdmission(CustomTestCase):
    def setUp(self):
        super().setUp()
        context = get_context().override_server_args(
            disaggregation_mode="decode",
            disaggregation_decode_enable_radix_cache=True,
            disaggregation_transfer_backend="mooncake",
            dsv4_attn_backend="flashmla",
            enable_encoder_swa_bounded_replay=False,
            enable_decoder_swa_bounded_replay=False,
            hicache_host_memory_mode="cache",
            hicache_write_policy="write_through",
            hicache_storage_prefetch_policy="wait_complete",
        )
        context.install()
        self.addCleanup(context.restore)
        storage = envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.override("")
        storage.__enter__()
        self.addCleanup(storage.__exit__, None, None, None)
        # Omit GPU allocation only; the native class and admission code are real.
        self.pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
        self.pool.__dict__.update(
            _unified_kv=False,
            device="cuda",
            kv_layout=KVLayout.V4,
            compression_ratios=[0, 2, 1],
            page_size=256,
            request_window=None,
            swa_req_ring_size=None,
            layer_router=SimpleNamespace(sub_pool=lambda _: None),
        )
        self.model = SimpleNamespace(
            is_compressed_kv_arch=True,
            is_hybrid_swa_compress=False,
            hf_config=SimpleNamespace(
                architectures=["DeepseekV4ForCausalLM"], model_type="deepseek_v41"
            ),
            hf_text_config=SimpleNamespace(dspark_block_size=5),
        )
        self.allocator = SimpleNamespace(
            page_size=256, alloc_extend_swa_tail=lambda: None
        )

    def validate(self, host=False):
        validate_decode_radix_cache(
            model_config=self.model,
            token_to_kv_pool=self.pool,
            is_hybrid_swa=True,
            enable_hierarchical_cache=host,
            token_to_kv_pool_allocator=self.allocator,
        )

    def test_l1_uses_fresh_swa_without_admitting_other_architectures(self):
        self.validate()
        self.assertFalse(swa_gates_tree_prefix_match(self.pool, self.allocator))
        with get_disagg().override(disaggregation_decode_enable_radix_cache=False):
            self.assertTrue(swa_gates_tree_prefix_match(self.pool, self.allocator))
        self.model.hf_config.architectures = ["Ernie5_5ForCausalLM"]
        with self.assertRaisesRegex(ValueError, "compressed-KV.*unified KV"):
            self.validate()

    def test_host_and_storage_require_complete_explicit_domains(self):
        domains = (
            (None, {}),
            ("file", {"file_storage_path": "/tmp/dsv41-cache-domain"}),
            (
                "mooncake",
                {
                    "protocol": "tcp",
                    "master_server_address": "127.0.0.1:59999",
                    "dsv4_cache_namespace": {
                        "checkpoint_id": "weights-v1",
                        "compute_recipe": "h20-v4",
                    },
                },
            ),
        )
        for backend, extra in domains:
            with (
                self.subTest(backend=backend),
                get_memory().override(
                    hicache_storage_backend=backend,
                    hicache_storage_backend_extra_config=json.dumps(extra),
                ),
            ):
                self.validate(host=True)
                with get_memory().override(hicache_write_policy="write_back"):
                    with self.assertRaisesRegex(ValueError, "write_through"):
                        self.validate(host=True)
        with get_memory().override(
            hicache_storage_backend="mooncake",
            hicache_storage_backend_extra_config=json.dumps({"protocol": "tcp"}),
        ):
            with self.assertRaisesRegex(ValueError, "checkpoint_id and compute_recipe"):
                self.validate(host=True)

    def test_dspark_and_parallelism_keep_the_validated_profile(self):
        with get_spec().override(
            speculative_algorithm="DSPARK", speculative_num_draft_tokens=6
        ):
            with envs.SGLANG_RAGGED_VERIFY_MODE.override("static"):
                self.validate()
                with get_spec().override(speculative_num_draft_tokens=5):
                    with self.assertRaisesRegex(ValueError, "six verify tokens"):
                        self.validate()
        with get_parallel().override(tp_size=2, attn_cp_size=2, moe_tp_size=2):
            with self.assertRaisesRegex(ValueError, "CP1/PP1"):
                self.validate()


if __name__ == "__main__":
    unittest.main()
