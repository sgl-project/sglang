import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class RecordingKVPool:
    calls = []

    def __init__(self, **kwargs):
        self.calls.append(kwargs)

    def get_kv_size_bytes(self):
        return 0, 0


class TestSWAKVPoolMemorySaver(CustomTestCase):
    def setUp(self):
        RecordingKVPool.calls = []

    def _build_pool(self, enable_memory_saver):
        with patch(
            "sglang.srt.mem_cache.swa_memory_pool.maybe_init_custom_mem_pool",
            return_value=(False, None, None),
        ):
            return SWAKVPool(
                size=8,
                size_swa=4,
                page_size=1,
                dtype=torch.float16,
                head_num=2,
                head_dim=4,
                swa_attention_layer_ids=[0],
                full_attention_layer_ids=[1],
                device="cpu",
                enable_memory_saver=enable_memory_saver,
                token_to_kv_pool_class=RecordingKVPool,
            )

    def test_forwards_memory_saver_setting_to_both_subpools(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                RecordingKVPool.calls = []
                self._build_pool(enabled)
                self.assertEqual(
                    [call["enable_memory_saver"] for call in RecordingKVPool.calls],
                    [enabled, enabled],
                )

    def test_defaults_memory_saver_off_for_existing_callers(self):
        with patch(
            "sglang.srt.mem_cache.swa_memory_pool.maybe_init_custom_mem_pool",
            return_value=(False, None, None),
        ):
            SWAKVPool(
                size=8,
                size_swa=4,
                page_size=1,
                dtype=torch.float16,
                head_num=2,
                head_dim=4,
                swa_attention_layer_ids=[0],
                full_attention_layer_ids=[1],
                device="cpu",
                token_to_kv_pool_class=RecordingKVPool,
            )
        self.assertEqual(
            [call["enable_memory_saver"] for call in RecordingKVPool.calls],
            [False, False],
        )

    def test_configurator_forwards_server_setting(self):
        configurator = object.__new__(KVCacheConfigurator)
        configurator.kv_cache_dtype = torch.float16
        configurator.post_capture_kv_active = False
        configurator.is_hybrid_swa_compress = False
        configurator.is_inkling_mtp_draft = False
        configurator.draft_swa_full_capacity = False
        configurator.device = "cpu"
        configurator.model_config = SimpleNamespace(
            head_dim=4,
            swa_attention_layer_ids=[0],
            full_attention_layer_ids=[1],
            get_num_kv_heads=lambda _tp_size: 2,
        )

        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                configurator.server_args = SimpleNamespace(
                    page_size=1,
                    kv_cache_dtype="auto",
                    enable_memory_saver=enabled,
                    speculative_algorithm=None,
                )
                with (
                    patch(
                        "sglang.srt.mem_cache.kv_cache_configurator.get_parallel",
                        return_value=SimpleNamespace(attn_tp_size=1),
                    ),
                    patch(
                        "sglang.srt.mem_cache.kv_cache_configurator.SWAKVPool"
                    ) as pool_cls,
                ):
                    configurator._build_hybrid_swa_kv_pool(
                        full_max_total_num_tokens=8,
                        swa_max_total_num_tokens=4,
                        mha_pool_class=RecordingKVPool,
                    )

                self.assertIs(pool_cls.call_args.kwargs["enable_memory_saver"], enabled)

    def test_ascend_configurator_forwards_server_setting(self):
        configurator = object.__new__(KVCacheConfigurator)
        configurator.kv_cache_dtype = torch.float16
        configurator.post_capture_kv_active = False
        configurator.is_hybrid_swa_compress = False
        configurator.device = "npu"
        configurator.model_config = SimpleNamespace(
            head_dim=4,
            swa_attention_layer_ids=[0],
            full_attention_layer_ids=[1],
            get_num_kv_heads=lambda _tp_size: 2,
        )
        fake_npu_pool = object()
        npu_module = ModuleType("sglang.srt.hardware_backend.npu.memory_pool_npu")
        npu_module.NPUMHATokenToKVPool = fake_npu_pool

        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                configurator.server_args = SimpleNamespace(
                    page_size=1,
                    enable_memory_saver=enabled,
                )
                with (
                    patch.dict(
                        sys.modules,
                        {"sglang.srt.hardware_backend.npu.memory_pool_npu": npu_module},
                    ),
                    patch(
                        "sglang.srt.mem_cache.kv_cache_configurator.get_parallel",
                        return_value=SimpleNamespace(attn_tp_size=1),
                    ),
                    patch(
                        "sglang.srt.mem_cache.kv_cache_configurator.SWAKVPool"
                    ) as pool_cls,
                ):
                    configurator._build_ascend_swa_kv_pool(
                        full_max_total_num_tokens=8,
                        swa_max_total_num_tokens=4,
                    )

                self.assertIs(pool_cls.call_args.kwargs["enable_memory_saver"], enabled)
                self.assertIs(
                    pool_cls.call_args.kwargs["token_to_kv_pool_class"],
                    fake_npu_pool,
                )


if __name__ == "__main__":
    unittest.main()
