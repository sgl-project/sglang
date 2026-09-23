"""CPU coverage of MTP pool construction against a shared sharded allocator."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache import page_interleave
from sglang.srt.mem_cache.allocator.page_interleave import PageInterleavePoolAllocator
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.mem_cache.page_interleave import (
    PageShardSpec,
    compute_page_shard_scratch_bytes,
    get_kv_shard_group_info,
    get_shared_kv_shard_pool,
    make_page_shard_spec,
)
from sglang.srt.mem_cache.page_interleave_pool import (
    PageInterleaveMHATokenToKVPool,
    PageInterleaveMLATokenToKVPool,
)
from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig
from sglang.srt.runtime_context import (
    get_context,
    get_memory,
    get_parallel,
    get_schedule,
    get_spec,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_draft_configurator(*, use_mla=True):
    # Keep the production pool/allocator types, but allocate only the CPU
    # allocator metadata. Pool construction itself is mocked in these tests.
    pool_class = (
        PageInterleaveMLATokenToKVPool if use_mla else PageInterleaveMHATokenToKVPool
    )
    target_pool = pool_class.__new__(pool_class)
    target_pool.shard_spec = PageShardSpec(
        shard_rank=1,
        shard_size=2,
        page_size=16,
        max_prefix_tokens=128,
        chunk_tokens=64,
    )
    target_pool.shard_group = SimpleNamespace(rank_in_group=1, world_size=2)
    target_pool.size = 128
    target_pool.page_size = 16
    target_pool.dtype = torch.bfloat16
    target_pool.kv_lora_rank = 8
    target_pool.qk_rope_head_dim = 4
    target_pool.head_num = 2
    target_pool.head_dim = 4
    target_pool.v_head_dim = 6
    allocator = PageInterleavePoolAllocator(
        size=target_pool.size,
        physical_page_size=16,
        shard_size=2,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=target_pool,
        need_sort=False,
        shard_spec=target_pool.shard_spec,
    )
    kvc = KVCacheConfigurator.__new__(KVCacheConfigurator)
    kvc.device = "cpu"
    kvc.gpu_id = 0
    kvc.is_draft_worker = True
    kvc.page_size = 16
    kvc.use_mla_backend = use_mla
    kvc.kv_cache_dtype = torch.bfloat16
    kvc.kv_cache_dtype_str = "bfloat16"
    kvc.token_to_kv_pool_allocator = allocator
    kvc.req_to_token_pool = None
    kvc.spec_algorithm = SpeculativeAlgorithm.EAGLE
    kvc.model_config = SimpleNamespace(
        num_nextn_predict_layers=1,
        context_len=100,
        kv_lora_rank=8,
        qk_rope_head_dim=4,
        head_dim=4,
        v_head_dim=6,
        get_num_kv_heads=lambda tp, dcp=1: 4 // tp,
        hf_config=SimpleNamespace(architectures=["DeepseekV3ForCausalLM"]),
    )
    kvc.layer_info = SimpleNamespace(start_layer=5, end_layer=6, num_effective_layers=1)
    kvc.spec_aux_config = SimpleNamespace(eagle_draft_num_layers=1)
    kvc.is_hybrid_swa = False
    kvc.is_hybrid_swa_compress = False
    kvc.sliding_window_size = None
    kvc.mambaish_config = None
    kvc.hybrid_gdn_config = None
    kvc.hybrid_kda_config = None
    kvc.post_capture_kv_active = False
    kvc.memory_pool_config = MemoryPoolConfig(
        max_total_num_tokens=128, max_running_requests=4
    )
    return kvc, target_pool


def _build_draft_pool(kvc, *, is_dsa_model=False, is_dsv4_model=False):
    sizes = kvc._derive_pool_sizes(config=kvc.memory_pool_config)
    return kvc._build_token_to_kv_pool(
        sizes=sizes,
        is_dsa_model=is_dsa_model,
        is_dsv4_model=is_dsv4_model,
        req_to_token_pool=None,
    )


class TestMTPKVShardConfig(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.enterContext(
            get_context().override_server_args(
                page_size=16,
                chunked_prefill_size=64,
                enable_multi_layer_eagle=False,
            )
        )
        self.enterContext(
            get_parallel().override(
                tp_size=2, attn_tp_size=2, moe_tp_size=2, attn_dcp_size=1
            )
        )

    def test_inherits_exact_target_spec_without_resolving_current_group(self):
        for use_mla in (True, False):
            with self.subTest(use_mla=use_mla):
                kvc, target = _make_draft_configurator(use_mla=use_mla)
                with patch.object(
                    page_interleave, "get_kv_shard_group", side_effect=AssertionError
                ):
                    self.assertIs(get_shared_kv_shard_pool(kvc), target)
                    self.assertEqual(get_kv_shard_group_info(kvc), (1, 2))
                    self.assertIs(make_page_shard_spec(kvc), target.shard_spec)

    def test_builds_separate_mtp_pool_with_target_layout_and_draft_layer_offsets(self):
        for use_mla in (True, False):
            with self.subTest(use_mla=use_mla):
                kvc, target = _make_draft_configurator(use_mla=use_mla)
                pool_class = type(target)
                # Patch __init__, not the class: target validation still sees
                # the real type, and construction returns a distinct instance.
                with patch.object(pool_class, "__init__", return_value=None) as init:
                    pool = _build_draft_pool(kvc)
                self.assertIsInstance(pool, pool_class)
                self.assertIsNot(pool, target)
                kwargs = init.call_args.kwargs
                self.assertEqual(kwargs["size"], target.size)
                self.assertIs(kwargs["shard_spec"], target.shard_spec)
                self.assertIs(kwargs["shard_group"], target.shard_group)
                self.assertEqual(kwargs["page_size"], 16)
                self.assertEqual(kwargs["layer_num"], 1)
                self.assertEqual(kwargs["start_layer"], 5)
                self.assertEqual(kwargs["end_layer"], 6)
                self.assertEqual(kwargs["dtype"], torch.bfloat16)
                sizes = kvc._derive_pool_sizes(config=kvc.memory_pool_config)
                self.assertIs(
                    kvc._build_token_to_kv_pool_allocator(
                        sizes=sizes,
                        token_to_kv_pool=pool,
                        is_dsv4_model=False,
                        req_to_token_pool=None,
                        token_to_kv_pool_allocator=kvc.token_to_kv_pool_allocator,
                    ),
                    kvc.token_to_kv_pool_allocator,
                )

    def test_draft_without_shared_sharded_allocator_keeps_existing_pool(self):
        for allocator in (None, object()):
            with self.subTest(allocator=allocator):
                kvc, _ = _make_draft_configurator()
                kvc.token_to_kv_pool_allocator = allocator
                self.assertIsNone(get_shared_kv_shard_pool(kvc))
                self.assertEqual(get_kv_shard_group_info(kvc), (None, 1))
                self.assertIsNone(make_page_shard_spec(kvc))
                self.assertEqual(compute_page_shard_scratch_bytes(kvc), 0)
                with patch.object(KVCacheConfigurator, "_build_mla_kv_pool") as build:
                    self.assertIs(_build_draft_pool(kvc), build.return_value)
                build.assert_called_once_with(max_total_num_tokens=128)

    def test_target_without_shared_allocator_keeps_existing_pool(self):
        kvc, _ = _make_draft_configurator()
        kvc.is_draft_worker = False
        kvc.token_to_kv_pool_allocator = None
        # Ordinary target construction does not require the global sharding
        # flag that has not yet been integrated into server arguments.
        with patch.object(KVCacheConfigurator, "_build_mla_kv_pool") as build:
            self.assertIs(_build_draft_pool(kvc), build.return_value)
        build.assert_called_once_with(max_total_num_tokens=128)

    def test_rejects_invalid_shared_target_pool_and_spec(self):
        kvc, _ = _make_draft_configurator()
        kvc.token_to_kv_pool_allocator._kvcache = object()
        with self.assertRaisesRegex(ValueError, "target's sharded KV pool"):
            get_shared_kv_shard_pool(kvc)
        mismatched_spec = PageShardSpec(
            shard_rank=0,
            shard_size=2,
            page_size=16,
            max_prefix_tokens=128,
            chunk_tokens=64,
        )
        for attribute, value in (
            ("shard_spec", mismatched_spec),
            ("shard_size", 4),
            ("page_size", 32),
            ("size", 128),
        ):
            with self.subTest(attribute=attribute):
                kvc, _ = _make_draft_configurator()
                setattr(kvc.token_to_kv_pool_allocator, attribute, value)
                with self.assertRaisesRegex(ValueError, "different shard layouts"):
                    get_shared_kv_shard_pool(kvc)

    def test_rejects_page_dtype_and_attention_mode_mismatches(self):
        for attribute, value, message in (
            ("page_size", 32, "physical page size"),
            ("kv_cache_dtype", torch.float16, "geometry and dtype"),
            ("use_mla_backend", False, "attention geometry"),
        ):
            with self.subTest(attribute=attribute):
                kvc, _ = _make_draft_configurator()
                setattr(kvc, attribute, value)
                with self.assertRaisesRegex(ValueError, message):
                    _build_draft_pool(kvc)

    def test_rejects_mla_and_mha_geometry_mismatches(self):
        for use_mla, attribute, value in (
            (True, "kv_lora_rank", 16),
            (True, "qk_rope_head_dim", 8),
            (False, "head_dim", 8),
            (False, "v_head_dim", 8),
            (False, "get_num_kv_heads", lambda tp, dcp=1: 4),
        ):
            with self.subTest(use_mla=use_mla, attribute=attribute):
                kvc, _ = _make_draft_configurator(use_mla=use_mla)
                setattr(kvc.model_config, attribute, value)
                with self.assertRaisesRegex(ValueError, "geometry and dtype"):
                    _build_draft_pool(kvc)

    def test_rejects_draft_capacity_different_from_target(self):
        for capacity in (64, 256):
            with self.subTest(capacity=capacity):
                kvc, _ = _make_draft_configurator()
                kvc.memory_pool_config = MemoryPoolConfig(
                    max_total_num_tokens=capacity, max_running_requests=4
                )
                with self.assertRaisesRegex(ValueError, "same per-rank capacity"):
                    _build_draft_pool(kvc)

    def test_rejects_other_draft_algorithms_and_multi_runner_mode(self):
        for algorithm in (
            SpeculativeAlgorithm.EAGLE3,
            SpeculativeAlgorithm.DFLASH,
            SpeculativeAlgorithm.FROZEN_KV_MTP,
        ):
            with self.subTest(algorithm=algorithm):
                kvc, _ = _make_draft_configurator()
                kvc.spec_algorithm = algorithm
                with self.assertRaisesRegex(ValueError, "EAGLE MTP"):
                    _build_draft_pool(kvc)
        kvc, _ = _make_draft_configurator()
        with get_spec().override(enable_multi_layer_eagle=True):
            with self.assertRaisesRegex(ValueError, "EAGLE MTP"):
                _build_draft_pool(kvc)
        for layers in (None, 0):
            with self.subTest(layers=layers):
                kvc.model_config.num_nextn_predict_layers = layers
                with self.assertRaisesRegex(ValueError, "EAGLE MTP"):
                    _build_draft_pool(kvc)

    def test_rejects_unsupported_model_pool_layouts(self):
        for options in ({"is_dsa_model": True}, {"is_dsv4_model": True}):
            with self.subTest(options=options):
                kvc, _ = _make_draft_configurator()
                with self.assertRaisesRegex(ValueError, "dense MLA and MHA"):
                    _build_draft_pool(kvc, **options)
        for attribute, value in (
            ("is_hybrid_swa", True),
            ("mambaish_config", object()),
            ("sliding_window_size", 128),
        ):
            with self.subTest(attribute=attribute):
                kvc, _ = _make_draft_configurator()
                setattr(kvc, attribute, value)
                with self.assertRaisesRegex(ValueError, "dense MLA and MHA"):
                    _build_draft_pool(kvc)

    def test_rejects_dcp(self):
        kvc, _ = _make_draft_configurator()
        with get_parallel().override(attn_dcp_size=2):
            with self.assertRaisesRegex(ValueError, "incompatible with DCP"):
                _build_draft_pool(kvc)

    def test_rejects_special_kv_storage_layouts(self):
        for attribute, value in (
            ("post_capture_kv_active", True),
            ("kv_cache_dtype_str", "mxfp8"),
        ):
            with self.subTest(attribute=attribute):
                kvc, _ = _make_draft_configurator()
                setattr(kvc, attribute, value)
                with self.assertRaisesRegex(ValueError, "plain per-layer KV layout"):
                    _build_draft_pool(kvc)
        kvc, _ = _make_draft_configurator()
        with get_memory().override(enable_page_major_kv_layout=True):
            with self.assertRaisesRegex(ValueError, "plain per-layer KV layout"):
                _build_draft_pool(kvc)

    def test_draft_scratch_uses_inherited_bounds_and_has_one_buffer_pair(self):
        for use_mla in (True, False):
            with self.subTest(use_mla=use_mla):
                kvc, _ = _make_draft_configurator(use_mla=use_mla)
                kvc.model_config.context_len = 9999
                row_bytes = (8 + 4) * 2 if use_mla else 2 * (4 + 6) * 2
                expected = 2 * (128 + 64 + 16) * row_bytes
                with get_schedule().override(chunked_prefill_size=1024):
                    self.assertEqual(compute_page_shard_scratch_bytes(kvc), expected)
                    with self.assertRaisesRegex(ValueError, "target"):
                        compute_page_shard_scratch_bytes(kvc, include_mtp=True)

    def test_target_plus_mtp_scratch_preserves_eight_context_budget(self):
        kvc, target = _make_draft_configurator()
        kvc.is_draft_worker = False
        kvc.token_to_kv_pool_allocator = None
        # Some architectures populate num_nextn_predict_layers only in their
        # draft config; target sizing uses the resolved draft layer count.
        kvc.model_config.num_nextn_predict_layers = None
        kvc.layer_info = SimpleNamespace(
            start_layer=0, end_layer=8, num_effective_layers=8
        )
        parallel = SimpleNamespace(
            enable_kv_cache_sharding=True,
            attn_cp_group=SimpleNamespace(world_size=1),
            attn_tp_group=target.shard_group,
            attn_tp_size=2,
        )
        # 100 tokens rounds to 128 per context before multiplying by eight.
        single_pool_scratch = 2 * (8 * 128 + 64 + 16) * (8 + 4) * 2
        with patch.object(page_interleave, "get_parallel", return_value=parallel):
            for layers in (1, 3):
                with self.subTest(layers=layers):
                    kvc.spec_aux_config.eagle_draft_num_layers = layers
                    self.assertEqual(
                        compute_page_shard_scratch_bytes(kvc), single_pool_scratch
                    )
                    self.assertEqual(
                        compute_page_shard_scratch_bytes(kvc, include_mtp=True),
                        2 * single_pool_scratch,
                    )
            kvc.spec_aux_config.eagle_draft_num_layers = None
            with self.assertRaisesRegex(ValueError, "EAGLE MTP"):
                compute_page_shard_scratch_bytes(kvc, include_mtp=True)


if __name__ == "__main__":
    unittest.main()
