"""CPU unit tests for DSV4 HiSparse request-level C4 sizing."""

import json
import unittest
from dataclasses import dataclass

from sglang.srt.mem_cache.hisparse_c4_sizing import HiSparseC4Geometry
from sglang.srt.model_executor.pool_configurator import (
    DSV4PoolConfigurator,
    MemoryPoolConfig,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@dataclass(frozen=True, slots=True)
class _TextConfig:
    index_topk: int | None = 512


@dataclass(frozen=True, slots=True)
class _HFConfig:
    architectures: tuple[str, ...] = ("DeepseekV4ForCausalLM",)


@dataclass(frozen=True, slots=True)
class _ModelConfig:
    context_len: int = 8192
    compress_ratios: tuple[int, ...] = (4, 128, 4, 128)
    qk_nope_head_dim: int = 448
    qk_rope_head_dim: int = 64
    index_head_dim: int = 128
    window_size: int = 256
    hf_config: _HFConfig = _HFConfig()
    hf_text_config: _TextConfig = _TextConfig()


@dataclass(frozen=True, slots=True)
class _LayerInfo:
    start_layer: int = 0
    end_layer: int = 4


@dataclass(frozen=True, slots=True)
class _ParallelState:
    attn_dp_size: int = 1
    pp_size: int = 1


@dataclass(frozen=True, slots=True)
class _PPGroup:
    rank_in_group: int = 0


@dataclass(frozen=True, slots=True)
class _KVConfiguratorFixture:
    model_config: _ModelConfig
    server_args: ServerArgs
    layer_info: _LayerInfo
    ps: _ParallelState
    kv_cache_dtype_str: str = "auto"
    pp_group: _PPGroup = _PPGroup()
    spec_algorithm: SpeculativeAlgorithm = SpeculativeAlgorithm.NONE
    page_size: int = 256


def _make_configurator(
    *,
    enable_hisparse: bool = True,
    disaggregation_mode: str = "decode",
    max_running_requests: int = 64,
    dp_size: int = 1,
    extra_slots: int = 0,
    host_to_device_ratio: int = 15,
    model_top_k: int | None = 512,
) -> DSV4PoolConfigurator:
    server_args = ServerArgs(
        model_path="dummy",
        enable_hisparse=enable_hisparse,
        disaggregation_mode=disaggregation_mode,
        disaggregation_decode_extra_slots=extra_slots,
        max_running_requests=max_running_requests,
        page_size=256,
        swa_full_tokens_ratio=0.01,
        hisparse_config=json.dumps(
            {
                "top_k": 64,
                "device_buffer_size": 2048,
                "host_to_device_ratio": host_to_device_ratio,
            }
        ),
    )
    fixture = _KVConfiguratorFixture(
        model_config=_ModelConfig(hf_text_config=_TextConfig(model_top_k)),
        server_args=server_args,
        layer_info=_LayerInfo(),
        ps=_ParallelState(attn_dp_size=dp_size),
    )
    return DSV4PoolConfigurator(fixture)


class TestHiSparseC4Geometry(CustomTestCase):
    def test_request_capacity_includes_the_reserve_row(self):
        for max_running_requests, extra_slots, expected_slots, expected_capacity in (
            (64, 0, 65, 137280),
            (32, 64, 97, 204864),
        ):
            with self.subTest(
                max_running_requests=max_running_requests,
                extra_slots=extra_slots,
            ):
                layout = HiSparseC4Geometry.create(
                    context_len=262144,
                    c4_page_size=64,
                    device_buffer_size=2048,
                    top_k=512,
                    local_c4_layers=2,
                    pd_extra_slots=extra_slots,
                    coordinator_instances=1,
                ).finalize(max_running_requests)

                self.assertEqual(layout.req_slot_count, expected_slots)
                self.assertEqual(layout.resident_request_capacity, expected_slots)
                self.assertEqual(
                    layout.c4_device_slot_capacity,
                    expected_capacity,
                )

    def test_persistent_budget_matches_allocated_tensor_shapes(self):
        layout = HiSparseC4Geometry.create(
            context_len=1028,
            c4_page_size=64,
            device_buffer_size=128,
            top_k=32,
            local_c4_layers=2,
            pd_extra_slots=0,
            coordinator_instances=2,
        ).finalize(3)
        persistent = layout.persistent_bytes

        self.assertEqual(persistent.c4_kv, 2 * 13 * 37440)
        self.assertEqual(persistent.allocator_free_pages, 12 * 8)
        self.assertEqual(persistent.req_to_token, 4 * 1028 * 4)
        self.assertEqual(persistent.coordinator_host_pool_ptrs, 2 * 2 * 16)
        self.assertEqual(persistent.coordinator, 2 * 32548)

    def test_device_buffer_size_keeps_existing_non_aligned_behavior(self):
        layout = HiSparseC4Geometry.create(
            context_len=1028,
            c4_page_size=64,
            device_buffer_size=130,
            top_k=32,
            local_c4_layers=2,
            pd_extra_slots=0,
            coordinator_instances=1,
        ).finalize(3)

        self.assertEqual(layout.padded_per_req, 194)
        self.assertEqual(layout.c4_device_slot_capacity, 776)


class TestDSV4HiSparsePoolConfigurator(CustomTestCase):
    def test_request_level_sizing_is_scoped_to_hisparse_pd_decode(self):
        for mode, enabled, shrink_factor in (
            ("null", True, 15),
            ("prefill", True, 15),
            ("decode", False, 1),
        ):
            with self.subTest(mode=mode, enabled=enabled):
                configurator = _make_configurator(
                    enable_hisparse=enabled,
                    disaggregation_mode=mode,
                )
                config = configurator.calculate_pool_sizes(2_000_000_000, 256)
                config.max_running_requests = 64
                configurator.finalize_with_max_running_requests(config)

                self.assertIsNone(config.hisparse_c4_layout)
                self.assertEqual(
                    config.c4_max_total_num_tokens,
                    config.full_max_total_num_tokens // (4 * shrink_factor),
                )

    def test_finalized_layout_uses_per_dp_request_capacity(self):
        configurator = _make_configurator(
            max_running_requests=96,
            dp_size=3,
            extra_slots=64,
            model_top_k=777,
        )
        config = configurator.calculate_pool_sizes(20_000_000_000, 256)
        config.max_running_requests = 32

        configurator.finalize_with_max_running_requests(config)

        layout = config.hisparse_c4_layout
        self.assertEqual(layout.req_slot_count, 97)
        self.assertEqual(layout.c4_device_slot_capacity, 204864)
        self.assertEqual(layout.top_k, 777)
        self.assertEqual(config.c4_max_total_num_tokens, 204864)

    def test_fixed_hisparse_footprint_is_removed_from_token_budget(self):
        configurator = _make_configurator(max_running_requests=64)
        layout = configurator.hisparse_c4_geometry.finalize(64)
        available_bytes = 20_000_000_000

        config = configurator.calculate_pool_sizes(available_bytes, 256)

        c128_bytes = configurator._get_c128_state_fixed_bytes(64)
        expected = int(
            (available_bytes - c128_bytes - layout.persistent_bytes.total)
            / configurator.bytes_per_full_token
        )
        self.assertEqual(config.max_total_num_tokens, expected // 256 * 256)

    def test_insufficient_budget_reports_persistent_components(self):
        configurator = _make_configurator(max_running_requests=64)

        with self.assertRaisesRegex(
            RuntimeError,
            "c4_kv=.*req_to_token=.*coordinator=.*total=",
        ):
            configurator.calculate_pool_sizes(1, 256)

    def test_draft_pool_sizes_do_not_take_target_layout_ownership(self):
        from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

        config = MemoryPoolConfig(
            max_total_num_tokens=4096,
            full_max_total_num_tokens=4096,
            swa_max_total_num_tokens=256,
            c4_max_total_num_tokens=137280,
            c128_max_total_num_tokens=32,
            c4_state_pool_size=64,
            c128_state_pool_size=128,
            hisparse_c4_layout=HiSparseC4Geometry.create(
                context_len=1028,
                c4_page_size=64,
                device_buffer_size=2048,
                top_k=512,
                local_c4_layers=2,
                pd_extra_slots=0,
                coordinator_instances=1,
            ).finalize(64),
        )

        class _DraftFixture:
            is_hybrid_swa = True
            is_draft_worker = True
            loc_space_scale = 1
            model_config = _ModelConfig(
                hf_config=_HFConfig(("DeepseekV4ForCausalLMNextN",))
            )

        sizes = KVCacheConfigurator._derive_pool_sizes(_DraftFixture(), config=config)

        self.assertIsNone(sizes.hisparse_c4_layout)
        self.assertEqual(sizes.c4_max_total_num_tokens, 0)
        self.assertEqual(sizes.c128_max_total_num_tokens, 0)
        self.assertEqual(sizes.c4_state_pool_size, 0)
        self.assertEqual(sizes.c128_state_pool_size, 0)


if __name__ == "__main__":
    unittest.main()
