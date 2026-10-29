import tempfile
import unittest
from datetime import timedelta

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")
from types import SimpleNamespace as NS

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, path):
    from sglang.test.unified_allocator_fixtures import setup_allocator_context

    setup_allocator_context()
    from sglang.srt.configs.qwen3_next import Qwen3NextConfig
    from sglang.srt.mem_cache.kv.decl import MHAGeometry, StateGeometry, kv_layer_decls
    from sglang.srt.mem_cache.kv.plan import plan_kv_storage
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
    from sglang.srt.runtime_context import get_exec, get_parallel, get_schedule

    mc = Qwen3NextConfig(full_attention_interval=4)
    lo, hi = ((0, 20), (20, 48))[rank]
    full = tuple(i for i in mc.full_attention_layer_ids if lo <= i < hi)
    state = tuple(i for i in mc.linear_layer_ids if lo <= i < hi)
    cfg = object.__new__(KVCacheConfigurator)
    cfg._unified_plan = plan_kv_storage(
        kv_layer_decls(
            full_layer_ids=full,
            full_geometry=MHAGeometry(
                head_num=mc.num_key_value_heads,
                head_dim=mc.head_dim,
                v_head_dim=mc.head_dim,
                kv_cache_dtype=torch.bfloat16,
            ),
            state_layer_ids=state,
            state_geometry=StateGeometry.of(mc.mamba2_cache_params),
        ),
        page_size=4,
    )
    cfg._unified_sized = None
    cfg.mambaish_config = mc
    cfg.is_hybrid_swa = False
    cfg.attn_dp_size = 1
    cfg.model_config = NS(context_len=8192, sliding_window_size=None)
    dist.init_process_group(
        "gloo",
        init_method="file://" + path,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    with (
        get_parallel().override(
            pp_size=2, pp_rank=rank, world_group=NS(cpu_group=dist.group.WORLD)
        ),
        get_schedule().override(
            max_running_requests=128,
            mamba_full_memory_ratio=1 / 9,
            disable_overlap_schedule=False,
        ),
        get_exec().mamba.override(enable_mamba_extra_buffer=True),
    ):
        initial = cfg._unified_config_from_budget(8 << 30)
        initial_rows = (
            cfg._unified_sized.request_rows,
            cfg._unified_sized.scratch_rows,
        )
        resized = cfg._unified_config_from_budget(
            6 << 30, cap_tokens=initial.max_total_num_tokens
        )
        record = dict(
            initial_rows=initial_rows,
            resized_rows=(
                cfg._unified_sized.request_rows,
                cfg._unified_sized.scratch_rows,
            ),
            rank=rank,
            full_layers=len(full),
            state_layers=len(state),
            initial_tokens=initial.max_total_num_tokens,
            initial_requests=initial.max_running_requests,
            microbatch=max(1, initial.max_running_requests // 2),
            resized_tokens=resized.max_total_num_tokens,
            resized_requests=resized.max_running_requests,
        )
        results = [None, None]
        dist.all_gather_object(results, record)
        for key in (
            "initial_tokens",
            "initial_requests",
            "microbatch",
            "resized_tokens",
            "resized_requests",
            "initial_rows",
            "resized_rows",
        ):
            assert results[0][key] == results[1][key], (key, results)
        assert record["initial_requests"] == 3, results
        assert record["resized_requests"] == 2, results
        assert record["initial_rows"] == record["resized_rows"], results
    dist.destroy_process_group()


class TestPipelineCapacity(CustomTestCase):
    def test_unequal_recurrent_stages_agree_before_and_after_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(worker, args=(tmp + "/rendezvous",), nprocs=2, join=True)


def _small_configurator(rank, state_counts=(6, 9)):
    from sglang.srt.mem_cache.kv.decl import MHAGeometry, StateGeometry, kv_layer_decls
    from sglang.srt.mem_cache.kv.plan import plan_kv_storage
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

    full_count = (2, 3)[rank]
    full = tuple(range(full_count))
    state = tuple(range(full_count, full_count + state_counts[rank]))
    cfg = object.__new__(KVCacheConfigurator)
    cfg._unified_plan = plan_kv_storage(
        kv_layer_decls(
            full_layer_ids=full,
            full_geometry=MHAGeometry(
                head_num=1, head_dim=8, v_head_dim=8, kv_cache_dtype=torch.float16
            ),
            state_layer_ids=state,
            state_geometry=StateGeometry(
                conv_state_shapes=((3, 8),),
                conv_dtype=torch.float16,
                temporal_state_shape=(1, 4, 8),
                temporal_dtype=torch.float32,
            ),
        ),
        page_size=4,
    )
    cfg._unified_sized = None
    cfg.mambaish_config = NS()
    cfg.is_hybrid_swa = False
    cfg.attn_dp_size = 1
    cfg.model_config = NS(context_len=4096, sliding_window_size=None)
    return cfg


def _build_storage(cfg):
    from sglang.srt.mem_cache.kv.build import UnifiedBuildContext, build_kv_storage

    return build_kv_storage(
        cfg._unified_plan,
        cfg._unified_sized,
        UnifiedBuildContext(
            device="cpu",
            start_layer=0,
            end_layer=max(i for s in cfg._unified_plan.sub_pools for i in s.layer_ids)
            + 1,
            enable_memory_saver=False,
            need_sort=False,
            head_num=1,
            head_dim=8,
            model_context_len=cfg.model_config.context_len,
            extra_max_context_len=4,
            state_cache_params=NS(layers=cfg._unified_plan.sub_pool("mamba").layer_ids),
        ),
    )


def _admission(allocator, input_tokens, max_requests):
    tree = NS(
        supports_mamba=lambda: True,
        full_evictable_size=lambda: 0,
        mamba_evictable_size=lambda: 0,
    )
    budget = allocator.create_prefill_budget(tree)
    admitted = 0
    for _ in range(max_requests):
        fits, _ = budget.check_prefill(
            extend_input_len=input_tokens,
            total_tokens=input_tokens + 1,
            max_new_tokens=1,
            input_tokens=input_tokens,
            swa_host_hit_length=0,
            chunk_limit=None,
            state_slots=1,
        )
        if not fits or not budget.state_slots_fit(1):
            break
        if not budget.can_allocate_prefill(
            paged_input=budget.ceil_paged_tokens(input_tokens),
            extend_input_len=input_tokens,
            max_new_tokens=1,
            chunk_limit=None,
            state_slots=1,
        ):
            break
        budget.reserve(input_tokens, 1, state_slots=1)
        admitted += 1
    assert admitted > 0
    slots = allocator.alloc_state_slots(admitted)
    assert slots is not None
    tokens = allocator.alloc(admitted * budget.ceil_paged_tokens(input_tokens + 1))
    assert tokens is not None
    assert allocator.verify_byte_accounting() == []
    allocator.free(tokens)
    allocator.mamba_allocator.free(slots)
    return admitted


def _resize_cpu_storage(storage, config):
    # Only CUDA physical backing is replaced; production finalization and resize
    # update the real CPU arena, views and allocation/admission policies.
    arena = storage.allocator.arena
    arena.backed_after_capture = True
    arena._vmm_owner = NS(ensure_prefix=lambda _: None)
    storage.allocator.get_kvcache().finalize_backing(config)
    storage.allocator.resize(config)


def admission_worker(rank, path):
    from sglang.srt.runtime_context import get_exec, get_parallel, get_schedule
    from sglang.test.unified_allocator_fixtures import setup_allocator_context

    setup_allocator_context()
    dist.init_process_group(
        "gloo",
        init_method="file://" + path,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        with (
            get_parallel().override(
                pp_size=2, pp_rank=rank, world_group=NS(cpu_group=dist.group.WORLD)
            ),
            get_schedule().override(
                page_size=4,
                max_running_requests=8,
                max_total_tokens=None,
                mamba_full_memory_ratio=1 / 9,
                disable_overlap_schedule=True,
            ),
            get_exec().mamba.override(enable_mamba_extra_buffer=False),
        ):
            cfg = _small_configurator(rank)
            initial = cfg._unified_config_from_budget(1 << 20)
            storage = _build_storage(cfg)
            allocator = storage.allocator
            for phase in ("initial", "resized"):
                config = initial
                if phase == "resized":
                    config = cfg._unified_config_from_budget(
                        3 << 18, cap_tokens=initial.max_total_num_tokens
                    )
                    _resize_cpu_storage(storage, config)
                    assert config.max_total_num_tokens < initial.max_total_num_tokens
                input_tokens = config.max_total_num_tokens // 3 + 8
                record = dict(
                    tokens=config.max_total_num_tokens,
                    requests=config.max_running_requests,
                    available=allocator.available_size(),
                    admitted=_admission(allocator, input_tokens, 8),
                )
                results = [None, None]
                dist.all_gather_object(results, record)
                assert results[0] == results[1], (phase, results)
                assert record["admitted"] == 2, (phase, results)
                assert record["available"] == record["tokens"], (phase, results)
    finally:
        dist.destroy_process_group()


class TestPipelineAdmission(CustomTestCase):
    def test_unequal_stages_admit_identical_batches_after_build_and_resize(self):
        """Unequal PP stages must select the same batch, including the MIN stage."""
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(admission_worker, args=(tmp + "/rendezvous",), nprocs=2, join=True)

    def test_single_stage_keeps_borrowing_and_explicit_caps_after_resize(self):
        from sglang.srt.runtime_context import get_exec, get_schedule
        from sglang.test.unified_allocator_fixtures import setup_allocator_context

        setup_allocator_context()
        for user_cap in (None, 20000, 6000):
            with (
                self.subTest(user_cap=user_cap),
                get_schedule().override(
                    page_size=4,
                    max_running_requests=8,
                    max_total_tokens=user_cap,
                    mamba_full_memory_ratio=1 / 9,
                    disable_overlap_schedule=True,
                ),
                get_exec().mamba.override(enable_mamba_extra_buffer=False),
            ):
                cfg = _small_configurator(1)
                initial = cfg._unified_config_from_budget(1 << 20)
                storage = _build_storage(cfg)
                final = cfg._unified_config_from_budget(
                    3 << 18, cap_tokens=initial.max_total_num_tokens
                )
                _resize_cpu_storage(storage, final)
                self.assertEqual(
                    storage.allocator.full_attn_allocator.max_live_tokens, user_cap
                )
                if user_cap == 6000:
                    self.assertEqual(storage.allocator.available_size(), user_cap)
                    expected_requests = 2
                else:
                    self.assertGreater(
                        storage.allocator.available_size(), final.max_total_num_tokens
                    )
                    expected_requests = 3
                input_tokens = final.max_total_num_tokens // 3 + 8
                self.assertEqual(
                    _admission(storage.allocator, input_tokens, 8), expected_requests
                )


if __name__ == "__main__":
    unittest.main()
