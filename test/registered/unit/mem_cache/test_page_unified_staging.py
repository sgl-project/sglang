"""Page-unified transfer scratch must survive final KV sizing and host assembly."""

import unittest
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt import runtime_context
from sglang.srt.mem_cache import kv_cache_builder
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool
from sglang.srt.mem_cache.pool_host import base, common, mha, mla
from sglang.srt.mem_cache.pool_host.page_unified import PageUnifiedLayout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PAGE_SIZE = 16


def _device_pool(is_mla, *, layer_num=4, dtype=torch.bfloat16):
    kwargs = dict(
        size=PAGE_SIZE * 8,
        page_size=PAGE_SIZE,
        dtype=dtype,
        layer_num=layer_num,
        device="cpu",
        enable_memory_saver=False,
    )
    if is_mla:
        return MLATokenToKVPool(kv_lora_rank=16, qk_rope_head_dim=16, **kwargs)
    return MHATokenToKVPool(head_num=4, head_dim=16, enable_alt_stream=False, **kwargs)


def _geometry(is_mla, *, layer_num=4):
    return PageUnifiedLayout(
        page_size=PAGE_SIZE,
        layer_num=layer_num,
        head_num=1 if is_mla else 4,
        head_group_num=1 if is_mla else 2,
        head_dim=32 if is_mla else 16,
        itemsize=torch.bfloat16.itemsize,
        is_mla=is_mla,
    )


def _worker(pool, drafts=()):
    return SimpleNamespace(
        model_runner=SimpleNamespace(
            token_to_kv_pool=pool, mtp_draft_device_pools=drafts
        )
    )


@contextmanager
def _cpu_staging_environment(*, dcp_size=1, **memory_overrides):
    memory = SimpleNamespace(
        hicache_mem_layout="page_unified",
        hicache_ratio=2.0,
        hicache_size=0,
        enable_hierarchical_cache=True,
        hicache_storage_key_scheme="unified",
        hicache_storage_head_partition=2,
    )
    vars(memory).update(memory_overrides)
    platform = SimpleNamespace(
        capabilities=SimpleNamespace(hicache_device_kernels=True)
    )

    def allocate_host(dims, *, dtype, device, **kwargs):
        return torch.zeros(dims, dtype=dtype, device=device)

    with ExitStack() as stack:
        for module in (runtime_context, kv_cache_builder, hybrid_pool_assembler):
            stack.enter_context(
                mock.patch.object(module, "get_memory", return_value=memory)
            )
        stack.enter_context(
            mock.patch.object(
                kv_cache_builder,
                "get_schedule",
                return_value=SimpleNamespace(page_size=PAGE_SIZE),
            )
        )
        stack.enter_context(
            mock.patch.object(
                kv_cache_builder,
                "get_disagg",
                return_value=SimpleNamespace(
                    disaggregation_decode_retraction_backup=None
                ),
            )
        )
        stack.enter_context(
            mock.patch.object(kv_cache_builder, "is_npu", return_value=False)
        )
        for module in (kv_cache_builder, hybrid_pool_assembler):
            stack.enter_context(
                mock.patch.object(
                    module,
                    "get_parallel",
                    return_value=SimpleNamespace(
                        dcp_enabled=dcp_size > 1,
                        attn_dcp_size=dcp_size,
                        attn_dcp_rank=0,
                    ),
                )
            )
        stack.enter_context(
            mock.patch.object(
                hybrid_pool_assembler, "_get_allocator_type", return_value="default"
            )
        )
        stack.enter_context(mock.patch.object(base, "_is_cuda", True))
        stack.enter_context(
            mock.patch.object(
                base, "can_use_page_unified_write_back_jit_kernel", return_value=True
            )
        )
        stack.enter_context(
            mock.patch.object(base, "host_memory_budget_bytes", return_value=1 << 40)
        )
        stack.enter_context(
            mock.patch.dict(common.ALLOC_MEMORY_FUNCS, {"cpu": allocate_host})
        )
        for module in (mha, mla):
            stack.enter_context(mock.patch.object(module, "current_platform", platform))
            stack.enter_context(mock.patch.object(module, "_is_cuda", False))
            stack.enter_context(mock.patch.object(module, "_is_hip", False))
        yield memory


class TestPageUnifiedStaging(CustomTestCase):
    def test_prepared_staging_survives_kv_shrink_and_host_assembly(self):
        """Final KV sizing must not be followed by a second scratch allocation."""
        for is_mla in (False, True):
            for draft_layers in (0, 2):
                with self.subTest(is_mla=is_mla, draft_layers=draft_layers):
                    pool = _device_pool(is_mla)
                    drafts = tuple(
                        _device_pool(is_mla, layer_num=1) for _ in range(draft_layers)
                    )
                    worker = _worker(pool, drafts)
                    geometry = _geometry(is_mla, layer_num=4 + draft_layers)
                    with _cpu_staging_environment():
                        kv_cache_builder.prepare_hicache_staging(tp_worker=worker)
                        prepared = pool.hicache_write_back_staging
                        self.assertIsInstance(prepared, torch.Tensor)
                        self.assertEqual(prepared.shape, geometry.page_dims(17))
                        self.assertEqual(prepared.nbytes, 17 * geometry.bytes_per_page)

                        # The post-capture budget can reduce the device capacity.
                        pool.size //= 2
                        with mock.patch.object(
                            torch,
                            "empty",
                            side_effect=AssertionError("late staging allocation"),
                        ):
                            kv_cache_builder.prepare_hicache_staging(tp_worker=worker)
                            host = hybrid_pool_assembler.build_kv_host_pool(
                                kv_pool=pool,
                                page_size=PAGE_SIZE,
                                mtp_draft_device_pools=drafts,
                            )
                        self.assertIs(pool.hicache_write_back_staging, prepared)
                        self.assertEqual(host.staging_page_capacity, 9)
                        self.assertEqual(
                            host.staging_buffer.shape, geometry.page_dims(9)
                        )
                        self.assertEqual(
                            host.staging_buffer.data_ptr(), prepared.data_ptr()
                        )
                        self.assertEqual(
                            host.staging_buffer.untyped_storage().nbytes(),
                            prepared.nbytes,
                        )

    def test_fixed_host_size_reserves_full_transfer_chunk(self):
        for is_mla in (False, True):
            with self.subTest(is_mla=is_mla), _cpu_staging_environment(hicache_size=1):
                pool = _device_pool(is_mla)
                kv_cache_builder.prepare_hicache_staging(tp_worker=_worker(pool))
                self.assertEqual(
                    pool.hicache_write_back_staging.shape,
                    _geometry(is_mla).page_dims(64),
                )

    def test_host_capacity_override_survives_device_shrink(self):
        for is_mla in (False, True):
            with self.subTest(is_mla=is_mla), _cpu_staging_environment():
                pool = _device_pool(is_mla)
                pool.host_capacity_tokens = pool.size * 2
                worker = _worker(pool)
                kv_cache_builder.prepare_hicache_staging(tp_worker=worker)
                prepared = pool.hicache_write_back_staging
                self.assertEqual(prepared.shape, _geometry(is_mla).page_dims(33))
                pool.size //= 2
                with mock.patch.object(
                    torch,
                    "empty",
                    side_effect=AssertionError("late staging allocation"),
                ):
                    kv_cache_builder.prepare_hicache_staging(tp_worker=worker)
                    host = hybrid_pool_assembler.build_kv_host_pool(
                        kv_pool=pool, page_size=PAGE_SIZE
                    )
                self.assertEqual(host.staging_page_capacity, 33)
                self.assertEqual(host.staging_buffer.data_ptr(), prepared.data_ptr())

    def test_fp8_staging_uses_physical_storage_dtype(self):
        for is_mla in (False, True):
            with self.subTest(is_mla=is_mla), _cpu_staging_environment():
                pool = _device_pool(is_mla, dtype=torch.float8_e4m3fn)
                kv_cache_builder.prepare_hicache_staging(tp_worker=_worker(pool))
                prepared = pool.hicache_write_back_staging
                self.assertEqual(prepared.dtype, torch.uint8)
                with mock.patch.object(
                    torch,
                    "empty",
                    side_effect=AssertionError("late staging allocation"),
                ):
                    host = hybrid_pool_assembler.build_kv_host_pool(
                        kv_pool=pool, page_size=PAGE_SIZE
                    )
                self.assertEqual(host.staging_buffer.data_ptr(), prepared.data_ptr())
                self.assertEqual(
                    prepared.nbytes, host.page_unified_layout.bytes_per_page * 17
                )

    def test_dcp_reuses_physical_page_staging(self):
        pool = _device_pool(is_mla=True)
        worker = _worker(pool)
        worker.model_runner.token_to_kv_pool_allocator = SimpleNamespace(
            page_size=PAGE_SIZE * 2
        )
        with _cpu_staging_environment(dcp_size=2):
            kv_cache_builder.prepare_hicache_staging(tp_worker=worker)
            prepared = pool.hicache_write_back_staging
            self.assertEqual(prepared.shape, _geometry(is_mla=True).page_dims(17))
            with mock.patch.object(
                torch, "empty", side_effect=AssertionError("late staging allocation")
            ):
                host = hybrid_pool_assembler.build_kv_host_pool(
                    kv_pool=pool, page_size=PAGE_SIZE * 2
                )
            self.assertEqual(host.page_size, PAGE_SIZE)
            self.assertEqual(host.staging_buffer.data_ptr(), prepared.data_ptr())

    def test_disabled_hicache_or_other_layout_does_not_allocate(self):
        for is_mla in (False, True):
            for memory in (
                dict(enable_hierarchical_cache=False),
                dict(hicache_mem_layout="layer_first"),
            ):
                with self.subTest(is_mla=is_mla, memory=memory):
                    pool = _device_pool(is_mla)
                    with (
                        _cpu_staging_environment(**memory),
                        mock.patch.object(
                            torch,
                            "empty",
                            side_effect=AssertionError("unexpected staging allocation"),
                        ),
                    ):
                        kv_cache_builder.prepare_hicache_staging(
                            tp_worker=_worker(pool)
                        )
                    self.assertIsNone(pool.hicache_write_back_staging)

    def test_incompatible_prepared_staging_is_rejected(self):
        for is_mla in (False, True):
            geometry = _geometry(is_mla)
            shape = geometry.page_dims(4)
            contiguous = torch.empty(shape, dtype=torch.bfloat16)
            candidates = {
                "dtype": contiguous.float(),
                "capacity": contiguous[:2],
                "shape": torch.empty(
                    geometry.page_dims(4)[:-1] + (64,), dtype=torch.bfloat16
                ),
                "strides": torch.empty(shape + (2,), dtype=torch.bfloat16)[..., 0],
                "device": torch.empty(shape, dtype=torch.bfloat16, device="meta"),
                "old split staging": (contiguous, contiguous),
            }
            for reason, staging in candidates.items():
                with (
                    self.subTest(is_mla=is_mla, reason=reason),
                    _cpu_staging_environment(),
                ):
                    with (
                        mock.patch.object(
                            torch,
                            "empty",
                            side_effect=AssertionError("replacement allocation"),
                        ),
                        self.assertRaisesRegex(ValueError, "geometry changed"),
                    ):
                        base.prepare_page_unified_write_back_staging(
                            geometry,
                            dtype=torch.bfloat16,
                            device=torch.device("cpu"),
                            page_capacity=4,
                            staging=staging,
                        )


if __name__ == "__main__":
    unittest.main()
