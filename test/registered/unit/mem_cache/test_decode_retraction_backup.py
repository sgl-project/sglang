import ctypes
import unittest
from array import array
from itertools import product
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.arg_groups.hicache_hook import handle_hicache
from sglang.srt.arg_groups.pd_disaggregation_hook import handle_pd_disaggregation
from sglang.srt.disaggregation.base.conn import KVPoll, KVTransferDestination
from sglang.srt.disaggregation.decode import DecodePreallocQueue, DecodeRequest
from sglang.srt.disaggregation.fake.conn import FakeKVManager
from sglang.srt.disaggregation.utils import ReqToMetadataIdxAllocator, TransferBackend
from sglang.srt.managers.schedule_batch import Req, ReqKvInfo
from sglang.srt.managers.scheduler_components.pool_stats_observer import (
    SchedulerPoolStatsObserver,
)
from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import (
    backup_kv_cache,
    release_kv_cache,
    restore_kv_cache,
)
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.kv_cache_builder import maybe_register_hicache_draft
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_parallel
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.speculative.base_spec_worker import (
    HiCacheDraftMode,
    HiCacheDraftPlan,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class TestDecodeRetractionBackup(CustomTestCase):
    pool_size = 32
    num_tokens = 8
    dtype = torch.bfloat16
    device = "cuda"

    def test_host_receive_dense_subclass_checks_actual_buffer_geometry(self):
        class DensePool(MHATokenToKVPool):
            pass

        with patch(f"{__name__}.MHATokenToKVPool", DensePool):
            env = self._build_cache(hicache_ratio=2.0, shared_receive=True)
        group = env.cache.host_pool_group
        device, host = group.get_contiguous_buf_infos()
        self.assertEqual(device, env.target_pool.get_contiguous_buf_infos())
        self.assertEqual(device[2], host[2])

        wrong_stride = (device[0], device[1], [n + 4 for n in device[2]])
        with patch.object(
            env.target_pool, "get_contiguous_buf_infos", return_value=wrong_stride
        ):
            with self.assertRaisesRegex(ValueError, "without embedded state"):
                group.get_contiguous_buf_infos()

    def _make_pool(self, layer_num: int, *, page_size=1) -> MHATokenToKVPool:
        return MHATokenToKVPool(
            size=self.pool_size,
            page_size=page_size,
            head_num=2,
            head_dim=64,
            dtype=self.dtype,
            layer_num=layer_num,
            device=self.device,
            enable_memory_saver=False,
        )

    def _seed_pool(
        self, pool: MHATokenToKVPool, indices: torch.Tensor, base: int
    ) -> None:
        for layer_id, (key, value) in enumerate(
            zip(pool.k_buffer, pool.v_buffer, strict=True)
        ):
            pattern = torch.arange(
                key[indices].numel(), device=self.device, dtype=torch.float32
            ).reshape_as(key[indices])
            key[indices] = (pattern + base + layer_id * 100).to(self.dtype)
            value[indices] = (pattern + base + 50 + layer_id * 100).to(self.dtype)

    @staticmethod
    def _snapshot_pool(
        pool: MHATokenToKVPool, indices: torch.Tensor
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [
            (key[indices].clone(), value[indices].clone())
            for key, value in zip(pool.k_buffer, pool.v_buffer, strict=True)
        ]

    def _assert_pool_equal(
        self,
        pool: MHATokenToKVPool,
        indices: torch.Tensor,
        expected: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        for (key, value), (expected_key, expected_value) in zip(
            zip(pool.k_buffer, pool.v_buffer, strict=True), expected, strict=True
        ):
            self.assertTrue(torch.equal(key[indices], expected_key))
            self.assertTrue(torch.equal(value[indices], expected_value))

    def _build_cache(
        self,
        hicache_ratio: float,
        *,
        shared_receive=False,
        page_size=1,
        io_backend="kernel",
        draft_mode=None,
        radix_cache=False,
        host_receive_threshold=0.5,
        hybrid_swa=False,
    ):
        """Bring up a UnifiedRadixCache over fresh pools, optionally with draft KV."""
        server_args = ServerArgs(
            model_path="dummy",
            page_size=page_size,
            hicache_ratio=hicache_ratio,
            hicache_io_backend=io_backend,
            hicache_mem_layout="page_first",
            disable_radix_cache=not radix_cache,
            hicache_write_policy="write_back",
            disaggregation_mode="decode" if shared_receive else "null",
            disaggregation_decode_host_receive_threshold=(
                host_receive_threshold if shared_receive else 1.0
            ),
            disaggregation_decode_enable_radix_cache=radix_cache,
        )
        if shared_receive:
            # Exercise pool layout selection from the normal HiCache default.
            handle_pd_disaggregation(server_args)
            handle_hicache(server_args)
        set_global_server_args_for_scheduler(server_args)

        req_to_token_pool = ReqToTokenPool(
            size=2,
            max_context_len=self.pool_size,
            device=self.device,
            enable_memory_saver=False,
        )
        target_pool = (
            SWAKVPool(
                size=self.pool_size,
                size_swa=self.pool_size,
                page_size=page_size,
                dtype=self.dtype,
                head_num=2,
                head_dim=64,
                swa_attention_layer_ids=[0],
                full_attention_layer_ids=[1],
                device=self.device,
            )
            if hybrid_swa
            else self._make_pool(layer_num=2, page_size=page_size)
        )
        if draft_mode is None:
            draft_mode = (
                HiCacheDraftMode.NONE if shared_receive else HiCacheDraftMode.SIDECAR
            )
        draft_pool = (
            self._make_pool(layer_num=1, page_size=page_size)
            if draft_mode != HiCacheDraftMode.NONE
            else None
        )
        allocator_cls = (
            PagedTokenToKVPoolAllocator if page_size > 1 else TokenToKVPoolAllocator
        )
        if hybrid_swa:
            allocator_cls = SWATokenToKVPoolAllocator
        allocator = allocator_cls(
            size=self.pool_size,
            dtype=self.dtype,
            device=self.device,
            kvcache=target_pool,
            need_sort=False,
            **({"page_size": page_size} if page_size > 1 else {}),
            **({"size_swa": self.pool_size} if hybrid_swa else {}),
        )
        params = CacheInitParams(
            disable=not radix_cache,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=page_size,
            is_eagle=draft_pool is not None,
            tree_components=(
                (ComponentType.FULL, ComponentType.SWA)
                if hybrid_swa
                else (ComponentType.FULL,)
            ),
            sliding_window_size=8 if hybrid_swa else None,
            mtp_draft_device_pools=(
                (draft_pool,) if draft_mode == HiCacheDraftMode.PACKED else ()
            ),
        )
        cache = UnifiedRadixCache(params)
        cache.init_hicache(server_args, params)
        self.addCleanup(cache.release_host_resources)

        if draft_mode == HiCacheDraftMode.SIDECAR:
            maybe_register_hicache_draft(
                tree_cache=cache,
                draft_plan=HiCacheDraftPlan(
                    mode=HiCacheDraftMode.SIDECAR,
                    device_pools=(draft_pool,),
                ),
            )
            self.assertIn(PoolName.DRAFT, cache.host_pool_group.entry_map)
        cache.validate_retraction_host_capacity()
        return SimpleNamespace(
            server_args=server_args,
            req_to_token_pool=req_to_token_pool,
            allocator=allocator,
            target_pool=target_pool,
            draft_pool=draft_pool,
            cache=cache,
        )

    def _admit_req(self, env, num_tokens: int):
        req = SimpleNamespace(rid="request", kv=ReqKvInfo(), seqlen=num_tokens + 1)
        self.assertIsNotNone(env.req_to_token_pool.alloc([req]))
        source_indices = env.allocator.alloc(num_tokens)
        self.assertIsNotNone(source_indices)
        env.req_to_token_pool.write(
            (req.kv.req_pool_idx, slice(0, num_tokens)), source_indices
        )
        return req, source_indices

    def test_backup_declined_when_host_pool_too_small(self):
        # A backup-only host pool is deliberately smaller than the device pool,
        # so a large enough request cannot be preserved.
        env = self._build_cache(hicache_ratio=0.1)
        self.assertLess(env.cache.host_pool_group.available_size(), self.num_tokens)

        req, source_indices = self._admit_req(env, self.num_tokens)
        host_free_before = env.cache.host_pool_group.available_size()

        self.assertIsNone(env.cache.backup_kv_cache(req))
        # The declined backup must not leak host slots.
        self.assertEqual(env.cache.host_pool_group.available_size(), host_free_before)

        # This is the signal release_req propagates so retract_decode aborts.
        self.assertFalse(
            backup_kv_cache(
                req,
                env.cache,
                env.req_to_token_pool,
                env.allocator,
                "host_pool",
            )
        )

        env.allocator.free(source_indices)
        env.req_to_token_pool.free(req)

    def test_restores_target_and_draft_kv(self):
        env = self._build_cache(hicache_ratio=1.0)
        req_to_token_pool = env.req_to_token_pool
        allocator = env.allocator
        target_pool = env.target_pool
        draft_pool = env.draft_pool
        cache = env.cache

        req, source_indices = self._admit_req(env, self.num_tokens)
        virtual_to_physical = torch.roll(
            torch.arange(self.pool_size, device=self.device),
            shifts=self.pool_size // 2,
        )
        target_pool.host_transfer_translate = lambda indices: virtual_to_physical[
            indices
        ]
        source_physical_indices = target_pool.host_transfer_translate(source_indices)
        self.assertFalse(torch.equal(source_indices, source_physical_indices))

        self._seed_pool(target_pool, source_physical_indices, base=1000)
        self._seed_pool(draft_pool, source_indices, base=3000)
        target_expected = self._snapshot_pool(target_pool, source_physical_indices)
        draft_expected = self._snapshot_pool(draft_pool, source_indices)

        host_free_before = cache.host_pool_group.available_size()
        backup = cache.backup_kv_cache(req)
        self.assertEqual(
            {transfer.name for transfer in backup.pool_transfers or []},
            {PoolName.DRAFT},
        )
        self.assertLess(cache.host_pool_group.available_size(), host_free_before)

        for buffer in (*target_pool.k_buffer, *target_pool.v_buffer):
            buffer.fill_(-1)
        for buffer in (*draft_pool.k_buffer, *draft_pool.v_buffer):
            buffer.fill_(-2)

        allocator.free(source_indices)
        blocker_indices = allocator.alloc(self.num_tokens)
        destination_indices = allocator.alloc(self.num_tokens)
        self.assertIsNotNone(blocker_indices)
        self.assertIsNotNone(destination_indices)
        self.assertFalse(torch.equal(source_indices, destination_indices))
        req_to_token_pool.write(
            (req.kv.req_pool_idx, slice(0, self.num_tokens)), destination_indices
        )
        destination_physical_indices = target_pool.host_transfer_translate(
            destination_indices
        )

        cache.restore_kv_cache(req, backup)

        self._assert_pool_equal(
            target_pool, destination_physical_indices, target_expected
        )
        self._assert_pool_equal(draft_pool, destination_indices, draft_expected)
        self.assertEqual(cache.host_pool_group.available_size(), host_free_before)

        allocator.free(blocker_indices)
        allocator.free(destination_indices)
        req_to_token_pool.free(req)

    def _receive_queue(self, env):
        hybrid_swa = isinstance(env.target_pool, SWAKVPool)
        queue = object.__new__(DecodePreallocQueue)
        queue.__dict__.update(
            tree_cache=env.cache,
            token_to_kv_pool=env.target_pool,
            draft_token_to_kv_pool=env.draft_pool,
            token_to_kv_pool_allocator=env.allocator,
            req_to_token_pool=env.req_to_token_pool,
            req_to_metadata_buffer_idx_allocator=ReqToMetadataIdxAllocator(2),
            metadata_buffers=SimpleNamespace(get_buf_infos=lambda: ([], [], [])),
            tp_rank=0,
            pp_rank=0,
            pp_size=1,
            gloo_group=None,
            transfer_backend=TransferBackend.FAKE,
            is_mla_backend=False,
            enable_staging=False,
            transfer_queue=SimpleNamespace(queue=[], enable_staging=False),
            queue=[],
            pending_reqs=[],
            retracted_queue=[],
            _prefill_dp_rank_queries={},
            _num_published_destinations=0,
            num_reserved_decode_tokens=0,
            scheduler=SimpleNamespace(
                server_args=env.server_args,
                ps=SimpleNamespace(dp_rank=0, gpu_id=0),
                model_config=SimpleNamespace(
                    num_hidden_layers=env.target_pool.layer_num
                ),
                running_batch=SimpleNamespace(reqs=[]),
                waiting_queue=[],
                last_batch=None,
                sliding_window_size=8 if hybrid_swa else None,
                enable_hisparse=False,
                enable_decode_hicache=False,
                enable_lora=False,
                enable_priority_scheduling=False,
                metrics_reporter=SimpleNamespace(
                    current_scheduler_metrics_enabled=False
                ),
                tp_worker=SimpleNamespace(
                    is_hybrid_swa=hybrid_swa,
                    model_runner=SimpleNamespace(kv_cache_dtype_str="bfloat16"),
                ),
            ),
        )
        with (
            patch.object(FakeKVManager, "supports_host_destination", True),
            get_parallel().override(dp_rank=0),
        ):
            queue.kv_manager = queue._init_kv_manager()
        queue.scheduler.pool_stats_observer = SchedulerPoolStatsObserver(
            tree_cache=env.cache,
            token_to_kv_pool_allocator=env.allocator,
            req_to_token_pool=env.req_to_token_pool,
            session_controller=None,
            hisparse_coordinator=None,
            is_hybrid_swa=hybrid_swa,
            is_hybrid_ssm=False,
            enable_hisparse=False,
            full_tokens_per_layer=self.pool_size,
            swa_tokens_per_layer=self.pool_size if hybrid_swa else None,
            max_total_num_tokens=self.pool_size,
        )
        return queue, queue.kv_manager.kv_args

    @patch("torch.distributed.get_world_size", return_value=1)
    def test_host_receive_swa_tail_capacity_restore_and_release(self, _world_size):
        env = self._build_cache(
            hicache_ratio=2.0,
            shared_receive=True,
            page_size=4,
            hybrid_swa=True,
            host_receive_threshold=0,
        )
        queue, _ = self._receive_queue(env)
        group = env.cache.host_pool_group
        full_host = group.get_pool(PoolName.KV)
        swa_host = group.get_pool(PoolName.SWA)
        capacities = (full_host.available_size(), swa_host.available_size())
        req = Req(
            rid="swa-host-receive",
            origin_input_text="",
            bootstrap_host="localhost",
            origin_input_ids=array("q", [1] * 23),
            sampling_params=SamplingParams(max_new_tokens=1),
        )
        receiver = Mock()
        decode_req = DecodeRequest(req=req, kv_receiver=receiver)
        blocker = swa_host.alloc(
            swa_host.available_size() - queue.host_swa_reserved_tokens
        )
        self.assertFalse(queue._pre_alloc_host(decode_req))
        self.assertEqual(full_host.available_size(), capacities[0])
        receiver.send_metadata.assert_not_called()
        swa_host.free(blocker)
        self.assertTrue(queue._pre_alloc_host(decode_req))
        backup = req.kv.retraction_backup
        self.assertEqual(len(backup.host_indices), 24)
        self.assertEqual(len(backup.pool_transfers[0].host_indices), 12)
        self.assertEqual(len(receiver.send_metadata.call_args.args[2][0]), 3)

        # Independent SWA pressure must delay device admission even when full KV fits.
        swa_blocker = env.allocator.swa_attn_allocator.alloc(self.pool_size)
        self.assertFalse(queue.allocate_host_staged(decode_req))
        self.assertIsNone(req.kv.req_pool_idx)
        self.assertIs(req.kv.retraction_backup, backup)
        env.allocator.swa_attn_allocator.free(swa_blocker)

        expected = []
        for pool_name, host_indices, num_tokens in (
            (PoolName.KV, backup.host_indices, 23),
            (PoolName.SWA, backup.pool_transfers[0].host_indices, 11),
        ):
            entry = group.get_entry(pool_name)
            pool = entry.device_pool
            for index, (buffer, host_buffer) in enumerate(
                zip(
                    pool.k_buffer + pool.v_buffer,
                    entry.host_pool.host_kv_data_refs,
                    strict=True,
                )
            ):
                values = torch.arange(len(host_indices) * buffer[0].numel()).reshape(
                    len(host_indices), *buffer.shape[1:]
                )
                values = ((values + index * 37) % 251).to(self.dtype)
                host_buffer[host_indices] = values
                expected.append((pool_name, buffer, values[:num_tokens]))
                buffer.fill_(-1)

        self.assertTrue(queue.allocate_host_staged(decode_req))
        req.output_ids.append(99)
        env.cache.restore_kv_cache(req, backup)
        full_indices = env.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, :23
        ].long()
        swa_indices = env.allocator.translate_swa_indices_for_transfer(
            full_indices[12:]
        )
        for pool_name, buffer, values in expected:
            indices = full_indices if pool_name == PoolName.KV else swa_indices
            self.assertTrue(torch.equal(buffer[indices].cpu(), values))
        self.assertEqual(
            (full_host.available_size(), swa_host.available_size()), capacities
        )
        req.kv.retraction_backup = None
        env.allocator.free(full_indices)
        env.req_to_token_pool.free(req)

        # An abort before device admission returns both independent allocations.
        self.assertTrue(queue._pre_alloc_host(decode_req))
        env.cache.discard_kv_cache_backup(req.kv.retraction_backup)
        self.assertEqual(
            (full_host.available_size(), swa_host.available_size()), capacities
        )

    @patch("torch.distributed.get_world_size", return_value=1)
    def test_host_receive_threshold_controls_device_allocation(self, _world_size):
        """Zero always stages, one disables, and intermediate values gate on usage."""
        for threshold, used_tokens, host_staged in (
            (0.0, 0, True),
            (1.0, 16, False),
            (0.5, 15, False),
            (0.5, 16, True),
        ):
            with self.subTest(threshold=threshold, used_tokens=used_tokens):
                env = self._build_cache(
                    hicache_ratio=2.0,
                    shared_receive=True,
                    host_receive_threshold=threshold,
                )
                queue, _ = self._receive_queue(env)
                host_admissions = []
                queue.scheduler.metrics_reporter.current_scheduler_metrics_enabled = (
                    True
                )
                queue.scheduler.metrics_collector = SimpleNamespace(
                    increment_decode_host_receive_reqs=lambda: host_admissions.append(1)
                )
                pressure = env.allocator.alloc(used_tokens)
                req = Req(
                    rid="threshold",
                    origin_input_text="",
                    bootstrap_host="localhost",
                    origin_input_ids=array("q", [1]),
                    sampling_params=SamplingParams(max_new_tokens=1),
                )
                receiver = Mock(supports_host_destination=True)
                receiver.poll.return_value = KVPoll.WaitingForInput
                decode_req = DecodeRequest(req=req, kv_receiver=receiver)
                queue.queue = [decode_req]
                self.assertEqual(queue.pop_preallocated(), ([decode_req], []))
                self.assertEqual(decode_req.host_staged, host_staged)
                self.assertEqual(len(host_admissions), int(host_staged))
                self.assertEqual(req.kv.req_pool_idx is None, host_staged)
                self.assertEqual(
                    env.allocator.available_size(),
                    self.pool_size - used_tokens - (0 if host_staged else 1),
                )
                if host_staged:
                    env.cache.discard_kv_cache_backup(req.kv.retraction_backup)
                else:
                    release_kv_cache(req, env.cache, checkpoint=False)
                env.allocator.free(pressure)
                self.assertEqual(env.allocator.available_size(), self.pool_size)

    @patch("torch.distributed.get_world_size", return_value=1)
    def test_host_receive_restores_target_and_draft_kv(self, _world_size):
        """Wire-order writes must restore both pools, including packed MHA K/V.

        Sidecar KV must follow the primary host indices through restore and
        release without allocating or freeing those indices a second time.
        """
        for draft_mode, io_backend in product(
            (HiCacheDraftMode.SIDECAR, HiCacheDraftMode.PACKED),
            ("kernel", "direct"),
        ):
            with self.subTest(draft_mode=draft_mode, io_backend=io_backend):
                page_size = num_slots = 16
                num_tokens = num_slots - 1
                env = self._build_cache(
                    hicache_ratio=2.0,
                    shared_receive=True,
                    page_size=page_size,
                    io_backend=io_backend,
                    draft_mode=draft_mode,
                )
                cache = env.cache
                queue, kv_args = self._receive_queue(env)

                req = Req(
                    rid="host-receive",
                    origin_input_text="",
                    bootstrap_host="localhost",
                    origin_input_ids=array("q", [1] * num_tokens),
                    sampling_params=SamplingParams(max_new_tokens=1),
                )
                receiver = Mock(supports_host_destination=False)
                receiver.poll.return_value = KVPoll.WaitingForInput
                decode_req = DecodeRequest(req=req, kv_receiver=receiver)
                queue.queue = [decode_req]
                host_free_before = cache.host_pool_group.available_size()
                blocker = env.allocator.alloc(self.pool_size)
                self.assertEqual(queue.pop_preallocated(), ([], []))
                receiver.send_metadata.assert_not_called()
                self.assertEqual(
                    cache.host_pool_group.available_size(), host_free_before
                )
                env.allocator.free(blocker)
                pressure = env.allocator.alloc(num_slots)
                receiver.supports_host_destination = True
                self.assertEqual(queue.pop_preallocated(), ([decode_req], []))
                self.assertIsNone(req.kv.req_pool_idx)
                self.assertEqual(
                    env.allocator.available_size(), self.pool_size - num_slots
                )
                self.assertTrue(decode_req.host_staged)
                backup = req.kv.retraction_backup
                self.assertEqual(
                    cache.host_pool_group.available_size(), host_free_before - num_slots
                )
                page_indices = receiver.send_metadata.call_args.args[0]
                self.assertEqual(
                    receiver.send_metadata.call_args.kwargs["destination"],
                    KVTransferDestination.HOST,
                )

                # Emulate the transport's page writes using the advertised
                # addresses and strides, independently of host-pool ordering.
                device_buffers = [
                    buffer
                    for pool in (env.target_pool, env.draft_pool)
                    for buffer in pool.k_buffer + pool.v_buffer
                ]
                expected = []
                for index, (buffer, host_ptr, host_len, item_len) in enumerate(
                    zip(
                        device_buffers,
                        kv_args.host_kv_data_ptrs,
                        kv_args.host_kv_data_lens,
                        kv_args.host_kv_item_lens,
                        strict=True,
                    )
                ):
                    values = torch.arange(
                        num_slots * buffer[0].numel(), dtype=torch.float32
                    ).reshape(num_slots, *buffer.shape[1:])
                    values = ((values + 37 * index) % 251).to(self.dtype)
                    for page, host_page in enumerate(page_indices):
                        page_values = values[page * page_size : (page + 1) * page_size]
                        offset = int(host_page) * item_len
                        self.assertEqual(page_values.nbytes, item_len)
                        self.assertLessEqual(offset + item_len, host_len)
                        ctypes.memmove(
                            host_ptr + offset, page_values.data_ptr(), item_len
                        )
                    expected.append(values[:num_tokens])
                    buffer.fill_(-1)

                # Incoming KV and retraction must coexist even with the host pool full.
                group = cache.host_pool_group
                blockers = group.alloc(group.available_size() - num_slots)
                pending_req = Req(
                    rid="pending",
                    origin_input_text="",
                    bootstrap_host="localhost",
                    origin_input_ids=array("q", [1] * num_tokens),
                    sampling_params=SamplingParams(max_new_tokens=1),
                )
                pending = DecodeRequest(req=pending_req, kv_receiver=receiver)
                queue.queue = [pending]
                self.assertEqual(queue.pop_preallocated(), ([], []))
                self.assertEqual(queue.queue, [pending])
                self.assertIsNone(pending_req.kv.req_pool_idx)
                self.assertIsNone(pending_req.kv.retraction_backup)
                self.assertEqual(
                    env.allocator.available_size(), self.pool_size - num_slots
                )
                self.assertEqual(group.available_size(), num_slots)
                retracted, source_indices = self._admit_req(env, num_slots)
                for index, buffer in enumerate(device_buffers):
                    buffer[source_indices] = index + 100
                retraction = cache.backup_kv_cache(retracted)
                self.assertIsNotNone(retraction)
                self.assertEqual(group.available_size(), 0)
                for buffer in device_buffers:
                    buffer[source_indices] = -1
                cache.restore_kv_cache(retracted, retraction)
                for index, buffer in enumerate(device_buffers):
                    self.assertTrue(torch.all(buffer[source_indices] == index + 100))
                group.free(blockers)
                env.allocator.free(source_indices)
                env.req_to_token_pool.free(retracted)

                # Device slots are assigned only after the host transfer.
                blocker = env.allocator.alloc(num_slots)
                self.assertFalse(queue.allocate_host_staged(decode_req))
                self.assertIsNone(req.kv.req_pool_idx)
                self.assertIs(req.kv.retraction_backup, backup)
                env.allocator.free(blocker)
                self.assertTrue(queue.allocate_host_staged(decode_req))
                self.assertFalse(decode_req.host_staged)
                received_indices = env.req_to_token_pool.req_to_token[
                    req.kv.req_pool_idx, :num_tokens
                ].to(torch.int64)
                req.output_ids.append(99)
                cache.restore_kv_cache(req, backup)
                for buffer, values in zip(device_buffers, expected, strict=True):
                    self.assertTrue(
                        torch.equal(buffer[received_indices[:num_tokens]].cpu(), values)
                    )
                self.assertEqual(
                    cache.host_pool_group.available_size(), host_free_before
                )
                # Abort cleanup uses the same descriptor, including sidecars.
                self.assertEqual(queue.pop_preallocated(), ([pending], []))
                cache.discard_kv_cache_backup(pending_req.kv.retraction_backup)
                self.assertEqual(
                    cache.host_pool_group.available_size(), host_free_before
                )
                env.allocator.free(received_indices)
                env.req_to_token_pool.free(req)
                env.allocator.free(pressure)

    @patch("torch.distributed.get_world_size", return_value=1)
    def test_host_receive_merges_existing_radix_prefix_after_restore(self, _world_size):
        """Receiving a cached prefix must not lock it or leak duplicate KV."""
        env = self._build_cache(
            hicache_ratio=2.0, shared_receive=True, radix_cache=True
        )
        cache = env.cache
        queue, _ = self._receive_queue(env)

        def make_req(rid, num_tokens):
            return Req(
                rid=rid,
                origin_input_text="",
                origin_input_ids=array("q", range(num_tokens)),
                sampling_params=SamplingParams(max_new_tokens=1),
                bootstrap_host="localhost",
            )

        cached = make_req("cached", 4)
        cached.output_ids.append(99)
        cached.last_node = cache.root_node_handle()
        cached_indices = queue._pre_alloc(cached)
        self._seed_pool(env.target_pool, cached_indices, base=500)
        cached_values = self._snapshot_pool(env.target_pool, cached_indices)
        cache.checkpoint(cached, up_to=cached.extend_end)
        pressure = env.allocator.alloc(self.pool_size // 2 - 4)

        req = make_req("receiving", 8)
        receiver = Mock(supports_host_destination=True)
        receiver.poll.return_value = KVPoll.WaitingForInput
        decode_req = DecodeRequest(req=req, kv_receiver=receiver)
        queue.queue = [decode_req]
        host_free_before = cache.host_pool_group.available_size()
        self.assertEqual(queue.pop_preallocated(), ([decode_req], []))
        self.assertIsNone(req.kv.req_pool_idx)
        self.assertEqual(cache.protected_size(), 4)
        env.allocator.free(pressure)
        backup = req.kv.retraction_backup
        for buffer in queue.host_pool.host_kv_data_refs:
            buffer[backup.host_indices] = 7

        self.assertTrue(queue.allocate_host_staged(decode_req))
        received_indices = env.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, :8
        ].clone()
        req.output_ids.append(99)
        # Prebuilt preparation retains the full-transfer/root state, then
        # restores before normal cache insertion deduplicates the prefix.
        req.init_next_round_input(None)
        req.extend_end = 8
        restore_kv_cache(req, cache, env.req_to_token_pool, env.allocator, "host_pool")
        self.assertEqual(cache.host_pool_group.available_size(), host_free_before)
        free_before_insert = env.allocator.available_size()
        cache.checkpoint(req, up_to=req.extend_end)

        row = env.req_to_token_pool.req_to_token[req.kv.req_pool_idx, :8]
        self.assertTrue(torch.equal(row[:4], cached_indices))
        self.assertTrue(torch.equal(row[4:], received_indices[4:]))
        self._assert_pool_equal(env.target_pool, cached_indices, cached_values)
        self.assertEqual(env.allocator.available_size(), free_before_insert + 4)
        self.assertEqual(cache.protected_size(), 8)
        release_kv_cache(req, cache, checkpoint=False)
        self.assertEqual(cache.protected_size(), 4)
        release_kv_cache(cached, cache, checkpoint=False)
        self.assertEqual(cache.protected_size(), 0)


DCP_SIZE = 4
DCP_RANK = 1
DCP_ROWS = 8


def _bare_mla_pool() -> MLATokenToKVPool:
    pool = object.__new__(MLATokenToKVPool)
    pool.layer_num = 2
    pool.cpu_offloading_chunk_size = 3
    pool.kv_buffer = [
        (torch.arange(DCP_ROWS, dtype=torch.float32) + 100 * layer).view(DCP_ROWS, 1, 1)
        for layer in range(pool.layer_num)
    ]
    return pool


def _dcp():
    return get_parallel().override(
        dcp_enabled=True, attn_dcp_size=DCP_SIZE, attn_dcp_rank=DCP_RANK
    )


class TestDcpRetractionBackup(unittest.TestCase):
    """`req_to_token` names KV slots in the widened DCP id space while
    `kv_buffer` holds only this rank's rows; a widened id used as a row index
    reads past the buffer or copies another token's row."""

    def test_restore_lands_on_new_owned_rows(self):
        pool = _bare_mla_pool()
        before = [buf.clone() for buf in pool.kv_buffer]
        old_widened = torch.arange(0, 12, dtype=torch.int64)
        new_widened = torch.arange(12, 24, dtype=torch.int64)

        with _dcp():
            pool.load_cpu_copy(pool.get_cpu_copy(old_widened), new_widened)

        old_rows = old_widened[DCP_RANK::DCP_SIZE] // DCP_SIZE
        new_rows = new_widened[DCP_RANK::DCP_SIZE] // DCP_SIZE
        self.assertEqual(new_rows.tolist(), [3, 4, 5])
        untouched = torch.tensor(
            [r for r in range(DCP_ROWS) if r not in new_rows.tolist()]
        )
        for layer in range(pool.layer_num):
            torch.testing.assert_close(
                pool.kv_buffer[layer][new_rows], before[layer][old_rows]
            )
            torch.testing.assert_close(
                pool.kv_buffer[layer][untouched], before[layer][untouched]
            )

    def test_resolved_pool_takes_ids_as_rows(self):
        pool = _bare_mla_pool()
        pool.write_loc_is_dcp_resolved = True
        rows = torch.arange(DCP_ROWS, dtype=torch.int64)

        with _dcp():
            kv_cpu = pool.get_cpu_copy(rows)

        for layer in range(pool.layer_num):
            torch.testing.assert_close(torch.cat(kv_cpu[layer]), pool.kv_buffer[layer])


if __name__ == "__main__":
    unittest.main()
