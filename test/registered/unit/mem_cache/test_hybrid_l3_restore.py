"""Compose checkpoint donation, file L3, radix publication and actual GPU copies.

The small Mamba fixture exercises the shared state/tree protocol with widened
128/512-token pages. Model-level MLA/DCP kernels are qualified separately.
No final publication callback or host/device copy is replaced in this test.
"""

import tempfile
import threading
import time
import unittest
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from parameterized import parameterized
from test_unified_radix_cache_unittest import (
    CacheConfig,
    UnifiedRadixCacheSuite,
    build_fixture,
)

from sglang.srt.disaggregation.decode_hicache_mixin import (
    DecodeHiCacheTransferMixin,
    DecodePrefixMatch,
)
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestHandle,
    DecLockRefParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components.base import ComponentType
from sglang.srt.runtime_context import reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=35, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "requires real GPU host transfers")
class TestHybridL3Restore(CustomTestCase):
    # Reuse fixture mechanics, without inheriting the unrelated full tree suite.
    _init_hicache = UnifiedRadixCacheSuite._init_hicache
    _backup_node = UnifiedRadixCacheSuite._backup_node
    _path_chain = UnifiedRadixCacheSuite._path_chain
    _write_path_to_l3 = UnifiedRadixCacheSuite._write_path_to_l3
    _flush_l3_backups = UnifiedRadixCacheSuite._flush_l3_backups
    _ongoing_l3_backups = UnifiedRadixCacheSuite._ongoing_l3_backups
    _run_prefetch_to_completion = UnifiedRadixCacheSuite._run_prefetch_to_completion
    _load_back_node = UnifiedRadixCacheSuite._load_back_node
    _get_full_kv_pool = UnifiedRadixCacheSuite._get_full_kv_pool
    _fill_full_kv = UnifiedRadixCacheSuite._fill_full_kv
    _snapshot_full_kv = UnifiedRadixCacheSuite._snapshot_full_kv

    def setUp(self):
        self.addCleanup(reset_context)

    def _request(self, cache, allocator, pool, tokens, rid):
        req = Req(
            rid=rid,
            origin_input_text="",
            origin_input_ids=array("q", tokens),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=32),
        )
        pool.alloc([req])
        req.output_ids = array("q")
        req.full_untruncated_fill_ids = array("q", tokens)
        req.set_extend_range(0, len(tokens))
        req.kv.kv_committed_len = req.kv.kv_allocated_len = len(tokens)
        req.last_node = cache.root_node_handle()
        req.lock_receipt = DecLockRefParams()
        indices = allocator.alloc(len(tokens))
        self.assertIsNotNone(indices)
        pool.write((req.kv.req_pool_idx, slice(0, len(tokens))), indices)
        return req

    def _forward_snapshot(self, cache, pool, req):
        batch = ScheduleBatch(reqs=[req])
        batch.tree_cache = cache
        batch.req_to_token_pool = pool
        batch.model_config = SimpleNamespace(
            hf_text_config=SimpleNamespace(mamba_chunk_size=64)
        )
        with patch(
            "sglang.srt.managers.schedule_batch.get_parallel",
            return_value=SimpleNamespace(dcp_enabled=True),
        ):
            entry = batch._mamba_radix_cache_v2_req_prepare_for_extend(req)
        self.assertTrue(entry.track_mask)
        depth = req.kv.mamba_last_track_seqlen
        # Independent prefix-dependent byte oracle at the forward boundary.
        signature = sum(req.origin_input_ids[:depth]) % 16000
        expected = []
        for component, buffer in enumerate(
            (pool.mamba_pool.mamba_cache.temporal, *pool.mamba_pool.mamba_cache.conv)
        ):
            value = buffer[:, entry.track_index]
            words = torch.arange(value.numel(), dtype=torch.int32).reshape(value.shape)
            value.copy_(words + signature + component * 31)
            expected.append(value.clone())
        return entry.track_index, depth, expected

    def _fixture(self, directory, tree_backend="python"):
        with patch(
            "test_unified_radix_cache_unittest._TREE_CORE_TEST_BACKEND", tree_backend
        ):
            cache, allocator, pool = build_fixture(self.cfg, mamba_cache_chunk_size=64)
        self._init_hicache(
            cache,
            storage_backend="file",
            storage_dir=directory,
            prefetch_threshold=1,
            storage_extra={"enable_metadata_cache": False},
        )
        return cache, allocator, pool

    def test_chunk_checkpoint_reaches_storage_without_a_later_snapshot(self):
        """A short final chunk cannot re-donate the preceding chunk's state."""
        for backend in ("python", "rust"):
            with (
                self.subTest(backend=backend),
                tempfile.TemporaryDirectory() as directory,
            ):
                page = 128
                self.cfg = CacheConfig(
                    page_size=page,
                    components=(ComponentType.FULL, ComponentType.MAMBA),
                    enable_mamba_extra_buffer=True,
                    num_layers=2,
                    full_attention_layer_ids=(0,),
                    kv_size=page * 16,
                    max_context_len=page * 8,
                )
                cache, allocator, pool = self._fixture(directory, backend)
                cache.write_through_threshold = 1
                tokens = list(range(4 * page + 1))
                req = self._request(cache, allocator, pool, tokens, "chunked")
                req.set_extend_range(0, 4 * page)
                _, depth, expected = self._forward_snapshot(cache, pool, req)
                self.assertEqual(depth, 4 * page)
                self._fill_full_kv(
                    allocator, pool.req_to_token[req.kv.req_pool_idx, :depth], marker=7
                )
                cache.cache_unfinished_req(req, chunked=True)
                self.assertIsNone(req.kv.mamba_last_track_seqlen)
                # No explicit backup_node/write_storage calls: drive the real
                # write-through completion path used by the scheduler.
                cache.writing_check(write_back=True)
                self._flush_l3_backups(cache)
                reader, _, _ = self._fixture(directory, backend)
                handle = CacheRequestHandle("reader", 0)
                reader.prefetch_from_storage(
                    handle,
                    reader.root_node_handle(),
                    array("q", tokens[:depth]),
                    None,
                    None,
                )
                self._run_prefetch_to_completion(reader, handle)
                match = reader.match_prefix(
                    MatchPrefixParams(key=RadixKey(array("q", tokens[:depth])))
                )
                self.assertEqual(match.host_hit_length, depth)
                slot = reader.tree_core.get_component_host_value(
                    match.last_host_node, ComponentType.MAMBA
                )[0]
                for actual, wanted in zip(
                    reader.host_pool_group.get_pool(
                        PoolName.MAMBA
                    ).get_hybrid_pool_buffer(),
                    expected,
                ):
                    torch.testing.assert_close(
                        actual[slot, :, 0], wanted.cpu(), rtol=0, atol=0
                    )
                reader.cache_controller._stop_storage_threads()
                cache.cache_controller._stop_storage_threads()

    def _cancel_during_state_read(self, cache, tokens, host, available):
        backend = cache.cache_controller.storage_backend
        original = backend.batch_get_v2
        entered, release = threading.Event(), threading.Event()

        def delayed_read(transfers, *args, **kwargs):
            if any(t.name == PoolName.MAMBA for t in transfers):
                entered.set()
                if not release.wait(10):
                    raise TimeoutError("Test did not release the paused state read")
            return original(transfers, *args, **kwargs)

        handle = CacheRequestHandle("reader", 0)
        with patch.object(backend, "batch_get_v2", side_effect=delayed_read):
            try:
                cache.prefetch_from_storage(
                    handle, cache.root_node_handle(), tokens, None, None
                )
                deadline = time.monotonic() + 10
                while not entered.is_set() and time.monotonic() < deadline:
                    cache.check_hicache_events()
                    time.sleep(0.01)
                self.assertTrue(entered.is_set(), "State IO did not reach the pause")
                self.assertEqual(host.available_size(), available - 1)
                cache.release_aborted_request(handle)
                cache.check_hicache_events()
                self.assertEqual(
                    host.available_size(),
                    available - 1,
                    "Canceled request freed an in-flight state destination",
                )
            finally:
                release.set()
            deadline = time.monotonic() + 10
            while host.available_size() != available and time.monotonic() < deadline:
                cache.check_hicache_events()
                time.sleep(0.01)
            self.assertEqual(host.available_size(), available)
            match = cache.match_prefix(MatchPrefixParams(key=RadixKey(tokens)))
            self.assertEqual(match.host_hit_length, 0)
            self.assertEqual(len(match.device_indices), 0)
            self.assertNotIn(handle, cache.ongoing_prefetch)

    @parameterized.expand(["python", "rust"])
    def test_decode_restore_preserves_already_received_live_state(self, tree_backend):
        """An older L3 checkpoint must not overwrite P's newer live state."""
        with tempfile.TemporaryDirectory() as directory:
            page = 128
            self.cfg = CacheConfig(
                page_size=page,
                components=(ComponentType.FULL, ComponentType.MAMBA),
                enable_mamba_extra_buffer=True,
                num_layers=2,
                full_attention_layer_ids=(0,),
                kv_size=page * 16,
                max_context_len=page * 8,
            )
            writer, wa, wp = self._fixture(directory, tree_backend)
            tokens = list(range(4 * page + 1))
            req = self._request(writer, wa, wp, tokens, "writer")
            _, depth, old_state = self._forward_snapshot(writer, wp, req)
            self._fill_full_kv(
                wa, wp.req_to_token[req.kv.req_pool_idx, :depth], marker=7
            )
            writer.cache_unfinished_req(req)
            self._backup_node(writer, req.last_node)
            self._write_path_to_l3(writer, req.last_node)
            self._flush_l3_backups(writer)
            reader, ra, rp = self._fixture(directory, tree_backend)
            live = self._request(reader, ra, rp, tokens, "decode")
            reader.prefetch_from_storage(
                live.cache_request_handle,
                reader.root_node_handle(),
                array("q", tokens[:depth]),
                None,
                None,
            )
            self._run_prefetch_to_completion(reader, live.cache_request_handle)
            # The transfer boundary has already delivered the state at the
            # full prompt length. Local prefix DMA is allowed to complete later.
            buffers = (
                rp.mamba_pool.mamba_cache.temporal,
                *rp.mamba_pool.mamba_cache.conv,
            )
            for buffer in buffers:
                buffer[:, live.kv.mamba_pool_idx] = -17
            incoming = [b[:, live.kv.mamba_pool_idx].clone() for b in buffers]
            match = reader.match_prefix(
                MatchPrefixParams(key=RadixKey(array("q", tokens[:depth])))
            )
            pm = DecodePrefixMatch(
                prefix_indices=match.device_indices,
                l2_host_hit_length=depth,
                l3_storage_hit_length=0,
                last_device_node=reader.root_node_handle(),
                last_host_node=match.last_host_node,
            )
            dr = SimpleNamespace(req=live, prefix_match=pm)
            harness = SimpleNamespace(tree_cache=reader)
            self.assertTrue(
                DecodeHiCacheTransferMixin._try_hicache_queue_load_back(harness, dr)
            )
            self.assertGreaterEqual(reader.ready_to_load_host_cache(), 0)
            for ack in list(reader.cache_controller.ack_load_queue):
                ack.finish_event.synchronize()
            reader.loading_check()
            canonical = reader.tree_core.get_component_device_value(
                dr.hicache_restored_node, ComponentType.MAMBA
            )[0]
            for buffer, old, new in zip(buffers, old_state, incoming):
                torch.testing.assert_close(buffer[:, canonical], old, rtol=0, atol=0)
                torch.testing.assert_close(
                    buffer[:, live.kv.mamba_pool_idx], new, rtol=0, atol=0
                )
            reader.cache_controller._stop_storage_threads()
            writer.cache_controller._stop_storage_threads()

    def _seed_two_checkpoints(self, prod, pa, pp, *, page):
        oracles = {}
        for pages in (2, 4):
            tokens = list(range(pages * page + 31))
            req = self._request(prod, pa, pp, tokens, f"writer-{pages}")
            _, depth, state = self._forward_snapshot(prod, pp, req)
            self.assertEqual(depth, pages * page)
            indices = pp.req_to_token[req.kv.req_pool_idx, :depth]
            self._fill_full_kv(pa, indices, marker=7)
            prod.cache_unfinished_req(req)
            match = prod.match_prefix(
                MatchPrefixParams(key=RadixKey(array("q", tokens[:depth])))
            )
            leaf = match.last_device_node
            oracles[pages] = (
                state,
                self._snapshot_full_kv(pa, match.device_indices),
            )
            self._backup_node(prod, leaf)
            self._write_path_to_l3(prod, leaf)
            self._flush_l3_backups(prod)
        return oracles, leaf

    @parameterized.expand(["python", "rust"])
    def test_fresh_partial_and_duplicate_restore_preserve_checkpoint_bytes(
        self, tree_backend
    ):
        for page in (128, 512):
            for scenario in ("complete", "missing_latest_state", "duplicate", "cancel"):
                with (
                    self.subTest(page=page, scenario=scenario),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    self.cfg = CacheConfig(
                        page_size=page,
                        components=(ComponentType.FULL, ComponentType.MAMBA),
                        enable_mamba_extra_buffer=True,
                        num_layers=2,
                        full_attention_layer_ids=(0,),
                        kv_size=page * 16,
                        max_context_len=page * 8,
                    )
                    prod, pa, pp = self._fixture(directory, tree_backend)
                    oracles, leaf = self._seed_two_checkpoints(prod, pa, pp, page=page)
                    if scenario == "missing_latest_state":
                        backend = prod.cache_controller.storage_backend
                        terminal = prod.tree_core.get_hash_values(leaf)[-1]
                        path = Path(directory) / (
                            backend._get_component_key(terminal, PoolName.MAMBA)
                            + ".bin"
                        )
                        self.assertTrue(path.is_file())
                        path.unlink()
                    expected_pages = 2 if scenario == "missing_latest_state" else 4
                    expected_tokens = expected_pages * page
                    cons, ca, cp = self._fixture(directory, tree_backend)
                    host = cons.host_pool_group.get_pool(PoolName.MAMBA)
                    available = host.available_size()
                    tokens = array("q", range(4 * page))
                    # Decode advertises this probe before asynchronous IO.
                    # KV beyond the latest complete state is not restorable.
                    self.assertEqual(
                        cons.query_storage_hit_length(cons.root_node_handle(), tokens),
                        expected_tokens,
                    )
                    if scenario == "cancel":
                        self._cancel_during_state_read(cons, tokens, host, available)
                    handles = [
                        CacheRequestHandle("reader", 1 if scenario == "cancel" else 0)
                    ]
                    if scenario == "duplicate":
                        handles.append(CacheRequestHandle("reader-duplicate", 0))
                    for handle in handles:
                        cons.prefetch_from_storage(
                            handle, cons.root_node_handle(), tokens, None, None
                        )
                    for handle in handles:
                        self._run_prefetch_to_completion(cons, handle)
                    cons.check_hicache_events()
                    match = cons.match_prefix(MatchPrefixParams(key=RadixKey(tokens)))
                    self.assertEqual(len(match.device_indices), 0)
                    self.assertEqual(match.host_hit_length, expected_tokens)
                    self.assertEqual(host.available_size(), available - 1)
                    state_host = cons.tree_core.get_component_host_value(
                        match.last_host_node, ComponentType.MAMBA
                    )
                    self.assertEqual(len(state_host), 1)
                    state, (wanted_k, wanted_v) = oracles[expected_pages]
                    for buf, wanted in zip(host.get_hybrid_pool_buffer(), state):
                        torch.testing.assert_close(
                            buf[state_host[0], :, 0], wanted.cpu(), rtol=0, atol=0
                        )
                    req = Req(
                        rid="load",
                        origin_input_text="",
                        origin_input_ids=list(tokens),
                        sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
                    )
                    self._load_back_node(cons, match.last_host_node, req=req)
                    loaded = cons.match_prefix(MatchPrefixParams(key=RadixKey(tokens)))
                    self.assertEqual(len(loaded.device_indices), expected_tokens)
                    actual_k, actual_v = self._snapshot_full_kv(
                        ca, loaded.device_indices
                    )
                    torch.testing.assert_close(actual_k, wanted_k, rtol=0, atol=0)
                    torch.testing.assert_close(actual_v, wanted_v, rtol=0, atol=0)
                    canonical = cons.tree_core.get_component_device_value(
                        loaded.last_device_node, ComponentType.MAMBA
                    )
                    self.assertNotEqual(int(canonical[0]), int(req.kv.mamba_pool_idx))
                    buffers = [
                        cp.mamba_pool.mamba_cache.temporal,
                        *cp.mamba_pool.mamba_cache.conv,
                    ]
                    for buf, wanted in zip(buffers, state):
                        torch.testing.assert_close(
                            buf[:, canonical[0]], wanted, rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            buf[:, req.kv.mamba_pool_idx], wanted, rtol=0, atol=0
                        )
                        buf[:, req.kv.mamba_pool_idx] = -9
                        torch.testing.assert_close(
                            buf[:, canonical[0]], wanted, rtol=0, atol=0
                        )
                    cons.sanity_check()
                    # Drain/stop while the temporary directory still exists.
                    cons.cache_controller._stop_storage_threads()
                    prod.cache_controller._stop_storage_threads()


if __name__ == "__main__":
    unittest.main()
