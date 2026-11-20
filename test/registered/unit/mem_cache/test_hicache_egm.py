"""TP2 HiCache L2 backed by the other rank's Grace host memory."""

import ctypes
import socket
import unittest
from array import array
from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=180, stage="base-c", runner_config="4-gpu-gb300")


def _worker(rank, port, scenario, devices):
    torch.cuda.set_device(devices[0] if scenario == "same_numa" else devices[rank])
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        from sglang.srt.mem_cache.base_prefix_cache import (
            EvictParams,
            InsertParams,
            MatchPrefixParams,
        )
        from sglang.srt.mem_cache.pool_host import egm
        from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
        from sglang.srt.mem_cache.radix_cache import RadixKey
        from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
        from sglang.test.kits.unified_radix_cache.config import CacheConfig
        from sglang.test.kits.unified_radix_cache.fixtures import build_fixture

        group = SimpleNamespace(
            world_size=2, rank_in_group=rank, cpu_group=dist.group.WORLD
        )
        with patch(
            "sglang.srt.distributed.parallel_state.get_tp_group", return_value=group
        ):
            if scenario == "same_numa":
                try:
                    egm.EgmRemoteTensorAllocator()
                except ValueError as exc:
                    assert "distinct hosts or host NUMA nodes" in str(exc)
                else:
                    raise AssertionError("expected same-NUMA rejection")
                return
            if scenario == "pool_preflight":
                from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

                device = MHATokenToKVPool(
                    size=16,
                    page_size=1,
                    dtype=torch.float16,
                    head_num=2,
                    head_dim=64,
                    layer_num=2,
                    device=f"cuda:{devices[1]}",
                    enable_memory_saver=False,
                )
                try:
                    MHATokenToKVPoolHost(
                        buffer_info=device.get_device_pool_infos()[0].buffer_info,
                        host_to_device_ratio=2,
                        host_size=0,
                        page_size=1,
                        layout="layer_first",
                        allocator_type="egm_remote",
                        device_capacity_tokens=device.size,
                        start_layer=device.start_layer,
                        end_layer=device.end_layer,
                    )
                except RuntimeError as exc:
                    assert "remote EGM preflight failed" in str(exc)
                else:
                    raise AssertionError("expected synchronized pool preflight failure")
                return
            if scenario not in ("transfer", "transfer_page16"):
                pool = egm.EgmRemoteTensorAllocator()
                if scenario == "preflight":
                    target = f"cuda:{devices[1]}"
                    try:
                        pool.preflight([4096], target)
                    except RuntimeError as exc:
                        assert "preflight failed" in str(exc)
                    else:
                        raise AssertionError("expected synchronized preflight failure")
                    pool.close()
                    return
                pool.preflight([2 * 2 * 33 * 2 * 64 * 2], f"cuda:{devices[rank]}")
                if scenario == "shape":
                    try:
                        pool.allocate((2, 2, 33 + rank, 2, 64), torch.float16, "cpu")
                    except ValueError as exc:
                        assert "matching layer_first MHA" in str(exc)
                    else:
                        raise AssertionError("expected metadata mismatch")
                    pool.close()
                    return
                if scenario == "allocation":
                    override = (
                        patch.object(
                            egm,
                            "EgmAllocation",
                            side_effect=RuntimeError("injected owner failure"),
                        )
                        if rank == 0
                        else nullcontext()
                    )
                    with override:
                        try:
                            pool.allocate((2, 2, 33, 2, 64), torch.float16, "cpu")
                        except RuntimeError as exc:
                            assert "remote EGM allocation failed" in str(exc)
                        else:
                            raise AssertionError("expected peer allocation failure")
                    assert not pool._regions
                    pool.close()
                    return
                if scenario in ("import", "descriptor"):
                    from cuda.bindings import driver as drv

                    original = egm.EgmAllocation
                    owned = []

                    class TrackingAllocation(original):
                        def __init__(self, *args):
                            super().__init__(*args)
                            owned.append(self)

                        def export(self):
                            if scenario == "descriptor" and rank == 0:
                                return b"bad"
                            return super().export()

                    failure = (
                        patch.object(
                            drv,
                            "cuMemImportFromShareableHandle",
                            return_value=(drv.CUresult.CUDA_ERROR_INVALID_HANDLE,),
                        )
                        if rank == 0 and scenario == "import"
                        else nullcontext()
                    )
                    with (
                        patch.object(egm, "EgmAllocation", TrackingAllocation),
                        failure,
                    ):
                        try:
                            pool.allocate((2, 2, 33, 2, 64), torch.float16, "cpu")
                        except RuntimeError as exc:
                            assert "remote EGM import failed" in str(exc)
                        else:
                            raise AssertionError("expected peer import failure")
                    assert len(owned) == 1 and owned[0].handle is None
                    assert owned[0].address is None and not pool._regions
                    pool.close()
                    return
                if scenario in ("drain", "unmap"):
                    from cuda.bindings import driver as drv

                    tensor = pool.allocate((2, 2, 33, 2, 64), torch.float16, "cpu")
                    owner = pool._regions[0].owned
                    if rank == 0 and scenario == "drain":
                        failure = patch.object(
                            torch.cuda,
                            "synchronize",
                            side_effect=RuntimeError("injected drain"),
                        )
                    elif rank == 0:
                        failure = patch.object(
                            drv,
                            "cuMemUnmap",
                            return_value=(drv.CUresult.CUDA_ERROR_INVALID_VALUE,),
                        )
                    else:
                        failure = nullcontext()
                    with failure:
                        try:
                            pool.close()
                        except RuntimeError as exc:
                            assert f"remote EGM {scenario} failed" in str(exc)
                        else:
                            raise AssertionError(f"expected {scenario} failure")
                    assert owner.handle is not None and not pool._closed
                    del tensor
                    pool.close()
                    assert owner.handle is None and pool._closed
                    return
                raise AssertionError(scenario)

            page_size = 16 if scenario == "transfer_page16" else 1
            token_count = 32 if page_size == 16 else 8
            cache, allocator, _ = build_fixture(
                CacheConfig(
                    page_size=page_size,
                    kv_size=128 if page_size == 16 else 64,
                    num_layers=2,
                    head_num=2,
                    head_dim=64,
                )
            )
            device = allocator.get_kvcache()
            host = MHATokenToKVPoolHost(
                buffer_info=device.get_device_pool_infos()[0].buffer_info,
                host_to_device_ratio=2,
                host_size=0,
                page_size=page_size,
                layout="layer_first",
                pin_memory=True,
                allocator_type="egm_remote",
                device_capacity_tokens=device.size,
                start_layer=device.start_layer,
                end_layer=device.end_layer,
            )

        assert host.kv_buffer.is_cuda
        if scenario == "transfer":
            assert host.can_use_jit, "expected the HiCache JIT transfer path"
        else:
            host.can_use_jit = False  # Exercise the AOT fallback independently.
        key = RadixKey(array("q", range(token_count)))
        if page_size > 1:
            allocator.alloc(page_size - 1)
        src = allocator.alloc(token_count)
        slots = host.alloc(token_count)
        expected = []
        for layer, (k, v) in enumerate(zip(device.k_buffer, device.v_buffer)):
            values = torch.arange(k.numel(), device=k.device).reshape(k.shape)
            k.copy_(((values + rank * 71 + layer * 17) % 503).to(k.dtype))
            v.copy_(((values * 3 + rank * 83 + layer * 23) % 503).to(v.dtype))
            expected.append((k[src].cpu().tolist(), v[src].cpu().tolist()))
        peers = [None, None]
        dist.all_gather_object(peers, (slots.tolist(), expected))
        cache.insert(InsertParams(key=key, value=src))
        node = cache.match_prefix(MatchPrefixParams(key=key)).last_device_node
        cache.tree_core.set_hicache_enabled()
        host.backup_from_device_all_layer(device, slots.cuda(), src, "kernel")
        torch.cuda.synchronize()
        cache.tree_core.set_component_host_value_raw(node, ComponentType.FULL, slots)
        cache.tree_core.update_duplicate_tracking(node)
        cache.host_pool_group = SimpleNamespace(
            get_pool=lambda _: host, free=lambda ids, **kwargs: host.free(ids)
        )
        assert (
            cache.evict(EvictParams(num_tokens=token_count)).num_tokens_evicted
            == token_count
        )
        dist.barrier()
        owner = host.allocator._regions[0].owned
        nbytes = host.kv_buffer.numel() * host.kv_buffer.element_size()
        raw = (ctypes.c_uint8 * nbytes).from_address(owner.address)
        physical = torch.frombuffer(raw, dtype=host.kv_buffer.dtype).reshape(
            host.kv_buffer.shape
        )
        peer_slots, peer_expected = peers[1 - rank]
        for layer in range(2):
            for kind in range(2):
                want = torch.tensor(peer_expected[layer][kind], dtype=physical.dtype)
                assert torch.equal(physical[kind, layer, peer_slots], want)
        match = cache.match_prefix(MatchPrefixParams(key=key))
        assert match.host_hit_length == token_count
        receipt = cache.inc_host_lock_ref(match.best_match_node).to_dec_params()
        assert cache.evict_host(token_count) == 0
        dst = allocator.alloc(token_count)
        for k, v in zip(device.k_buffer, device.v_buffer):
            k.zero_()
            v.zero_()
        for layer in range(2):
            host.load_to_device_per_layer(device, slots.cuda(), dst, layer, "kernel")
        assert cache.evict_host(token_count) == 0
        torch.cuda.synchronize()
        for layer, (k, v) in enumerate(zip(device.k_buffer, device.v_buffer)):
            assert torch.equal(
                k[dst].cpu(), torch.tensor(expected[layer][0], dtype=k.dtype)
            )
            assert torch.equal(
                v[dst].cpu(), torch.tensor(expected[layer][1], dtype=v.dtype)
            )
        cache.dec_host_lock_ref(match.best_match_node, receipt)
        assert cache.evict_host(token_count) == token_count
        del physical, raw
        host.destroy()
        assert owner.handle is None and owner.address is None
    finally:
        dist.destroy_process_group()


class TestRemoteEgmHiCache(CustomTestCase):
    def test_two_rank_transfer_and_failures(self):
        from sglang.srt.mem_cache.pool_host.egm import _allocation_properties

        devices = None
        for first in range(torch.cuda.device_count()):
            for second in range(first + 1, torch.cuda.device_count()):
                if (
                    _allocation_properties(first)[1]
                    != _allocation_properties(second)[1]
                ):
                    devices = (first, second)
                    break
            if devices is not None:
                break
        self.assertIsNotNone(
            devices, "EGM test requires GPUs on distinct Grace NUMA nodes"
        )
        for scenario in (
            "transfer",
            "transfer_page16",
            "same_numa",
            "pool_preflight",
            "preflight",
            "shape",
            "allocation",
            "import",
            "descriptor",
            "drain",
            "unmap",
        ):
            with self.subTest(scenario=scenario):
                with socket.socket() as s:
                    s.bind(("127.0.0.1", 0))
                    port = s.getsockname()[1]
                mp.spawn(_worker, args=(port, scenario, devices), nprocs=2, join=True)

    def test_runtime_storage_attach_rejects_remote_egm(self):
        from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
            HybridCacheController,
        )
        from sglang.srt.mem_cache.pool_host.egm import EgmRemoteTensorAllocator

        controller = HybridCacheController.__new__(HybridCacheController)
        pool = SimpleNamespace(
            allocator=EgmRemoteTensorAllocator.__new__(EgmRemoteTensorAllocator)
        )
        controller.mem_pool_host = SimpleNamespace(
            entry_map={"kv": SimpleNamespace(host_pool=pool)}
        )
        with self.assertRaisesRegex(ValueError, "EGM host pools"):
            controller.attach_storage_backend("mooncake")


if __name__ == "__main__":
    unittest.main()
