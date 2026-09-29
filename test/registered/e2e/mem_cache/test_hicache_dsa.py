"""Weight-free byte regressions for DSA HiCache assembly and L2 transfers."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    _DsaStrategy,
    build_hicache_draft_sidecars,
)
from sglang.srt.mem_cache.l2_transfer import L2TransferEngine
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.mem_cache.pool_host.dsa import DSAIndexerPoolHost
from sglang.srt.mem_cache.unified_cache.components import CacheTransferPhase
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker, HiCacheDraftMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _pool(dtype, dim, live, compressed):
    return DSATokenToKVPool(
        size=1024,
        page_size=64,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        kv_cache_dim=dim,
        dtype=dtype,
        layer_num=len(live),
        device="cuda",
        index_head_dim=128,
        enable_memory_saver=False,
        index_kpool=4 if compressed else 1,
        index_kpool_compress=compressed,
        max_running_requests=4,
        skip_topk_layers=[not value for value in live],
    )


def _tokens(pages):
    return (pages[:, None] * 64 + torch.arange(64, device=pages.device)).flatten()


def _host_layers(host, index):
    """Independent layout views, each [physical page, raw bytes]."""
    if isinstance(host, DSAIndexerPoolHost):
        buffer = host.index_k_with_scale_buffer
    else:
        buffer = host.kv_buffer
    if host.layout == "layer_first":
        layers = list(buffer)
    else:
        layers = [buffer[:, layer] for layer in range(host.layer_num)]
    # Index rows already represent pages; KV rows represent 64 tokens per page.
    return [
        layer.view(torch.uint8).reshape(
            host.page_num if not index else layer.shape[0], -1
        )
        for layer in layers
    ]


@pytest.mark.parametrize(
    "layout,backend",
    [
        ("page_first", "kernel"),
        ("layer_first", "kernel"),
        ("layer_first", "direct"),
        ("page_first_direct", "direct"),
    ],
)
@pytest.mark.parametrize(
    "mode", ["no_draft", "packed", "sidecar_live", "sidecar_empty", "uncompressed"]
)
@torch.inference_mode()
def test_dsa_transfer_bytes(layout, backend, mode):
    """Missing/incorrect copies must disagree at the host or device boundary.

    Uses production strategy/spec discovery, allocation resolution, index
    preparation and asynchronous copies. The controller constructor/background
    scheduling are bypassed; queued operations are drained with explicit
    completion, and a controlled device allocator selects the restore pages.
    """
    publish(
        ServerArgs(
            model_path="dummy",
            enable_hierarchical_cache=True,
            hicache_ratio=2,
            hicache_mem_layout=layout,
            hicache_io_backend=backend,
        ),
        role="scheduler",
    )
    group = None
    try:
        compressed = mode != "uncompressed"
        # Empty target layers exercise compressed-layer compaction. The legacy
        # uncompressed route has no skipped index layers in this fixture.
        target = _pool(
            torch.float8_e4m3fn,
            656,
            (True, False, True) if compressed else (True, True),
            compressed,
        )
        if mode == "no_draft":
            drafts = ()
        elif mode == "packed":
            drafts = tuple(
                _pool(torch.float8_e4m3fn, 656, (True,), compressed) for _ in range(2)
            )
        else:
            drafts = (
                _pool(torch.bfloat16, 576, (mode != "sidecar_empty",), compressed),
            )
        target_runner = SimpleNamespace(
            token_to_kv_pool=target, spec_algorithm=SpeculativeAlgorithm.EAGLE
        )
        runners = tuple(
            SimpleNamespace(
                token_to_kv_pool=p,
                model_config=SimpleNamespace(
                    num_nextn_predict_layers=1,
                    hf_config=SimpleNamespace(architectures=["Glm5NextForCausalLMMTP"]),
                ),
            )
            for p in drafts
        )
        if drafts:
            plan = BaseSpecWorker._build_hicache_draft_plan(
                SimpleNamespace(
                    target_worker=SimpleNamespace(model_runner=target_runner),
                    _draft_model_runners=lambda: runners,
                )
            )
            assert plan.mode == (
                HiCacheDraftMode.PACKED
                if mode == "packed"
                else HiCacheDraftMode.SIDECAR
            )
        params = SimpleNamespace(
            mtp_draft_device_pools=getattr(target_runner, "mtp_draft_device_pools", ()),
            page_size=64,
            token_to_kv_pool_allocator=None,
            tp_cache_group=None,
            attn_cp_cache_group=None,
            attn_tp_cache_group=None,
            pp_cache_group=None,
        )
        prefix = "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler."
        with (
            patch(prefix + "_get_allocator_type", return_value="default"),
            patch(prefix + "HybridCacheController"),
        ):
            assembled = _DsaStrategy().build(
                cache=None,
                kvcache=target,
                params=params,
                server_args=None,
                load_cache_event=None,
            )
            group = assembled.host_pool_group
            controller = object.__new__(HybridCacheController)
            controller.mem_pool_host = group
            controller.page_size, controller.layer_num = 64, target.layer_num
            controller.device, controller.io_backend = "cuda", backend
            controller.write_queue, controller.load_queue = [], []
            controller.start_writing = lambda: None
            cache = SimpleNamespace(
                cache_controller=controller, sidecar_pool_specs=assembled.sidecars
            )
            if drafts and mode != "packed":
                specs, entries = build_hicache_draft_sidecars(
                    draft_device_pools=plan.device_pools, tree_cache=cache
                )
                cache.sidecar_pool_specs += specs
                for entry in entries:
                    group.add_entry(entry)

        generator = torch.Generator().manual_seed(38212)
        originals = {}
        for pool in (target, *drafts):
            for buffer in (*pool.kv_buffer, *pool.index_k_with_scale_buffer):
                if buffer.numel():
                    view = buffer.view(torch.uint8)
                    expected = torch.randint(
                        256, view.shape, dtype=torch.uint8, generator=generator
                    )
                    view.copy_(expected)
                    originals[id(buffer)] = (buffer, expected)

        # Expected host contents are assembled from device buffers, independently
        # of transfer lists and layer_mapper. An omitted spec cannot hide a pool.
        host_checks = []
        for entry in group.entries:
            index = entry.name not in (
                PoolName.KV,
                PoolName.DRAFT,
            ) and "indexer" in str(entry.name)
            pools = (entry.device_pool, *entry.packed_draft_device_pools)
            sources = [
                b
                for p in pools
                for b in (p.index_k_with_scale_buffer if index else p.kv_buffer)
                if b.numel()
            ]
            actual_layers = _host_layers(entry.host_pool, index)
            assert len(sources) == len(actual_layers)
            storage = (
                entry.host_pool.index_k_with_scale_buffer
                if isinstance(entry.host_pool, DSAIndexerPoolHost)
                else entry.host_pool.kv_buffer
            )
            for buffer in storage if isinstance(storage, list) else [storage]:
                buffer.view(torch.uint8).fill_(91)
            for layer, (actual, source) in enumerate(
                zip(actual_layers, sources, strict=True)
            ):
                expected = torch.full_like(actual, 91)
                source_bytes = originals[id(source)][1]
                source_pages = source_bytes.reshape(
                    source_bytes.shape[0] if index else source_bytes.shape[0] // 64, -1
                )
                host_checks.append(
                    (entry.host_pool, index, layer, expected, source_pages)
                )

        source_pages = torch.tensor([5, 1, 7, 3, 0, 6, 2, 4])
        dest_pages = torch.tensor([10, 14, 8, 12, 15, 9, 13, 11])
        engine = L2TransferEngine(backend)
        cached_parts = []
        # Recycle selected pages through the real allocator. Their nonmonotonic
        # order makes direct-backend sorting exercise the paired device reorder.
        assert group.alloc(group.logical_size) is not None
        group.free(_tokens(torch.tensor([9, 4, 13, 1, 8, 14, 3, 7])))
        for start in (0, 4):
            source = _tokens(source_pages[start : start + 4].cuda())
            extras = UnifiedRadixCache._build_sidecar_transfers(
                cache,
                CacheTransferPhase.BACKUP_HOST,
                PoolTransfer(PoolName.KV, device_indices=source),
                {},
            )
            cached = controller.write(source, extra_pools=extras)
            assert cached is not None
            assert not torch.equal(cached, cached.sort()[0])
            cached_parts.append(cached)
            op = controller.write_queue.pop()
            completion = engine.submit_device_to_host(
                controller._l2_transfers(*controller._move_write_operation(op))
            )
            completion.finish_event.synchronize()
            pages = cached.reshape(-1, 64)[:, 0] // 64
            for host, index, layer, expected, source_bytes in host_checks:
                expected[pages] = source_bytes[source_pages[start : start + 4]]
                assert torch.equal(_host_layers(host, index)[layer], expected), (
                    "backup payload or untouched host page changed"
                )

        for buffer, _ in originals.values():
            buffer.view(torch.uint8).fill_(173)
        cached = torch.cat(cached_parts)
        destination = _tokens(dest_pages.cuda())
        controller.mem_pool_device_allocator = SimpleNamespace(
            alloc=lambda size: destination if size == len(destination) else None
        )
        extras = UnifiedRadixCache._build_sidecar_transfers(
            cache,
            CacheTransferPhase.LOAD_BACK,
            PoolTransfer(PoolName.KV, host_indices=cached),
            {},
        )
        assert controller.load(cached, extra_pools=extras) is destination
        op = controller.load_queue.pop()
        completion = engine.submit_host_to_device(
            controller._l2_load_transfers(*controller._move_op_indices(op)),
            layer_num=target.layer_num,
        )
        completion.finish_event.synchronize()
        for pool in (target, *drafts):
            for index, buffers in (
                (False, pool.kv_buffer),
                (True, pool.index_k_with_scale_buffer),
            ):
                for buffer in buffers:
                    if not buffer.numel():
                        continue
                    expected = torch.full_like(originals[id(buffer)][1], 173)
                    src, dst = (
                        (source_pages, dest_pages)
                        if index
                        else (_tokens(source_pages), _tokens(dest_pages))
                    )
                    expected[dst] = originals[id(buffer)][1][src]
                    assert torch.equal(buffer.view(torch.uint8).cpu(), expected), (
                        "restored payload or untouched device page changed"
                    )
    finally:
        if group is not None:
            for entry in group.entries:
                entry.host_pool.destroy()
        reset_context()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
