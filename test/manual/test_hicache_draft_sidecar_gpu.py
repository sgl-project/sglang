#!/usr/bin/env python3
"""Byte-exact mixed-geometry HiCache fallback test; run on an isolated GPU.

Uses real DSA device/host pools, the draft planner and sidecar assembler, and
the controller's transfer lists and asynchronous L2 engine. Backs up in two
pieces, poisons device storage, then restores onto shuffled physical pages.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    _build_hybrid_dsa_index_entry,
    _build_mha_mla_host_pool,
    build_hicache_draft_sidecars,
    build_pool_entry,
)
from sglang.srt.mem_cache.l2_transfer import L2TransferEngine
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, HybridLinearKVPool
from sglang.srt.mem_cache.pool_host import HostPoolGroup
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker, HiCacheDraftMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


def make_pool(dtype, dim, live, compress):
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
        index_kpool=4 if compress else 1,
        index_kpool_compress=compress,
        max_running_requests=4,
        skip_topk_layers=[not x for x in live],
    )


def wrap(pool, hybrid):
    if not hybrid:
        return pool
    result = object.__new__(HybridLinearKVPool)
    result.full_kv_pool = pool
    return result


def tokens(pages):
    return (pages[:, None] * 64 + torch.arange(64, device=pages.device)).flatten()


def run_case(layout, backend, target_geometry, draft_geometry, hybrid, live, compress):
    publish(
        ServerArgs(
            model_path="dummy",
            enable_hierarchical_cache=True,
            hicache_mem_layout=layout,
            hicache_io_backend=backend,
        ),
        role="scheduler",
    )
    target = make_pool(*target_geometry, (True, False, True), compress)
    draft = make_pool(*draft_geometry, (live,), compress)
    target_runner = SimpleNamespace(
        token_to_kv_pool=wrap(target, hybrid), spec_algorithm=SpeculativeAlgorithm.EAGLE
    )
    runner = SimpleNamespace(
        token_to_kv_pool=wrap(draft, hybrid),
        model_config=SimpleNamespace(
            num_nextn_predict_layers=1,
            hf_config=SimpleNamespace(architectures=["Glm5NextForCausalLMMTP"]),
        ),
    )
    worker = SimpleNamespace(
        target_worker=SimpleNamespace(model_runner=target_runner),
        _draft_model_runners=lambda: (runner,),
    )
    plan = BaseSpecWorker._build_hicache_draft_plan(worker)
    assert plan.mode == HiCacheDraftMode.SIDECAR
    assert target_runner.mtp_draft_device_pools == ()
    prefix = "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler."
    with patch(prefix + "_get_allocator_type", return_value="default"):
        host = _build_mha_mla_host_pool(
            pool=target,
            host_to_device_ratio=2,
            page_size=64,
            layout=layout,
            allocator_type="default",
            pool_label="target",
        )
        mapping = dict(enumerate(range(target.layer_num)))
        anchor = build_pool_entry(
            name=PoolName.KV,
            host_pool=host,
            device_pool=target,
            layer_mapping=mapping,
            transfer_layer_num=target.layer_num,
            is_anchor=True,
        )
        index = _build_hybrid_dsa_index_entry(
            kv_pool=target,
            kv_host_pool=host,
            layer_mapping=mapping,
            transfer_layer_num=target.layer_num,
        )
        group = HostPoolGroup([anchor, index])
        controller = object.__new__(HybridCacheController)
        controller.mem_pool_host = group
        controller.page_size = 64
        controller.layer_num = target.layer_num
        specs, entries = build_hicache_draft_sidecars(
            draft_device_pools=plan.device_pools,
            tree_cache=SimpleNamespace(cache_controller=controller),
        )
        for entry in entries:
            group.add_entry(entry)
    assert group.get_pool(PoolName.DRAFT).dtype == draft.store_dtype
    assert (
        group.get_pool(PoolName.DRAFT)._host_kv_cache_dim(draft) == draft.kv_cache_dim
    )
    assert (PoolName.DRAFT_INDEXER in group.entry_map) == live
    assert all(spec.indices_from_pool == PoolName.KV for spec in specs)

    generator = torch.Generator().manual_seed(38212)
    originals = []
    for pool in (target, draft):
        for buffer in pool.kv_buffer:
            view = buffer.view(torch.uint8)
            expected = torch.randint(
                0, 256, view.shape, dtype=torch.uint8, generator=generator
            )
            view.copy_(expected)
            originals.append((view, expected, False))
        for buffer in pool.index_k_with_scale_buffer:
            if not buffer.numel():
                continue
            expected = torch.randint(
                0, 256, buffer.shape, dtype=torch.uint8, generator=generator
            )
            buffer.copy_(expected)
            originals.append((buffer, expected, True))
    source_pages = torch.tensor([0, 1, 2, 3, 8, 9, 10, 11])
    host_pages = torch.tensor([16, 17, 18, 19, 4, 5, 6, 7])
    dest_pages = torch.tensor([12, 13, 14, 15, 4, 5, 6, 7])
    index_device = "cuda" if backend == "kernel" else "cpu"
    src, cached, dst = [
        tokens(p.to(index_device)) for p in (source_pages, host_pages, dest_pages)
    ]
    engine = L2TransferEngine(backend)

    def transfers(host_indices, device_indices, backup):
        aux = [
            PoolTransfer(
                name,
                host_indices=host_indices,
                device_indices=device_indices,
                indices_from_pool=PoolName.KV,
            )
            for name in group.entry_map
            if name != PoolName.KV
        ]
        result = controller._l2_load_transfers(host_indices, device_indices, aux)
        if backup:
            result = [
                t._replace(host_indices=t.host_indices.cpu())
                if t.host_pool.can_use_write_back_jit
                else t
                for t in result
            ]
        return result

    for start in (0, 256):
        completion = engine.submit_device_to_host(
            transfers(cached[start : start + 256], src[start : start + 256], True)
        )
    torch.cuda.current_stream().wait_event(completion.finish_event)
    for buffer, _, _ in originals:
        buffer.fill_(173)
    restored = engine.submit_host_to_device(
        transfers(cached, dst, False), layer_num=target.layer_num
    )
    restored.finish_event.synchronize()
    compared = 0
    for buffer, expected, is_index in originals:
        actual = buffer.cpu()[dest_pages if is_index else tokens(dest_pages)]
        expected = expected[source_pages if is_index else tokens(source_pages)]
        assert torch.equal(actual, expected), (
            layout,
            backend,
            target_geometry,
            draft_geometry,
            is_index,
        )
        compared += actual.numel()
    for entry in group.entries:
        entry.host_pool.destroy()
    reset_context()
    print(
        json.dumps(
            dict(
                layout=layout,
                backend=backend,
                target=str(target_geometry),
                draft=str(draft_geometry),
                hybrid=hybrid,
                index_live=live,
                compressed=compress,
                bytes_compared=compared,
                passed=True,
            )
        ),
        flush=True,
    )


def main():
    assert torch.cuda.is_available(), "requires an isolated CUDA test environment"
    print(
        json.dumps(
            dict(
                gpu=torch.cuda.get_device_name(),
                capability=torch.cuda.get_device_capability(),
            )
        ),
        flush=True,
    )
    fp8, bf16 = torch.float8_e4m3fn, torch.bfloat16
    cases = 0
    with torch.inference_mode():
        for layout, backend in (
            ("page_first", "kernel"),
            ("layer_first", "kernel"),
            ("layer_first", "direct"),
            ("page_first_direct", "direct"),
        ):
            for target, draft in (
                ((fp8, 656), (fp8, 528)),
                ((fp8, 656), (bf16, 576)),
                ((bf16, 576), (fp8, 576)),
            ):
                for hybrid in (False, True):
                    for live in (False, True):
                        run_case(layout, backend, target, draft, hybrid, live, True)
                        cases += 1
            run_case(layout, backend, (fp8, 656), (bf16, 576), False, True, False)
            cases += 1
    print(f"HiCache draft sidecar GPU byte oracle: {cases}/{cases} passed", flush=True)


if __name__ == "__main__":
    main()
