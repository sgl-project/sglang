import sys
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from sglang.srt.configs.inkling import (
    InklingConvCacheParams,
    InklingConvStateShape,
    InklingStateDType,
)
from sglang.srt.managers.cache_controller import CacheOperation
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as assembler
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.l2_transfer import L2TransferEngine
from sglang.srt.mem_cache.memory_pool import (
    MambaPool,
    MHATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.pool_host import HostPoolGroup
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.runtime_context import get_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _mamba_pool(layer_ids: list[int], with_temporal: bool) -> MambaPool:
    return MambaPool(
        size=16,
        spec_state_size=4,
        cache_params=InklingConvCacheParams(
            shape=InklingConvStateShape(
                conv=[(3, 256), (3, 256), (3, 128), (3, 128), (3, 256), (3, 256)],
                temporal=(2, 8, 16) if with_temporal else (0, 0, 0),
            ),
            layers=layer_ids,
            dtype=InklingStateDType(conv=torch.bfloat16, temporal=torch.bfloat16),
        ),
        mamba_layer_ids=layer_ids,
        device="cuda",
    )


@pytest.mark.parametrize(
    ("io_backend", "layout"),
    [("kernel", "page_first"), ("direct", "page_first_direct")],
)
@pytest.mark.parametrize(
    ("local_drafts", "draft_full_indices"),
    [([0, 1, 2], True), ([0, 2], True), ([], True), ([0, 1, 2], False)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("target_layers", [2, 4])
@pytest.mark.parametrize("with_temporal", [False, True])
def test_hicache_restores_all_mtp_kv_and_conv_state(
    monkeypatch: pytest.MonkeyPatch,
    layout: str,
    io_backend: str,
    local_drafts: list[int],
    draft_full_indices: bool,
    dtype: torch.dtype,
    target_layers: int,
    with_temporal: bool,
) -> None:
    # Cover both ratio sizing and a fixed budget, including draft sidecars.
    host_size_gb = 0.01 if target_layers == 4 else 0
    get_context().set_server_args(
        ServerArgs(
            model_path="test",
            hicache_ratio=1.0,
            hicache_size=host_size_gb,
            hicache_mem_layout=layout,
            hicache_io_backend=io_backend,
        )
    )

    def kv_pool(
        full: list[int], swa: list[int], full_indices: bool = False
    ) -> SWAKVPool:
        pool = SWAKVPool(
            size=128,
            size_swa=128 if full_indices else 64,
            page_size=16,
            dtype=dtype,
            head_num=2,
            head_dim=128,
            full_attention_layer_ids=full,
            swa_attention_layer_ids=swa,
            device="cuda",
            token_to_kv_pool_class=MHATokenToKVPool,
            swa_kv_pool_kwargs={
                "head_num": 1,
                "head_dim": 128,
                "page_size": 16,
                "device": "cuda",
                "enable_memory_saver": False,
            },
        )
        pool.swa_uses_full_indices = full_indices
        return pool

    full_layers = list(range(1, target_layers, 2))
    swa_layers = list(range(0, target_layers, 2))
    target_kv = kv_pool(full_layers, swa_layers)
    drafts = tuple(
        kv_pool(
            [] if i in local_drafts else [i],
            [i] if i in local_drafts else [],
            draft_full_indices,
        )
        for i in range(3)
    )
    target_mamba = _mamba_pool(list(range(target_layers)), with_temporal)
    draft_mamba = tuple(_mamba_pool([i], with_temporal) for i in range(3))
    allocator = SimpleNamespace(alloc=lambda n: None, free=lambda ids: None)
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=cast(
            ReqToTokenPool,
            SimpleNamespace(mamba_pool=target_mamba, mamba_allocator=allocator),
        ),
        token_to_kv_pool_allocator=cast(
            BaseTokenToKVPoolAllocator, SimpleNamespace(swa_attn_allocator=allocator)
        ),
        page_size=16,
        mtp_draft_device_pools=drafts,
        mtp_draft_mamba_pools=draft_mamba,
    )

    # Exercise the real assembly and transfer engine without storage threads.
    def controller_without_threads(
        _token_allocator: object,
        host_group: HostPoolGroup,
        *_args: object,
        transfer_layer_id_max: int,
        **_kwargs: object,
    ) -> HybridCacheController:
        controller = object.__new__(HybridCacheController)
        monkeypatch.setattr(controller, "mem_pool_host", host_group, raising=False)
        controller.transfer_layer_id_max = transfer_layer_id_max
        controller.io_backend = io_backend
        controller.device = "cuda"
        return controller

    monkeypatch.setattr(assembler, "HybridCacheController", controller_without_threads)
    group, controller = assembler.build_hybrid_mamba_swa_stack(
        params=params,
        full_kv_pool=target_kv.full_kv_pool,
        swa_kv_pool=target_kv.swa_kv_pool,
        mamba_pool=target_mamba,
        full_layer_mapping={layer: i for i, layer in enumerate(full_layers)},
        swa_layer_mapping={layer: i for i, layer in enumerate(swa_layers)},
        mamba_layer_mapping={i: i for i in range(target_layers)},
        page_size=16,
        tp_group=None,
        load_cache_event=None,
        storage_backend=None,
    )
    try:
        if host_size_gb:
            host_bytes = sum(
                entry.host_pool.size * entry.host_pool.size_per_token
                for entry in group.entries
            )
            # Each pool rounds up by a page. A derived sidecar also inherits
            # the anchor's rounding before rounding its own capacity.
            page_padding = sum(
                entry.host_pool.page_size
                * entry.host_pool.size_per_token
                * (2 if entry.name == PoolName.DRAFT_SWA else 1)
                for entry in group.entries
            )
            assert host_bytes <= host_size_gb * 1e9 + page_padding
        if PoolName.DRAFT_SWA in group.entry_map:
            assert (
                group.get_entry(PoolName.DRAFT_SWA).host_pool.size
                >= group.anchor_entry.host_pool.size
            )
        full_src, full_dst = (
            torch.arange(16, 32, device="cuda"),
            torch.arange(64, 80, device="cuda"),
        )
        swa_src, swa_dst = (
            torch.arange(32, 48, device="cuda"),
            torch.arange(16, device="cuda"),
        )
        state_src, state_dst = (
            torch.tensor([3, 5], device="cuda"),
            torch.tensor([8, 10], device="cuda"),
        )
        active_state = torch.tensor([12, 13], device="cuda")
        # Token-major views let the same check cover every KV layer and all six
        # convolution buffers and any temporal state.
        buffers = []
        for pool in (target_kv, *drafts):
            # Inkling drafts use identity mapping, while the target's SWA
            # allocation has different source AND destination token IDs.
            draft_swa_src, draft_swa_dst = (
                (full_src, full_dst)
                if pool.swa_uses_full_indices
                else (swa_src, swa_dst)
            )
            for subpool, src, dst in (
                (pool.full_kv_pool, full_src, full_dst),
                (pool.swa_kv_pool, draft_swa_src, draft_swa_dst),
            ):
                assert isinstance(subpool, MHATokenToKVPool)
                assert subpool.k_buffer is not None and subpool.v_buffer is not None
                buffers.extend(
                    (buf, src, dst) for buf in (*subpool.k_buffer, *subpool.v_buffer)
                )
        for pool in (target_mamba, *draft_mamba):
            buffers.extend(
                (buf.transpose(0, 1), state_src, active_state)
                for buf in (*pool.mamba_cache.conv, pool.mamba_cache.temporal)
                if buf.numel()
            )
        expected = []
        for i, (buf, src, _) in enumerate(buffers):
            raw = buf.view(torch.uint8)
            raw.copy_(
                (torch.arange(raw.numel(), device="cuda").reshape(raw.shape) + i)
                .remainder(251)
                .to(torch.uint8)
            )
            expected.append(raw[src].clone())

        full_host, swa_host, state_host = (
            torch.arange(16),
            torch.arange(16, 32),
            torch.tensor([1, 2]),
        )

        def transfers(
            full_indices: torch.Tensor,
            swa_indices: torch.Tensor,
            state_indices: torch.Tensor,
        ) -> list[PoolTransfer]:
            result = [
                PoolTransfer(
                    PoolName.SWA, host_indices=swa_host, device_indices=swa_indices
                ),
                PoolTransfer(
                    PoolName.MAMBA,
                    host_indices=state_host,
                    device_indices=state_indices,
                ),
            ]
            specs = assembler._mtp_swa_sidecar_specs(group)
            assert bool(specs) == (draft_full_indices and bool(local_drafts))
            for spec in specs:
                assert spec.indices_from_pool == PoolName.KV
                result.append(
                    PoolTransfer(
                        spec.pool_name, indices_from_pool=spec.indices_from_pool
                    )
                )
            return group.resolve_host_transfers(
                result,
                primary_device_indices=full_indices,
                primary_host_indices=full_host,
            )

        engine = L2TransferEngine(io_backend)
        backup = CacheOperation(
            full_host,
            full_src,
            1,
            pool_transfers=transfers(full_src, swa_src, state_src),
        )
        engine.submit_device_to_host(
            controller._l2_transfers(*controller._move_write_operation(backup))
        ).finish_event.synchronize()
        for buf, _, _ in buffers:
            buf.zero_()
        restore = CacheOperation(
            full_host,
            full_dst,
            1,
            pool_transfers=transfers(full_dst, swa_dst, state_dst),
        )
        layer_done = [
            torch.cuda.Event() for _ in range(controller.transfer_layer_id_max)
        ]
        engine.submit_host_to_device(
            controller._l2_load_transfers(*controller._move_op_indices(restore)),
            transfer_layer_id_max=controller.transfer_layer_id_max,
            on_layer_done=lambda layer: layer_done[layer].record(),
        )
        # Prefix reuse copies all target and draft convolution state on the
        # forward stream after waiting for the final target-layer event.
        layer_done[-1].wait()
        for pool in (target_mamba, *draft_mamba):
            pool.copy_from(state_dst, active_state)
        torch.cuda.synchronize()

        for (buf, _, dst), reference in zip(buffers, expected, strict=True):
            assert torch.equal(buf.view(torch.uint8)[dst], reference)
    finally:
        for entry in group.entries:
            entry.host_pool.destroy()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-x", *sys.argv[1:]]))
