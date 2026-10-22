"""HiCache moves a fused draft's rows with the target's.

A fused draft's KV is one more part of every target slot's entry. On a Mamba
host the HiCache host pool copies rows layer by layer, so the draft's layers
pack after the target's, and the transfer engine resolves their virtual ids
through the draft pool to the host's physical slots. This backs a request up
through the real controller and engine, wipes the device buffer, loads the
request back at another request's slots, and compares every target and draft
row bit for bit.

    python -m pytest test/registered/kernels/ops/kvcache/test_fused_draft_hicache_transfer.py -v
"""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.managers.cache_controller import CacheOperation
from sglang.srt.mem_cache.allocator.unified_mamba import (
    UnifiedMambaTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as assembler
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.l2_transfer import L2TransferEngine
from sglang.srt.mem_cache.layout.fused_draft import (
    DenseDraftRegion,
    FusedDraftPlacement,
    LatentDraftRegion,
)
from sglang.srt.mem_cache.unified_draft_pool import (
    UnifiedDraftKVPool,
    UnifiedDraftMLAKVPool,
)
from sglang.srt.mem_cache.unified_memory_pool import (
    MambaSubPoolSpec,
    MHASubPoolSpec,
    MLASubPoolSpec,
    UnifiedKVPool,
    UnifiedMHATokenToKVPool,
    UnifiedMLATokenToKVPool,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

_PS = 16
_PAGES = 16
_DTYPE = torch.bfloat16


class _FakeKV:
    def attach_allocator(self, allocator):
        self.allocator = allocator


def _pools(mla: bool):
    """A full + mamba unified buffer whose full entries carry one draft lane,
    its allocator, the target's full pool and the draft pool over the lane."""
    if mla:
        region = LatentDraftRegion(
            lane_num=1, kv_lora_rank=512, qk_rope_head_dim=64, store_dtype=_DTYPE
        )
        full = MLASubPoolSpec(
            name="full",
            layer_num=2,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            store_dtype=_DTYPE,
            grow_direction="down",
            draft_region=region,
        )
    else:
        region = DenseDraftRegion(
            lane_num=1, head_num=2, head_dim=128, store_dtype=_DTYPE
        )
        full = MHASubPoolSpec(
            name="full",
            layer_num=2,
            head_num=2,
            head_dim=128,
            store_dtype=_DTYPE,
            grow_direction="down",
            draft_region=region,
        )
    mamba = MambaSubPoolSpec(
        name="mamba",
        layer_num=1,
        conv_state_shapes=((2, 2),),
        conv_dtype=torch.float32,
        temporal_state_shape=(2,),
        temporal_dtype=torch.float32,
        grow_direction="up",
    )
    pool = UnifiedKVPool(
        total_bytes=_PAGES * _PS * full.entry_bytes() + 4 * mamba.entry_bytes(),
        sub_pool_specs=[full, mamba],
        device="cuda",
        enable_memory_saver=False,
        page_size=_PS,
        fused_draft=FusedDraftPlacement.from_counts(
            counts={"full": [1]}, regions={"full": region}
        ),
    )
    if mla:
        target = UnifiedMLATokenToKVPool(
            unified_buffer=pool,
            sub_pool_name="full",
            page_size=_PS,
            kv_cache_dtype=_DTYPE,
        )
    else:
        target = UnifiedMHATokenToKVPool(
            unified_buffer=pool, sub_pool_name="full", page_size=_PS
        )
    alloc = UnifiedMambaTokenToKVPoolAllocator(
        unified_buffer=pool,
        kvcache=SimpleNamespace(full_kv_pool=target, mamba_pool=_FakeKV()),
        device="cuda",
        page_size=_PS,
        need_sort=False,
        forward_stream=None,
        lazy_compaction=False,
    )
    draft = (UnifiedDraftMLAKVPool if mla else UnifiedDraftKVPool)(
        unified_buffer=pool,
        host_sub_pool_name="full",
        host_allocator=alloc,
        layer_lanes={0: 0},
        page_size=_PS,
    )
    return pool, alloc, target, draft


@pytest.mark.parametrize("layout", ["layer_first", "page_first"])
@pytest.mark.parametrize("mla", [False, True])
def test_fused_draft_rows_reload_with_the_target(layout: str, mla: bool) -> None:
    get_context().set_server_args(
        ServerArgs(
            model_path="test",
            hicache_ratio=2.0,
            hicache_size=0,
            hicache_mem_layout=layout,
            hicache_io_backend="kernel",
        )
    )
    pool, alloc, target, draft = _pools(mla)
    group = assembler.build_kv_only_group(
        page_size=_PS,
        kv_pool=target,
        full_layer_mapping={0: 0, 1: 1},
        use_mla=mla,
        mtp_draft_device_pools=(draft,),
    )
    try:
        host = group.anchor_entry.host_pool
        if not host.can_use_jit:
            pytest.skip("strided device rows need the JIT HiCache kernels")
        # The real controller's transfer assembly, without its storage threads.
        controller = object.__new__(HybridCacheController)
        controller.mem_pool_host = group
        controller.transfer_layer_id_max = target.layer_num
        controller.io_backend = "kernel"
        controller.device = "cuda"

        src = alloc.alloc(2 * _PS)
        dst = alloc.alloc(2 * _PS)
        translate = alloc.full_attn_allocator.translate_kv_loc
        src_rows, dst_rows = translate(src), translate(dst)
        assert not torch.equal(src_rows, src), "ids must translate"
        buffers = (
            [*target.kv_buffer, *draft.kv_buffer]
            if mla
            else [*target.k_buffer, *target.v_buffer, *draft.k_buffer, *draft.v_buffer]
        )
        expected = []
        for value, buffer in enumerate(buffers, 1):
            raw = buffer.view(torch.uint8)
            raw.copy_(
                (torch.arange(raw.numel(), device="cuda").reshape(raw.shape) + value)
                .remainder(251)
                .to(torch.uint8)
            )
            expected.append(raw[src_rows].clone())

        host_indices = host.alloc(len(src))
        engine = L2TransferEngine("kernel")
        backup = CacheOperation(host_indices, src, 1)
        engine.submit_device_to_host(
            controller._l2_transfers(*controller._move_write_operation(backup))
        ).finish_event.synchronize()
        pool._raw.zero_()
        restore = CacheOperation(host_indices, dst, 1)
        engine.submit_host_to_device(
            controller._l2_load_transfers(*controller._move_op_indices(restore)),
            transfer_layer_id_max=controller.transfer_layer_id_max,
        )
        torch.cuda.synchronize()

        for buffer, rows in zip(buffers, expected, strict=True):
            assert torch.equal(buffer.view(torch.uint8)[dst_rows], rows)
    finally:
        for entry in group.entries:
            entry.host_pool.destroy()


if __name__ == "__main__":
    args = [arg for arg in sys.argv[1:] if arg != "-f"]
    raise SystemExit(pytest.main([__file__, "-v", "-x", *args]))
