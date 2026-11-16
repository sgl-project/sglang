"""Unified L3 keys that the controller derives from real hybrid device pools.

Writer and reader deployments of different PP and TP splits build their storage
config from a real ``HybridLinearKVPool`` and Mamba state pool. Every page is a
pure function of the model-global layers, heads and tokens it holds, so a hit
is correct only if each chunk lands where the reader's own topology puts it.
"""

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.hicache_storage import HiCacheFile, PoolName, PoolTransfer
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, MambaPool
from sglang.srt.mem_cache.pool_host.page_unified import (
    PageUnifiedLayout,
    build_mamba_page_unified_layout,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

FULL_LAYERS, LINEAR_LAYERS = (3, 7), (0, 1, 2, 4, 5, 6)
KEYS, PAGE, KV_HEADS, HEAD_DIM, MLA_DIM = ["p0", "p1"], 2, 2, 4, 8
# GDN-shaped state: 8 heads; conv is cat(q, k, v) along axis 0.
TEMPORAL, CONV, CONV_WIDTHS = (8, 2, 4), (64, 4), (16, 16, 32)


def _truth(*shape, offset=0.0):
    return offset + torch.arange(torch.Size(shape).numel(), dtype=torch.float32).view(
        shape
    )


KV = _truth(len(KEYS), 8, 2, PAGE, KV_HEADS, HEAD_DIM)
MLA = _truth(len(KEYS), 8, PAGE, MLA_DIM, offset=1e4)
STATES = (
    _truth(len(LINEAR_LAYERS), len(KEYS), *TEMPORAL, offset=2e4),
    _truth(len(LINEAR_LAYERS), len(KEYS), *CONV, offset=3e4),
)


class _HostPool:
    """The page interface the file backend uses: one byte row per key."""

    page_size, layout, storage_format_tag = 1, "page_unified", None

    def __init__(self, layout, pages, **attrs):
        self.page_unified_layout, self.pages = layout, pages
        self.__dict__.update(attrs)

    def get_data_page(self, index, flat=True):
        return self.pages[index]

    def get_dummy_flat_data_page(self):
        return torch.empty_like(self.pages[0])

    def set_from_flat_data_page(self, index, page):
        self.pages[index].copy_(page.view(torch.uint8).flatten())


def _pages(tensor):
    return tensor.contiguous().view(torch.uint8).reshape(len(KEYS), -1)


def _rank(mla, tp_size, tp_rank, pp_size, pp_rank):
    stage = range(pp_rank * 8 // pp_size, (pp_rank + 1) * 8 // pp_size)
    full = [layer for layer in FULL_LAYERS if layer in stage]
    linear = [layer for layer in LINEAR_LAYERS if layer in stage]
    heads = max(1, KV_HEADS // tp_size)
    kv_device = HybridLinearKVPool(
        size=8,
        page_size=PAGE,
        dtype=torch.float32,
        head_num=heads,
        head_dim=HEAD_DIM,
        full_attention_layer_ids=full,
        device="cpu",
        mamba_pool=None,
        **(dict(use_mla=True, kv_lora_rank=4, qk_rope_head_dim=4) if mla else {}),
    )
    if mla:
        kv_pages = _pages(MLA[:, full])
    else:
        # Consecutive ranks replicate a head past TP = KV_HEADS.
        first = tp_rank // (tp_size * heads // KV_HEADS) * heads
        kv_pages = _pages(KV[:, full, :, :, first : first + heads])
    kv_layout = PageUnifiedLayout(
        page_size=PAGE,
        layer_num=len(full),
        head_num=1 if mla else heads,
        head_group_num=1,
        head_dim=MLA_DIM if mla else HEAD_DIM,
        itemsize=4,
        is_mla=mla,
    )
    kv_host = _HostPool(
        kv_layout, kv_pages, layer_num=len(full), head_num=heads, head_group_num=1
    )

    rows = [LINEAR_LAYERS.index(layer) for layer in linear]
    state_heads = TEMPORAL[0] // tp_size
    blocks, start = [], 0
    for width in CONV_WIDTHS:
        step = width // tp_size
        blocks.append(STATES[1][rows, :, start + tp_rank * step :][:, :, :step])
        start += width
    state_device = object.__new__(MambaPool)
    state_device.mamba_layer_ids = linear
    state_device.mamba_cache = MambaPool.State(
        temporal=STATES[0][rows, :, tp_rank * state_heads :][:, :, :state_heads],
        conv=[torch.cat(blocks, dim=2)],
    )
    state_device.conv_shard_groups, state_device.conv_slice_axis = CONV_WIDTHS, 0
    (info,) = state_device.get_device_pool_infos()
    # --hicache-storage-state-partition 1: one state head per chunk.
    layout = build_mamba_page_unified_layout(
        info.buffer_info,
        layer_num=len(linear),
        attn_tp_size=tp_size,
        group_num=state_heads,
    )
    states = (info.buffer_info.temporal, info.buffer_info.conv[0])
    state_pages = torch.empty(len(KEYS), layout.bytes_per_page, dtype=torch.uint8)
    for slot in range(len(KEYS)):
        for part, layer, offset, page_offset, length in layout.pieces():
            source = states[part][layer, slot].contiguous().view(torch.uint8).flatten()
            state_pages[slot, page_offset : page_offset + length] = source[
                offset : offset + length
            ]
    state_host = _HostPool(
        layout,
        state_pages,
        temporal_dtype=torch.float32,
        temporal_state_elem_size=1,
        conv_dtype=torch.float32,
        conv_state_shapes=[CONV],
    )

    controller = object.__new__(HiCacheController)
    controller.page_size, controller.attn_cp_group = PAGE, None
    controller.mem_pool_device, controller.storage_host_pool = kv_device, kv_host
    controller.enable_storage_metrics = False
    kv_entry = SimpleNamespace(host_pool=kv_host, device_pool=kv_device)
    state_entry = SimpleNamespace(host_pool=state_host, device_pool=state_device)
    controller.mem_pool_host = SimpleNamespace(
        layout="page_unified",
        entries=[kv_entry, state_entry],
        entry_map={PoolName.KV: kv_entry, PoolName.MAMBA: state_entry},
        anchor_entry=kv_entry,
    )
    memory = SimpleNamespace(
        hicache_storage_key_scheme="unified",
        hicache_storage_head_partition=None,
        hicache_storage_state_partition=1,
        hicache_storage_layer_partition=4,
    )
    parallel = SimpleNamespace(
        tp_rank=tp_rank,
        tp_size=tp_size,
        attn_tp_rank=tp_rank,
        attn_tp_size=tp_size,
        pp_rank=pp_rank,
        pp_size=pp_size,
        attn_dcp_size=1,
        attn_dcp_rank=0,
    )
    model_config = SimpleNamespace(get_total_num_kv_heads=lambda: KV_HEADS)
    module = "sglang.srt.managers.cache_controller"
    with (
        patch(f"{module}.get_memory", return_value=memory),
        patch(f"{module}.get_parallel", return_value=parallel),
        patch(f"{module}.is_dp_attention_enabled", return_value=False),
        patch(f"{module}.process_model_config", return_value=model_config),
    ):
        config = controller._generate_storage_config(
            "org/hybrid",
            {
                "enable_metadata_cache": True,
                "metadata_ttl": -1,
                "max_size": 0,
                "min_free_space": 0,
            },
        )
    backend = HiCacheFile(config)
    backend.register_mem_pool_host(kv_host)
    backend.register_mem_host_pool_v2(kv_host, PoolName.KV)
    backend.register_mem_host_pool_v2(state_host, PoolName.MAMBA)
    return config, backend, (kv_host.pages, state_host.pages)


def _transfer(name):
    return PoolTransfer(name=name, keys=KEYS, host_indices=torch.arange(len(KEYS)))


class TestUnifiedHybridAcrossTopologies(CustomTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        context = envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.override(directory.name)
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)

    def _round_trip(self, mla, writer, reader):
        (w_tp, w_pp), (r_tp, r_pp) = writer, reader
        written = []
        for rank in range(w_tp * w_pp):
            config, backend, _ = _rank(mla, w_tp, rank % w_tp, w_pp, rank // w_tp)
            # Every rank writes its state shard; KV only its elected writer.
            names = [PoolName.MAMBA]
            if config.is_storage_writer:
                written += config.unified_suffixes
                names.append(PoolName.KV)
            results = backend.batch_set_v2([_transfer(name) for name in names])
            self.assertEqual(results, {name: [True] * len(KEYS) for name in names})
        # Replicas of one KV chunk elect exactly one writer.
        self.assertEqual(len(written), len(set(written)))

        for rank in range(r_tp * r_pp):
            config, backend, pages = _rank(mla, r_tp, rank % r_tp, r_pp, rank // r_tp)
            self.assertLessEqual(set(config.unified_suffixes), set(written))
            expected = [page.clone() for page in pages]
            for page in pages:
                page.zero_()
            result = backend.batch_exists_v2(KEYS, [_transfer(PoolName.MAMBA)])
            self.assertEqual(result.kv_hit_pages, len(KEYS))
            names = [PoolName.KV, PoolName.MAMBA]
            results = backend.batch_get_v2([_transfer(name) for name in names])
            self.assertEqual(results, {name: [True] * len(KEYS) for name in names})
            for page, want in zip(pages, expected):
                torch.testing.assert_close(page, want, rtol=0, atol=0)

    def test_mla_hybrid_across_pp_and_tp(self):
        # Compact MLA child pools must name chunks by model layer.
        self._round_trip(mla=True, writer=(2, 2), reader=(4, 1))

    def test_gqa_hybrid_with_replicated_kv_heads(self):
        # 2 KV heads: TP8 replicates each head on 4 ranks, TP4 on 2.
        self._round_trip(mla=False, writer=(8, 1), reader=(4, 2))

    def test_conv_without_declared_sub_blocks_is_rejected(self):
        # Standard Mamba2 declares none; a stale declaration must not mis-tile.
        for widths in ((), (8, 8, 8), (9, 7, 16)):
            info = SimpleNamespace(
                temporal=STATES[0][:, :, :4],
                conv=[STATES[1][:, :, :32]],
                conv_shard_groups=widths,
                conv_slice_axis=0,
            )
            with self.subTest(widths=widths), self.assertRaises(NotImplementedError):
                build_mamba_page_unified_layout(
                    info, layer_num=6, attn_tp_size=2, group_num=1
                )


if __name__ == "__main__":
    unittest.main()
