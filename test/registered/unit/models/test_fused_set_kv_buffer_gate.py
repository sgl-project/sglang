"""Which pools the fused RoPE + KV store may write on ROCm.

The fused store writes window layers through `full_to_swa_index_mapping`. A
unified pool's window view takes ids only its write door derives, so the fused
store must step aside for it and leave the write to `set_kv_buffer`.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.kv.alloc.id_space import IdSpaceKind
from sglang.srt.mem_cache.kv.router import PublicSubPool
from sglang.srt.models import utils as model_utils
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    UnifiedGeometry,
    build_unified_bundle,
    setup_allocator_context,
)

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

_P16 = dict(page_size=16, full_tokens=64 * 16, swa_tokens=64 * 16)


def _unified_pool(**geometry):
    setup_allocator_context()
    return build_unified_bundle(UnifiedGeometry(**geometry, **_P16)).token_to_kv_pool


# Full-side slots 0..7 hold window rows at other window-side slots.
_FULL_TO_SWA = torch.tensor([0, 5, 7, 2, 6, 1, 3, 4], dtype=torch.int32)
_PAGE, _HEADS, _DIM = 2, 1, 4


def _static_window_pool():
    """A static full + window pool: its window view writes at ids the mapping gives."""
    window = PublicSubPool(name=IdSpaceKind.SLIDING_WINDOW, layer_ids=(1,))
    views = {
        IdSpaceKind.SLIDING_WINDOW: SimpleNamespace(requires_physical_write_loc=False)
    }
    buffer = torch.zeros(_FULL_TO_SWA.numel(), _HEADS, _DIM)
    layer_kv = SimpleNamespace(key=lambda: buffer, value=lambda: buffer)
    router = SimpleNamespace(
        sub_pool=lambda name: window if name == window.name else None,
        view=lambda name: views[name],
        read=lambda layer_id: layer_kv,
    )
    return SimpleNamespace(
        layer_router=router,
        page_size=_PAGE,
        full_to_swa_index_mapping=_FULL_TO_SWA,
        get_key_buffer=lambda layer_id: buffer,
        get_value_buffer=lambda layer_id: buffer,
    )


def _layer(layer_id, sliding_window_size):
    return SimpleNamespace(
        layer_id=layer_id,
        sliding_window_size=sliding_window_size,
        tp_k_head_num=_HEADS,
        tp_v_head_num=_HEADS,
        qk_head_dim=_DIM,
        v_head_dim=_DIM,
        k_scale=None,
        v_scale=None,
    )


def _hip_write_slots(pool, layer, out_cache_loc):
    """The slots the fused store writes ``layer`` at: ``out_cache_loc``,
    mapped through ``swa_slot_mapping`` when it has one."""
    with (
        patch.object(model_utils, "_is_hip", True),
        patch.object(model_utils, "get_token_to_kv_pool", return_value=pool),
        patch.object(model_utils, "get_kv_layers", return_value=pool.layer_router),
    ):
        arg = model_utils.create_fused_set_kv_buffer_arg(
            torch.zeros(out_cache_loc.numel(), _HEADS * _DIM),
            layer,
            SimpleNamespace(out_cache_loc=out_cache_loc),
        )
    slots = arg["slot_mapping"]
    if arg["swa_slot_mapping"] is not None:
        slots = arg["swa_slot_mapping"][slots]
    return slots


def _hip_gate(pool, dcp_kv_mask=None) -> bool:
    with (
        patch.object(model_utils, "_is_hip", True),
        patch.object(model_utils, "_is_cuda", False),
        patch.object(model_utils, "get_token_to_kv_pool", return_value=pool),
    ):
        return model_utils.enable_fused_set_kv_buffer(
            SimpleNamespace(dcp_kv_mask=dcp_kv_mask)
        )


class TestHipFusedStoreGate(CustomTestCase):
    def test_unified_window_pool_takes_the_write_door(self):
        pool = _unified_pool(chain="swa", full_layer_ids=(0, 1), swa_layer_ids=(2, 3))
        self.assertFalse(_hip_gate(pool))

    def test_unified_three_band_pool_takes_the_write_door(self):
        pool = _unified_pool(
            chain="tri",
            full_layer_ids=(0, 1),
            swa_layer_ids=(2, 3),
            state_layer_ids=(0, 2),
        )
        self.assertFalse(_hip_gate(pool))

    def test_unified_pool_without_a_window_stays_fused(self):
        pool = _unified_pool(
            chain="mamba", full_layer_ids=(0, 1), state_layer_ids=(2, 3)
        )
        self.assertTrue(_hip_gate(pool))

    def test_static_window_pool_stays_fused(self):
        self.assertTrue(_hip_gate(_static_window_pool()))

    def test_each_layer_of_a_static_window_pool_writes_at_its_own_ids(self):
        """Full and window layers of one batch: the full layer writes at the
        full-side ids, the window layer at the window-side ids the mapping
        gives them, which differ."""
        pool = _static_window_pool()
        loc = torch.tensor([1, 2, 6])
        full = _hip_write_slots(pool, _layer(0, sliding_window_size=-1), loc)
        window = _hip_write_slots(pool, _layer(1, sliding_window_size=4), loc)
        torch.testing.assert_close(full, loc)
        torch.testing.assert_close(window, _FULL_TO_SWA[loc].long())
        self.assertFalse(torch.equal(full, window))

    def test_a_unified_window_pool_writes_neither_layer_fused(self):
        """BUG REGRESSION. On ROCm the fused store served a unified window
        pool, which has no full -> window mapping, so window layers were
        written at the full side's ids. Its full and window ids differ, and
        the gate is per batch, so neither layer of the batch takes the fused
        store: both write through the pool."""
        pool = _unified_pool(chain="swa", full_layer_ids=(0, 1), swa_layer_ids=(2, 3))
        router = pool.layer_router
        self.assertTrue(
            router.view(
                router.sub_pool(IdSpaceKind.SLIDING_WINDOW).name
            ).requires_physical_write_loc
        )
        self.assertFalse(_hip_gate(pool))

    def test_decode_context_parallel_mask_declines(self):
        self.assertFalse(_hip_gate(_static_window_pool(), dcp_kv_mask=object()))


if __name__ == "__main__":
    unittest.main()
