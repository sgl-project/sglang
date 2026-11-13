"""Fused RoPE stores use each pool's physical ids and reserved-sink contract."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.kv.alloc.id_space import IdSpaceKind
from sglang.srt.mem_cache.kv.router import PublicSubPool
from sglang.srt.model_executor.forward_mode import ForwardMode
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
_PAGE, _HEADS, _DIM = 2, 1, 8


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


def _hip_fused_args(pool, layer, batch, backend=None):
    with (
        patch.object(model_utils, "_is_hip", True),
        patch.object(model_utils, "get_token_to_kv_pool", return_value=pool),
        patch.object(model_utils, "get_kv_layers", return_value=pool.layer_router),
        patch.object(model_utils, "get_attn_backend", return_value=backend),
    ):
        return model_utils.create_fused_set_kv_buffer_arg(
            torch.zeros(batch.out_cache_loc.numel(), _HEADS * _DIM),
            layer,
            batch,
        )


def _hip_write_slots(pool, layer, out_cache_loc):
    """Static pools map full slots to their window rail inside the kernel."""
    arg = _hip_fused_args(pool, layer, SimpleNamespace(out_cache_loc=out_cache_loc))
    slots = arg["slot_mapping"]
    if arg["swa_slot_mapping"] is not None:
        slots = arg["swa_slot_mapping"][slots]
    return slots


def _hip_gate(pool, dcp_kv_mask=None) -> bool:
    with (
        patch.object(model_utils, "_is_hip", True),
        patch.object(model_utils, "_is_cuda", False),
        patch.object(
            model_utils,
            "get_parallel",
            return_value=SimpleNamespace(dcp_enabled=False),
        ),
        patch.object(model_utils, "get_token_to_kv_pool", return_value=pool),
    ):
        return model_utils.enable_fused_set_kv_buffer(
            SimpleNamespace(dcp_kv_mask=dcp_kv_mask)
        )


class TestHipFusedStoreGate(CustomTestCase):
    def test_unified_window_pool_stays_fused(self):
        pool = _unified_pool(chain="swa", full_layer_ids=(0, 1), swa_layer_ids=(2, 3))
        self.assertTrue(_hip_gate(pool))

    def test_unified_three_band_pool_stays_fused(self):
        pool = _unified_pool(
            chain="tri",
            full_layer_ids=(0, 1),
            swa_layer_ids=(2, 3),
            state_layer_ids=(0, 2),
        )
        self.assertTrue(_hip_gate(pool))

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

    def test_arena_full_and_window_borrow_physical_replay_buffers(self):
        pool = _unified_pool(
            chain="swa",
            full_layer_ids=(0,),
            swa_layer_ids=(1,),
            head_dim=_DIM,
            swa_head_dim=_DIM,
        )
        full_ids = torch.tensor([17, 18, 0])
        window_ids = torch.tensor([65, 66, 0, 0])
        batch = SimpleNamespace(
            out_cache_loc=full_ids,
            out_cache_loc_is_physical=True,
            forward_mode=ForwardMode.DECODE,
        )
        backend = SimpleNamespace(
            forward_metadata=SimpleNamespace(swa_out_cache_loc=window_ids)
        )
        args = [
            _hip_fused_args(pool, _layer(i, -1 if i == 0 else 4), batch, backend)
            for i in (0, 1)
        ]
        for arg, ids in zip(args, (full_ids, window_ids)):
            self.assertEqual(arg["slot_mapping"].data_ptr(), ids.data_ptr())
            self.assertEqual(arg["slot_mapping"].numel(), full_ids.numel())
            self.assertIsNone(arg["swa_slot_mapping"])
            self.assertEqual(arg["reserved_skip_index"], 0)
            self.assertGreater(arg["key_cache"].stride(1), _HEADS * _DIM)
        # A captured kernel keeps these addresses while metadata moves pages.
        full_ids.copy_(torch.tensor([81, 82, 0]))
        window_ids[:3].copy_(torch.tensor([33, 34, 0]))
        torch.testing.assert_close(args[0]["slot_mapping"], full_ids)
        torch.testing.assert_close(args[1]["slot_mapping"], window_ids[:3])

    def test_full_only_arena_preserves_sink_and_static_pool_keeps_slot_zero(self):
        pool = _unified_pool(
            chain="mamba", full_layer_ids=(0,), state_layer_ids=(1,), head_dim=_DIM
        )
        batch = SimpleNamespace(
            out_cache_loc=torch.tensor([17, 0]), out_cache_loc_is_physical=True
        )
        arg = _hip_fused_args(pool, _layer(0, -1), batch)
        self.assertEqual(arg["reserved_skip_index"], 0)
        static = _hip_fused_args(_static_window_pool(), _layer(0, -1), batch)
        self.assertEqual(static["reserved_skip_index"], -1)

    def test_window_metadata_routes_through_active_wrappers(self):
        from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
        from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
            HybridLinearAttnBackend,
        )
        from sglang.srt.layers.attention.tbo_backend import TboAttnBackend

        prefill_ids, decode_ids = torch.tensor([33, 0]), torch.tensor([65, 0])
        hybrid = object.__new__(HybridAttnBackend)
        hybrid.prefill_backend = SimpleNamespace(
            forward_metadata=SimpleNamespace(swa_out_cache_loc=prefill_ids)
        )
        hybrid.decode_backend = SimpleNamespace(
            forward_metadata=SimpleNamespace(swa_out_cache_loc=decode_ids)
        )
        hybrid.spec_attn_is_decode = True
        linear = object.__new__(HybridLinearAttnBackend)
        linear.full_attn_backend = hybrid
        tbo = object.__new__(TboAttnBackend)
        tbo.primary = linear
        for mode, expected in (
            (ForwardMode.EXTEND, prefill_ids),
            (ForwardMode.DECODE, decode_ids),
            (ForwardMode.TARGET_VERIFY, decode_ids),
        ):
            with (
                self.subTest(mode=mode),
                patch.object(model_utils, "get_attn_backend", return_value=tbo),
            ):
                loc = model_utils._fused_window_cache_loc(
                    SimpleNamespace(out_cache_loc=torch.zeros(2), forward_mode=mode),
                    "window",
                )
                self.assertEqual(loc.data_ptr(), expected.data_ptr())

    def test_named_window_rail_selection_and_missing_metadata_fail_closed(self):
        from sglang.srt.layers.attention.triton_backend import ForwardMetadata

        selected = torch.tensor([49, 50, 0])
        metadata = object.__new__(ForwardMetadata)
        metadata.windows = {
            "other": SimpleNamespace(out_cache_loc=torch.tensor([1, 2, 0])),
            "selected": SimpleNamespace(out_cache_loc=selected),
        }
        batch = SimpleNamespace(out_cache_loc=torch.zeros(3))
        with patch.object(
            model_utils,
            "get_attn_backend",
            return_value=SimpleNamespace(forward_metadata=metadata),
        ):
            result = model_utils._fused_window_cache_loc(batch, "selected")
            self.assertEqual(result.data_ptr(), selected.data_ptr())
            with self.assertRaises(KeyError):
                model_utils._fused_window_cache_loc(batch, "missing")
            metadata.windows["selected"].out_cache_loc = None
            with self.assertRaises(AssertionError):
                model_utils._fused_window_cache_loc(batch, "selected")

    def test_arena_refuses_unmarked_virtual_locations(self):
        pool = _unified_pool(
            chain="mamba", full_layer_ids=(0,), state_layer_ids=(1,), head_dim=_DIM
        )
        with self.assertRaises(AssertionError):
            _hip_fused_args(
                pool,
                _layer(0, -1),
                SimpleNamespace(
                    out_cache_loc=torch.tensor([17]), out_cache_loc_is_physical=False
                ),
            )

    def test_decode_context_parallel_mask_declines(self):
        self.assertFalse(_hip_gate(_static_window_pool(), dcp_kv_mask=object()))


class TestDcpFusedStoreGate(CustomTestCase):
    def test_dcp_without_a_batch_mask_uses_the_backend_store(self):
        pool = SimpleNamespace(
            dtype=torch.bfloat16,
            layer_router=SimpleNamespace(sub_pool=lambda name: None),
        )
        for cuda in (True, False):
            for dcp_size in (1, 2, 4):
                with (
                    self.subTest(cuda=cuda, dcp_size=dcp_size),
                    patch.object(model_utils, "_is_cuda", cuda),
                    patch.object(model_utils, "_is_hip", not cuda),
                    patch.object(model_utils, "is_cp_active", return_value=False),
                    patch.object(
                        model_utils, "get_token_to_kv_pool", return_value=pool
                    ),
                    patch.object(
                        model_utils,
                        "get_parallel",
                        return_value=SimpleNamespace(dcp_enabled=dcp_size > 1),
                    ),
                ):
                    self.assertEqual(
                        model_utils.enable_fused_set_kv_buffer(
                            SimpleNamespace(dcp_kv_mask=None)
                        ),
                        dcp_size == 1,
                    )


if __name__ == "__main__":
    unittest.main()
