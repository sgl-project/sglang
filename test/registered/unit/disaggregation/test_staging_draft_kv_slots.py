"""Staging slot ids stay aligned once a draft KV pool is registered.

The staging gather writes every k_buffer and then every v_buffer, while
kv_data_ptrs (and therefore kv_layer_ids) is ordered
[K target, V target, K draft, V draft]. Labelling slots with kv_layer_ids
silently pairs a layer's KV with another layer's staging slot as soon as a
draft pool exists.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.utils import (
    build_staging_slot_metadata,
    setup_state_kv_args,
)
from sglang.srt.mem_cache.kv.alloc.id_space import IdSpaceKind
from sglang.srt.mem_cache.kv.decl import Region
from sglang.srt.mem_cache.kv.router import PlainPoolRouter
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class _Pool(MHATokenToKVPool):
    def __init__(self, tag, layer_ids):
        self.k_buffer = [f"{tag}K{i}" for i in layer_ids]
        self.v_buffer = [f"{tag}V{i}" for i in layer_ids]
        self.start_layer = 0
        self.layer_num = len(layer_ids)
        self.model_layer_ids = tuple(layer_ids)


class _NoBuffers:
    """A pool of another layout: one sub-pool, its own view, no K/V tensors."""

    start_layer = 0
    layer_num = 0

    def __init__(self):
        self.layer_router = PlainPoolRouter(self, IdSpaceKind.FULL)


def _kv_layer_ids(target_ids, draft_ids):
    """kv_data_ptrs order: K target, V target, K draft, V draft."""
    return list(target_ids) + list(target_ids) + list(draft_ids) + list(draft_ids)


class TestStagingDraftKvSlots(CustomTestCase):
    def test_swa_model_ids_are_opt_in(self):
        cases = (
            (2, False, [3, 9], [1, 2, 3, 4], []),
            (2, True, [3, 9], [1, 2, 3, 4], [3, 9, 3, 9]),
            (2, True, [3], [1, 2, 3, 4], None),
            (2, True, [], [], []),
            (1, True, [], [1, 2, 3, 4], []),
        )
        for pp, label, model_ids, ptrs, expected in cases:
            with (
                self.subTest(pp=pp, label=label, model_ids=model_ids, ptrs=ptrs),
                get_context().override_server_args(model_path="dummy"),
                get_parallel().override(pp_size=pp),
            ):
                region = Region(
                    ptrs=tuple(ptrs),
                    lens=(256,) * len(ptrs),
                    item_lens=(64,) * len(ptrs),
                    kv_layout="nhd",
                )
                view = SimpleNamespace(model_layer_ids=model_ids)
                entry = SimpleNamespace(
                    sub_pool="swa", part=None, state_type=StateType.SWA
                )
                storage = SimpleNamespace(
                    owner="full",
                    view=lambda name: object() if name == "full" else view,
                    wire_entries=lambda: [entry],
                    regions=lambda *args, **kwargs: region,
                    entry_sub_pool=lambda item: "swa",
                    entry_wire_key=lambda item: "swa",
                )
                args = SimpleNamespace()
                if expected is None:
                    with self.assertRaisesRegex(RuntimeError, "model layer id"):
                        setup_state_kv_args(args, storage, label_swa_model_ids=label)
                else:
                    setup_state_kv_args(args, storage, label_swa_model_ids=label)
                    self.assertEqual(args.state_layer_ids, [expected])
                    self.assertEqual(args.state_data_ptrs, [ptrs])

    def test_global_model_ids_recover_empty_unified_wire_metadata(self):
        target = _Pool("t", [3, 9])
        draft = _Pool("d", [0, 2])
        k_buffers, v_buffers, slot_ids = build_staging_slot_metadata(
            kv_layer_ids=[],
            num_draft_entries=4,
            kv_pool=target,
            draft_kv_pool=draft,
            num_hidden_layers=62,
        )
        self.assertEqual(k_buffers, ["tK3", "tK9", "dK0", "dK2"])
        self.assertEqual(v_buffers, ["tV3", "tV9", "dV0", "dV2"])
        self.assertEqual(slot_ids, [3, 9, 62, 63, 3, 9, 62, 63])

    def test_model_ids_preserve_existing_static_gather_slot_order(self):
        target, draft = _Pool("t", [3, 9]), _Pool("d", [0, 2])
        kwargs = dict(
            kv_layer_ids=_kv_layer_ids([3, 9], [62, 63]),
            num_draft_entries=4,
            kv_pool=target,
            draft_kv_pool=draft,
        )
        legacy = build_staging_slot_metadata(**kwargs)
        kwargs["kv_layer_ids"] = []
        labelled = build_staging_slot_metadata(**kwargs, num_hidden_layers=62)
        self.assertEqual(labelled, legacy)

    def test_registered_draft_entry_count_must_match_physical_buffers(self):
        with self.assertRaisesRegex(RuntimeError, "count must match"):
            build_staging_slot_metadata(
                kv_layer_ids=[],
                num_draft_entries=4,
                kv_pool=_Pool("t", [3]),
                draft_kv_pool=_Pool("d", [0]),
                num_hidden_layers=62,
            )

    def test_complete_wire_ids_do_not_require_pool_model_ids(self):
        target, draft = _Pool("t", [3, 9]), _Pool("d", [0, 2])
        del target.model_layer_ids
        del draft.model_layer_ids
        kwargs = dict(
            kv_layer_ids=_kv_layer_ids([3, 9], [62, 63]),
            num_draft_entries=4,
            kv_pool=target,
            draft_kv_pool=draft,
        )
        legacy = build_staging_slot_metadata(**kwargs)
        self.assertEqual(
            build_staging_slot_metadata(**kwargs, num_hidden_layers=62), legacy
        )

    def test_undescribable_draft_still_yields_target_buffers(self):
        # Returning nothing here left the caller skipping set_kv_buffer_tensors
        # entirely, and staging then came up with no buffers at all.
        k_buffers, v_buffers, slot_ids = build_staging_slot_metadata(
            kv_layer_ids=_kv_layer_ids([87], [92]),
            num_draft_entries=2,
            kv_pool=_Pool("t", [87]),
            draft_kv_pool=_NoBuffers(),
        )
        self.assertEqual(k_buffers, ["tK87"])
        self.assertEqual(v_buffers, ["tV87"])
        self.assertEqual(slot_ids, [])

    def test_pool_without_contiguous_tensors_is_declined(self):
        # MLA pools have no k_buffer/v_buffer to stage; the caller relies on None
        # to skip the registration rather than register empty lists.
        self.assertIsNone(
            build_staging_slot_metadata(
                kv_layer_ids=[],
                num_draft_entries=0,
                kv_pool=_NoBuffers(),
                draft_kv_pool=None,
            )
        )


if __name__ == "__main__":
    unittest.main()
