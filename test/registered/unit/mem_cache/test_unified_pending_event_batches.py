"""Preserve all moved sources until their forward event can release them."""

import itertools
import unittest
from types import SimpleNamespace

import torch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from unified_allocator_fixtures import (
    build_swa_pool,
    build_tri_pool,
    reset_context,
    setup_allocator_context,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _Event:
    def __init__(self):
        self.fired = False

    def query(self):
        return self.fired


def _views(bundle, allocator, slots, states):
    result = []
    if slots is not None:
        kv = bundle.token_to_kv_pool
        for layer, member in enumerate(
            (allocator.full_attn_allocator, allocator.swa_attn_allocator)
        ):
            for value in (False, True):
                buf = kv.get_value_buffer(layer) if value else kv.get_key_buffer(layer)
                result.append((member, buf, slots, layer * 2 + int(value)))
    if states is not None:
        cache = bundle.req_to_token_pool.mamba_pool.mamba_cache
        result.extend(
            (allocator.mamba_allocator, view[0], states, 4 + index)
            for index, view in enumerate([*cache.conv, cache.temporal])
        )
    return result


def _payload(ids, buffer, tag, generation=0):
    width = buffer[0].numel()
    digits = min(width, 8)
    ordinal = (generation * 4096 + ids.long()) * 16 + tag
    assert ordinal.max() < 16**digits
    data = torch.arange(width).expand(len(ids), width).clone() + 32
    data[:, :digits] = ordinal[:, None] // (16 ** torch.arange(digits)) % 16
    encoded = data.to(buffer.dtype)
    assert torch.equal(encoded.long(), data)
    return encoded.reshape(len(ids), *buffer.shape[1:])


class TestUnifiedPendingEventBatches(CustomTestCase):
    def test_multiple_flushes_preserve_event_sources(self):
        # Events model reader completion only; these CPU checks make no CUDA
        # stream-ordering claim. The allocator selects the actual move geometry.
        for (layout, owner), page_size, mode in itertools.product(
            ((2, "full"), (3, "full"), (3, "mamba")),
            (1, 4, 16),
            ("same", "different", "fired_between"),
        ):
            with self.subTest(
                layout=layout, owner=owner, page_size=page_size, mode=mode
            ):
                try:
                    setup_allocator_context()
                    if layout == 3:
                        bundle, allocator, _ = build_tri_pool(
                            lazy=True,
                            page_size=page_size,
                            temporal=(1, 4, 8),
                            state_cache=False,
                        )
                        states = allocator.mamba_allocator.alloc(8)
                        self.assertIsNotNone(states)
                    else:
                        allocator, _ = build_swa_pool(
                            occupancy=0, lazy=True, page_size=page_size
                        )
                        bundle = SimpleNamespace(
                            token_to_kv_pool=allocator.get_kvcache()
                        )
                        states = None
                    slots = allocator.alloc(80 * page_size)
                    self.assertIsNotNone(slots)
                    retained_slots = slots
                    retained_states = states
                    for member, buf, ids, tag in _views(
                        bundle, allocator, slots, states
                    ):
                        buf[member.translate_kv_loc_for_kernel(ids)] = _payload(
                            ids, buf, tag
                        )
                    member = (
                        allocator.full_attn_allocator
                        if owner == "full"
                        else allocator.mamba_allocator
                    )
                    first, second = _Event(), _Event()
                    expected = {}
                    all_sources = set()

                    def assert_pending():
                        self.assertEqual(set(member._pending_reuse), set(expected))
                        union = set().union(*expected.values()) if expected else set()
                        self.assertEqual(member._pending_reuse_pages_cpu, union)
                        free = member._free_phys_pages.tolist()
                        self.assertTrue(union.isdisjoint(free))
                        self.assertEqual(len(free), len(set(free)))
                        for event, sources in expected.items():
                            cpu, device = member._pending_reuse[event]
                            self.assertEqual(set(cpu), sources)
                            self.assertEqual(len(cpu), len(sources))
                            self.assertEqual(device.tolist(), cpu)

                    def assert_payload():
                        for current, buf, ids, tag in _views(
                            bundle, allocator, retained_slots, retained_states
                        ):
                            self.assertTrue(
                                torch.equal(
                                    buf[current.translate_kv_loc_for_kernel(ids)],
                                    _payload(ids, buf, tag),
                                ),
                                (current.sub_pool_name, tag),
                            )
                        self.assertEqual(allocator.verify_byte_accounting(), [])

                    for index in (0, 1):
                        event = second if mode == "different" and index else first
                        if index and mode == "fired_between":
                            first.fired = True
                            expected.clear()
                        member.set_latest_forward_done_event(event)
                        member.set_inflight_forward(event, None)
                        if owner == "full":
                            drop = slots[
                                (10 + index * 10) * page_size : (11 + index * 10)
                                * page_size
                            ]
                            allocator.free(drop)
                            retained_slots = retained_slots[
                                ~torch.isin(retained_slots, drop)
                            ]
                        else:
                            drop = states[1 + index * 2 : 2 + index * 2]
                            member.free(drop)
                            retained_states = retained_states[
                                ~torch.isin(retained_states, drop)
                            ]
                        before = member.virtual_to_physical.clone()
                        self.assertGreater(member._flush(urgent=False), 0)
                        after = member.virtual_to_physical
                        moved = (
                            (before >= member.min_page_index)
                            & (after >= member.min_page_index)
                            & (before != after)
                        )
                        # Derive sources from retained virtual mappings, independently
                        # of either pending representation being checked.
                        sources = set(before[moved].tolist())
                        self.assertTrue(sources)
                        all_sources.update(sources)
                        if not event.fired:
                            expected.setdefault(event, set()).update(sources)
                        assert_pending()
                        assert_payload()

                    member._drain_pending_reuse(urgent=False)
                    assert_pending()
                    for event in (second, first):
                        event.fired = True
                        expected.pop(event, None)
                        member._drain_pending_reuse(urgent=False)
                        assert_pending()
                        assert_payload()
                    self.assertFalse(member._pending_reuse)
                    self.assertFalse(member._pending_reuse_pages_cpu)
                    if mode != "fired_between":
                        self.assertEqual(
                            set(member._free_phys_pages.tolist()), all_sources
                        )
                        if owner == "full":
                            fresh = allocator.alloc(2 * page_size)
                            self.assertIsNotNone(fresh)
                            fresh_views = _views(bundle, allocator, fresh, None)
                            self.assertFalse(torch.isin(fresh, retained_slots).any())
                        else:
                            fresh = member.alloc(2)
                            self.assertIsNotNone(fresh)
                            fresh_views = _views(bundle, allocator, None, fresh)
                            self.assertFalse(torch.isin(fresh, retained_states).any())
                        self.assertIsNotNone(fresh)
                        self.assertEqual(
                            set(
                                member.virtual_to_physical[
                                    fresh // member.pool_page_size
                                ].tolist()
                            ),
                            all_sources,
                        )
                        for current, buf, ids, tag in fresh_views:
                            buf[current.translate_kv_loc_for_kernel(ids)] = _payload(
                                ids, buf, tag, generation=1
                            )
                        assert_payload()
                finally:
                    reset_context()


if __name__ == "__main__":
    unittest.main()
