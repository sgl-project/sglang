"""`MambaPoolHost` must notice an envelope-strided device state view.

Every state-transfer kernel in `pool_host/mamba.py` addresses a slot as
``ptr + index * item_size``, i.e. it assumes the slot stride equals the slot's
own size. The unified memory pool stores conv/SSM state ENVELOPE-strided: one
slot's stride spans every state tensor of every layer, so slot `i` does not
start at `i * numel_per_slot`. The mis-addressing stays inside the buffer, so
it corrupts silently instead of faulting -- which is why the predicate that
routes those views through the contiguous staging path is worth pinning.

    python -m pytest test/registered/unit/mem_cache/test_unified_hicache_strided_state.py -v
"""

import unittest

import torch

from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


class TestStridedStateDetection(CustomTestCase):
    def test_contiguous_slots_are_not_strided(self):
        """A plain per-slot array is what the kernels already handle."""
        for shape in ((8, 4), (8, 4, 3), (1, 5)):
            with self.subTest(shape=shape):
                self.assertFalse(
                    MambaPoolHost._slots_are_strided(torch.zeros(shape)),
                    "a contiguous slot array must take the direct path",
                )

    def test_envelope_strided_slots_are_detected(self):
        """One slot's stride spanning a wider envelope is the unified layout."""
        num_slots, per_slot, envelope = 6, 4, 10
        raw = torch.zeros(num_slots * envelope)
        view = torch.as_strided(raw, size=(num_slots, per_slot), stride=(envelope, 1))
        self.assertTrue(MambaPoolHost._slots_are_strided(view))

    def test_empty_tensor_is_not_strided(self):
        """No slots, nothing to address -- must not divide by or index slot 0."""
        self.assertFalse(MambaPoolHost._slots_are_strided(torch.zeros((0, 4))))

    def test_staging_round_trip_preserves_slot_contents(self):
        """The property the staging path relies on: gathering the wanted slots
        out of a strided view and scattering them back is the identity, so the
        kernel can run against a contiguous copy in between."""
        num_slots, per_slot, envelope = 6, 4, 10
        raw = torch.arange(num_slots * envelope, dtype=torch.float32)
        view = torch.as_strided(raw, size=(num_slots, per_slot), stride=(envelope, 1))
        indices = torch.tensor([4, 1, 3])

        staged = view.index_select(0, indices)
        self.assertTrue(staged.is_contiguous())
        for row, slot in enumerate(indices.tolist()):
            self.assertTrue(torch.equal(staged[row], view[slot]))

        dst = torch.zeros_like(raw)
        dst_view = torch.as_strided(
            dst, size=(num_slots, per_slot), stride=(envelope, 1)
        )
        dst_view.index_copy_(0, indices, staged)
        for slot in indices.tolist():
            self.assertTrue(torch.equal(dst_view[slot], view[slot]))
        # Slots outside the index set must be untouched, or a partial backup
        # would clobber a neighbour's envelope.
        for slot in set(range(num_slots)) - set(indices.tolist()):
            self.assertTrue(torch.all(dst_view[slot] == 0))


if __name__ == "__main__":
    unittest.main()


class TestMambaSlotWiringIsShared(CustomTestCase):
    """Wrapping the mamba end in the slot allocator and installing its v2p
    translate must happen together, in one place.

    HiCache holds VIRTUAL slot ids while the state pool is a pure PHYSICAL
    store, so `L2TransferEngine` applies `host_transfer_translate` just before
    each transfer. A factory that wraps the allocator without installing the
    translate hands raw virtual ids to that store -- which reads correctly
    until the first compaction moves a slot, and then silently transfers the
    wrong state. That is what happened to the tri-pool factory: it grew its own
    copy of the wrapping and never got the translate.

    AST-level because the factories build real pools and need a GPU.
    """

    @staticmethod
    def _assignments_to(attr: str):
        """Enclosing function name for every `<x>.mamba_allocator = <allocator>`
        in the unified pool module.

        `= None` is excluded: declaring the slot up front is what lets the rest
        of the code do a None check instead of a defensive `getattr`, so it is
        the opposite of a second wrapping site.
        """
        import ast
        import inspect

        from sglang.srt.mem_cache import unified_memory_pool

        tree = ast.parse(inspect.getsource(unified_memory_pool))
        found = []
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(fn):
                if not isinstance(node, ast.Assign):
                    continue
                if isinstance(node.value, ast.Constant) and node.value.value is None:
                    continue
                for tgt in node.targets:
                    if isinstance(tgt, ast.Attribute) and tgt.attr == attr:
                        found.append(fn.name)
        return found

    def test_only_the_shared_hook_wraps_the_slot_allocator(self):
        self.assertEqual(
            sorted(set(self._assignments_to("mamba_allocator"))),
            ["_wire_mamba_slot_allocator"],
            "a factory wraps the mamba end itself; it will miss "
            "host_transfer_translate exactly as the tri-pool factory did",
        )

    def test_every_unified_mamba_factory_calls_the_shared_hook(self):
        import ast
        import inspect

        from sglang.srt.mem_cache import unified_memory_pool

        tree = ast.parse(inspect.getsource(unified_memory_pool))
        # Factories whose composite holds a mamba end.
        expected = {"init_unified_mamba_pools", "init_unified_mamba_swa_pools"}
        callers = {
            fn.name
            for fn in ast.walk(tree)
            if isinstance(fn, ast.FunctionDef)
            for node in ast.walk(fn)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_wire_mamba_slot_allocator"
        }
        self.assertEqual(expected, expected & callers, f"missing: {expected - callers}")
