# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""The SWA cache component must reach its allocator only through the allocator API.

There are two SWA allocators. The static `SWATokenToKVPoolAllocator` owns a
`full_to_swa_index_mapping` tensor; `UnifiedSWATokenToKVPoolAllocator` has NO
such tensor — it skips the parent `__init__` on purpose (that init allocates the
static-partition sub-pools unified memory replaces) and uses its swa sub-pool's
virtual->physical table as the mapping. Both implement
`set_full_to_swa_mapping`.

So any component line that pokes the tensor instead of calling the method works
on one allocator and raises `AttributeError` on the other. That is not a
theoretical hazard: `RecoverSWAWithLockedFull` did exactly that, and it killed
the scheduler on every hybrid-SWA model run with --enable-unified-memory and the
radix cache.

CPU-only — no GPU / Triton needed.

    python -m pytest test/registered/unit/mem_cache/test_swa_component_unified_alloc.py -v
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.unified_cache.cache_action import RecoverSWAWithLockedFull
from sglang.srt.mem_cache.unified_cache.components.swa_component import SWAComponent
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _UnifiedShapedAllocator:
    """The unified SWA allocator's SHAPE: the method, and no mapping tensor.

    Deliberately not a `SimpleNamespace` — attribute access must RAISE for
    anything the real class does not define, which is the whole point here.
    """

    def __init__(self):
        self.mapped = []
        self.freed = []
        self.full_attn_allocator = SimpleNamespace(free=self.freed.append)

    def set_full_to_swa_mapping(self, full_indices, swa_indices):
        self.mapped.append((full_indices.clone(), swa_indices.clone()))


class _StaticShapedAllocator(_UnifiedShapedAllocator):
    """The static allocator's shape: the same method, backed by a real tensor."""

    def __init__(self, size=16):
        super().__init__()
        self.full_to_swa_index_mapping = torch.arange(size, dtype=torch.int64)

    def set_full_to_swa_mapping(self, full_indices, swa_indices):
        super().set_full_to_swa_mapping(full_indices, swa_indices)
        self.full_to_swa_index_mapping[full_indices.to(torch.int64)] = swa_indices.to(
            self.full_to_swa_index_mapping.dtype
        )


def _component(alloc):
    """A bare SWAComponent with only what `apply_component_action` touches."""
    comp = object.__new__(SWAComponent)
    comp.cache = SimpleNamespace(token_to_kv_pool_allocator=alloc)
    comp.tree_core = SimpleNamespace(
        set_component_device_value=lambda *a, **kw: None,
    )
    comp._translate_full_to_swa = lambda full: full * 2  # any stable stand-in
    return comp


def _action(kept, incoming):
    return RecoverSWAWithLockedFull(
        node_id=0,
        kept_full=torch.tensor(kept, dtype=torch.int64),
        incoming_full=torch.tensor(incoming, dtype=torch.int64),
    )


class TestSWAComponentUsesAllocatorAPI(CustomTestCase):
    def test_recover_locked_full_runs_on_the_unified_allocator(self):
        """RED before the fix: the component wrote
        `alloc.full_to_swa_index_mapping[...] = 0`, which the unified allocator
        does not have, so this raised AttributeError and took the scheduler
        down mid-forward."""
        alloc = _UnifiedShapedAllocator()
        _component(alloc).apply_component_action(_action([3, 4], [7, 8]))

        # Both writes went through the method: the kept full gets its swa
        # translation, the incoming full is tombstoned to 0.
        self.assertEqual(len(alloc.mapped), 2)
        self.assertTrue(torch.equal(alloc.mapped[0][0], torch.tensor([3, 4])))
        self.assertTrue(torch.equal(alloc.mapped[1][0], torch.tensor([7, 8])))
        self.assertTrue(torch.equal(alloc.mapped[1][1], torch.tensor([0, 0])))
        # ...and only the incoming full is freed.
        self.assertEqual(len(alloc.freed), 1)
        self.assertTrue(torch.equal(alloc.freed[0], torch.tensor([7, 8])))

    def test_static_allocator_tensor_is_written_exactly_as_before(self):
        """The fix must not change the static path: the same entries end up
        zeroed, which is what the direct assignment used to do."""
        alloc = _StaticShapedAllocator(size=16)
        before = alloc.full_to_swa_index_mapping.clone()
        _component(alloc).apply_component_action(_action([3, 4], [7, 8]))

        # The component remaps the KEPT full onto the INCOMING full's swa
        # translation, then tombstones the incoming one.
        want = before.clone()
        want[torch.tensor([3, 4])] = torch.tensor([7, 8]) * 2
        want[torch.tensor([7, 8])] = 0
        self.assertTrue(torch.equal(alloc.full_to_swa_index_mapping, want))

    def test_no_raw_mapping_writes_left_in_the_component(self):
        """Enforcement scan: a future edit that pokes the tensor again would
        pass every unified unit test that does not exercise this one action,
        then crash a real hybrid-SWA server. Keep the boundary explicit."""
        import inspect

        src = inspect.getsource(SWAComponent)
        offenders = [
            line.strip()
            for line in src.splitlines()
            if "full_to_swa_index_mapping" in line and not line.strip().startswith("#")
        ]
        self.assertEqual(
            offenders,
            [],
            "SWAComponent must reach the mapping through "
            "`set_full_to_swa_mapping`; the unified allocator has no tensor:\n  "
            + "\n  ".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
