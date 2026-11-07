"""CPU-only regression for the misaligned Mamba checkpoint attach (#38815).

A shared insert truncates the key to the smallest component boundary (e.g.
the SWA branch), while the Mamba value was produced at its own later track
seqlen. Stamping that value on the shorter leaf attaches a later state to an
earlier key. The component now skips the stamp, and cleanup treats the value
as not-inserted.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import Mock

import torch

from sglang.srt.mem_cache.base_prefix_cache import InsertParams, InsertResult
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.mamba import (
    MambaComponent,
)
from sglang.srt.mem_cache.unified_cache.unified_tree_core import (
    UnifiedLRUList,
    UnifiedTreeNode,
)


class _FakeTreeCore:
    tree_components = (ComponentType.FULL, ComponentType.MAMBA)
    is_eagle = False

    def __init__(self):
        self.root_node = UnifiedTreeNode(self.tree_components)
        self.component_evictable_size_ = {ComponentType.MAMBA: 0}
        self.lru_lists = {
            ComponentType.MAMBA: UnifiedLRUList(
                ComponentType.MAMBA, self.tree_components
            )
        }


def _component():
    core = _FakeTreeCore()
    component = object.__new__(MambaComponent)
    component.tree_core = core
    component.mamba_max_states_per_path = 1000  # no cap eviction in these tests
    return component, core


class TestMambaValueAlignment(unittest.TestCase):
    def _params(self, key_len, seqlen):
        params = InsertParams(
            key=RadixKey(list(range(key_len))),
            mamba_value=torch.tensor([7]),
            mamba_value_seqlen=seqlen,
        )
        return params

    def test_misaligned_value_is_not_stamped(self):
        component, core = _component()
        node = UnifiedTreeNode(core.tree_components)
        result = InsertResult(prefix_len=0)

        component.commit_insert_component_data(
            node, True, self._params(key_len=96, seqlen=192), result, []
        )

        assert node.component_data[ComponentType.MAMBA].value is None
        assert core.component_evictable_size_[ComponentType.MAMBA] == 0

    def test_misaligned_host_value_is_not_stamped(self):
        component, core = _component()
        node = UnifiedTreeNode(core.tree_components)
        result = InsertResult(prefix_len=0)
        params = self._params(key_len=96, seqlen=192)
        params.mamba_value = None
        params.mamba_host_value = torch.tensor([11])

        component.commit_insert_component_data(node, True, params, result, [])

        assert node.component_data[ComponentType.MAMBA].host_value is None

    def test_misaligned_host_value_is_released(self):
        component, _ = _component()
        result = InsertResult(prefix_len=0)
        params = self._params(key_len=96, seqlen=192)
        params.mamba_value = None
        params.mamba_host_value = torch.tensor([11])
        component.free_host_values = Mock()

        component._release_unused_host_checkpoint(result, params)

        component.free_host_values.assert_called_once()
        freed = component.free_host_values.call_args.args[0]
        torch.testing.assert_close(freed[0], torch.tensor([11]))

    def test_misaligned_host_source_skips_the_device_to_host_copy(self):
        component, _ = _component()
        params = self._params(key_len=96, seqlen=192)
        params.mamba_value = None
        params.mamba_host_src = torch.tensor([11])
        component._destination_has_mamba = Mock(return_value=False)
        component._copy_mamba_slot_to_host = Mock()

        reused = component.materialize_host_only_checkpoint(params, params.key)

        assert not reused
        assert params.mamba_host_src is None
        component._copy_mamba_slot_to_host.assert_not_called()

    def test_aligned_value_is_stamped(self):
        component, core = _component()
        node = UnifiedTreeNode(core.tree_components)
        result = InsertResult(prefix_len=0)

        component.commit_insert_component_data(
            node, True, self._params(key_len=192, seqlen=192), result, []
        )

        torch.testing.assert_close(
            node.component_data[ComponentType.MAMBA].value, torch.tensor([7])
        )
        assert core.component_evictable_size_[ComponentType.MAMBA] == 1

    def test_untracked_value_keeps_legacy_behavior(self):
        # No seqlen recorded (non-tracked inserts): no alignment check fires.
        component, core = _component()
        node = UnifiedTreeNode(core.tree_components)
        result = InsertResult(prefix_len=0)
        params = InsertParams(
            key=RadixKey(list(range(96))), mamba_value=torch.tensor([7])
        )

        component.commit_insert_component_data(node, True, params, result, [])

        torch.testing.assert_close(
            node.component_data[ComponentType.MAMBA].value, torch.tensor([7])
        )


if __name__ == "__main__":
    unittest.main()
