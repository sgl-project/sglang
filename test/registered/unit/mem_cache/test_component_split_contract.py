"""The node-split contract every tree component must satisfy."""

import dataclasses
import unittest
from array import array
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_factory import (
    DEFAULT_COMPONENT_FACTORY_KEYS,
)
from sglang.srt.mem_cache.unified_cache.components.base import (
    ComponentData,
    ComponentType,
)
from sglang.srt.mem_cache.unified_cache.components.registry import (
    get_python_tree_component,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedTreeNode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestComponentSplitFieldCoverage(unittest.TestCase):
    """`redistribute_on_node_split` must decide every ComponentData field.

    Each component hand-writes this hook, and the recurring bug is a field
    nobody remembered: SWA dropped `host_lock_ref` (#38138) and FULL dropped it
    again (#38480), in both cases letting an eviction reclaim a host slice an
    in-flight L3 write was still reading. The table below is the decision
    record; adding a field to ComponentData without classifying it here fails.
    """

    # How the new parent must end up, per field:
    #   copy    inherit the child's value (state covering the whole path)
    #   divide  the child's tensor splits; the parent takes the head
    #   reset   stay at the field default (state that belongs to the leaf)
    SPLIT_POLICY = {
        ComponentType.FULL: {
            "value": "divide",
            "host_value": "divide",
            "lock_ref": "copy",
            "host_lock_ref": "copy",
            "session_ref": "copy",
            "session_ids": "reset",
            "metadata": "migrate",
        },
        ComponentType.SWA: {
            "value": "divide",
            "host_value": "divide",
            "lock_ref": "copy",
            "host_lock_ref": "copy",
            "session_ref": "copy",
            "session_ids": "reset",
            "metadata": "migrate",
        },
        # Mamba state is attached to one exact node, so a prefix parent owns
        # none of it.
        ComponentType.MAMBA: {
            "value": "reset",
            "host_value": "reset",
            "lock_ref": "reset",
            "host_lock_ref": "reset",
            "session_ref": "reset",
            "session_ids": "reset",
            "metadata": "stay",
        },
    }

    # Segment-boundary uuids mark a node's older edge, which the split moves to
    # the parent; "stay" components keep whatever they had on the child.
    MIGRATED_METADATA_KEYS = {
        ComponentType.FULL: {"host_uuid"},
        ComponentType.SWA: {"uuid", "host_uuid"},
        ComponentType.MAMBA: set(),
    }

    SPLIT_LEN = 2
    CHILD_LEN = 5

    def test_policy_covers_every_component_and_field(self):
        self.assertEqual(
            set(self.SPLIT_POLICY),
            set(DEFAULT_COMPONENT_FACTORY_KEYS),
            "a default component has no split policy: decide what "
            "its split does with each ComponentData field and record it here",
        )
        field_names = {f.name for f in dataclasses.fields(ComponentData)}
        for component_type, policy in self.SPLIT_POLICY.items():
            self.assertEqual(
                set(policy),
                field_names,
                f"{component_type.name}: ComponentData fields and the split "
                "policy disagree. A new field must be classified copy / divide "
                "/ reset / migrate / stay, and the hook must implement it",
            )
            self.assertIn(component_type, self.MIGRATED_METADATA_KEYS)

    def _marked_child(self, component_type):
        """A child node whose every ComponentData field is distinguishable."""
        node = UnifiedTreeNode((component_type,))
        node.key = RadixKey(array("q", range(self.CHILD_LEN)))
        cd = node.component_data[component_type]
        cd.value = torch.arange(self.CHILD_LEN, dtype=torch.int64)
        cd.host_value = torch.arange(100, 100 + self.CHILD_LEN, dtype=torch.int64)
        cd.lock_ref = 3
        cd.host_lock_ref = 2
        cd.session_ref = 4
        cd.session_ids = None
        cd.metadata = {"uuid": 7, "host_uuid": 9, "unrelated": "keep"}
        return node

    def _split(self, component_type):
        """Run the real hook over a fresh parent/child pair."""
        component = object.__new__(
            get_python_tree_component(DEFAULT_COMPONENT_FACTORY_KEYS[component_type])
        )
        # The hooks only reach the tree core for the host LRU, and the marked
        # child keeps both halves out of it (device value present, host lock held).
        component.tree_core = mock.Mock(
            host_lru_lists={ct: mock.Mock() for ct in DEFAULT_COMPONENT_FACTORY_KEYS}
        )
        component.cache = SimpleNamespace(tree_core=component.tree_core)
        child = self._marked_child(component_type)
        # The hook pops migrated keys out of the child's dict, which
        # dataclasses.replace() would share with the snapshot.
        before = dataclasses.replace(
            child.component_data[component_type],
            metadata=dict(child.component_data[component_type].metadata),
        )
        parent = UnifiedTreeNode((component_type,))
        parent.key = RadixKey(array("q", range(self.SPLIT_LEN)))
        component.redistribute_on_node_split(parent, child)
        return (
            before,
            parent.component_data[component_type],
            child.component_data[component_type],
        )

    def test_hook_follows_its_policy(self):
        defaults = ComponentData()
        for component_type, policy in self.SPLIT_POLICY.items():
            with self.subTest(component=component_type.name):
                before, parent_cd, child_cd = self._split(component_type)
                for name, rule in policy.items():
                    if name == "metadata":
                        continue  # checked key by key below
                    got = getattr(parent_cd, name)
                    if rule == "copy":
                        self.assertEqual(
                            got,
                            getattr(before, name),
                            f"{component_type.name}.{name} must reach the parent: "
                            "a holder of the child holds the parent's range too",
                        )
                    elif rule == "divide":
                        self.assertIsNotNone(got, f"{component_type.name}.{name}")
                        self.assertTrue(
                            torch.equal(got, getattr(before, name)[: self.SPLIT_LEN])
                        )
                        self.assertTrue(
                            torch.equal(
                                getattr(child_cd, name),
                                getattr(before, name)[self.SPLIT_LEN :],
                            )
                        )
                    elif rule == "reset":
                        self.assertEqual(
                            got,
                            getattr(defaults, name),
                            f"{component_type.name}.{name} must not reach the parent",
                        )
                    else:
                        self.fail(f"unknown split rule {rule!r} for {name}")

    def test_metadata_boundary_uuids_migrate(self):
        for component_type, migrated in self.MIGRATED_METADATA_KEYS.items():
            with self.subTest(component=component_type.name):
                before, parent_cd, child_cd = self._split(component_type)
                for key in migrated:
                    self.assertEqual(
                        parent_cd.metadata.get(key),
                        before.metadata[key],
                        f"{component_type.name}: the {key} boundary must move to "
                        "the parent, or the paired release walks past its segment",
                    )
                    self.assertNotIn(
                        key,
                        child_cd.metadata,
                        f"{component_type.name}: {key} must not stay on the child too",
                    )
                for key in set(before.metadata) - migrated:
                    self.assertEqual(child_cd.metadata.get(key), before.metadata[key])


if __name__ == "__main__":
    unittest.main()
