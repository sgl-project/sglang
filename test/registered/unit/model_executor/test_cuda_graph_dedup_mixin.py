"""CPU coverage for CUDA graph executable dedup lifecycle."""

import unittest
from unittest.mock import Mock, call, patch

import sglang.srt.model_executor.runner_backend.cuda_graph_dedup_mixin as dedup_module
from sglang.srt.model_executor.runner_backend.cuda_graph_dedup_mixin import (
    DedupedCudaGraphRegistry,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _CapturedGraph:
    def __init__(self, raw_graph):
        self._raw_graph = raw_graph
        self.reset = Mock()

    def raw_cuda_graph(self):
        return self._raw_graph


class TestDedupedCudaGraphRegistry(CustomTestCase):
    def test_compat_exec_is_lazy_until_a_duplicate_graph_arrives(self):
        registry = DedupedCudaGraphRegistry()
        registry.instantiate = Mock(side_effect=[1001, 1002])

        with (
            patch.object(dedup_module, "graph_signature", return_value=("same",)),
            patch.object(
                dedup_module, "dedup_update", return_value=(True, "")
            ) as update,
        ):
            first = registry.register(_CapturedGraph(101))

            self.assertEqual(registry.instantiate.call_args_list, [call(101)])
            self.assertIsNone(first.group.compat_exec)

            second = registry.register(_CapturedGraph(102))
            third = registry.register(_CapturedGraph(103))

        self.assertIs(first.group, second.group)
        self.assertIs(first.group, third.group)
        self.assertEqual(
            registry.instantiate.call_args_list,
            [call(101), call(101)],
        )
        self.assertEqual(first.group.compat_exec, 1002)
        self.assertEqual(
            update.call_args_list,
            [call(1002, 102), call(1002, 103)],
        )

    def test_unique_signatures_never_allocate_compat_execs(self):
        registry = DedupedCudaGraphRegistry()
        registry.instantiate = Mock(side_effect=[1001, 1002])

        with patch.object(
            dedup_module,
            "graph_signature",
            side_effect=[("first",), ("second",)],
        ):
            first = registry.register(_CapturedGraph(101))
            second = registry.register(_CapturedGraph(102))

        self.assertEqual(
            registry.instantiate.call_args_list,
            [call(101), call(102)],
        )
        self.assertIsNone(first.group.compat_exec)
        self.assertIsNone(second.group.compat_exec)

    def test_seal_destroys_only_allocated_compat_execs(self):
        registry = DedupedCudaGraphRegistry()
        registry.instantiate = Mock(side_effect=[1001, 1002, 1003])
        registry.destroy_exec = Mock()

        with (
            patch.object(
                dedup_module,
                "graph_signature",
                side_effect=[("singleton",), ("duplicate",), ("duplicate",)],
            ),
            patch.object(dedup_module, "dedup_update", return_value=(True, "")),
        ):
            singleton = registry.register(_CapturedGraph(101))
            duplicate = registry.register(_CapturedGraph(102))
            registry.register(_CapturedGraph(103))

        registry.seal()

        registry.destroy_exec.assert_called_once_with(1003)
        self.assertIsNone(singleton.group.compat_exec)
        self.assertIsNone(duplicate.group.compat_exec)

    def test_singleton_close_destroys_only_the_persistent_exec(self):
        registry = DedupedCudaGraphRegistry()
        registry.instantiate = Mock(return_value=1001)
        registry.destroy_exec = Mock()
        captured = _CapturedGraph(101)

        with patch.object(dedup_module, "graph_signature", return_value=("singleton",)):
            graph = registry.register(captured)

        registry.close()
        registry.close()

        registry.destroy_exec.assert_called_once_with(1001)
        captured.reset.assert_called_once_with()
        self.assertIsNone(graph.group)
        self.assertEqual(registry.groups, {})

    def test_failed_first_compatibility_check_remains_closeable(self):
        registry = DedupedCudaGraphRegistry()
        registry.instantiate = Mock(side_effect=[1001, 1002])
        registry.destroy_exec = Mock()
        first_captured = _CapturedGraph(101)
        rejected_captured = _CapturedGraph(102)

        with (
            patch.object(dedup_module, "graph_signature", return_value=("same",)),
            patch.object(
                dedup_module,
                "dedup_update",
                return_value=(False, "incompatible"),
            ),
        ):
            first = registry.register(first_captured)
            group = first.group
            with self.assertRaisesRegex(AssertionError, "incompatible"):
                registry.register(rejected_captured)

        self.assertEqual(group.graphs, [first])
        self.assertEqual(group.compat_exec, 1002)

        registry.close()

        self.assertEqual(
            registry.destroy_exec.call_args_list,
            [call(1002), call(1001)],
        )
        first_captured.reset.assert_called_once_with()
        self.assertIsNone(first.group)


if __name__ == "__main__":
    unittest.main()
