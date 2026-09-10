"""Unit tests for fail-closed C++ radix-cache request validation."""

import importlib
import sys
import types
import unittest
from array import array
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=60, suite="base-a-test-cpu")


class TestRadixCacheCpp(CustomTestCase):
    @contextmanager
    def _import_with_fake_extension(self):
        extension_name = "sglang.srt.mem_cache.cpp_radix_tree.radix_tree"
        module_name = "sglang.srt.mem_cache.radix_cache_cpp"
        fake_extension = types.ModuleType(extension_name)
        fake_extension.IOHandle = object
        fake_extension.RadixTreeCpp = object
        fake_extension.TreeNodeCpp = object

        original_module = sys.modules.pop(module_name, None)
        try:
            with patch.dict(sys.modules, {extension_name: fake_extension}):
                yield importlib.import_module(module_name)
        finally:
            sys.modules.pop(module_name, None)
            if original_module is not None:
                sys.modules[module_name] = original_module

    def test_cache_unfinished_req_uses_combined_insert_and_match(self):
        with self._import_with_fake_extension() as module:
            cache = module.RadixCacheCpp.__new__(module.RadixCacheCpp)
            cache.cache_controller = None
            cache.device = torch.device("cpu")
            cache.page_size = 1
            cache.tree = MagicMock()
            old_node, new_node = 7, 9
            cache.tree.writing_through_and_match_prefix.return_value = (
                [],
                2,
                [torch.tensor([10, 11]), torch.tensor([12, 13])],
                new_node,
            )
            cache.token_to_kv_pool_allocator = SimpleNamespace(free=MagicMock())
            cache.req_to_token_pool = SimpleNamespace(
                req_to_token=torch.tensor([[10, 101, 102, 103]])
            )
            req = SimpleNamespace(
                cache_salt=None,
                extra_key=None,
                kv=SimpleNamespace(holds_kv=True, req_pool_idx=0),
                get_fill_ids=MagicMock(return_value=array("q", [1, 2, 3, 4])),
                prefix_indices=torch.tensor([10]),
                last_node=old_node,
            )

            cache.cache_unfinished_req(req)

            cache.tree.writing_through_and_match_prefix.assert_called_once()
            key_arg, indices_arg = (
                cache.tree.writing_through_and_match_prefix.call_args.args
            )
            self.assertEqual(key_arg, array("q", [1, 2, 3, 4]))
            self.assertTrue(torch.equal(indices_arg, torch.tensor([10, 101, 102, 103])))
            cache.tree.writing_through.assert_not_called()
            cache.tree.match_prefix.assert_not_called()
            self.assertEqual(
                cache.tree.lock_ref.call_args_list,
                [call(old_node, False), call(new_node, True)],
            )
            freed = cache.token_to_kv_pool_allocator.free.call_args.args[0]
            self.assertTrue(torch.equal(freed, torch.tensor([101])))
            self.assertTrue(
                torch.equal(
                    cache.req_to_token_pool.req_to_token[0],
                    torch.tensor([10, 11, 102, 103]),
                )
            )
            self.assertTrue(
                torch.equal(req.prefix_indices, torch.tensor([10, 11, 12, 13]))
            )
            self.assertEqual(req.last_node, new_node)

    def test_cache_salt_is_rejected_without_loading_cpp_extension(self):
        with self._import_with_fake_extension() as module:
            module.RadixCacheCpp._reject_cache_salt(None)
            with self.assertRaisesRegex(ValueError, "experimental C\\+\\+"):
                module.RadixCacheCpp._reject_cache_salt("tenant-a")


class TestRadixTreeCppFusedInsertMatch(CustomTestCase):
    @staticmethod
    def _flatten(indices):
        return torch.cat(indices) if indices else torch.empty(0, dtype=torch.int64)

    def test_fused_insert_match_semantics(self):
        extension = importlib.import_module(
            "sglang.srt.mem_cache.cpp_radix_tree.radix_tree"
        )
        legacy_tree = extension.RadixTreeCpp(False, None, 1, 2)
        legacy_actions, legacy_match = legacy_tree.writing_through(
            [8, 9], torch.tensor([80, 90])
        )
        self.assertEqual((legacy_actions, legacy_match), ([], 0))
        self.assertTrue(
            torch.equal(
                self._flatten(legacy_tree.match_prefix([8, 9])[0]),
                torch.tensor([80, 90]),
            )
        )

        tree = extension.RadixTreeCpp(False, None, 1, 2)

        actions, matched, indices, first_node = tree.writing_through_and_match_prefix(
            [1, 2, 3, 4], torch.tensor([10, 11, 12, 13])
        )
        self.assertEqual(actions, [])
        self.assertEqual(matched, 0)
        self.assertTrue(
            torch.equal(self._flatten(indices), torch.tensor([10, 11, 12, 13]))
        )
        standalone_indices, host_length, device_node, host_node = tree.match_prefix(
            [1, 2, 3, 4]
        )
        self.assertEqual(
            (host_length, device_node, host_node), (0, first_node, first_node)
        )
        self.assertTrue(
            torch.equal(self._flatten(standalone_indices), self._flatten(indices))
        )

        _, matched, duplicate_indices, duplicate_node = (
            tree.writing_through_and_match_prefix(
                [1, 2, 3, 4], torch.tensor([20, 21, 22, 23])
            )
        )
        self.assertEqual((matched, duplicate_node), (4, first_node))
        self.assertTrue(
            torch.equal(
                self._flatten(duplicate_indices), torch.tensor([10, 11, 12, 13])
            )
        )

        _, matched, branch_indices, branch_node = tree.writing_through_and_match_prefix(
            [1, 2, 5, 6], torch.tensor([30, 31, 32, 33])
        )
        self.assertEqual(matched, 2)
        self.assertNotEqual(branch_node, first_node)
        self.assertTrue(
            torch.equal(self._flatten(branch_indices), torch.tensor([10, 11, 32, 33]))
        )
        self.assertEqual(tree.total_size(), 6)

        paged_tree = extension.RadixTreeCpp(False, None, 2, 2)
        _, matched, paged_indices, _ = paged_tree.writing_through_and_match_prefix(
            [1, 2, 3, 4, 5], torch.tensor([40, 41, 42, 43, 44])
        )
        self.assertEqual(matched, 0)
        self.assertTrue(
            torch.equal(self._flatten(paged_indices), torch.tensor([40, 41, 42, 43]))
        )
        self.assertEqual(paged_tree.total_size(), 4)

        disabled_tree = extension.RadixTreeCpp(True, None, 1, 2)
        actions, matched, indices, node = (
            disabled_tree.writing_through_and_match_prefix(
                [1], torch.tensor([1], dtype=torch.int64)
            )
        )
        self.assertEqual((actions, matched, indices, node), ([], 0, [], 0))


if __name__ == "__main__":
    unittest.main()
