"""Unit tests for C++ radix-cache request handling."""

import importlib
import sys
import types
import unittest
from array import array
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


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

    def test_cache_salt_is_rejected_without_loading_cpp_extension(self):
        with self._import_with_fake_extension() as module:
            module.RadixCacheCpp._reject_cache_salt(None)
            with self.assertRaisesRegex(ValueError, "experimental C\\+\\+"):
                module.RadixCacheCpp._reject_cache_salt("tenant-a")

    def test_skip_insert_keeps_progress_request_owned(self):
        """Skipped publication must advance progress and free only request-owned KV."""
        with self._import_with_fake_extension() as module:
            kv_indices = torch.tensor([[401, 402, 403, 404, 405, 406]])
            cache = module.RadixCacheCpp.__new__(module.RadixCacheCpp)
            cache.page_size = 1
            cache.req_to_token_pool = SimpleNamespace(req_to_token=kv_indices)
            cache.token_to_kv_pool_allocator = MagicMock()
            cache._insert = MagicMock()
            cache.dec_lock_ref = MagicMock()

            last_node = object()
            req = SimpleNamespace(
                cache_salt=None,
                req_pool_idx=0,
                prefix_indices=kv_indices[0, :2].clone(),
                cache_protected_len=2,
                origin_input_ids=array("q", [1, 2, 3, 4, 5, 6]),
                output_ids=array("q"),
                extra_key=None,
                last_node=last_node,
                get_fill_ids=lambda: array("q", [1, 2, 3, 4, 5, 6]),
            )

            cache.cache_unfinished_req(req, chunked=True, is_insert=False)

            torch.testing.assert_close(req.prefix_indices, kv_indices[0])
            self.assertEqual(req.cache_protected_len, 2)
            cache._insert.assert_not_called()

            cache.cache_finished_req(req, is_insert=False, kv_len_to_handle=6)

            cache.token_to_kv_pool_allocator.free.assert_called_once()
            torch.testing.assert_close(
                cache.token_to_kv_pool_allocator.free.call_args.args[0],
                kv_indices[0, 2:],
            )
            cache.dec_lock_ref.assert_called_once_with(last_node)


if __name__ == "__main__":
    unittest.main()
