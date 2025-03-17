import logging
import unittest
from unittest.mock import Mock

import torch

from sglang.srt.connector import create_remote_connector
from sglang.srt.connector.memkv import MemKVConnector


class TestMemKVStorageStorage(unittest.TestCase):
    def setUp(self):
        self.connector = MemKVConnector("memkv://1?size=1")  # 1 GB

    def test_initialization(self):
        self.assertEqual(self.connector.size, 1 * 1024 * 1024 * 1024)
        self.assertEqual(self.connector.total_size, 0)
        self.assertEqual(len(self.connector.tensor_cache), 0)

    def test_set_and_get(self):
        tensor = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        result = self.connector.set("key1", tensor)
        self.assertEqual(result, 0)
        self.assertEqual(len(self.connector.tensor_cache), 1)
        self.assertIn("key1", self.connector.tensor_cache)

        get_tensor = torch.zeros_like(tensor, device="cuda")
        self.connector.get("key1", get_tensor)
        self.assertTrue(torch.equal(get_tensor, tensor))

    def test_set_full_storage(self):
        large_tensor = torch.zeros(
            (1 * 1024 * 1024 * 1024 // 4), dtype=torch.float32, device="cuda"
        )

        result = self.connector.set("key1", large_tensor)
        self.assertEqual(result, 0)

        another_tensor = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        result = self.connector.set("key2", another_tensor)
        self.assertEqual(result, -1)

    def test_get_nonexistent_key(self):
        tensor = torch.zeros((3,), device="cuda")

        self.connector.get("nonexistent_key", tensor)
        self.assertTrue(torch.equal(tensor, torch.zeros((3,), device="cuda")))

    def test_get_storage_class(self):
        args = Mock()
        args.session_cache_connectors = ["memkv"]
        args.connector_memkv_size = 1
        connector = create_remote_connector("memkv://1?size=1")
        self.assertIsInstance(connector, MemKVConnector)
        self.assertEqual(connector.size, 1 * 1024 * 1024 * 1024)

        session_cache_connectors = "invalid_url"
        with self.assertRaises(ValueError):
            create_remote_connector(session_cache_connectors)


if __name__ == "__main__":
    unittest.main()
