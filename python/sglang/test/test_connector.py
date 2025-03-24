import shutil
import unittest

import torch

from sglang.srt.connector import create_remote_connector


class TestConnector(unittest.TestCase):
    def test_file_connector(self):
        path = "/tmp/sgl_file_connector_test"
        connector = create_remote_connector(f"file://{path}")

        original_value = torch.randn(10, 10)
        connector.set("test", original_value)
        value = connector.get("test")
        print(value, original_value)
        self.assertTrue(torch.allclose(value, original_value))

        original_str = "file_connector"
        connector.setstr("test_str", original_str)
        value = connector.getstr("test_str")
        self.assertEqual(value, original_str)

        key_list = connector.list("*")
        self.assertEqual(len(key_list), 2)
        self.assertIn("test", key_list)
        self.assertIn("test_str", key_list)

        shutil.rmtree(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
