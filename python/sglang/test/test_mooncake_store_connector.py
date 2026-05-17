import unittest

from sglang.srt.connector.mooncake_store import MooncakeStoreConnector


class FakeStore:
    def __init__(self):
        self.data = {}

    def get(self, key):
        return self.data.get(key, b"")

    def put(self, key, value, config=None):
        self.data[key] = value
        return 0

    def is_exist(self, key):
        return int(key in self.data)


class TestMooncakeStoreConnector(unittest.TestCase):
    def setUp(self):
        self.connector = object.__new__(MooncakeStoreConnector)
        self.connector.closed = True
        self.connector.store = FakeStore()
        self.connector._rep_config = object()

    def test_getstr_missing_key_returns_none(self):
        self.assertIsNone(self.connector.getstr("missing"))

    def test_getstr_existing_empty_value_returns_empty_string(self):
        self.connector.store.put("empty", b"")
        self.assertEqual(self.connector.getstr("empty"), "")

    def test_setstr_registers_index_for_existing_file(self):
        self.connector.setstr("model/files/config.json", '{"model_type":"opt"}')
        self.connector.setstr("model/files/tokenizer_config.json", "{}")

        self.assertEqual(
            self.connector.list("model/files/"),
            [
                "model/files/config.json",
                "model/files/tokenizer_config.json",
            ],
        )


if __name__ == "__main__":
    unittest.main()